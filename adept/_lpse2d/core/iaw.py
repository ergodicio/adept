"""Ion-acoustic-wave evolution for the Envelope-2D solver.

The real state variables are the fractional ion density perturbation ``n``
(MATLAB ``Nelf``) and the ion-velocity divergence ``w`` (MATLAB ``W``):

    d w / dt = -laplacian(PP)
    d n / dt = -w

where the ponderomotive potential is

    PP = cs^2 n + Z e^2/(4 me mi) [|E_epw|^2/wp0^2
                                    + |E0|^2/w0^2
                                    + |E1|^2/w1^2].

Two time integrators are available (``terms.iaw.solver``):

``explicit`` (the ``isEvolveIaw`` split-step path of ``m201805_matlabLpse_v11.m``):
kick ``w`` from the potential with the FD Laplacian, apply twice the amplitude
Landau rate to ``w``, drift ``n``; boundary damping acts on ``w`` and collisions
act on ``n`` after the split step. Conditionally stable, ``omega_max dt < 2``.

``spectral`` (the original LPSE ``iaw.solver = spectral`` path,
``ZakharovSolver.cpp`` lines 2083-2113): per k-mode the exact solution over
``dt`` of the damped oscillator

    dn/dt = -w,   dw/dt = cs^2 k^2 n - 2 gamma_k w

with ``gamma_k`` the amplitude Landau rate, i.e. with ``beta_k = sqrt(cs^2 k^2 - gamma_k^2)``

    n'  = e^{-gamma dt} [ cos(beta dt) n + sin(beta dt)/beta (gamma n - w) ]
    w'  = e^{-gamma dt} [ cos(beta dt) w + sin(beta dt)/beta (cs^2 k^2 n - gamma w) ]

times the Doppler phase ``e^{-i k . V0 dt}`` of a uniform background flow ``V0``,
followed by the split-step pieces LPSE applies in the same order: collisional
damping ``w' *= e^{-2 nu dt}`` (on the velocity divergence, not on ``n``), the
ponderomotive kick ``w' += dt k^2 PP_k`` (``PP`` without its ``cs^2 n`` part, which
is inside the propagator), and the fluctuation-dissipation noise source on ``w``.
Unconditionally stable and exact for the linear acoustic part at any ``dt``, so
the IAW may also be advanced every ``terms.iaw.stride`` EPW steps as LPSE does.
"""

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

IAW_SOLVERS = ("explicit", "spectral")
ION_LANDAU_FORMS = ("simplified", "full")


def ion_landau_rate(cfg: dict, k_sq: np.ndarray) -> np.ndarray:
    """Amplitude ion Landau damping rate ``gamma_k`` (1/ps) for every k mode.

    ``terms.iaw.damping.landau_form: simplified`` (default; LPSE ``isSimplified``):
    ``gamma = landau * cs * |k|``.

    ``full``: the Z-generalized Krall & Trivelpiece expression (p. 390) used by LPSE's
    IAW solver (``IawSolver.cpp``, modified by J. Palastro for the effective ionization
    state), with ``eta = 1 + 3 Ti/(Z Te)``, ``M = (mi/me)/(eta Z)``, ``L = 1 + k^2 lambda_D^2``:

        W_i = W_r sqrt(pi/8) L^{-3/2} [ (3/(eta-1))^{3/2} exp(-(3/(eta-1))/(2L)) + sqrt(1/(eta M)) ],
        W_r = cs |k| / sqrt(L),

    ``W_i`` being the damping rate of the velocity divergence, twice the amplitude rate.
    """
    derived = cfg["units"]["derived"]
    iaw = cfg["terms"]["iaw"]
    damping = iaw["damping"]
    cs = derived["cs"]
    k = np.sqrt(np.asarray(k_sq))
    form = str(damping.get("landau_form", "simplified"))
    if form == "simplified":
        return float(damping["landau"]) * cs * k
    if form != "full":
        raise ValueError(f"terms.iaw.damping.landau_form must be one of {ION_LANDAU_FORMS}, got {form!r}")

    from astropy.units import Quantity as _Q

    te = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
    ti = _Q(cfg["units"]["reference ion temperature"]).to("keV").value
    z = float(cfg["units"]["ionization state"])
    mi_over_me = derived["mi"] / derived["me"]
    eta = 1.0 + 3.0 * ti / (z * te)
    big_m = mi_over_me / (eta * z)
    lambda_d_sq = derived["vte_sq"] / derived["wp0"] ** 2
    big_l = 1.0 + k_sq * lambda_d_sq
    w_r = cs * k / np.sqrt(big_l)
    three_over = 3.0 / (eta - 1.0)
    w_i = (
        w_r
        * np.sqrt(np.pi / 8.0)
        / big_l**1.5
        * (three_over**1.5 * np.exp(-0.5 * three_over / big_l) + np.sqrt(1.0 / (eta * big_m)))
    )
    return 0.5 * w_i


class IonAcousticWave:
    """Advance the ion-acoustic density and velocity-divergence fields."""

    def __init__(self, cfg: dict):
        grid = cfg["grid"]
        derived = cfg["units"]["derived"]
        iaw = cfg["terms"]["iaw"]

        self.solver = str(iaw.get("solver", "explicit"))
        if self.solver not in IAW_SOLVERS:
            raise ValueError(f"terms.iaw.solver must be one of {IAW_SOLVERS}, got {self.solver!r}")
        self.stride = int(iaw.get("stride", 1) or 1)
        if self.stride < 1:
            raise ValueError("terms.iaw.stride must be a positive integer")
        if self.stride > 1 and self.solver != "spectral":
            raise ValueError(
                "terms.iaw.stride > 1 requires terms.iaw.solver: spectral (the explicit step is not stable)"
            )

        self.dt = grid["dt"] * self.stride
        self.dx = grid["dx"]
        self.dy = grid["dy"]
        self.nx = grid["nx"]
        self.ny = grid["ny"]
        self.kx = grid["kx"]
        self.ky = grid["ky"]
        self.k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2
        self.filter = grid["low_pass_filter_grid"] * grid["zero_mask"]
        self.boundary = grid["iaw_absorbing_boundaries"]

        self.cs = derived["cs"]
        self.wp0 = derived["wp0"]
        self.w0 = derived["w0"]
        self.w1 = derived["w1"]
        self.ponderomotive_prefactor = (
            cfg["units"]["ionization state"] * derived["e"] ** 2 / (4.0 * derived["me"] * derived["mi"])
        )

        damping = iaw["damping"]
        k_sq_np = np.asarray(self.k_sq)
        self.landau_rate = jnp.asarray(ion_landau_rate(cfg, k_sq_np))
        self.nu_coll = float(damping["collisions"])
        # explicit (MATLAB): collisions act on n as (1 - nu dt); spectral (LPSE): on w as e^{-2 nu dt}
        self.collisional_factor = 1.0 - self.nu_coll * self.dt
        self.max_density_perturbation = iaw["max_density_perturbation"]

        # uniform background flow (Mach number along x, y) -> Doppler phase in the propagator
        flow = iaw.get("flow")
        if flow is None:
            self.flow = None
        else:
            if self.solver != "spectral":
                raise ValueError("terms.iaw.flow requires terms.iaw.solver: spectral")
            self.flow = np.asarray(flow, dtype=np.float64) * self.cs
            if self.flow.shape != (2,):
                raise ValueError("terms.iaw.flow must be [Mach_x, Mach_y]")

        if self.solver == "spectral":
            gamma = np.asarray(self.landau_rate)
            omega_sq = self.cs**2 * k_sq_np
            # beta may be imaginary for over-damped modes (gamma > cs k): the complex
            # cos/sin form below is the exact solution in both regimes
            beta = np.sqrt((omega_sq - gamma**2).astype(np.complex128))
            beta_safe = np.where(np.abs(beta) > 0.0, beta, 1.0)
            cos_b = np.cos(beta * self.dt)
            sin_over_b = np.where(np.abs(beta) > 0.0, np.sin(beta * self.dt) / beta_safe, self.dt)
            phase = -1j * (k_sq_np * 0.0)
            if self.flow is not None:
                kx_np, ky_np = np.asarray(self.kx)[:, None], np.asarray(self.ky)[None, :]
                phase = -1j * (kx_np * self.flow[0] + ky_np * self.flow[1])
            coeff = np.exp((-gamma + phase) * self.dt)
            # propagator matrix on (n_k, w_k)
            self.p_nn = jnp.asarray(coeff * (cos_b + sin_over_b * gamma))
            self.p_nw = jnp.asarray(-coeff * sin_over_b)
            self.p_wn = jnp.asarray(coeff * sin_over_b * omega_sq)
            self.p_ww = jnp.asarray(coeff * (cos_b - sin_over_b * gamma))
            self.collisional_w_factor = float(np.exp(-2.0 * self.nu_coll * self.dt))

        # IAW noise (LPSE IawSolver::addNoise): random-phase source on the velocity
        # divergence with the fluctuation-dissipation amplitude
        #     A N sqrt(exp(2 dt (gamma_k + nu)) - 1)
        # on the retained band; LPSE refuses it without damping
        self.noise_enabled = bool(iaw.get("noise", False))
        if self.noise_enabled:
            amplitude = float(iaw.get("noise_amplitude", 1.0))
            total = np.asarray(self.landau_rate) + self.nu_coll
            band = np.asarray(self.filter) > 0.0
            if not np.any(total[band] > 0.0):
                raise ValueError("terms.iaw.noise requires IAW damping (landau and/or collisions)")
            kick = amplitude * float(self.nx * self.ny) * np.sqrt(np.expm1(2.0 * self.dt * total))
            self.noise_kick = jnp.asarray(np.where(band, kick, 0.0))
            seed = iaw.get("noise_seed")
            self.noise_key = jax.random.PRNGKey(int(seed) if seed is not None else 271828)
        else:
            self.noise_kick = None

    def laplacian(self, field: Array) -> Array:
        """Second-order periodic finite-difference Laplacian used by MATLAB."""
        lap = (jnp.roll(field, -1, axis=0) - 2.0 * field + jnp.roll(field, 1, axis=0)) / self.dx**2
        if self.ny > 1:
            lap = lap + (jnp.roll(field, -1, axis=1) - 2.0 * field + jnp.roll(field, 1, axis=1)) / self.dy**2
        return lap

    def epw_fields(self, phi_k: Array) -> tuple[Array, Array]:
        """Return the real-space EPW electric-field envelopes from ``phi_k``."""
        ex = jnp.fft.ifft2(-1j * self.kx[:, None] * phi_k)
        ey = jnp.fft.ifft2(-1j * self.ky[None, :] * phi_k)
        return ex, ey

    def ponderomotive_drive(self, phi_k: Array, E0: Array, E1: Array) -> Array:
        """The EPW/pump/Raman part of the ponderomotive potential (no acoustic term)."""
        ex, ey = self.epw_fields(phi_k)
        epw_intensity = jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2
        pump_intensity = jnp.sum(jnp.abs(E0) ** 2, axis=-1)
        raman_intensity = jnp.sum(jnp.abs(E1) ** 2, axis=-1)
        return self.ponderomotive_prefactor * (
            epw_intensity / self.wp0**2 + pump_intensity / self.w0**2 + raman_intensity / self.w1**2
        )

    def ponderomotive_potential(self, phi_k: Array, E0: Array, E1: Array, density: Array) -> Array:
        """Build the acoustic plus EPW/pump/Raman ponderomotive potential."""
        return self.cs**2 * density + self.ponderomotive_drive(phi_k, E0, E1)

    def get_noise(self, t: float) -> Array:
        step = jnp.round(t / self.dt).astype(int)
        key = jax.random.fold_in(self.noise_key, step)
        phases = 2.0 * np.pi * jax.random.uniform(key, (self.nx, self.ny))
        return self.noise_kick * jnp.exp(1j * phases)

    def _explicit_step(self, t: float, y: dict[str, Array]) -> tuple[Array, Array]:
        potential = self.ponderomotive_potential(y["epw"], y["E0"], y["E1"], y["iaw_density"])

        velocity_divergence = y["iaw_velocity_divergence"] - self.dt * self.laplacian(potential)
        velocity_k = jnp.fft.fft2(velocity_divergence)
        velocity_k = velocity_k * jnp.exp(-2.0 * self.landau_rate * self.dt) * self.filter
        if self.noise_enabled:
            velocity_k = velocity_k + self.get_noise(t)
        velocity_divergence = jnp.real(jnp.fft.ifft2(velocity_k)) * self.boundary

        density = (y["iaw_density"] - self.dt * velocity_divergence) * self.collisional_factor
        return density, velocity_divergence

    def _spectral_step(self, t: float, y: dict[str, Array]) -> tuple[Array, Array]:
        n_k = jnp.fft.fft2(y["iaw_density"])
        w_k = jnp.fft.fft2(y["iaw_velocity_divergence"])

        # exact damped-oscillator propagator with the flow Doppler phase
        n_new = self.p_nn * n_k + self.p_nw * w_k
        w_new = self.p_wn * n_k + self.p_ww * w_k

        # split-step pieces in LPSE's order: collisions on div v, ponderomotive kick, noise
        w_new = w_new * self.collisional_w_factor
        drive_k = jnp.fft.fft2(self.ponderomotive_drive(y["epw"], y["E0"], y["E1"]))
        w_new = w_new + self.dt * self.k_sq * drive_k
        if self.noise_enabled:
            w_new = w_new + self.get_noise(t)

        n_new = n_new * self.filter
        w_new = w_new * self.filter
        density = jnp.real(jnp.fft.ifft2(n_new))
        velocity_divergence = jnp.real(jnp.fft.ifft2(w_new)) * self.boundary
        return density, velocity_divergence

    def __call__(self, y: dict[str, Array], t: float = 0.0) -> dict[str, Array]:
        """Advance one IAW step (of length ``stride * grid.dt``) and return the full state."""
        if self.solver == "spectral":
            density, velocity_divergence = self._spectral_step(t, y)
        else:
            density, velocity_divergence = self._explicit_step(t, y)

        if self.max_density_perturbation is not None:
            density = jnp.clip(
                density,
                -self.max_density_perturbation,
                self.max_density_perturbation,
            )

        return {
            **y,
            "iaw_density": density,
            "iaw_velocity_divergence": velocity_divergence,
        }
