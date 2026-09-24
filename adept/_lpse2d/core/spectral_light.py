"""Spectral (k-space) light-wave propagators for the Envelope-2D solver.

This is the original LPSE ``{laser|raman}.solver = spectral`` path
(``LightSolver::evolveSpectral``). Per light sub-step, in LPSE's order:

1. injector source and the x-space split step of the scattering potential
   ``E *= exp(i dt Vo)`` -- ``Vo = w/2 (1 - wpe^2/w^2 n_tot/n_env)`` plus, with
   ``terms.light.absorption``, the imaginary part ``nu_abs (n/nc_w)^2`` (collisional
   absorption; ``LightSolver::calculateScatteringPotential``);
2. the EPW coupling terms added with forward Euler in x-space;
3. the exact k-space propagator ``exp(-i dt c^2 k^2/(2 w))`` applied to the
   transverse part of the field only (``k k/k^2`` longitudinal projector; the
   longitudinal part is left unpropagated, ``LightSolver.cpp:3684-3743``), with
   modes outside the retained band zeroed;
4. the absorbing layers.

Compared with the staggered finite-difference scheme of ``raman.py``/``light.py``
this has no CFL limit (``grid.light_substeps`` defaults to 1) and no grid
dispersion, so the group velocity at 7-8 cells per wavelength is exact and the
staggered scheme's exchange instability cannot occur. With the pump evolved the
SRS exchange between E0 and E1 is integrated with the exact local rotation of
``CoupledLight.couple`` on either side of the propagation (Strang), which is what
``terms.light.coupling: rotation`` does on the FD path.

The injector is a smooth (Gaussian, ``injector_width`` wide, half a local
wavelength by default as LPSE's ``injectorWidth``) source ``S(x) = A v_g g(x) e^{+-i k x}``, ``int g dx = 1``,
which by ``dA/dx = S/v_g`` launches the amplitude ``A`` in the chosen direction
(to 0.2 % at half a wavelength: the paraxial correction grows with the width) and a
leakage ``~exp(-2 k^2 sigma^2)`` in the other -- the smooth analogue of LPSE's
sigmoid spectral injector (``laser.spectral.injectorWidth``). The FD two-point
injector is not usable here: its calibration relies on the FD Green's function.
Keep the injector clear of the absorbing layer (its skirt damps the source region):
the seed's default offset of 1.6 boundary widths sits inside the tanh skirt, which
the FD path tolerates by design; with this solver use ``drivers.E1.offset`` of at
least ``2 * boundary_width``.
"""

import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._lpse2d.core.light import CoupledLight, super_gaussian_y
from adept._lpse2d.core.raman import RamanLight, transverse_part
from adept._lpse2d.core.timeline import as_linear
from adept._lpse2d.core.vector import fft2c, ifft2c, split_k, with_components


def gaussian_injector_profile(x: np.ndarray, x_inject: float, width: float, dx: float) -> np.ndarray:
    """Unit-integral (sum * dx = 1) Gaussian source profile centered at ``x_inject``."""
    g = np.exp(-0.5 * ((np.asarray(x) - x_inject) / width) ** 2)
    return g / (np.sum(g) * dx)


def transverse_propagate(
    field: Array,
    kx: Array,
    ky: Array,
    one_over_k_sq: Array,
    propagator: Array,
    band: Array | float = 1.0,
    keep_longitudinal: bool = True,
) -> Array:
    """Apply ``propagator`` (nx, ny) to the transverse part of ``field`` (nx, ny, nc) in k-space
    and return the result in x-space. The longitudinal part is not propagated; it is kept
    inside the retained light ``band`` (LPSE zeroes every component outside the band and the
    anti-aliasing region, ``LightSolver.cpp:3720-3735``) or dropped altogether when
    ``keep_longitudinal`` is False. A z component (k_z = 0) is entirely transverse."""
    longitudinal_k, transverse_k = split_k(fft2c(field), kx, ky, one_over_k_sq)
    l_factor = (jnp.asarray(band)[..., None] if jnp.ndim(band) else band) if keep_longitudinal else 0.0
    return ifft2c(l_factor * longitudinal_k + propagator[..., None] * transverse_k)


class SpectralRamanLight(RamanLight):
    """Raman light E1 with a prescribed pump, spectral propagator."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        grid = cfg["grid"]
        self.kx_arr = jnp.asarray(grid["kx"])
        self.ky_arr = jnp.asarray(grid["ky"])
        k_sq = np.asarray(grid["kx"])[:, None] ** 2 + np.asarray(grid["ky"])[None, :] ** 2
        self.one_over_k_sq = jnp.asarray(np.where(k_sq > 0, 1.0 / np.where(k_sq > 0, k_sq, 1.0), 0.0))
        self.light_band = jnp.asarray(self._light_band(cfg, k_sq))
        self.propagator1 = jnp.exp(-1j * self.dt_l * self.c**2 / (2.0 * self.w1) * jnp.asarray(k_sq)) * self.light_band
        # x-space scattering potential over one sub-step (detuning); the IAW part and the
        # absorption are applied per call
        self.detune1 = jnp.exp(self.dt_l * self.linear_coeff)
        if self.seed_enabled:
            seed = cfg["drivers"]["E1"]["derived"]
            width = seed.get("injector_width", np.pi / self.k1_inject)
            self.seed_profile = jnp.asarray(
                gaussian_injector_profile(np.asarray(self.x), float(self.x[self.i1]), width, self.dx)
            )

    @staticmethod
    def _light_band(cfg: dict, k_sq: np.ndarray) -> np.ndarray:
        """Retained light band: the grid's anti-aliasing mask, optionally capped at
        ``terms.light.max_wavenumber * k0`` (LPSE ``{laser|raman}.maxWavenumber``)."""
        band = np.asarray(cfg["grid"]["low_pass_filter_grid"]) > 0.0
        cap = cfg["terms"].get("light", {}).get("max_wavenumber")
        if cap is not None:
            k0 = cfg["units"]["derived"]["w0"] / cfg["units"]["derived"]["c"]
            band = band & (np.sqrt(k_sq) < float(cap) * k0)
        return np.where(band, 1.0, 0.0)

    def calc_seed_source(self, t: float, seed_args: dict) -> Array:
        """Smooth injector for the leftward seed: returns the (nx, ny) source added to E1_y."""
        dw1 = seed_args["delta_omega"]
        turn_on = 1.0 - jnp.exp(-((t / seed_args["turn_on_time"]) ** 2))
        if seed_args["yw"] > 0:
            envelope_y = jnp.exp(-((self.y / (seed_args["yw"] / 2.0)) ** 4))
        else:
            envelope_y = jnp.ones_like(self.y)
        k1 = self.w1 / self.c * jnp.sqrt((1.0 + dw1) ** 2 - self.wpe_sq_i1 / self.w1**2)
        v_g = self.c**2 * k1 / self.w1
        eps1 = 1.0 - self.wpe_sq_i1 / self.w1**2
        amplitude = seed_args["amplitude"] / eps1**0.25 * turn_on
        carrier = jnp.exp(-1j * k1 * (self.x - self.x[self.i1]) - 1j * self.w1 * dw1 * t)
        return (amplitude * v_g * self.seed_profile * carrier)[:, None] * envelope_y[None, :]

    def absorption_factor(self, rate, density_ratio, iaw_density, feedback=None):
        """``exp(-nu dt_l (n/nc_w)^2)`` with the IAW perturbation included (``feedback``: the wave's
        ``iaw_feedback_factor``, the Raman light's by default)."""
        if rate is None:
            return 1.0
        feedback = self.iaw_feedback if feedback is None else feedback
        if iaw_density is None:
            n_over_nc = density_ratio
        else:
            n_over_nc = density_ratio * (1.0 + iaw_density * feedback / self.n_over_env)
        return jnp.exp(-rate * self.dt_l * n_over_nc**2)[..., None]

    def __call__(self, t, E1, E0_fn, phi_k, seed_args, iaw_density=None):
        seed_args = seed_args if self.seed_enabled else None
        # phi_k / iaw_density: arrays (fixed) or timeline.Linear (read at each sub-step's middle)
        laplacian_tl = as_linear(phi_k).map(lambda p: jnp.fft.ifft2(-self.k_sq * p))
        iaw_tl = as_linear(iaw_density)

        def scattering(dn):
            detune = self.detune1
            if dn is not None:
                detune = detune * jnp.exp(-1j * self.wp0**2 / (2.0 * self.w1) * dn * self.iaw_feedback * self.dt_l)
            return detune[..., None] * self.absorption_factor(self.absorption_rate1, self.n_over_nc1, dn)

        fixed = scattering(iaw_tl.new) if iaw_tl.constant else None

        def substep(i, E1):
            t_i = t + i * self.dt_l
            laplacian_phi = laplacian_tl.substep(i, self.n_sub)
            # x-space: scattering potential (detuning + absorption), then the sources (Euler)
            E1 = E1 * (fixed if fixed is not None else scattering(iaw_tl.substep(i, self.n_sub)))
            E0 = E0_fn(t_i)
            coupling = self.srs_coeff * jnp.conj(laplacian_phi)[..., None] * E0
            if self.transverse_source:
                coupling = transverse_part(coupling, self.kx_arr, self.ky_arr, self.one_over_k_sq)
            E1 = E1 + self.dt_l * coupling
            if seed_args is not None:
                E1 = self.add_seed(E1, self.dt_l * self.calc_seed_source(t_i, seed_args))
            # k-space: exact transverse propagation; the longitudinal part is kept only
            # inside the light band
            E1 = transverse_propagate(
                E1, self.kx_arr, self.ky_arr, self.one_over_k_sq, self.propagator1, self.light_band
            )
            return E1 * self.sub_boundary[..., None]

        return lax.fori_loop(0, self.n_sub, substep, E1)


class SpectralCoupledLight(CoupledLight):
    """Evolved pump E0 and Raman light E1, spectral propagators, exact SRS exchange."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        grid = cfg["grid"]
        self.kx_arr = jnp.asarray(grid["kx"])
        self.ky_arr = jnp.asarray(grid["ky"])
        k_sq = np.asarray(grid["kx"])[:, None] ** 2 + np.asarray(grid["ky"])[None, :] ** 2
        self.one_over_k_sq = jnp.asarray(np.where(k_sq > 0, 1.0 / np.where(k_sq > 0, k_sq, 1.0), 0.0))
        self.light_band = jnp.asarray(SpectralRamanLight._light_band(cfg, k_sq))
        self.propagator0 = jnp.exp(-1j * self.dt_l * self.c**2 / (2.0 * self.w0) * jnp.asarray(k_sq)) * self.light_band
        self.propagator1 = jnp.exp(-1j * self.dt_l * self.c**2 / (2.0 * self.w1) * jnp.asarray(k_sq)) * self.light_band
        self.detune0 = jnp.exp(self.dt_l * self.linear_coeff0)
        self.detune1 = jnp.exp(self.dt_l * self.linear_coeff)
        pump = cfg["drivers"]["E0"]["derived"]
        # beams with |angle| > 90 deg are launched leftward from the x-max face (plan 2 L.4a):
        # their injector plane sits at xmax - offset, with that plane's density
        leftward = np.atleast_1d(np.asarray(pump.get("beam_leftward", [False]), dtype=bool))
        self.beam_sign = [-1 if left else 1 for left in leftward]
        x_inject_max = cfg["grid"]["xmax"] - pump["offset"]
        self.i0_max = int(np.argmin(np.abs(np.array(self.x) - x_inject_max)))
        self.n_src_max = float(cfg["grid"]["background_density"][self.i0_max, 0])
        if np.any(leftward) and self.n_src_max >= 1.0:
            raise ValueError(
                f"The x-max pump injector at x = {float(self.x[self.i0_max]):.2f} um sits at density "
                f"{self.n_src_max:.3f} nc, at or above critical."
            )
        self.beam_i_inject = [self.i0_max if left else self.i0 for left in leftward]
        self.beam_n_src = [self.n_src_max if left else self.n_src for left in leftward]
        k0_inject = self.w0 / self.c * np.sqrt(1.0 - self.n_src)
        # in-plane angle of incidence (drivers.E0.angle): the transverse wavenumber is snapped
        # to the periodic y grid (terms.light.snap_beam_ky, default), kx follows from the local
        # dispersion per color, and the field is polarised perpendicular to k (LPSE beam
        # direction / polarization 0). With snap_beam_ky false the injector carries the exact
        # k0 sin(angle) as LPSE's spectral injector does: on a box whose width does not fit
        # that wavelength the source has a phase kink at the periodic seam whose sidebands add
        # up to an intensity hot spot there (test_010: +45 %, plan-2 N.1) -- kept as an option
        # to reproduce such a reference run, not as a default.
        dky = 2.0 * np.pi / (cfg["grid"]["ny"] * cfg["grid"]["dy"])
        angles = np.atleast_1d(np.asarray(pump.get("beam_angle", [pump.get("angle", 0.0)]), dtype=np.float64))
        snap = bool(cfg["terms"].get("light", {}).get("snap_beam_ky", True))
        k0_beams = self.w0 / self.c * np.sqrt(1.0 - np.asarray(self.beam_n_src, dtype=np.float64))
        if cfg["grid"]["ny"] > 1:
            ky_beams = k0_beams * np.sin(angles)
            if snap:
                ky_beams = np.round(ky_beams / dky) * dky
        else:
            ky_beams = np.zeros_like(angles)
        if np.any(np.abs(ky_beams) >= k0_beams):
            raise ValueError(
                f"drivers.E0 beam angles {np.rad2deg(angles)} deg cannot be launched at densities {self.beam_n_src}"
            )
        self.ky_beams = jnp.asarray(ky_beams)
        self.ky_pump = float(ky_beams[0])
        self.beam_fraction = jnp.asarray(np.atleast_1d(pump.get("beam_fraction", [1.0])), dtype=jnp.float64)
        self.beam_phase = jnp.asarray(np.atleast_1d(pump.get("beam_phase", [0.0])), dtype=jnp.float64)
        self.beam_delta_omega = jnp.asarray(np.atleast_1d(pump.get("beam_delta_omega", [0.0])), dtype=jnp.float64)
        # per-beam polarization angle (rad) about the beam axis, drivers.E0.polarization / beams[].polarization
        self.beam_polarization = jnp.asarray(
            np.atleast_1d(pump.get("beam_polarization", [pump.get("polarization", 0.0)])), dtype=jnp.float64
        )
        self.y_arr = jnp.asarray(cfg["grid"]["y"])
        # transverse super-Gaussian of the injected beams (LPSE laser.N.width / sgOrder / offset)
        width = float(pump.get("beam_width", 0.0) or 0.0)
        if cfg["grid"]["ny"] > 1:
            self.beam_envelope_y = jnp.asarray(
                super_gaussian_y(
                    cfg["grid"]["y"], width, float(pump.get("beam_sg_order", 4.0)), float(pump.get("beam_offset", 0.0))
                )
            )
        else:
            self.beam_envelope_y = jnp.ones(cfg["grid"]["ny"])
        # Kubo-Anderson bandwidth: piecewise-constant random phase with correlation time 2 pi / (dW)
        self.kap_bandwidth = float(pump.get("kap_bandwidth", 0.0) or 0.0)
        self.kap_seed = int(pump.get("kap_seed", 0) or 0)
        width = pump.get("injector_width", np.pi / k0_inject)
        self.pump_profile = jnp.asarray(
            gaussian_injector_profile(np.asarray(self.x), float(self.x[self.i0]), width, self.dx)
        )
        width_max = pump.get("injector_width", np.pi / (self.w0 / self.c * np.sqrt(max(1.0 - self.n_src_max, 1e-12))))
        self.pump_profile_max = jnp.asarray(
            gaussian_injector_profile(np.asarray(self.x), float(self.x[self.i0_max]), width_max, self.dx)
        )
        if self.seed_enabled:
            seed = cfg["drivers"]["E1"]["derived"]
            width1 = seed.get("injector_width", np.pi / self.k1_inject)
            self.seed_profile = jnp.asarray(
                gaussian_injector_profile(np.asarray(self.x), float(self.x[self.i1]), width1, self.dx)
            )

    # the smooth injectors
    calc_seed_source = SpectralRamanLight.calc_seed_source
    absorption_factor = SpectralRamanLight.absorption_factor

    def calc_pump_source(self, t: float, pump_args: dict) -> Array:
        """Smooth injector for the (optionally oblique) pump beams, summed over colors: the
        (nx, ny, 3) source added to E0 (the FD two-point rows are replaced). Beams with
        |angle| > 90 deg are launched leftward from the x-max plane."""
        time_factor = self.pump_time_factor(t, pump_args)
        delta_omega = pump_args["delta_omega"]  # (nc,)
        intensities = pump_args["intensities"]  # (nc, ny)
        phases = pump_args["phases"]  # (nc, ny)
        amp = self.E0_source * jnp.sqrt(intensities) * time_factor  # (nc, ny)
        amp = amp * self.beam_envelope_y[None, :]
        color_phase = jnp.exp(-1j * self.w0 * delta_omega[:, None] * t + 1j * phases)  # (nc, ny)
        total = jnp.zeros((self.x.shape[0], self.y_arr.shape[0], 3), dtype=jnp.complex128)
        for b in range(int(self.ky_beams.shape[0])):
            ky_b = self.ky_beams[b]
            dw_b = self.beam_delta_omega[b]
            sign, n_src, i_inject = self.beam_sign[b], self.beam_n_src[b], self.beam_i_inject[b]
            profile = self.pump_profile_max if sign < 0 else self.pump_profile
            k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega + dw_b) ** 2 - n_src)  # (nc,)
            kx = sign * jnp.sqrt(k0**2 - ky_b**2)  # (nc,) longitudinal wavenumber of each color, signed
            v_gx = self.c**2 * jnp.abs(kx) / self.w0  # flux through the injector plane: the x group velocity
            beam_phase = self.beam_phase[b] - self.w0 * dw_b * t + self.kap_phase(t, b)
            amp_b = amp / (1.0 - n_src) ** 0.25 * jnp.sqrt(self.beam_fraction[b]) * jnp.exp(1j * beam_phase)
            carrier = jnp.exp(1j * kx[:, None] * (self.x[None, :] - self.x[i_inject]))  # (nc, nx)
            source = (amp_b * v_gx[:, None] * color_phase)[:, None, :] * (carrier * profile[None, :])[:, :, None]
            source = jnp.sum(source, axis=0) * jnp.exp(1j * ky_b * self.y_arr)[None, :]  # (nx, ny)
            k_mag = jnp.sqrt(kx[0] ** 2 + ky_b**2)
            # LPSE rotateBeam: cos(psi) along the in-plane transverse direction of the (first
            # color's) k, sin(psi) along z
            cos_psi, sin_psi = jnp.cos(self.beam_polarization[b]), jnp.sin(self.beam_polarization[b])
            pol = jnp.stack([-ky_b / k_mag * cos_psi, kx[0] / k_mag * cos_psi, sin_psi])
            total = total + source[..., None] * pol[None, None, :]
        return total

    def pump_source_for(self, E0: Array, t: float, pump_args: dict) -> Array:
        """``calc_pump_source`` with the component count of ``E0`` (refusing to drop a
        z-polarised beam into a two-component field)."""
        source = self.calc_pump_source(t, pump_args)
        if E0.shape[-1] == 2 and bool(np.any(np.sin(np.asarray(self.beam_polarization)) != 0.0)):
            raise ValueError("an out-of-plane (s-polarised) pump needs three-component light fields")
        return with_components(source, E0.shape[-1])

    def kap_phase(self, t, beam: int):
        """Kubo-Anderson process (LPSE laser.N.bandwidth.KAP.frequency): random dwell times per beam,
        2 Exp(1) / (bandwidth w0), each followed by a new uniform phase (core/kap.py)."""
        return self.kap.phase(t, beam)

    def __call__(self, t, E0, E1, phi_k, pump_args, seed_args, iaw_density=None):
        seed_args = seed_args if self.seed_enabled else None
        # phi_k / iaw_density: arrays (fixed) or timeline.Linear (read at each sub-step's middle)
        phi_tl = as_linear(phi_k)
        laplacian_tl = phi_tl.map(lambda p: jnp.fft.ifft2(-self.k_sq * p))
        iaw_tl = as_linear(iaw_density)

        def scattering(dn):
            detune0, detune1 = self.detune0, self.detune1
            if dn is not None:
                # in units of n_env (LPSE Nelf * n_b / No), per wave (ionAcousticPerturbations)
                dn0, dn1 = dn * self.iaw_feedback0, dn * self.iaw_feedback
                detune0 = detune0 * jnp.exp(-1j * self.wp0**2 / (2.0 * self.w0) * dn0 * self.dt_l)
                detune1 = detune1 * jnp.exp(-1j * self.wp0**2 / (2.0 * self.w1) * dn1 * self.dt_l)
            absorb0 = self.absorption_factor(self.absorption_rate0, self.n_over_nc0, dn, self.iaw_feedback0)
            absorb1 = self.absorption_factor(self.absorption_rate1, self.n_over_nc1, dn)
            return detune0[..., None] * absorb0, detune1[..., None] * absorb1

        fixed = scattering(iaw_tl.new) if iaw_tl.constant else None
        exchange = self.srs_enabled

        def substep(i, fields):
            E0, E1 = fields
            t_i = t + i * self.dt_l
            laplacian_phi = laplacian_tl.substep(i, self.n_sub)
            phi_i = phi_tl.substep(i, self.n_sub)
            if exchange:
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
            # x-space: scattering potentials, then the non-exchange sources (Euler)
            s0, s1 = fixed if fixed is not None else scattering(iaw_tl.substep(i, self.n_sub))
            E0 = E0 * s0
            E1 = E1 * s1
            if self.tpd_enabled:
                E0 = E0 + self.dt_l * with_components(self.calc_tpd_depletion(t_i, phi_i), E0.shape[-1])
            E0 = E0 + self.dt_l * self.pump_source_for(E0, t_i, pump_args)
            if seed_args is not None:
                E1 = self.add_seed(E1, self.dt_l * self.calc_seed_source(t_i, seed_args))
            # k-space: exact transverse propagation. The exchange is an exact local rotation
            # that cannot be projected term by term, so with transverse_source the fields
            # themselves are kept transverse here (equivalent to projecting every source)
            keep_l = not self.transverse_source
            E0 = transverse_propagate(
                E0, self.kx_arr, self.ky_arr, self.one_over_k_sq, self.propagator0, self.light_band, keep_l
            )
            if exchange:
                E1 = transverse_propagate(
                    E1, self.kx_arr, self.ky_arr, self.one_over_k_sq, self.propagator1, self.light_band, keep_l
                )
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
            E0 = E0 * self.sub_boundary0[..., None]
            E1 = E1 * self.sub_boundary[..., None]
            return (E0, E1)

        E0, E1 = lax.fori_loop(0, self.n_sub, substep, (E0, E1))
        if self.light_filter is not None:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.light_filter, axes=(0, 1))
            E1 = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.light_filter, axes=(0, 1))
        return E0, E1
