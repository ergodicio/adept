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

from adept._base_ import get_envelope
from adept._lpse2d.core.light import CoupledLight
from adept._lpse2d.core.raman import RamanLight, transverse_part


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
    """Apply ``propagator`` (nx, ny) to the transverse part of ``field`` (nx, ny, 2) in k-space
    and return the result in x-space. The longitudinal part is not propagated; it is kept
    inside the retained light ``band`` (LPSE zeroes every component outside the band and the
    anti-aliasing region, ``LightSolver.cpp:3720-3735``) or dropped altogether when
    ``keep_longitudinal`` is False."""
    fx_k = jnp.fft.fft2(field[..., 0])
    fy_k = jnp.fft.fft2(field[..., 1])
    kdote = (kx[:, None] * fx_k + ky[None, :] * fy_k) * one_over_k_sq
    lx_k, ly_k = kx[:, None] * kdote, ky[None, :] * kdote
    tx_k, ty_k = fx_k - lx_k, fy_k - ly_k
    l_factor = band if keep_longitudinal else 0.0
    fx_k = l_factor * lx_k + propagator * tx_k
    fy_k = l_factor * ly_k + propagator * ty_k
    return jnp.stack([jnp.fft.ifft2(fx_k), jnp.fft.ifft2(fy_k)], axis=-1)


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

    def absorption_factor(self, rate, density_ratio, iaw_density):
        """``exp(-nu dt_l (n/nc_w)^2)`` with the IAW perturbation included."""
        if rate is None:
            return 1.0
        n_over_nc = density_ratio if iaw_density is None else density_ratio * (1.0 + iaw_density / self.n_over_env)
        return jnp.exp(-rate * self.dt_l * n_over_nc**2)[..., None]

    def __call__(self, t, E1, E0_fn, phi_k, seed_args, iaw_density=None):
        seed_args = seed_args if self.seed_enabled else None
        laplacian_phi = jnp.fft.ifft2(-self.k_sq * phi_k)
        detune = self.detune1
        if iaw_density is not None:
            detune = detune * jnp.exp(-1j * self.wp0**2 / (2.0 * self.w1) * iaw_density * self.dt_l)
        absorb = self.absorption_factor(self.absorption_rate1, self.n_over_nc1, iaw_density)

        def substep(i, E1):
            t_i = t + i * self.dt_l
            # x-space: scattering potential (detuning + absorption), then the sources (Euler)
            E1 = E1 * detune[..., None] * absorb
            E0 = E0_fn(t_i)
            coupling = self.srs_coeff * jnp.conj(laplacian_phi)[..., None] * E0
            if self.transverse_source:
                coupling = transverse_part(coupling, self.kx_arr, self.ky_arr, self.one_over_k_sq)
            E1 = E1 + self.dt_l * coupling
            if seed_args is not None:
                E1 = E1.at[..., 1].add(self.dt_l * self.calc_seed_source(t_i, seed_args))
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
        k0_inject = self.w0 / self.c * np.sqrt(1.0 - self.n_src)
        # in-plane angle of incidence (drivers.E0.angle): the transverse wavenumber is snapped
        # to the periodic y grid, kx follows from the local dispersion per color, and the
        # field is polarised perpendicular to the snapped k (LPSE beam direction / polarization 0)
        angle = float(pump.get("angle", 0.0))
        dky = 2.0 * np.pi / (cfg["grid"]["ny"] * cfg["grid"]["dy"])
        self.ky_pump = float(np.round(k0_inject * np.sin(angle) / dky) * dky) if cfg["grid"]["ny"] > 1 else 0.0
        if abs(self.ky_pump) >= k0_inject:
            raise ValueError(
                f"drivers.E0.angle = {np.rad2deg(angle):.1f} deg cannot be launched at density {self.n_src}"
            )
        self.y_arr = jnp.asarray(cfg["grid"]["y"])
        width = pump.get("injector_width", np.pi / k0_inject)
        self.pump_profile = jnp.asarray(
            gaussian_injector_profile(np.asarray(self.x), float(self.x[self.i0]), width, self.dx)
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
        """Smooth injector for the rightward (optionally oblique) pump, summed over colors:
        the (nx, ny, 2) source added to E0 (the FD two-point rows are replaced)."""
        t_env = get_envelope(
            pump_args["tr"],
            pump_args["tr"],
            pump_args["tc"] - pump_args["tw"] / 2,
            pump_args["tc"] + pump_args["tw"] / 2,
            t,
        )
        turn_on = 1.0 - jnp.exp(-((t / self.pump_turn_on_time) ** 2))
        delta_omega = pump_args["delta_omega"]  # (nc,)
        intensities = pump_args["intensities"]  # (nc, ny)
        phases = pump_args["phases"]  # (nc, ny)
        k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega) ** 2 - self.n_src)  # (nc,)
        kx = jnp.sqrt(k0**2 - self.ky_pump**2)  # (nc,) longitudinal wavenumber of each color
        v_gx = self.c**2 * kx / self.w0  # flux through the injector plane is set by the x group velocity
        eps0 = 1.0 - self.n_src
        amp = self.E0_source * jnp.sqrt(intensities) / eps0**0.25 * t_env * turn_on  # (nc, ny)
        color_phase = jnp.exp(-1j * self.w0 * delta_omega[:, None] * t + 1j * phases)  # (nc, ny)
        carrier = jnp.exp(1j * kx[:, None] * (self.x[None, :] - self.x[self.i0]))  # (nc, nx)
        source = (amp * v_gx[:, None] * color_phase)[:, None, :] * (carrier * self.pump_profile[None, :])[:, :, None]
        source = jnp.sum(source, axis=0) * jnp.exp(1j * self.ky_pump * self.y_arr)[None, :]  # (nx, ny)
        k_mag = jnp.sqrt(kx[0] ** 2 + self.ky_pump**2)
        pol = jnp.stack([-self.ky_pump / k_mag, kx[0] / k_mag])  # perpendicular to the (first color's) k
        return source[..., None] * pol[None, None, :]

    def __call__(self, t, E0, E1, phi_k, pump_args, seed_args, iaw_density=None):
        seed_args = seed_args if self.seed_enabled else None
        laplacian_phi = jnp.fft.ifft2(-self.k_sq * phi_k)
        detune0, detune1 = self.detune0, self.detune1
        if iaw_density is not None:
            detune0 = detune0 * jnp.exp(-1j * self.wp0**2 / (2.0 * self.w0) * iaw_density * self.dt_l)
            detune1 = detune1 * jnp.exp(-1j * self.wp0**2 / (2.0 * self.w1) * iaw_density * self.dt_l)
        absorb0 = self.absorption_factor(self.absorption_rate0, self.n_over_nc0, iaw_density)
        absorb1 = self.absorption_factor(self.absorption_rate1, self.n_over_nc1, iaw_density)
        exchange = self.srs_enabled

        def substep(i, fields):
            E0, E1 = fields
            t_i = t + i * self.dt_l
            if exchange:
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
            # x-space: scattering potentials, then the non-exchange sources (Euler)
            E0 = E0 * detune0[..., None] * absorb0
            E1 = E1 * detune1[..., None] * absorb1
            if self.tpd_enabled:
                E0 = E0 + self.dt_l * self.calc_tpd_depletion(t_i, phi_k)
            E0 = E0 + self.dt_l * self.calc_pump_source(t_i, pump_args)
            if seed_args is not None:
                E1 = E1.at[..., 1].add(self.dt_l * self.calc_seed_source(t_i, seed_args))
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
            E0 = E0 * self.sub_boundary[..., None]
            E1 = E1 * self.sub_boundary[..., None]
            return (E0, E1)

        E0, E1 = lax.fori_loop(0, self.n_sub, substep, (E0, E1))
        if self.light_filter is not None:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.light_filter, axes=(0, 1))
            E1 = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.light_filter, axes=(0, 1))
        return E0, E1
