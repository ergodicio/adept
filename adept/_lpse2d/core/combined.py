"""Combined TPD + SRS solver (``terms.epw.solver: combined``).

This is the original LPSE ``lw.solver = combined`` formulation, which LPSE requires
whenever TPD and SRS are both on (users guide 1.5: the four separate envelope
equations "are not valid for simultaneous TPD and SRS"). One complex vector field
``E1``, enveloped at the reference plasma frequency ``wp0`` (= ``w0 - wp0`` at the
envelope density n_c/4, so the Raman light and the plasma wave share a carrier),
carries the Raman scattered light as its transverse part and the EPW as its
longitudinal part. The potential ``phi_k = i k . E1_k / k^2`` is *derived* from it
every step (``ZakharovSolver::makePotentialFromE_fft``) for the diagnostics, the IAW
ponderomotive drive and the HPE tracker; it is not an independent state variable.

Per light sub-step (``LightSolver::evolveSpectral``, ``LightSolver.cpp:3600-3779``):

1. x-space scattering potential on the whole field, ``exp(-i dt wp0/2 (n_tot/n_env - 1))``
   (the EPW density detuning, identical to the light detuning at carrier ``wp0``), and
   the collisional damping ``exp(-nu_R dt (n/n_env)^2)`` -- LPSE damps the combined field
   with the Raman light's absorption (``raman.evolution.absorption``,
   ``terms.light.raman_absorption``) at the field's own critical density, ``n_env``, and
   ignores ``lw.collisionalDampingRate`` (``LightSolver::calculateScatteringPotential``);
2. the unified source, forward Euler (``LightSolver.cpp:4239``):

       dE1/dt = -i e/(4 me w0) e^{-i (w0 - 2 wp0) t} [ grad(E0 . E1*) + (1 - w0/wp0) E0 (div E1)* ]

   whose longitudinal part is the TPD source of ``epw.py`` (both terms) plus the SRS
   source (from ``E0 . E1_T*``), and whose transverse part is the Raman-light
   generation of ``raman.py`` (from ``E0 (div E1_L)*``) -- one expression for all four
   couplings, with no separate-envelope approximation between them;
3. the k-space propagator with the 2x2 longitudinal/transverse projector
   (``LightSolver.cpp:3684-3743``): ``exp(-i dt 3 vte^2 k^2/(2 wp0) - gamma_L dt)`` on
   ``k k/k^2 E1`` (Bohm-Gross dispersion and Landau damping) and
   ``exp(-i dt c^2 k^2/(2 wp0))`` on ``(I - k k/k^2) E1`` (light near its cutoff), with
   modes outside the retained band zeroed;
4. the EPW noise source on the longitudinal part (``noise_model`` as in ``epw.py``), built
   for the light sub-step with the Raman rate as its collisional part
   (``LwSolver::addNoise_combinedSolver_spectral(dt)``, ``ZakharovSolver::addNoiseToPotential_fft``);
5. the absorbing layers.

With ``terms.light.pump_depletion`` the pump advances in the same sub-step with the
spectral propagator and LPSE's unified depletion term (``LightSolver.cpp:4265``)

    dE0/dt = i e/(2 me w0) e^{+i (w0 - 2 wp0) t} E1 (div E1)

which contains both the SRS depletion (``E1_T div E1``, ``i e/(4 me w1)`` at n_c/4)
and the TPD depletion (``E1_L div E1``), unprojected: LPSE takes the transverse part of
source terms only outside combined mode (``LightSolver.cpp:4325``), so
``terms.light.tpd_projection`` does not apply here. Otherwise the pump is the prescribed field
of ``laser.py``.
"""

import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._lpse2d.core.epw import analytic_landau_rate, noise_kick_spectrum
from adept._lpse2d.core.pulse import PulseShape
from adept._lpse2d.core.raman import light_absorption_rates
from adept._lpse2d.core.spectral_light import gaussian_injector_profile
from adept._lpse2d.core.vector import dot_conj, fft2c, ifft2c, k_dot, split_k


def longitudinal_transverse(field: Array, kx: Array, ky: Array, one_over_k_sq: Array) -> tuple[Array, Array]:
    """k-space split of an (nx, ny, nc) x-space field into its longitudinal and transverse
    parts, ``(L_k, T_k)`` each (nx, ny, nc); a z component (k_z = 0) is wholly transverse."""
    return split_k(fft2c(field), kx, ky, one_over_k_sq)


def transverse_part(field: Array, kx: Array, ky: Array, one_over_k_sq: Array) -> Array:
    """The transverse (Raman light) part of a combined field, in x-space."""
    _, transverse_k = longitudinal_transverse(field, kx, ky, one_over_k_sq)
    return ifft2c(transverse_k)


def potential_from_field(field: Array, kx: Array, ky: Array, one_over_k_sq: Array, band: Array) -> Array:
    """``phi_k = i k . E_k / k^2`` on the retained band (``makePotentialFromE_fft``)."""
    return 1j * k_dot(fft2c(field), kx, ky) * one_over_k_sq * band


class CombinedSolver:
    """Advance the combined Raman-light + EPW field E1 (and the pump when it is evolved)."""

    def __init__(self, cfg: dict):
        grid = cfg["grid"]
        derived = cfg["units"]["derived"]
        source_cfg = cfg["terms"]["epw"]["source"]
        light_cfg = cfg["terms"].get("light", {})

        self.tpd_enabled = bool(source_cfg.get("tpd", False))
        self.srs_enabled = bool(source_cfg.get("srs", False))
        if self.tpd_enabled != self.srs_enabled:
            raise ValueError(
                "terms.epw.solver: combined needs terms.epw.source.tpd and srs both on or both off "
                "(LPSE: 'When lw.solver=combined, both SRS and TPD must be enabled or neither')"
            )
        self.sources_on = self.tpd_enabled
        self.pump_depletion = bool(light_cfg.get("pump_depletion", False))

        self.nx, self.ny = int(grid["nx"]), int(grid["ny"])
        self.dt = grid["dt"]
        self.n_sub = int(grid.get("light_substeps", 1))
        self.dt_l = self.dt / self.n_sub
        self.x = grid["x"]
        self.y = grid["y"]
        self.dx = grid["dx"]
        self.kx = jnp.asarray(grid["kx"])
        self.ky = jnp.asarray(grid["ky"])
        k_sq_np = np.asarray(grid["kx"])[:, None] ** 2 + np.asarray(grid["ky"])[None, :] ** 2
        self.k_sq = jnp.asarray(k_sq_np)
        self.one_over_k_sq = jnp.asarray(np.where(k_sq_np > 0, 1.0 / np.where(k_sq_np > 0, k_sq_np, 1.0), 0.0))
        band = np.asarray(grid["low_pass_filter_grid"]) * np.where(k_sq_np > 0, 1.0, 0.0)
        cap = light_cfg.get("max_wavenumber")
        if cap is not None:
            band = band * np.where(np.sqrt(k_sq_np) < float(cap) * derived["w0"] / derived["c"], 1.0, 0.0)
        self.band = jnp.asarray(band)

        self.c = derived["c"]
        self.w0 = derived["w0"]
        self.wp0 = derived["wp0"]
        self.w1 = derived["w1"]
        self.vte_sq = derived["vte_sq"]
        self.e = derived["e"]
        self.me = derived["me"]
        # the Raman light's absorption at its own critical density (n_env for this field), not
        # the EPW's collisions (LightSolver::calculateScatteringPotential, raman class)
        _, raman_rate = light_absorption_rates(cfg)
        self.nu_raman = 0.0 if raman_rate is None else float(raman_rate)
        self.envelope_density = cfg["units"]["envelope density"]
        self.background_density = grid["background_density"]
        self.n_over_env = self.background_density / self.envelope_density
        # iaw_density (the local fraction delta n / n_b) in units of n_env: n_b / n_env (LPSE)
        from adept._lpse2d.core.iaw import iaw_feedback_factor

        self.iaw_feedback = iaw_feedback_factor(cfg)
        self.delta_w = self.w0 - 2.0 * self.wp0

        # propagators over one sub-step
        self.landau_enabled = bool(cfg["terms"]["epw"]["damping"].get("landau", True))
        self.hpe_enabled = bool(cfg["terms"].get("hpe", {}).get("active", False))
        # the state's evolved Landau rate: from the particles (HPE) or the quasilinear VDF with
        # landau_evolution, as the separate EPW solver (LPSE applies the evolved LDgammaE in both)
        qle = cfg["terms"].get("qle", {}) or {}
        self.evolved_landau = self.hpe_enabled or (
            bool(qle.get("active", False)) and bool(qle.get("landau_evolution", False))
        )
        self.landau_rate = analytic_landau_rate(cfg)
        self.disp_L = jnp.exp(-1j * self.dt_l * 1.5 * self.vte_sq / self.wp0 * self.k_sq) * self.band
        self.prop_T = jnp.exp(-1j * self.dt_l * self.c**2 / (2.0 * self.wp0) * self.k_sq) * self.band
        self.detune = jnp.exp(-1j * self.dt_l * self.wp0 / 2.0 * (self.n_over_env - 1.0))
        self.collisional = jnp.exp(-self.nu_raman * self.dt_l * self.n_over_env**2)
        self.boundary = grid["absorbing_boundaries"] ** (1.0 / self.n_sub)
        # the pump is LPSE's laser class: its own (5e3/ps) absorber, not the EPW one -- at the EPW
        # rate the injected pump reflects off both walls into a standing wave (see helpers)
        self.light_boundary = grid["light_absorbing_boundaries"] ** (1.0 / self.n_sub)

        # x-space source window on the unified source (terms.epw.source_window etc., plan 2 I.3)
        mask = grid.get("epw_source_mask")
        self.source_mask = None if mask is None or bool(np.all(np.asarray(mask) == 1.0)) else jnp.asarray(mask)
        # unified source coefficient (LightSolver.cpp Sc_srs for the Raman class = (q/m)/(4 W0))
        self.source_coeff = -1j * self.e / (4.0 * self.me * self.w0)
        self.rho_factor = 1.0 - self.w0 / self.wp0
        # k-filter on the pump entering the source (LPSE lw.kFilter, off by default there;
        # terms.epw.source.srs_k_filter / srs_k_filter_scale, see SpectralEPWSolver)
        if bool(source_cfg.get("srs_k_filter", True)):
            n_min = float(np.min(np.asarray(self.background_density)))
            scale = float(source_cfg.get("srs_k_filter_scale", 1.2))
            max_k0_sq = scale**2 * (self.w0 / self.c) ** 2 * max(1.0 - n_min, 0.0)
            self.E0_filter = jnp.asarray(np.where(k_sq_np > max_k0_sq, 0.0, 1.0))
        else:
            self.E0_filter = jnp.ones_like(self.k_sq)

        # noise on the longitudinal part, in field form: E_k += -i k kick e^{i theta}
        self.noise_enabled = bool(source_cfg.get("noise", False))
        if self.noise_enabled:
            import jax

            self.noise_kick = jnp.asarray(noise_kick_spectrum(cfg, dt=self.dt_l, nu_coll=self.nu_raman))
            seed = source_cfg.get("noise_seed")
            self.noise_key = jax.random.PRNGKey(int(seed) if seed is not None else np.random.randint(2**20))

        # evolved pump (spectral) with the unified depletion term
        if self.pump_depletion:
            self.E0_source = derived["E0_source"]
            self.linear_coeff0 = 1j * self.w0 / 2.0 * (1.0 - self.wp0**2 / self.w0**2 * self.n_over_env)
            self.detune0 = jnp.exp(self.dt_l * self.linear_coeff0)
            self.prop0 = jnp.exp(-1j * self.dt_l * self.c**2 / (2.0 * self.w0) * self.k_sq) * self.band
            self.depletion_coeff = 1j * self.e / (2.0 * self.me * self.w0)
            rate0, _ = light_absorption_rates(cfg)
            self.absorb0 = None if rate0 is None else jnp.exp(-rate0 * self.dt_l * self.background_density**2)
            pump = cfg["drivers"]["E0"]["derived"]
            x_inject = grid["xmin"] + pump["offset"]
            self.i0 = int(np.argmin(np.abs(np.asarray(self.x) - x_inject)))
            # pump polarization (drivers.E0.polarization): cos(psi) to y, sin(psi) to z (plan 2 F.2)
            psi = float(pump.get("polarization", 0.0))
            self.pump_weights = (float(np.cos(psi)), float(np.sin(psi)))
            if bool(np.any(cfg["drivers"]["E0"]["derived"].get("beam_leftward", [False]))):
                raise ValueError("the combined solver's pump injector launches from the x-min face only")
            self.n_src = float(self.background_density[self.i0, 0])
            if self.n_src >= 1.0:
                raise ValueError("The pump injector sits at or above critical density")
            self.pump_turn_on_time = pump["turn_on_time"]
            # LPSE laser.pulseShape: the injected source carries sqrt(shape) (core/pulse.py)
            self.pulse = PulseShape(pump)
            k0_inject = self.w0 / self.c * np.sqrt(1.0 - self.n_src)
            width = pump.get("injector_width", np.pi / k0_inject)
            self.pump_profile = jnp.asarray(
                gaussian_injector_profile(np.asarray(self.x), float(self.x[self.i0]), width, self.dx)
            )

    # ------------------------------------------------------------- helpers --

    def potential(self, E1: Array) -> Array:
        return potential_from_field(E1, self.kx, self.ky, self.one_over_k_sq, self.band)

    def transverse(self, E1: Array) -> Array:
        return transverse_part(E1, self.kx, self.ky, self.one_over_k_sq)

    def unified_source(self, t: float, E0: Array, E1: Array) -> Array:
        """``-i e/(4 me w0) e^{-i dw t} [grad(E0 . E1*) + (1 - w0/wp0) E0 (div E1)*]`` in x-space."""
        E0f = ifft2c(fft2c(E0) * self.E0_filter[..., None])
        # E0 . E1* over every component (E0z E1z* included), its gradient is in-plane
        scalar_k = jnp.fft.fft2(dot_conj(E0f, E1)) * self.band
        grad = jnp.zeros_like(E1)
        grad = grad.at[..., 0].set(jnp.fft.ifft2(1j * self.kx[:, None] * scalar_k))
        grad = grad.at[..., 1].set(jnp.fft.ifft2(1j * self.ky[None, :] * scalar_k))
        rho = jnp.fft.ifft2(1j * k_dot(fft2c(E1), self.kx, self.ky) * self.band)
        term = grad + self.rho_factor * E0f * jnp.conj(rho)[..., None]
        if self.source_mask is not None:
            term = term * self.source_mask[..., None]
        return self.source_coeff * jnp.exp(-1j * self.delta_w * t) * term

    def unified_depletion(self, t: float, E1: Array) -> Array:
        """``i e/(2 me w0) e^{+i dw t} [E1 div E1]`` (unprojected, as LPSE's combined path)."""
        rho = jnp.fft.ifft2(1j * k_dot(fft2c(E1), self.kx, self.ky) * self.band)
        return self.depletion_coeff * jnp.exp(1j * self.delta_w * t) * E1 * rho[..., None]

    def propagate_combined(self, E1: Array, gamma_landau) -> Array:
        longitudinal_k, transverse_k = longitudinal_transverse(E1, self.kx, self.ky, self.one_over_k_sq)
        exp_l = self.disp_L * jnp.exp(-gamma_landau * self.dt_l)
        return ifft2c(exp_l[..., None] * longitudinal_k + self.prop_T[..., None] * transverse_k)

    def add_noise(self, t: float, E1: Array) -> Array:
        import jax

        step = jnp.round(t / self.dt_l).astype(int)
        key = jax.random.fold_in(self.noise_key, step)
        phases = 2.0 * np.pi * jax.random.uniform(key, (self.nx, self.ny))
        phi_kick = self.noise_kick * jnp.exp(1j * phases)
        e_k = fft2c(E1)
        e_k = e_k.at[..., 0].add(-1j * self.kx[:, None] * phi_kick)
        e_k = e_k.at[..., 1].add(-1j * self.ky[None, :] * phi_kick)
        return ifft2c(e_k)

    def calc_pump_source(self, t: float, pump_args: dict) -> Array:
        t_env = get_envelope(
            pump_args["tr"],
            pump_args["tr"],
            pump_args["tc"] - pump_args["tw"] / 2,
            pump_args["tc"] + pump_args["tw"] / 2,
            t,
        )
        turn_on = 1.0 - jnp.exp(-((t / self.pump_turn_on_time) ** 2))
        delta_omega = pump_args["delta_omega"]
        intensities = pump_args["intensities"]
        phases = pump_args["phases"]
        k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega) ** 2 - self.n_src)
        v_g = self.c**2 * k0 / self.w0
        amp = self.E0_source * jnp.sqrt(intensities) / (1.0 - self.n_src) ** 0.25 * t_env * turn_on
        amp = amp * self.pulse.field_factor(t)
        color_phase = jnp.exp(-1j * self.w0 * delta_omega[:, None] * t + 1j * phases)
        carrier = jnp.exp(1j * k0[:, None] * (self.x[None, :] - self.x[self.i0]))
        source = (amp * v_g[:, None] * color_phase)[:, None, :] * (carrier * self.pump_profile[None, :])[:, :, None]
        return jnp.sum(source, axis=0)

    def propagate_pump(self, E0: Array) -> Array:
        longitudinal_k, transverse_k = longitudinal_transverse(E0, self.kx, self.ky, self.one_over_k_sq)
        return ifft2c(longitudinal_k + self.prop0[..., None] * transverse_k)

    # ---------------------------------------------------------------- step --

    def __call__(self, t: float, y: dict, driver_args: dict, E0_fn=None) -> tuple[Array, Array, Array]:
        """Advance over one EPW step. Returns (E0, E1, phi_k).

        ``E0_fn(t)`` gives the prescribed pump when it is not evolved; ``driver_args["E0"]``
        the injector parameters when it is."""
        E1 = y["E1"]
        E0 = y["E0"]
        iaw_density = y.get("iaw_density")
        detune = self.detune
        if iaw_density is not None:
            detune = detune * jnp.exp(-1j * self.dt_l * self.wp0 / 2.0 * iaw_density * self.iaw_feedback)
        if self.evolved_landau:
            gamma_landau = y["gamma_L"]
        elif self.landau_enabled:
            gamma_landau = self.landau_rate
        else:
            gamma_landau = jnp.zeros_like(self.k_sq)
        pump_args = driver_args.get("E0") if self.pump_depletion else None
        if self.pump_depletion:
            detune0 = self.detune0
            if iaw_density is not None:
                detune0 = detune0 * jnp.exp(
                    -1j * self.wp0**2 / (2.0 * self.w0) * iaw_density * self.iaw_feedback * self.dt_l
                )

        def substep(i, fields):
            E0, E1 = fields
            t_i = t + i * self.dt_l
            if not self.pump_depletion:
                E0 = E0_fn(t_i)
            # 1. scattering potential and collisional damping on the combined field
            E1 = E1 * (detune * self.collisional)[..., None]
            # 2. the unified source (and the pump's x-space step)
            if self.pump_depletion:
                E0 = E0 * detune0[..., None]
                if self.absorb0 is not None:
                    E0 = E0 * self.absorb0[..., None]
                if self.sources_on:
                    E0 = E0 + self.dt_l * self.unified_depletion(t_i, E1)
                pump_source = self.dt_l * self.calc_pump_source(t_i, pump_args)
                for c, w in zip((1, 2), self.pump_weights, strict=True):
                    if w == 0.0:
                        continue
                    if c >= E0.shape[-1]:
                        raise ValueError("an out-of-plane (s-polarised) pump needs three-component light fields")
                    E0 = E0.at[..., c].add(w * pump_source)
            if self.sources_on:
                E1 = E1 + self.dt_l * self.unified_source(t_i, E0, E1)
            # 3. k-space propagation with the L/T projector
            E1 = self.propagate_combined(E1, gamma_landau)
            if self.pump_depletion:
                E0 = self.propagate_pump(E0)
            # 4. noise on the longitudinal part
            if self.noise_enabled:
                E1 = self.add_noise(t_i, E1)
            # 5. absorbing layers
            E1 = E1 * self.boundary[..., None]
            if self.pump_depletion:
                E0 = E0 * self.light_boundary[..., None]
            return (E0, E1)

        E0, E1 = lax.fori_loop(0, self.n_sub, substep, (E0, E1))
        return E0, E1, self.potential(E1)
