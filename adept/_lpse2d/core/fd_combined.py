"""LPSE's combined TPD + SRS solver with the finite-difference light propagator (inventory A15).

With ``lw.solver = combined`` and ``{laser|raman}.solver = fd`` LPSE advances the combined field --
the Raman class, carrier ``wpe``, whose transverse part is the Raman light and whose longitudinal
part is the EPW -- with its FD Schroedinger step, the grad-div coefficient carrying the Bohm-Gross
term (``SchrodingerSolver3::step_2d``, ``gradDivCoeff = dt 3 Ve^2 / (2 wpe) - Cdt``):

    dE1/dt = i c^2/(2 wp0) [lap E1 - grad div E1] + i 3 vte^2/(2 wp0) grad div E1
             + i wp0/2 (1 - n/n_env) E1 + unified source,

the pump with the same FD step, and the sources from FD derivatives (``LightSolver::calculateSources``:
``divergence`` / ``gradE`` with the solver's stencil). Landau damping and the EPW noise act on the
longitudinal part in k-space once per Langmuir-wave step (``LwSolver::advanceLW_combinedTPDandSRS_FD``);
the spectral combined solver (``combined.py``) does both per light sub-step, as LPSE's spectral path.

This class is ``CoupledLight`` (the FD pump, its injectors and absorbers, the staggered sub-steps)
with the combined field's operator and the unified couplings of ``CombinedSolver`` in FD form.
"""

import numpy as np
from jax import Array
from jax import numpy as jnp

from adept._lpse2d.core.combined import CombinedSolver
from adept._lpse2d.core.light import CoupledLight
from adept._lpse2d.core.vector import fft2c, ifft2c, k_dot


class FDCombinedSolver(CoupledLight):
    """The combined field E1 and the evolved pump E0 on the FD light propagator."""

    def __init__(self, cfg: dict):
        if not cfg["terms"].get("light", {}).get("pump_depletion", False):
            raise ValueError(
                "terms.epw.solver: combined with terms.light.solver: fd needs terms.light.pump_depletion "
                "(LPSE's FD combined decks evolve the pump; the static pump runs on the spectral combined solver)"
            )
        super().__init__(cfg)
        # the k-space pieces of the combined formulation: band, potential, Landau rate, noise
        self.spectral = CombinedSolver(cfg)
        # the separate SRS / TPD couplings of CoupledLight are replaced by the unified ones
        self.srs_enabled = False
        self.tpd_enabled = False
        self.sources_on = self.spectral.sources_on
        derived = cfg["units"]["derived"]
        vte_sq = derived["vte_sq"]
        # the combined field: carrier wp0, Bohm-Gross longitudinal dispersion
        self.diffraction_coeff = 1j * self.c**2 / (2.0 * self.wp0)
        self.grad_div_coeff = 1j * 3.0 * vte_sq / (2.0 * self.wp0)
        self.linear_coeff = 1j * self.wp0 / 2.0 * (1.0 - self.n_over_env)
        self.n_over_nc1 = self.n_over_env  # the Raman class's critical density is n_env here
        self.w_carrier1 = self.wp0
        # LPSE's double-exponential layer on the combined field (setupAbsorbingBoundaries_doubleExponential)
        grid = cfg["grid"]
        self.sub_boundary = grid.get("combined_absorbing_boundaries", grid["absorbing_boundaries"]) ** (
            1.0 / self.n_sub
        )
        self.delta_w = self.w0 - 2.0 * self.wp0
        self.source_coeff = self.spectral.source_coeff  # -i e / (4 me w0)
        self.depletion_coeff = self.spectral.depletion_coeff if hasattr(self.spectral, "depletion_coeff") else None
        if self.depletion_coeff is None:
            self.depletion_coeff = 1j * self.e / (2.0 * self.me * self.w0)
        self.rho_factor = 1.0 - self.w0 / self.wp0
        # Landau damping / noise once per EPW step (the LW step)
        self.noise_enabled = self.spectral.noise_enabled
        if self.noise_enabled:
            from adept._lpse2d.core.epw import noise_kick_spectrum

            self.noise_kick_lw = jnp.asarray(noise_kick_spectrum(cfg, dt=self.dt, nu_coll=self.spectral.nu_raman))
        print("combined solver on the FD light propagator (LPSE lw.solver = combined, raman.solver = fd)")

    # ------------------------------------------------------------ FD pieces --

    def _div(self, E: Array) -> Array:
        return self._dx(E[..., 0]) + self._dy(E[..., 1])

    def _grad_div(self, E: Array) -> list[Array]:
        ex, ey = E[..., 0], E[..., 1]
        out = [self._d2x(ex) + self._dxdy(ey), self._dxdy(ex) + self._d2y(ey)]
        if E.shape[-1] == 3:
            out.append(jnp.zeros_like(ex))
        return out

    def unified_source_fd(self, t: float, E0: Array, E1: Array) -> Array:
        """``-i e/(4 me w0) e^{-i dw t} [grad(E0 . E1*) + (1 - w0/wp0) E0 (div E1)*]`` with the FD
        stencils (LPSE ``gradE`` / ``divergence``)."""
        scalar = jnp.sum(E0 * jnp.conj(E1), axis=-1)
        grad = [self._dx(scalar), self._dy(scalar)]
        if E1.shape[-1] == 3:
            grad.append(jnp.zeros_like(scalar))
        div1 = self._div(E1)
        term = jnp.stack(grad, axis=-1) + self.rho_factor * E0 * jnp.conj(div1)[..., None]
        return self.source_coeff * jnp.exp(-1j * self.delta_w * t) * term

    def unified_depletion_fd(self, t: float, E1: Array) -> Array:
        """``i e/(2 me w0) e^{+i dw t} E1 div E1`` with the FD divergence."""
        return self.depletion_coeff * jnp.exp(1j * self.delta_w * t) * E1 * self._div(E1)[..., None]

    def coupled_rhs(
        self,
        t: float,
        E0: Array,
        E1: Array,
        laplacian_phi: Array,
        pump_args: dict,
        seed_args: dict | None,
        iaw_density: Array | None = None,
        phi_k: Array | None = None,
        couple: bool = True,
        patterns: list[Array] | None = None,
    ) -> tuple[Array, Array]:
        # pump: CoupledLight's propagation, detuning and injector rows (its SRS / TPD terms are off)
        k_e0 = self.pump_rhs(t, E0, E1, laplacian_phi, pump_args, iaw_density, None, couple=False, patterns=patterns)
        linear_coeff = self.linear_coeff
        if iaw_density is not None:
            linear_coeff = linear_coeff - 1j * self.wp0 / 2.0 * iaw_density * self.iaw_feedback
        comps = [E1[..., i] for i in range(E1.shape[-1])]
        k_e1 = [
            self.diffraction_coeff * cc + self.grad_div_coeff * gd + linear_coeff * e
            for cc, gd, e in zip(self.curl_curl(E1), self._grad_div(E1), comps, strict=True)
        ]
        k_e1 = jnp.stack(k_e1, axis=-1)
        if self.sources_on:
            depletion = self.unified_depletion_fd(t, E1)
            source = self.unified_source_fd(t, E0, E1)
            if not isinstance(self.source_mask0, float):
                depletion = depletion * self.source_mask0[..., None]
            if not isinstance(self.source_mask1, float):
                source = source * self.source_mask1[..., None]
            k_e0 = k_e0 + depletion
            k_e1 = k_e1 + source
        return k_e0, k_e1

    # ------------------------------------------------------- the LW step --

    def langmuir_step(self, t: float, E1: Array, gamma_landau) -> Array:
        """Landau damping and noise on the longitudinal part, once per EPW step (LwSolver::
        advanceLW_combinedTPDandSRS_FD): ``E_L -= E_L (1 - exp(-gamma dt))`` (``gamma dt`` below 1e-3)
        and ``E += -i k phi_noise`` on the retained band."""
        import jax

        band = self.spectral.band
        e_k = fft2c(E1)
        rate_dt = gamma_landau * self.dt
        coeff = jnp.where(rate_dt < 1.0e-3, rate_dt, 1.0 - jnp.exp(-rate_dt)) * band
        kx, ky = self.spectral.kx[:, None], self.spectral.ky[None, :]
        k_dot_e = k_dot(e_k, self.spectral.kx, self.spectral.ky)
        e_long = [kx * k_dot_e * self.spectral.one_over_k_sq, ky * k_dot_e * self.spectral.one_over_k_sq]
        comps = [e_k[..., 0] - coeff * e_long[0], e_k[..., 1] - coeff * e_long[1]]
        if self.noise_enabled:
            step = jnp.round(t / self.dt).astype(int)
            key = jax.random.fold_in(self.spectral.noise_key, step)
            phases = 2.0 * np.pi * jax.random.uniform(key, (self.nx, self.ny))
            phi_kick = self.noise_kick_lw * jnp.exp(1j * phases) * band
            comps = [comps[0] - 1j * kx * phi_kick, comps[1] - 1j * ky * phi_kick]
        if e_k.shape[-1] == 3:
            comps.append(e_k[..., 2])
        return ifft2c(jnp.stack(comps, axis=-1))

    def __call__(self, t: float, y: dict, driver_args: dict, E0_fn=None) -> tuple[Array, Array, Array]:
        """One EPW step: the LW step on the longitudinal part, then the light sub-steps. Returns
        ``(E0, E1, phi_k)`` like ``CombinedSolver``."""
        if self.spectral.evolved_landau:
            gamma_landau = y["gamma_L"]
        elif self.spectral.landau_enabled:
            gamma_landau = self.spectral.landau_rate
        else:
            gamma_landau = jnp.zeros_like(self.spectral.k_sq)
        E1 = self.langmuir_step(t, y["E1"], gamma_landau)
        zero_phi = jnp.zeros((self.nx, self.ny), dtype=jnp.complex128)
        E0, E1 = CoupledLight.__call__(self, t, y["E0"], E1, zero_phi, driver_args["E0"], None, y.get("iaw_density"))
        return E0, E1, self.spectral.potential(E1)
