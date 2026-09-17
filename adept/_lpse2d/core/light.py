import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._lpse2d.core.raman import RamanLight, transverse_part


class CoupledLight(RamanLight):
    """
    Evolves the pump E0 and, when enabled, the Raman scattered light E1.

    This is the `isPumpDepletion` path of m201805_matlabLpse_v11.m: the pump is no longer
    prescribed analytically but advanced with the same staggered explicit scheme as the
    Raman light (lightSplitStep, lines 1377-1424), sourced by a boundary injector
    (lines 1707-1753) and coupled to the EPW through the conjugate-free SRS term
    (lines 1611-1648):

        dE0/dt = i c^2/(2 w0) * (Laplacian terms) E0
                 + i w0/2 * (1 - wp0^2/w0^2 * n/n_env) * E0
                 - i e/(4 w1 me) * (laplacian phi) * E1       (pump depletion)
                 + boundary source injector

        dE1/dt = i c^2/(2 w1) * (Laplacian terms) E1
                 + i w1/2 * (1 - wp0^2/w1^2 * n/n_env) * E1
                 - i e/(4 w0 me) * conj(laplacian phi) * E0   (SRS coupling)
                 [+ seed injector]

    Note the coupling denominators: each wave's SRS term carries the *partner* wave's
    frequency, and only E1's term conjugates the potential. Together with the EPW source
    (epw.py, prefactor e*wp0/(4 me w0 w1)) these satisfy Manley-Rowe exactly:
    d/dt Int(|E0|^2 + |E1|^2 + |grad phi|^2) = 0 for the coupling terms alone.

    Both fields must be advanced inside the *same* staggered update (both real parts with
    the RHS at t, then both imaginary parts with the RHS at t + dt/2) -- advancing them
    with two independent RamanLight-style calls would break the discrete conservation.

    The E1 half of the solver -- FD stencils, detuning/diffraction coefficients, SRS
    coupling, and the seed injector -- is inherited from ``RamanLight`` (``self.rhs``),
    so any numerics fix there applies to both the prescribed-pump and pump-depletion
    paths. This class adds the pump: its coefficients, boundary injector, reciprocal
    TPD/SRS couplings, and the coupled staggered loop.

    The TPD pump term is LPSE's (LightSolver.cpp, ``Sc_tpd``; LwSolver.cpp, the
    transverse projection; Follett's equation document Eqs 53/55):

        dE0/dt |_TPD = i e/(2 w0 me) exp(+i (w0 - 2 wp0)t) * [Eh div(Eh)]_T

    for every pump component, with ``[.]_T`` the transverse (divergence-free) part taken
    in k-space (``terms.light.tpd_projection``, default on) so that a longitudinal
    component is never injected into the light field; ``terms.light.tpd_k_filter``
    additionally restricts the term to ``|k| < 1.2 k0 sqrt(1 - n_min)`` (LPSE
    ``lw.kFilter``). The coefficient is exactly twice the EPW-side TPD source coefficient
    ``i e/(4 me w0)``: with that ratio the pair conserves ``|E0|^2 + (2 wp0/w0)|Eh|^2``
    for the coupling terms alone, i.e. the total wave energy at envelope density n_c/4
    (settled against the equation document on 2026-09-14; adept main previously carried
    half this coefficient).

    The pump injector amplitude is divided by sinc(k0 dx) so the launched amplitude is
    exactly E0_source * sqrt(intensity) / eps^(1/4) despite the two-point discrete
    source's sinc response (the E1 seed injector intentionally keeps the MATLAB
    calibration; see tests/test_lpse2d/test_srs.py::test_srs_seed_propagation).

    **Coupling scheme** (``terms.light.coupling``). The staggered real/imaginary update
    is a leapfrog only for a RHS operator that is i times a *real* matrix in the
    (Re, Im) basis. The propagation and detuning terms are; the SRS exchange between
    E0 and E1 is not: its matrix element is the complex, spatially rotating
    ``laplacian phi``. Written out, the real parts of both waves are advanced together
    with an explicit Euler step on the part of the exchange that is proportional to
    Im(laplacian phi), whose spectral radius per sub-step is
    1 + sin^2(arg laplacian phi) (Omega dt_l)^2 / 2 with
    Omega = e |laplacian phi| / (4 me sqrt(w0 w1)) the local exchange rate. Averaged over
    the EPW phase that is a growth rate Omega^2 dt_l / 4 of the light fields for *any*
    dt_l -- negligible at small EPW amplitude, but it scales with the EPW energy and
    feeds the EPW through the SRS/TPD sources, so a depleted-pump run runs away once
    Omega dt_l reaches ~0.01-0.1 (tests/test_lpse2d/test_light_coupling.py).

    ``coupling: explicit`` (default) keeps the MATLAB scheme. ``coupling: rotation``
    Strang-splits each sub-step as [exact exchange over dt_l/2] [staggered propagation
    with the exchange off] [exact exchange over dt_l/2]. With laplacian phi frozen, the
    exchange-only system has M^2 = A B |laplacian phi|^2 I = -Omega^2 I, so
    exp(tau M) = cos(Omega tau) I + sin(Omega tau)/Omega M exactly -- a rotation that
    conserves the action w1 |E0|^2 + w0 |E1|^2 pointwise and is stable for any dt_l.
    The TPD pump term and the IAW detuning are not part of the exchange and stay in
    the staggered RHS under both schemes; with SRS off there is no exchange and the
    two schemes coincide.
    """

    def __init__(self, cfg: dict):
        # E1 solver: coefficients, stencils, sub-stepping, seed injector
        super().__init__(cfg)

        derived = cfg["units"]["derived"]
        self.E0_source = derived["E0_source"]
        background_density = cfg["grid"]["background_density"]
        source_cfg = cfg["terms"]["epw"]["source"]
        self.srs_enabled = bool(source_cfg.get("srs", False))
        self.tpd_enabled = bool(source_cfg.get("tpd", False))
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]

        # pump detuning/diffraction (MATLAB lines 1616-1626); with wp0^2 = w0^2 * n_env
        # the pump coefficient reduces to i w0/2 (1 - n)
        self.linear_coeff0 = (
            1j * self.w0 / 2.0 * (1.0 - self.wp0**2 / self.w0**2 * background_density / self.envelope_density)
        )
        self.diffraction_coeff0 = 1j * self.c**2 / (2.0 * self.w0)
        self.srs_depletion_coeff0 = -1j * self.e / (4.0 * self.w1 * self.me)
        self.tpd_depletion_coeff0 = 1j * self.e / (2.0 * self.w0 * self.me)

        light_cfg = cfg["terms"].get("light", {})
        # transverse projection of E div(E) (LPSE LwSolver::makeExyzDivE, k-space
        # P_T = I - k k / k^2) and the optional LPSE lw.kFilter on the same term
        self.tpd_projection = bool(light_cfg.get("tpd_projection", True))
        self.tpd_k_filter = bool(light_cfg.get("tpd_k_filter", False))
        kx_arr = np.asarray(cfg["grid"]["kx"])
        ky_arr = np.asarray(cfg["grid"]["ky"])
        k_sq_np = kx_arr[:, None] ** 2 + ky_arr[None, :] ** 2
        k_sq_safe = np.where(k_sq_np > 0, k_sq_np, 1.0)
        self.kx_over_k_sq = jnp.asarray(np.where(k_sq_np > 0, kx_arr[:, None] / k_sq_safe, 0.0))
        self.ky_over_k_sq = jnp.asarray(np.where(k_sq_np > 0, ky_arr[None, :] / k_sq_safe, 0.0))
        if self.tpd_k_filter:
            n_min = float(np.min(np.asarray(background_density)))
            k0_max_sq = (1.2 * self.w0 / self.c) ** 2 * max(1.0 - n_min, 0.0)
            self.tpd_k_mask = jnp.asarray(np.where((k_sq_np > 0) & (k_sq_np < k0_max_sq), 1.0, 0.0))
        else:
            self.tpd_k_mask = None

        # how the SRS E0 <-> E1 exchange enters the light sub-step (see the class docstring)
        self.coupling = str(light_cfg.get("coupling", "explicit"))
        if self.coupling not in ("explicit", "rotation"):
            raise ValueError(f"terms.light.coupling must be 'explicit' or 'rotation', got {self.coupling!r}")
        # local E0 <-> E1 exchange rate per unit |laplacian phi|: Omega = sqrt(|A B|) |lap phi|
        self.omega_prefactor = self.e / (4.0 * self.me * np.sqrt(self.w0 * self.w1))

        # optional isotropic low-pass filter on both light fields, applied once per EPW
        # step (terms.light.filter = fraction of the grid Nyquist wavenumber; default off).
        # The physical light content is |k| <= ~1.2 k0, far below the grid Nyquist; grid-
        # scale light modes have FD group velocity c^2 sin(k dx)/(w dx) -> 0 and the
        # staggered scheme's phase per sub-step is largest there (diagnostic option).
        light_filter = light_cfg.get("filter", None)
        if light_filter is None or light_filter is False:
            self.light_filter = None
        else:
            frac = float(light_filter)
            if frac <= 0.0:
                raise ValueError(f"terms.light.filter must be a positive fraction of pi/dx, got {light_filter!r}")
            kx = np.asarray(cfg["grid"]["kx"])
            ky = np.asarray(cfg["grid"]["ky"])
            k_nyq = np.pi / float(self.dx)
            k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
            self.light_filter = jnp.asarray(np.where(k_mag <= frac * k_nyq, 1.0, 0.0))[..., None]

        # ---- pump injector (MATLAB lines 1707-1753, mirrored to the left edge) ----
        pump = cfg["drivers"]["E0"]["derived"]
        x_inject = cfg["grid"]["xmin"] + pump["offset"]
        self.i0 = int(np.argmin(np.abs(np.array(self.x) - x_inject)))
        # pump polarization (drivers.E0.polarization): the two-point injector writes cos(psi) to
        # the in-plane transverse component y and sin(psi) to z (plan 2 F.2)
        psi = float(cfg["drivers"]["E0"].get("derived", {}).get("polarization", 0.0))
        self.pump_weights = (float(np.cos(psi)), float(np.sin(psi)))
        n_src = float(background_density[self.i0, 0])
        permittivity0 = 1.0 - n_src
        if permittivity0 <= 0:
            raise ValueError(
                f"The pump injector at x = {float(self.x[self.i0]):.2f} um sits at density "
                f"{n_src:.3f} nc, at or above critical. Lower density.max or move drivers.E0.offset."
            )
        self.n_src = n_src
        self.pump_turn_on_time = pump["turn_on_time"]
        self.source_prefactor0 = self.c**2 / (2.0 * self.w0) / permittivity0**0.25 / self.dx**2

    def calc_pump_source(self, t: float, pump_args: dict) -> tuple[Array, Array]:
        """
        Two-point pump injector rows, summed over colors (MATLAB lines 1738-1750,
        with +k0 and the left edge instead of -k1 and the right edge).

        Returns the rows added to the E0_y RHS at self.i0 and self.i0 + 1.
        """
        t_env = get_envelope(
            pump_args["tr"],
            pump_args["tr"],
            pump_args["tc"] - pump_args["tw"] / 2,
            pump_args["tc"] + pump_args["tw"] / 2,
            t,
        )
        turn_on = 1.0 - jnp.exp(-((t / self.pump_turn_on_time) ** 2))

        delta_omega = pump_args["delta_omega"]  # (nc,)
        intensities = pump_args["intensities"]  # (nc, ny), fractions summing to 1
        phases = pump_args["phases"]  # (nc, ny)

        # local pump wavenumber per color (MATLAB kSource0). The two-point source
        # launches amplitude E_src * sin(k0 dx)/sin(k_grid dx) / eps^(1/4) -- a ~2%
        # deficit at 8 cells/wavelength from the grid dispersion; the budget metrics
        # normalize to the *measured* incident flux, so this bias cancels there.
        k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega) ** 2 - self.n_src)  # (nc,)

        amp = self.source_prefactor0 * self.E0_source * jnp.sqrt(intensities) * t_env * turn_on  # (nc, ny)

        color_phase = jnp.exp(-1j * self.w0 * delta_omega[:, None] * t + 1j * phases)  # (nc, ny)
        row_i0p1 = jnp.sum(-1j * amp * jnp.exp(1j * k0[:, None] * self.x[self.i0]) * color_phase, axis=0)
        row_i0 = jnp.sum(1j * amp * jnp.exp(1j * k0[:, None] * self.x[self.i0 + 1]) * color_phase, axis=0)
        return row_i0, row_i0p1

    def pump_rhs(
        self,
        t: float,
        E0: Array,
        E1: Array,
        laplacian_phi: Array,
        pump_args: dict,
        iaw_density: Array | None = None,
        phi_k: Array | None = None,
        couple: bool = True,
    ) -> Array:
        """Pump RHS: propagation + detuning (MATLAB lines 1616-1626), SRS pump depletion
        (lines 1640-1646: no conjugate, w1 denominator) unless ``couple`` is False (the
        rotation scheme integrates that exchange exactly outside the RHS), the TPD pump
        depletion, and the boundary injector."""
        e0x, e0y = E0[..., 0], E0[..., 1]
        linear_coeff0 = self.linear_coeff0
        if iaw_density is not None:
            # MATLAB: i*w0/2 * [1 - wp0^2/w0^2 * (n_b/n_env + Nelf)] E0
            linear_coeff0 = linear_coeff0 - 1j * self.wp0**2 / (2.0 * self.w0) * iaw_density

        # discrete curl-curl on the in-plane components, the plain Laplacian on E0z (k_z = 0)
        k_e0 = [
            self.diffraction_coeff0 * (self._d2y(e0x) - self._dxdy(e0y)) + linear_coeff0 * e0x,
            self.diffraction_coeff0 * (self._d2x(e0y) - self._dxdy(e0x)) + linear_coeff0 * e0y,
        ]
        if E0.shape[-1] == 3:
            e0z = E0[..., 2]
            k_e0.append(self.diffraction_coeff0 * (self._d2x(e0z) + self._d2y(e0z)) + linear_coeff0 * e0z)
        if self.srs_enabled and couple:
            depletion = (self.srs_depletion_coeff0 * laplacian_phi)[..., None] * E1
            if self.transverse_source:
                depletion = transverse_part(depletion, self.kx_arr, self.ky_arr, self.one_over_k_sq)
            k_e0 = [k + depletion[..., i] for i, k in enumerate(k_e0)]
        if self.tpd_enabled:
            if phi_k is None:
                raise ValueError("phi_k is required for TPD pump depletion")
            tpd_dep = self.calc_tpd_depletion(t, phi_k)  # in-plane only: E_h has no z component
            k_e0[0] = k_e0[0] + tpd_dep[..., 0]
            k_e0[1] = k_e0[1] + tpd_dep[..., 1]
        row_i0, row_i0p1 = self.calc_pump_source(t, pump_args)
        for c, w in zip((1, 2), self.pump_weights, strict=True):
            if w == 0.0:
                continue
            if c >= len(k_e0):
                raise ValueError("an out-of-plane (s-polarised) pump needs three-component light fields")
            k_e0[c] = k_e0[c].at[self.i0, :].add(w * row_i0)
            k_e0[c] = k_e0[c].at[self.i0 + 1, :].add(w * row_i0p1)

        return jnp.stack(k_e0, axis=-1)

    def tpd_depletion_vector(self, phi_k: Array) -> Array:
        """``[E_h div(E_h)]_T`` in real space, shape (nx, ny, 2), from the frozen potential.

        ``div_e`` uses the same potential convention as the EPW TPD source:
        ``div_e = ifft2(k^2 phi_k)``. The transverse projection and the optional k-filter
        are LPSE's ``LwSolver::makeExyzDivE``: ``F_T = F - k (k . F) / k^2`` mode by mode.
        """
        ex = jnp.fft.ifft2(-1j * self.kx[:, None] * phi_k)
        ey = jnp.fft.ifft2(-1j * self.ky[None, :] * phi_k)
        div_e = jnp.fft.ifft2(self.k_sq * phi_k)
        fx, fy = ex * div_e, ey * div_e
        if self.tpd_projection or self.tpd_k_mask is not None:
            fx_k = jnp.fft.fft2(fx)
            fy_k = jnp.fft.fft2(fy)
            if self.tpd_projection:
                longitudinal = self.kx[:, None] * fx_k + self.ky[None, :] * fy_k
                fx_k = fx_k - self.kx_over_k_sq * longitudinal
                fy_k = fy_k - self.ky_over_k_sq * longitudinal
            if self.tpd_k_mask is not None:
                fx_k = fx_k * self.tpd_k_mask
                fy_k = fy_k * self.tpd_k_mask
            fx, fy = jnp.fft.ifft2(fx_k), jnp.fft.ifft2(fy_k)
        return jnp.stack([fx, fy], axis=-1)

    def calc_tpd_depletion(self, t: float, phi_k: Array) -> Array:
        """Return the reciprocal TPD term for both pump components, shape (nx, ny, 2)."""
        phase = jnp.exp(1j * (self.w0 - 2.0 * self.wp0) * t)
        return self.tpd_depletion_coeff0 * phase * self.tpd_depletion_vector(phi_k)

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
    ) -> tuple[Array, Array]:
        # the E1 RHS (propagation + detuning + SRS coupling + seed rows) is exactly
        # the RamanLight one
        pump_rhs = self.pump_rhs(t, E0, E1, laplacian_phi, pump_args, iaw_density, phi_k, couple=couple)
        if not self.srs_enabled:
            return pump_rhs, jnp.zeros_like(E1)
        return pump_rhs, self.rhs(t, E1, E0, laplacian_phi, seed_args, iaw_density, couple=couple)

    def couple(self, E0: Array, E1: Array, laplacian_phi: Array, tau: float) -> tuple[Array, Array]:
        """
        Exact solution over ``tau`` of the SRS exchange alone with laplacian phi frozen,

            dE0/dt = A L E1,   dE1/dt = B L* E0,   A = srs_depletion_coeff0, B = srs_coeff,

        i.e. exp(tau M) = cos(Omega tau) I + sin(Omega tau)/Omega M with
        Omega = sqrt(|A B|) |L| = e |L| / (4 me sqrt(w0 w1)). Conserves w1|E0|^2 + w0|E1|^2
        at every point (Manley-Rowe for the light pair) for any tau.
        """
        omega = self.omega_prefactor * jnp.abs(laplacian_phi)
        omega_safe = jnp.where(omega > 0.0, omega, 1.0)
        cos_ = jnp.cos(omega * tau)[..., None]
        sinc_ = jnp.where(omega > 0.0, jnp.sin(omega * tau) / omega_safe, tau)
        a = (sinc_ * self.srs_depletion_coeff0 * laplacian_phi)[..., None]
        b = (sinc_ * self.srs_coeff * jnp.conj(laplacian_phi))[..., None]
        return cos_ * E0 + a * E1, cos_ * E1 + b * E0

    def __call__(
        self,
        t: float,
        E0: Array,
        E1: Array,
        phi_k: Array,
        pump_args: dict,
        seed_args: dict | None,
        iaw_density: Array | None = None,
    ):
        """
        Advance (E0, E1) over one EPW step: self.n_sub light sub-steps with the EPW
        potential held fixed.

        ``coupling == "explicit"`` matches MATLAB lightSplitStep: both real parts are
        updated with the full RHS at t_i, then both imaginary parts with the RHS at
        t_i + dt/2. ``coupling == "rotation"`` applies the exact SRS-exchange rotation
        for dt_l/2 on either side of the same staggered update with the exchange
        switched off (Strang splitting); see the class docstring for why. With SRS off
        there is no exchange and both settings run the plain staggered update.
        """
        seed_args = seed_args if self.seed_enabled else None
        laplacian_phi = jnp.fft.ifft2(-self.k_sq * phi_k)
        rotate = self.coupling == "rotation" and self.srs_enabled
        couple_in_rhs = not rotate

        def propagate(t_i, E0, E1):
            k_e0, k_e1 = self.coupled_rhs(
                t_i, E0, E1, laplacian_phi, pump_args, seed_args, iaw_density, phi_k, couple=couple_in_rhs
            )
            E0 = E0 + self.dt_l * jnp.real(k_e0)
            E1 = E1 + self.dt_l * jnp.real(k_e1)
            k_e0, k_e1 = self.coupled_rhs(
                t_i + self.dt_l / 2.0,
                E0,
                E1,
                laplacian_phi,
                pump_args,
                seed_args,
                iaw_density,
                phi_k,
                couple=couple_in_rhs,
            )
            E0 = E0 + 1j * self.dt_l * jnp.imag(k_e0)
            E1 = E1 + 1j * self.dt_l * jnp.imag(k_e1)
            return E0, E1

        def substep(i, fields):
            E0, E1 = fields
            t_i = t + i * self.dt_l
            if rotate:
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
                E0, E1 = propagate(t_i, E0, E1)
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
            else:
                E0, E1 = propagate(t_i, E0, E1)
            E0 = E0 * self.sub_boundary[..., None]
            E1 = E1 * self.sub_boundary[..., None]
            if absorb0 is not None:
                E0 = E0 * absorb0
                E1 = E1 * absorb1
            return (E0, E1)

        absorb0 = absorb1 = None
        if self.absorption_rate0 is not None:
            n0 = self.n_over_nc0 if iaw_density is None else self.n_over_nc0 * (1.0 + iaw_density / self.n_over_env)
            n1 = self.n_over_nc1 if iaw_density is None else self.n_over_nc1 * (1.0 + iaw_density / self.n_over_env)
            absorb0 = jnp.exp(-self.absorption_rate0 * self.dt_l * n0**2)[..., None]
            absorb1 = jnp.exp(-self.absorption_rate1 * self.dt_l * n1**2)[..., None]
        E0, E1 = lax.fori_loop(0, self.n_sub, substep, (E0, E1))
        if self.light_filter is not None:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.light_filter, axes=(0, 1))
            E1 = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.light_filter, axes=(0, 1))
        if self.transverse_fields:
            # drop the longitudinal part the FD curl-curl generated over the sub-steps (see
            # RamanLight); a y-uniform plane-wave pump is unchanged
            E0 = transverse_part(E0, self.kx_arr, self.ky_arr, self.one_over_k_sq)
            E1 = transverse_part(E1, self.kx_arr, self.ky_arr, self.one_over_k_sq)
        return E0, E1
