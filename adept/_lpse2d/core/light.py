import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._lpse2d.core.raman import RamanLight


class CoupledLight(RamanLight):
    """
    Evolves the pump E0 and the Raman scattered light E1 together, with pump depletion.

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
    paths. This class adds only the pump: its coefficients, its boundary injector, and
    the coupled staggered loop.

    The pump injector amplitude is divided by sinc(k0 dx) so the launched amplitude is
    exactly E0_source * sqrt(intensity) / eps^(1/4) despite the two-point discrete
    source's sinc response (the E1 seed injector intentionally keeps the MATLAB
    calibration; see tests/test_lpse2d/test_srs.py::test_srs_seed_propagation).

    **Coupling scheme** (``terms.light.coupling``). The staggered real/imaginary update
    is a leapfrog only for a RHS operator that is i times a *real* matrix in the
    (Re, Im) basis. The propagation and detuning terms are; the coupling is not: its
    matrix element is the complex, spatially rotating ``laplacian phi``. Written out,
    the real parts of both waves are advanced together with an explicit Euler step on
    the part of the coupling that is proportional to Im(laplacian phi), whose spectral
    radius per sub-step is 1 + sin^2(arg laplacian phi) (Omega dt_l)^2 / 2 with
    Omega = e |laplacian phi| / (4 me sqrt(w0 w1)) the local exchange rate. Averaged over
    the EPW phase that is a growth rate Omega^2 dt_l / 4 of the light fields for *any*
    dt_l -- negligible at small EPW amplitude, but it scales with the EPW energy and
    feeds the EPW through the SRS/TPD sources, so a depleted-pump run runs away once
    Omega dt_l reaches ~0.01-0.1 (tests/test_lpse2d/test_light_coupling.py).

    ``coupling: explicit`` (default) keeps the MATLAB scheme. ``coupling: rotation``
    Strang-splits each sub-step as [exact coupling over dt_l/2] [staggered propagation
    with the coupling off] [exact coupling over dt_l/2]. With laplacian phi frozen, the
    coupling-only system has M^2 = A B |laplacian phi|^2 I = -Omega^2 I, so
    exp(tau M) = cos(Omega tau) I + sin(Omega tau)/Omega M exactly -- a rotation that
    conserves the action w1 |E0|^2 + w0 |E1|^2 pointwise and is stable for any dt_l.
    """

    def __init__(self, cfg: dict):
        # E1 solver: coefficients, stencils, sub-stepping, seed injector
        super().__init__(cfg)

        derived = cfg["units"]["derived"]
        self.E0_source = derived["E0_source"]
        background_density = cfg["grid"]["background_density"]

        # pump detuning/diffraction (MATLAB lines 1616-1626); with wp0^2 = w0^2 * n_env
        # the pump coefficient reduces to i w0/2 (1 - n)
        self.linear_coeff0 = (
            1j * self.w0 / 2.0 * (1.0 - self.wp0**2 / self.w0**2 * background_density / self.envelope_density)
        )
        self.diffraction_coeff0 = 1j * self.c**2 / (2.0 * self.w0)
        self.depletion_coeff0 = -1j * self.e / (4.0 * self.w1 * self.me)  # in dE0/dt, lap phi * E1

        # how the EPW coupling enters the light sub-step (see the class docstring)
        self.coupling = str(cfg["terms"].get("light", {}).get("coupling", "explicit"))
        if self.coupling not in ("explicit", "rotation"):
            raise ValueError(f"terms.light.coupling must be 'explicit' or 'rotation', got {self.coupling!r}")
        # local E0 <-> E1 exchange rate per unit |laplacian phi|: Omega = sqrt(|A B|) |lap phi|
        self.omega_prefactor = self.e / (4.0 * self.me * np.sqrt(self.w0 * self.w1))

        # TPD pump depletion (terms.light.tpd_depletion). The EPW potential source
        # S_k = i e/(8 wp0 me) exp(-i (w0 - 2 wp0) t) [F(E0_y conj(E_y)) + i ky/k^2 F(E0_y conj(rho))]
        # (epw.py calc_tpd_source, rho = div E = F^-1(k^2 phi_k)) changes the EPW energy
        # Int |grad phi|^2 at the rate -(e/(2 wp0 me)) Im[exp(-i dw t) Int E0_y rho* E_y*],
        # which the pump equation of the MATLAB original never pays back (its pump
        # depletion is SRS-only, line 1634). The conjugate term
        #     dE0_y/dt += i e/(4 wp0 me) exp(+i (w0 - 2 wp0) t) rho E_y
        # returns exactly that rate to Int |E0|^2 (tests/test_lpse2d/test_light_coupling.py),
        # so Int(|E0|^2 + |grad phi|^2) is conserved by the TPD pair as
        # Int(|E0|^2 + |E1|^2 + |grad phi|^2) already is by the SRS terms.
        self.tpd_depletion = bool(cfg["terms"].get("light", {}).get("tpd_depletion", False))
        if self.tpd_depletion and not cfg["terms"]["epw"]["source"].get("tpd", False):
            raise ValueError("terms.light.tpd_depletion requires terms.epw.source.tpd: true")
        self.tpd_depletion_coeff = 1j * self.e / (4.0 * self.wp0 * self.me)
        self.tpd_delta_w = self.w0 - 2.0 * self.wp0
        self.ky = cfg["grid"]["ky"]

        # optional isotropic low-pass filter on both light fields, applied once per EPW
        # step (terms.light.filter = fraction of the grid Nyquist wavenumber; default off).
        # The physical light content is |k| <= ~1.2 k0, far below the grid Nyquist; grid-
        # scale light modes have FD group velocity c^2 sin(k dx)/(w dx) -> 0 and the
        # staggered scheme's phase per sub-step is largest there (diagnostic option, see
        # srs-campaign srs-2d-testbed NOTES 'Blowup investigation').
        light_filter = cfg["terms"].get("light", {}).get("filter", None)
        if light_filter is None or light_filter is False:
            self.light_filter = None
        else:
            frac = float(light_filter)
            kx = np.asarray(cfg["grid"]["kx"])
            ky = np.asarray(cfg["grid"]["ky"])
            k_nyq = np.pi / float(self.dx)
            k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
            self.light_filter = jnp.asarray(np.where(k_mag <= frac * k_nyq, 1.0, 0.0))[..., None]

        # ---- pump injector (MATLAB lines 1707-1753, mirrored to the left edge) ----
        pump = cfg["drivers"]["E0"]["derived"]
        x_inject = cfg["grid"]["xmin"] + pump["offset"]
        self.i0 = int(np.argmin(np.abs(np.array(self.x) - x_inject)))
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

    def tpd_depletion_term(self, phi_k: Array) -> Array:
        """rho * E_y (real space) for the TPD pump-depletion term, from the frozen EPW
        potential; multiplied by i e/(4 wp0 me) exp(i (w0 - 2 wp0) t) in pump_rhs."""
        rho = jnp.fft.ifft2(self.k_sq * phi_k)
        ey = jnp.fft.ifft2(-1j * self.ky[None, :] * phi_k)
        return self.tpd_depletion_coeff * rho * ey

    def pump_rhs(
        self,
        t: float,
        E0: Array,
        E1: Array,
        laplacian_phi: Array,
        pump_args: dict,
        couple: bool = True,
        tpd_dep: Array | None = None,
    ) -> Array:
        """Pump RHS: propagation + detuning (MATLAB lines 1616-1626), SRS pump depletion
        (lines 1640-1646: no conjugate, w1 denominator) unless ``couple`` is False, the
        TPD pump depletion when ``tpd_dep`` (from tpd_depletion_term) is given, and the
        boundary injector."""
        e0x, e0y = E0[..., 0], E0[..., 1]
        e1x, e1y = E1[..., 0], E1[..., 1]

        k_e0x = self.diffraction_coeff0 * (self._d2y(e0x) - self._dxdy(e0y)) + self.linear_coeff0 * e0x
        k_e0y = self.diffraction_coeff0 * (self._d2x(e0y) - self._dxdy(e0x)) + self.linear_coeff0 * e0y
        if couple:
            k_e0x += self.depletion_coeff0 * laplacian_phi * e1x
            k_e0y += self.depletion_coeff0 * laplacian_phi * e1y
        if tpd_dep is not None:
            k_e0y += tpd_dep * jnp.exp(1j * self.tpd_delta_w * t)
        row_i0, row_i0p1 = self.calc_pump_source(t, pump_args)
        k_e0y = k_e0y.at[self.i0, :].add(row_i0)
        k_e0y = k_e0y.at[self.i0 + 1, :].add(row_i0p1)

        return jnp.stack([k_e0x, k_e0y], axis=-1)

    def coupled_rhs(
        self,
        t: float,
        E0: Array,
        E1: Array,
        laplacian_phi: Array,
        pump_args: dict,
        seed_args: dict | None,
        couple: bool = True,
        tpd_dep: Array | None = None,
    ) -> tuple[Array, Array]:
        # the E1 RHS (propagation + detuning + SRS coupling + seed rows) is exactly
        # the RamanLight one
        return (
            self.pump_rhs(t, E0, E1, laplacian_phi, pump_args, couple=couple, tpd_dep=tpd_dep),
            self.rhs(t, E1, E0, laplacian_phi, seed_args, couple=couple),
        )

    def couple(self, E0: Array, E1: Array, laplacian_phi: Array, tau: float) -> tuple[Array, Array]:
        """
        Exact solution over ``tau`` of the coupling-only system with laplacian phi frozen,

            dE0/dt = A L E1,   dE1/dt = B L* E0,   A = depletion_coeff0, B = srs_coeff,

        i.e. exp(tau M) = cos(Omega tau) I + sin(Omega tau)/Omega M with
        Omega = sqrt(|A B|) |L| = e |L| / (4 me sqrt(w0 w1)). Conserves w1|E0|^2 + w0|E1|^2
        at every point (Manley-Rowe for the light pair) for any tau.
        """
        omega = self.omega_prefactor * jnp.abs(laplacian_phi)
        omega_safe = jnp.where(omega > 0.0, omega, 1.0)
        cos_ = jnp.cos(omega * tau)[..., None]
        sinc_ = jnp.where(omega > 0.0, jnp.sin(omega * tau) / omega_safe, tau)
        a = (sinc_ * self.depletion_coeff0 * laplacian_phi)[..., None]
        b = (sinc_ * self.srs_coeff * jnp.conj(laplacian_phi))[..., None]
        return cos_ * E0 + a * E1, cos_ * E1 + b * E0

    def __call__(self, t: float, E0: Array, E1: Array, phi_k: Array, pump_args: dict, seed_args: dict | None):
        """
        Advance (E0, E1) over one EPW step: self.n_sub light sub-steps with the EPW
        potential held fixed.

        ``coupling == "explicit"`` matches MATLAB lightSplitStep: both real parts are
        updated with the full RHS at t_i, then both imaginary parts with the RHS at
        t_i + dt/2. ``coupling == "rotation"`` applies the exact coupling rotation for
        dt_l/2 on either side of the same staggered update with the coupling switched
        off (Strang splitting); see the class docstring for why.
        """
        seed_args = seed_args if self.seed_enabled else None
        laplacian_phi = jnp.fft.ifft2(-self.k_sq * phi_k)
        couple_in_rhs = self.coupling == "explicit"
        tpd_dep = self.tpd_depletion_term(phi_k) if self.tpd_depletion else None

        def propagate(t_i, E0, E1):
            k_e0, k_e1 = self.coupled_rhs(
                t_i, E0, E1, laplacian_phi, pump_args, seed_args, couple=couple_in_rhs, tpd_dep=tpd_dep
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
                couple=couple_in_rhs,
                tpd_dep=tpd_dep,
            )
            E0 = E0 + 1j * self.dt_l * jnp.imag(k_e0)
            E1 = E1 + 1j * self.dt_l * jnp.imag(k_e1)
            return E0, E1

        def substep(i, fields):
            E0, E1 = fields
            t_i = t + i * self.dt_l
            if couple_in_rhs:
                E0, E1 = propagate(t_i, E0, E1)
            else:
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
                E0, E1 = propagate(t_i, E0, E1)
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
            E0 = E0 * self.sub_boundary[..., None]
            E1 = E1 * self.sub_boundary[..., None]
            return (E0, E1)

        E0, E1 = lax.fori_loop(0, self.n_sub, substep, (E0, E1))
        if self.light_filter is not None:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.light_filter, axes=(0, 1))
            E1 = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.light_filter, axes=(0, 1))
        return E0, E1
