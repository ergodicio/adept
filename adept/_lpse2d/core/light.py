import jax
import numpy as np
from jax import Array, lax
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._lpse2d.core.kap import KapPhases
from adept._lpse2d.core.pulse import PulseShape
from adept._lpse2d.core.raman import RamanLight, light_source_mask, transverse_part
from adept._lpse2d.core.timeline import as_linear


def super_gaussian_y(y, width: float, order: float, offset: float) -> np.ndarray:
    """LPSE's transverse beam profile (``SchrodingerSolver3::superGaussian``) across y:
    ``exp(-(|y - offset| / W)^order)`` with ``W = sqrt(2) width`` (``beam_width`` is the Gaussian
    standard deviation, the translator's ``W / sqrt(2)``); 1 everywhere when the width or the order
    is 0, LPSE's periodic-injection case."""
    y = np.asarray(y, dtype=np.float64)
    if width <= 0.0 or order == 0.0:
        return np.ones_like(y)
    return np.exp(-((np.abs(y - offset) / (np.sqrt(2.0) * width)) ** order))


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

    The pump injector is the stencil's plane-wave injector (``RamanLight.injector_rows``:
    the two-point source at second order, wider at ``terms.light.fd_order`` 4 / 6) with
    amplitude E0_source * sqrt(intensity) / eps^(1/4); the launched wave carries the
    stencil's dispersion (``stencils.launched_amplitude_ratio``, a ~2 % deficit at 8 cells
    per wavelength at second order, 0.1 % at fourth), which the flux probes normalise out
    (tests/test_lpse2d/test_srs.py::test_pump_injector_calibration, test_fd_order.py).

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
        self.nx, self.ny = int(cfg["grid"]["nx"]), int(cfg["grid"]["ny"])
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
        # the pump's coupling sources are zeroed at the seed injector rows (and in the pump's layer on
        # request), the Raman light's at the pump's (LPSE LightSolver::calculateSources); the exact
        # exchange (coupling: rotation) cannot act one-sidedly and is switched off where either is
        self.source_mask0 = light_source_mask(cfg, 0)
        self.exchange_mask = self.source_mask0 * self.source_mask1

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

        # optional DIRECTIONAL (one-way) mask on the pump only: keep kx >= 0.
        # The pump's x-operator here is i c^2/(2 w0) d2x, whose numerical dispersion
        # w(k) = c^2 k^2 / (2 w0) is EVEN in kx, so -k0 is a degenerate, freely
        # propagating mode of the discretisation and the real-space SRS source
        # lap(phi)*E1 drives it as resonantly as +k0. Measured in srs-2d-testbed run 6:
        # the source carries 4-5x more power near -k0 than near +k0 where the backward
        # pump grows, and E0 reaches ~48% backward power by 10 ps, which is what drives
        # the incident-flux probe through zero. This option removes the backward half of
        # the pump spectrum so that contribution can be tested directly. Diagnostic,
        # default off; E1 is untouched (SRS backscatter is legitimately kx < 0).
        one_way = cfg["terms"].get("light", {}).get("one_way", False)
        if one_way:
            kx_ow = np.asarray(cfg["grid"]["kx"])
            self.one_way_mask = jnp.asarray(np.where(kx_ow >= 0.0, 1.0, 0.0))[:, None, None]
        else:
            self.one_way_mask = None

        # ---- pump injector (MATLAB lines 1707-1753, mirrored to the left edge) ----
        # a leftward beam (drivers.E0.angle 180, plan 2 L.4a) is launched from the x-max face
        # at xmax - offset instead, as LPSE's x-max injector
        pump = cfg["drivers"]["E0"]["derived"]
        leftward = np.atleast_1d(np.asarray(pump.get("beam_leftward", [False]), dtype=bool))
        # (the spectral subclass keeps this x-min plane for its rightward beams when the beams
        # are mixed, and adds its own x-max plane)
        self.pump_direction = -1 if bool(np.all(leftward)) else 1
        if self.pump_direction == 1:
            x_inject = cfg["grid"]["xmin"] + pump["offset"]
        else:
            x_inject = cfg["grid"]["xmax"] - pump["offset"]
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
        # LPSE laser.pulseShape: every injected source carries sqrt(shape) (core/pulse.py)
        self.pulse = PulseShape(pump)
        # LPSE KAP bandwidth on every injector (core/kap.py)
        self.kap = KapPhases(
            float(pump.get("kap_bandwidth", 0.0) or 0.0),
            self.w0,
            len(np.atleast_1d(pump.get("beam_fraction", [1.0]))),
            cfg["grid"]["tmax"],
            int(pump.get("kap_seed", 0) or 0),
        )
        self.source_prefactor0 = self.c**2 / (2.0 * self.w0) / permittivity0**0.25 / self.dx**2

        # ---- resonance absorption (plan 2 L.3; LPSE laser.evolution.resonanceAbsorption) on the
        # pump, applied every light sub-step after the propagation update
        from adept._lpse2d.core.epw import analytic_landau_rate
        from adept._lpse2d.core.resonance import ResonanceAbsorption

        ra_cfg = light_cfg.get("resonance_absorption", None)
        if ra_cfg:
            # LPSE builds the Landau table whenever RA is on, whatever lw.landauDamping.enable
            cfg_ld = {
                **cfg,
                "terms": {
                    **cfg["terms"],
                    "epw": {**cfg["terms"]["epw"], "damping": {**cfg["terms"]["epw"]["damping"], "landau": True}},
                },
            }
            landau_rate = analytic_landau_rate(cfg_ld)
        else:
            landau_rate = None
        self.resonance = ResonanceAbsorption(cfg, self, self.dt_l, landau_rate)

        # ---- general FD injector (plan 2 L.4): oblique, multiple or transversely profiled beams
        # use the same commutator on the full 2-D operator, S = D[H V] - H D[V], with V the
        # analytic beam (polarised perpendicular to its k, super-Gaussian in y); the axial
        # single-beam default keeps the row form above (bit-identical to the prototype)
        angles = np.atleast_1d(np.asarray(pump.get("beam_angle", [pump.get("angle", 0.0)]), dtype=np.float64))
        self.beam_width = float(pump.get("beam_width", 0.0) or 0.0)
        self.fd_general_injector = bool(
            angles.size > 1 or np.any(np.abs(np.sin(angles)) > 1e-12) or (self.beam_width > 0.0 and self.ny > 1)
        )
        if self.fd_general_injector:
            x_np, y_np = np.asarray(self.x), np.asarray(self.y)
            self.i0_max = int(np.argmin(np.abs(x_np - (cfg["grid"]["xmax"] - pump["offset"]))))
            self.beam_i_inject = [self.i0_max if left else self.i0 for left in leftward]
            self.beam_n_src = [float(background_density[i, 0]) for i in self.beam_i_inject]
            if any(n >= 1.0 for n in self.beam_n_src):
                raise ValueError("a pump injector plane sits at or above the critical density")
            k0_beams = self.w0 / self.c * np.sqrt(1.0 - np.asarray(self.beam_n_src))
            dky = 2.0 * np.pi / (self.ny * self.dy)
            snap = bool(light_cfg.get("snap_beam_ky", True))
            ky_beams = k0_beams * np.sin(angles) if self.ny > 1 else np.zeros_like(angles)
            if snap and self.ny > 1:
                ky_beams = np.round(ky_beams / dky) * dky
            # every colour of every beam must have kx^2 = k0(dw)^2 - ky^2 > 0 at its plane
            dw_min = float(np.min(np.atleast_1d(pump.get("beam_delta_omega", [0.0])))) - float(
                cfg["drivers"]["E0"].get("delta_omega_max", 0.0) or 0.0
            )
            k0_low = self.w0 / self.c * np.sqrt(np.maximum((1.0 + dw_min) ** 2 - np.asarray(self.beam_n_src), 0.0))
            if np.any(np.abs(ky_beams) >= 0.999 * k0_low):
                bad = np.rad2deg(angles[np.abs(ky_beams) >= 0.999 * k0_low])
                raise ValueError(
                    f"drivers.E0 beams at {np.round(bad, 2)} deg graze the x face (|ky| >= k0 for some colour): "
                    "adept injects from the x faces only"
                )
            self.ky_beams = ky_beams
            self.beam_sign = [-1 if left else 1 for left in leftward]
            self.beam_fraction = np.atleast_1d(np.asarray(pump.get("beam_fraction", [1.0]), dtype=np.float64))
            self.beam_phase = np.atleast_1d(np.asarray(pump.get("beam_phase", [0.0]), dtype=np.float64))
            self.beam_delta_omega = np.atleast_1d(np.asarray(pump.get("beam_delta_omega", [0.0]), dtype=np.float64))
            self.beam_polarization = np.atleast_1d(
                np.asarray(pump.get("beam_polarization", [pump.get("polarization", 0.0)]), dtype=np.float64)
            )
            if self.ny > 1:
                self.beam_envelope_y = jnp.asarray(
                    super_gaussian_y(
                        y_np,
                        self.beam_width,
                        float(pump.get("beam_sg_order", 4.0)),
                        float(pump.get("beam_offset", 0.0)),
                    )
                )
            else:
                self.beam_envelope_y = jnp.ones(self.ny)
            m = self.fd_order // 2
            # the commutator is non-zero on the stencil's rows about each plane only: offsets
            # -m+1 .. m from the plane for either direction -- the mask edge lies between rows i
            # and i+1 for a rightward (H = 1 above i) and a leftward (H = 1 up to i) beam alike
            # (stencils.injector_offsets)
            self.beam_rows = [np.arange(i - m + 1, i + m + 1) for i in self.beam_i_inject]
            self.beam_masks = [
                jnp.asarray(
                    np.where(np.arange(self.nx) >= i + 1, 1.0, 0.0)
                    if sign > 0
                    else np.where(np.arange(self.nx) <= i, 1.0, 0.0)
                )
                for i, sign in zip(self.beam_i_inject, self.beam_sign, strict=True)
            ]

        # ---- injector from LPSE files (plan 2 L.4c; drivers.E0.injector_file)
        self.file_injector = "injector_planes" in pump
        if self.file_injector:
            self.file_times, self.file_patterns, self.file_rows = self.file_injector_patterns(
                pump["injector_times"], pump["injector_planes"]
            )

    def file_injector_patterns(self, times, planes) -> tuple[Array, Array, np.ndarray]:
        """The file injector's source rows per file time, ``(T, 2, ny, 3)``, and their row indices.

        LPSE's ``SchrodingerSolver3::getInjectorSource`` (second order): with W the field that is
        the file's first plane on the first injected row p, its second plane on the next row inward
        and zero elsewhere, the source is ``-(L W)`` on the row outside the plane (p - s) and on p,
        L the pump's propagation operator (diffraction on the curl-curl + detuning, background
        density only). It is the total-field / scattered-field source ``S = H L V - L (H V)``
        with the incident field V's row outside the plane eliminated by assuming ``L V = 0`` at p,
        i.e. that the file holds a steady solution -- the only form the two planes allow."""
        s = self.pump_direction
        p = self.i0 + 1 if s > 0 else self.i0
        rows = np.array([p - s, p])
        planes = jnp.asarray(planes)
        linear = jnp.asarray(self.linear_coeff0)[..., None]

        def source(pl):
            w = jnp.zeros((self.nx, self.ny, 3), dtype=planes.dtype).at[p].set(pl[0]).at[p + s].set(pl[1])
            lw = self.diffraction_coeff0 * jnp.stack(self.curl_curl(w), axis=-1) + linear * w
            return -lw[rows]

        return jnp.asarray(times), jax.vmap(source)(planes), rows

    def pump_time_factor(self, t: float, pump_args: dict):
        """The scalar time factor of every pump source: the driver envelope, the turn-on ramp
        ``1 - exp(-(t / turn_on_time)^2)`` and the square root of the pulse shape's power factor,
        LPSE's ``temporalSourceAmplitudeMultiplier`` (``SchrodingerSolver3::addInjectorSources``)."""
        t_env = get_envelope(
            pump_args["tr"],
            pump_args["tr"],
            pump_args["tc"] - pump_args["tw"] / 2,
            pump_args["tc"] + pump_args["tw"] / 2,
            t,
        )
        turn_on = 1.0 - jnp.exp(-((t / self.pump_turn_on_time) ** 2))
        return t_env * turn_on * self.pulse.field_factor(t)

    def file_pump_rows(self, t: float, pump_args: dict) -> Array:
        """The file injector's rows ``(2, ny, 3)`` at time ``t``: LPSE interpolates linearly between
        the file times and repeats the table with the last time as period
        (``SchrodingerSolver3::addInjectorSources``, loadInjector); one time is held constant. As
        every pump source, the rows carry the pump envelope and the turn-on ramp (LPSE
        ``laser.evolution.riseTime``)."""
        time_factor = self.pump_time_factor(t, pump_args)
        if self.file_times.size == 1:
            pattern = self.file_patterns[0]
        else:
            period = self.file_times[-1]
            t_c = t - jnp.floor(t / period) * period
            k = jnp.clip(jnp.searchsorted(self.file_times, t_c, side="right") - 1, 0, self.file_times.size - 2)
            a = (t_c - self.file_times[k]) / (self.file_times[k + 1] - self.file_times[k])
            pattern = (1.0 - a) * self.file_patterns[k] + a * self.file_patterns[k + 1]
        return time_factor * pattern

    def calc_pump_source(self, t: float, pump_args: dict) -> list[tuple[int, Array]]:
        """
        Pump injector rows, summed over colors (MATLAB lines 1738-1750 at second order,
        with +k0 and the left edge instead of -k1 and the right edge; ``injector_rows``
        for the stencil order), launching the rightward wave into ``x >= x[i0 + 1]`` (or, for
        a leftward pump, ``e^{-i k0 x}`` into ``x <= x[i0]`` from the x-max face).

        Returns ``(row index, values)`` pairs added to the E0 RHS.
        """
        time_factor = self.pump_time_factor(t, pump_args)

        delta_omega = pump_args["delta_omega"]  # (nc,)
        intensities = pump_args["intensities"]  # (nc, ny), fractions summing to 1
        phases = pump_args["phases"]  # (nc, ny)

        # local pump wavenumber per color (MATLAB kSource0). The two-point source
        # launches amplitude E_src * sin(k0 dx)/sin(k_grid dx) / eps^(1/4) -- a ~2%
        # deficit at 8 cells/wavelength from the grid dispersion; the budget metrics
        # normalize to the *measured* incident flux, so this bias cancels there.
        k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega) ** 2 - self.n_src)  # (nc,)

        amp = self.source_prefactor0 * self.E0_source * jnp.sqrt(intensities) * time_factor  # (nc, ny)

        color_phase = jnp.exp(
            -1j * self.w0 * delta_omega[:, None] * t + 1j * (phases + self.kap.phase(t, 0))
        )  # (nc, ny)

        sign = self.pump_direction

        def wave(i):
            return jnp.sum(1j * amp * jnp.exp(1j * sign * k0[:, None] * self.x[i]) * color_phase, axis=0)

        return self.injector_rows(self.i0, sign, wave)

    def pump_pattern(self, pump_args: dict) -> list[Array]:
        """The general injector's spatial source per beam, ``(nc, n_rows, ny, 3)``: the commutator
        ``D[H V] - H D[V]`` of the curl-curl operator with the beam's mask, for the analytic beam
        ``V = sqrt(I_c(y)) e^{i phases_c(y)} env(y) e^{i (kx (x - x_inj) + ky y)} pol`` of every
        colour, restricted to the stencil's rows about the plane. Time enters only through the
        scalar factors applied in ``pump_rhs``, so this is evaluated once per EPW step."""
        delta_omega = pump_args["delta_omega"]  # (nc,)
        amp_y = jnp.sqrt(pump_args["intensities"]) * jnp.exp(1j * pump_args["phases"])  # (nc, ny)
        amp_y = amp_y * self.beam_envelope_y[None, :]
        x = self.x[None, :, None]
        y = self.y[None, None, :]
        patterns = []
        for b in range(len(self.beam_sign)):
            sign, n_src, i_inject = self.beam_sign[b], self.beam_n_src[b], self.beam_i_inject[b]
            ky_b = float(self.ky_beams[b])
            k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega + self.beam_delta_omega[b]) ** 2 - n_src)  # (nc,)
            kx = sign * jnp.sqrt(k0**2 - ky_b**2)  # (nc,)
            carrier = jnp.exp(1j * (kx[:, None, None] * (x - self.x[i_inject]) + ky_b * y))  # (nc, nx, ny)
            v = carrier * amp_y[:, None, :] / (1.0 - n_src) ** 0.25
            k_mag = jnp.sqrt(kx[0] ** 2 + ky_b**2)
            cos_psi, sin_psi = np.cos(self.beam_polarization[b]), np.sin(self.beam_polarization[b])
            # LPSE rotateBeam: cos(psi) along the in-plane transverse direction of the first
            # colour's k, sin(psi) along z
            pol = jnp.stack([-ky_b / k_mag * cos_psi, kx[0] / k_mag * cos_psi, sin_psi])
            vvec = v[..., None] * pol[None, None, None, :]  # (nc, nx, ny, 3)
            h = self.beam_masks[b][:, None, None]
            rows = self.beam_rows[b]

            def commutator(vc, h=h, rows=rows):
                d_hv = jnp.stack(self.curl_curl(h * vc), axis=-1)
                d_v = jnp.stack(self.curl_curl(vc), axis=-1)
                return (d_hv - h * d_v)[rows]

            patterns.append(jax.vmap(commutator)(vvec))  # (nc, n_rows, ny, 3)
        return patterns

    def general_pump_rows(self, t: float, pump_args: dict, patterns: list[Array]) -> list[tuple[int, Array]]:
        """Rows ``(index, (ny, 3) values)`` of the general injector at time ``t``: the patterns
        times the colour and beam time factors and the propagation coefficient."""
        time_factor = self.pump_time_factor(t, pump_args)
        delta_omega = pump_args["delta_omega"]
        color_time = jnp.exp(-1j * self.w0 * delta_omega * t)  # (nc,)
        rows: dict[int, Array] = {}
        for b, pattern in enumerate(patterns):
            beam_time = jnp.exp(
                1j * (self.beam_phase[b] - self.w0 * self.beam_delta_omega[b] * t + self.kap.phase(t, b))
            )
            scale = self.diffraction_coeff0 * self.E0_source * time_factor * np.sqrt(self.beam_fraction[b]) * beam_time
            block = scale * jnp.sum(pattern * color_time[:, None, None, None], axis=0)  # (n_rows, ny, 3)
            for r, i in enumerate(self.beam_rows[b]):
                rows[int(i)] = block[r] if int(i) not in rows else rows[int(i)] + block[r]
        return list(rows.items())

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
        patterns: list[Array] | None = None,
    ) -> Array:
        """Pump RHS: propagation + detuning (MATLAB lines 1616-1626), SRS pump depletion
        (lines 1640-1646: no conjugate, w1 denominator) unless ``couple`` is False (the
        rotation scheme integrates that exchange exactly outside the RHS), the TPD pump
        depletion, and the boundary injector."""
        linear_coeff0 = self.linear_coeff0
        if iaw_density is not None:
            # MATLAB: i*w0/2 * [1 - wp0^2/w0^2 * (n_b/n_env + Nelf)] E0
            linear_coeff0 = linear_coeff0 - 1j * self.wp0**2 / (2.0 * self.w0) * iaw_density * self.iaw_feedback0

        # discrete curl-curl on the in-plane components, the plain Laplacian on E0z (k_z = 0)
        comps = [E0[..., i] for i in range(E0.shape[-1])]
        k_e0 = [
            self.diffraction_coeff0 * cc + linear_coeff0 * e for cc, e in zip(self.curl_curl(E0), comps, strict=True)
        ]
        if self.srs_enabled and couple:
            depletion = (self.srs_depletion_coeff0 * laplacian_phi * self.source_mask0)[..., None] * E1
            if self.transverse_source:
                depletion = transverse_part(depletion, self.kx_arr, self.ky_arr, self.one_over_k_sq)
            k_e0 = [k + depletion[..., i] for i, k in enumerate(k_e0)]
        if self.tpd_enabled:
            if phi_k is None:
                raise ValueError("phi_k is required for TPD pump depletion")
            tpd_dep = self.calc_tpd_depletion(t, phi_k)  # in-plane only: E_h has no z component
            if not isinstance(self.source_mask0, float):
                tpd_dep = tpd_dep * self.source_mask0[..., None]
            k_e0[0] = k_e0[0] + tpd_dep[..., 0]
            k_e0[1] = k_e0[1] + tpd_dep[..., 1]
        if self.fd_general_injector:
            if patterns is None:
                patterns = self.pump_pattern(pump_args)
            for i, block in self.general_pump_rows(t, pump_args, patterns):
                for c in range(len(k_e0)):
                    k_e0[c] = k_e0[c].at[i, :].add(block[:, c])
                if len(k_e0) == 2 and bool(np.any(np.sin(self.beam_polarization) != 0.0)):
                    raise ValueError("an out-of-plane (s-polarised) pump needs three-component light fields")
        elif self.file_injector:
            block = self.file_pump_rows(t, pump_args)
            for r, i in enumerate(self.file_rows):
                for c in range(len(k_e0)):
                    k_e0[c] = k_e0[c].at[int(i), :].add(block[r, :, c])
        else:
            rows = self.calc_pump_source(t, pump_args)
            for c, w in zip((1, 2), self.pump_weights, strict=True):
                if w == 0.0:
                    continue
                if c >= len(k_e0):
                    raise ValueError("an out-of-plane (s-polarised) pump needs three-component light fields")
                for i, row in rows:
                    k_e0[c] = k_e0[c].at[i, :].add(w * row)

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
        patterns: list[Array] | None = None,
    ) -> tuple[Array, Array]:
        # the E1 RHS (propagation + detuning + SRS coupling + seed rows) is exactly
        # the RamanLight one
        pump_rhs = self.pump_rhs(
            t, E0, E1, laplacian_phi, pump_args, iaw_density, phi_k, couple=couple, patterns=patterns
        )
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
        laplacian_phi = laplacian_phi * self.exchange_mask
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
        # the EPW potential and the ion density at the middle of each sub-step (LPSE
        # interpolateSourcesInTime; plain arrays are held fixed); the Laplacian is interpolated
        # in x-space so the sub-steps need no FFT
        phi_tl = as_linear(phi_k)
        laplacian_tl = phi_tl.map(lambda p: jnp.fft.ifft2(-self.k_sq * p))
        iaw_tl = as_linear(iaw_density)
        rotate = self.coupling == "rotation" and self.srs_enabled
        couple_in_rhs = not rotate
        # the general injector's spatial pattern once per EPW step (time factors per sub-step)
        patterns = self.pump_pattern(pump_args) if self.fd_general_injector else None

        def propagate(t_i, E0, E1, laplacian_phi, dn, phi_i):
            k_e0, k_e1 = self.coupled_rhs(
                t_i,
                E0,
                E1,
                laplacian_phi,
                pump_args,
                seed_args,
                dn,
                phi_i,
                couple=couple_in_rhs,
                patterns=patterns,
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
                dn,
                phi_i,
                couple=couple_in_rhs,
                patterns=patterns,
            )
            E0 = E0 + 1j * self.dt_l * jnp.imag(k_e0)
            E1 = E1 + 1j * self.dt_l * jnp.imag(k_e1)
            return E0, E1

        def absorption(dn):
            # each field at its own rate (LPSE {laser|raman}.evolution.absorption); an unset rate is 1
            dn0 = None if dn is None else dn * self.iaw_feedback0 / self.n_over_env
            dn1 = None if dn is None else dn * self.iaw_feedback / self.n_over_env
            n0 = self.n_over_nc0 if dn0 is None else self.n_over_nc0 * (1.0 + dn0)
            n1 = self.n_over_nc1 if dn1 is None else self.n_over_nc1 * (1.0 + dn1)
            a0 = (
                1.0 if self.absorption_rate0 is None else jnp.exp(-self.absorption_rate0 * self.dt_l * n0**2)[..., None]
            )
            a1 = (
                1.0 if self.absorption_rate1 is None else jnp.exp(-self.absorption_rate1 * self.dt_l * n1**2)[..., None]
            )
            return a0, a1

        def substep(i, fields):
            E0, E1 = fields
            t_i = t + i * self.dt_l
            laplacian_phi = laplacian_tl.substep(i, self.n_sub)
            phi_i = phi_tl.substep(i, self.n_sub)
            dn = iaw_tl.substep(i, self.n_sub)
            if rotate:
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
                E0, E1 = propagate(t_i, E0, E1, laplacian_phi, dn, phi_i)
                E0, E1 = self.couple(E0, E1, laplacian_phi, 0.5 * self.dt_l)
            else:
                E0, E1 = propagate(t_i, E0, E1, laplacian_phi, dn, phi_i)
            if self.resonance.enabled:
                E0 = self.resonance(t_i, i, E0)
            E0 = E0 * self.sub_boundary0[..., None]
            E1 = E1 * self.sub_boundary[..., None]
            if absorbing:
                a0, a1 = absorbs if iaw_tl.constant else absorption(dn)
                E0 = E0 * a0
                E1 = E1 * a1
            return (E0, E1)

        # the pump's and the Raman light's rates are independent (A21): either may be unset
        absorbing = self.absorption_rate0 is not None or self.absorption_rate1 is not None
        absorbs = absorption(iaw_tl.new) if absorbing and iaw_tl.constant else None
        E0, E1 = lax.fori_loop(0, self.n_sub, substep, (E0, E1))
        if self.one_way_mask is not None:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.one_way_mask, axes=(0, 1))
        if self.light_filter is not None:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.light_filter, axes=(0, 1))
            E1 = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.light_filter, axes=(0, 1))
        if self.transverse_fields:
            # drop the longitudinal part the FD curl-curl generated over the sub-steps (see
            # RamanLight); a y-uniform plane-wave pump is unchanged
            E0 = transverse_part(E0, self.kx_arr, self.ky_arr, self.one_over_k_sq)
            E1 = transverse_part(E1, self.kx_arr, self.ky_arr, self.one_over_k_sq)
        return E0, E1
