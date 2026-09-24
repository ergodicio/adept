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

**Normalisation of the feedback.** The drive above is density-independent, so ``n`` is the
*local* fractional perturbation ``delta n / n_b(x)`` -- LPSE's ``Nelf``. The waves see the
density ``n_b (1 + n)``: in units of the envelope density the perturbation is
``n * n_b / n_env`` (LPSE ``densityPerturbation += Nelf * backgroundDensity``,
``ZakharovSolver::densityUpdateOfE_fft`` and ``LightSolver.cpp:4063``). The MATLAB prototype adds
``Nelf`` to ``n_b / n_env`` directly, which is exact only where ``n_b = n_env``; it is kept as
``terms.iaw.feedback: envelope`` (``iaw_feedback_factor``).
"""

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

IAW_SOLVERS = ("explicit", "spectral", "fd")
ION_LANDAU_FORMS = ("simplified", "full")


IAW_WAVES = ("epw", "pump", "raman")


def iaw_feedback_factor(cfg: dict, wave: str = "epw"):
    """The factor that turns ``iaw_density`` (the local fraction ``delta n / n_b``) into the
    perturbation ``wave`` (``epw``, ``pump`` or ``raman``) sees in units of the envelope density:
    ``n_b / n_env`` (LPSE, ``terms.iaw.feedback: local``, the default) or 1 (the MATLAB prototype's
    ``n_b / n_env + Nelf``, ``feedback: envelope``); 0 when ``terms.iaw.perturbs.<wave>`` is off
    (LPSE ``{lw|laser|raman}.ionAcousticPerturbations.enable``: that wave's scattering potential and
    absorption then ignore the IAW density). Shape ``(nx, ny)`` or a scalar."""
    iaw = cfg["terms"].get("iaw") or {}
    mode = str(iaw.get("feedback", "local"))
    if mode not in ("local", "envelope"):
        raise ValueError(f"terms.iaw.feedback must be local or envelope, got {mode!r}")
    if wave not in IAW_WAVES:
        raise ValueError(f"iaw_feedback_factor: wave must be one of {IAW_WAVES}, got {wave!r}")
    if iaw.get("active", False) and not bool((iaw.get("perturbs") or {}).get(wave, True)):
        return 0.0
    if not iaw.get("active", False) or mode == "envelope":
        return 1.0
    return jnp.asarray(np.asarray(cfg["grid"]["background_density"]) / float(cfg["units"]["envelope density"]))


def _Q_kev(value) -> float:
    from adept._lpse2d.helpers import _Q

    return float(_Q(value).to("keV").value)


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
        # x-space window on the ponderomotive drive, squared (plan 2 I.3); 1.0 when none
        mask = grid.get("iaw_source_mask")
        self.source_mask_sq = 1.0 if mask is None or bool(np.all(np.asarray(mask) == 1.0)) else jnp.asarray(mask) ** 2
        # terms.iaw.t_start / t_stop (ps): the IAW step only acts inside this interval (LPSE
        # iaw.startEvolvingTime / stopEvolvingTime); None means unbounded
        self.t_start = float(iaw["t_start"]) if iaw.get("t_start") is not None else None
        self.t_stop = float(iaw["t_stop"]) if iaw.get("t_stop") is not None else None
        if self.stride > 1 and self.solver == "explicit":
            # the fd solver sub-cycles its IAW step to its own stability limit (iaw_fd.FDIonAcoustic)
            raise ValueError(
                "terms.iaw.stride > 1 requires terms.iaw.solver: spectral or fd (the explicit step is not stable)"
            )

        self.dt = grid["dt"] * self.stride
        self.dx = grid["dx"]
        self.dy = grid["dy"]
        self.nx = grid["nx"]
        self.ny = grid["ny"]
        self.kx = grid["kx"]
        self.ky = grid["ky"]
        self.k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2
        # the IAW's own band: the anti-aliased band and terms.iaw.max_wavenumber, not the EPW's cap
        self.filter = grid.get("iaw_low_pass_filter_grid", grid["low_pass_filter_grid"]) * grid["zero_mask"]
        self.boundary = grid["iaw_absorbing_boundaries"]

        self.cs = derived["cs"]
        self.wp0 = derived["wp0"]
        self.w0 = derived["w0"]
        self.w1 = derived["w1"]
        # the combined solver's field is the Raman class with carrier wp0 and carries the EPW as its
        # longitudinal part (LightSolver.cpp:2405-2420): the drive and the heating take the whole
        # field at wp0 and no separate EPW term (IawSolver.cpp:211-217; ZakharovSolver.cpp
        # getThermalFilamentationSource, thermalFil.lw only without the combined solver)
        self.combined = cfg["terms"]["epw"].get("solver", "separate") == "combined"
        self.w_raman = self.wp0 if self.combined else self.w1
        self.ponderomotive_prefactor = (
            cfg["units"]["ionization state"] * derived["e"] ** 2 / (4.0 * derived["me"] * derived["mi"])
        )
        self._init_thermal_filamentation(cfg)
        # the ponderomotive drive's channels (LPSE iaw.sourceTerm.{lw|laser|raman}.enable) and the
        # waves the IAW density feeds back into ({lw|laser|raman}.ionAcousticPerturbations.enable)
        drive_cfg = iaw.get("drive") or {}
        self.drive_on = {w: bool(drive_cfg.get(w, True)) for w in IAW_WAVES}
        perturbs_cfg = iaw.get("perturbs") or {}
        if cfg["terms"]["epw"].get("solver", "separate") == "combined":
            # LPSE: the combined field is both the EPW and the Raman light (LightSolver.cpp:1158,
            # IawSolver.cpp:211-216)
            if bool(perturbs_cfg.get("epw", True)) != bool(perturbs_cfg.get("raman", True)):
                raise ValueError("terms.iaw.perturbs: epw and raman must agree with the combined solver (LPSE)")
            if self.drive_on["epw"] != self.drive_on["raman"]:
                raise ValueError("terms.iaw.drive: epw and raman must agree with the combined solver (LPSE)")

        damping = iaw["damping"]
        k_sq_np = np.asarray(self.k_sq)
        self.landau_rate = jnp.asarray(ion_landau_rate(cfg, k_sq_np))
        self.nu_coll = float(damping["collisions"])
        # explicit (MATLAB): collisions act on n as (1 - nu dt); spectral (LPSE): on w as e^{-2 nu dt}
        self.collisional_factor = 1.0 - self.nu_coll * self.dt
        self.max_density_perturbation = iaw["max_density_perturbation"]

        # uniform background flow (Mach number along x, y) -> Doppler phase in the propagator;
        # the fd solver takes a profile too (a mapping; iaw_fd.flow_profile)
        flow = iaw.get("flow")
        if flow is None:
            self.flow = None
        elif self.solver == "fd":
            self.flow = flow
        else:
            if self.solver != "spectral" or isinstance(flow, dict):
                raise ValueError(
                    "a uniform terms.iaw.flow requires terms.iaw.solver: spectral; a profile the fd solver"
                )
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

        # the finite-difference solver with flow profiles (plan 2 I.1 / I.2); built last, it reads
        # the damping and noise set up above
        if self.solver == "fd":
            from adept._lpse2d.core.iaw_fd import FDIonAcoustic

            self.fd = FDIonAcoustic(self, cfg)
        else:
            self.fd = None

    def _init_thermal_filamentation(self, cfg: dict) -> None:
        """LPSE ``thermalFil.{laser,raman,lw}`` (``ZakharovSolver::getThermalFilamentationSource``):
        inverse-bremsstrahlung heating by the spatially varying part of each wave's intensity,
        balanced by Spitzer heat conduction, drives the ion flow through the electron pressure:

            d(div v)/dt += Z Q_w / (m_i kappa'),   Q_w = nu_w(n) (|E_w|^2 - <|E_w|^2>) / (8 pi)

        with ``kappa'`` the Spitzer conductivity over k_B (1/(cm s)), ``nu_w`` the wave's energy
        damping rate (light: 2 nu_abs(n_c) (n/n_c)^2; EPW: 2 nu_coll n/n_env), ``<.>`` the box
        average. ``nonlocal: true`` adds LPSE's k^(4/3) correction: the source's k-space content is
        multiplied by ``1 + (k lambda_nl)^(4/3)`` with ``lambda_nl = 30 (k_B T_e)^2 / (4 pi e^4
        sqrt(Z+1) ln Lambda n_e)``. The LPSE form is kept; the normalization is adept's own
        (derived from the heat and momentum equations, not LPSE's ZAK constants)."""
        tf = cfg["terms"]["iaw"].get("thermal_filamentation") or {}
        self.thermal_waves = [w for w in ("laser", "raman", "lw") if tf.get(w, False)]
        if self.combined and "lw" in self.thermal_waves:
            # LPSE: thermalFil.lw acts only without the combined solver, whose field (thermalFil.raman)
            # carries the EPW
            self.thermal_waves.remove("lw")
            print("NOTE: terms.iaw.thermal_filamentation.lw is ignored with the combined solver (LPSE); use raman")
        if not self.thermal_waves:
            return
        derived = cfg["units"]["derived"]
        units = cfg["units"]
        from adept._lpse2d.core.raman import light_absorption_rates

        rate0, rate1 = light_absorption_rates(cfg)
        z = float(units["ionization state"])
        te_kev = float(derived.get("Te", 0.0)) or float(_Q_kev(units["reference electron temperature"]))
        # cgs constants
        e_cgs, me_cgs, mp_cgs, kb_cgs = 4.8032068e-10, 9.10938291e-28, 1.6726219e-24, 1.380649e-16
        te_k = te_kev * 1.0e3 * 1.16045e4
        lambda_um = 2.0 * np.pi * derived["c"] / derived["w0"]
        nc_cgs = 1.1148e21 / lambda_um**2
        n_cgs = np.asarray(cfg["grid"]["background_density"]) * nc_cgs
        n_mean = float(np.mean(n_cgs))
        log_lambda = max(
            23.5 - np.log(np.sqrt(n_mean) * te_kev**-1.25 * 1e3**-1.25 * 1e3)
            if False
            else 6.68 + np.log(lambda_um * te_kev),
            2.0,
        )
        g_factor = 1.0 / (1.0 + 3.3 / z)
        kappa = (8.0 / np.pi) ** 1.5 * g_factor * (kb_cgs * te_k) ** 2.5 / (z * e_cgs**4 * np.sqrt(me_cgs) * log_lambda)
        kappa *= float(tf.get("conductivity_multiplier", 1.0))
        mi_cgs = mp_cgs * float(units["atomic number"])
        field_scale = float(derived["fieldScale"])  # statV/cm per code field unit
        # d(div v)/dt [1/s^2] = Z nu(n)[1/s] |E|^2_cgs / (8 pi m_i kappa'); -> 1/ps^2 per code |E|^2
        base = z * field_scale**2 / (8.0 * np.pi * mi_cgs * kappa) * 1.0e-24 * 1.0e12  # per (1/ps rate)
        n_over_nc = np.asarray(cfg["grid"]["background_density"])
        self.thermal_coeff = {}
        if "laser" in self.thermal_waves:
            if rate0 is None:
                raise ValueError("terms.iaw.thermal_filamentation.laser needs terms.light.absorption")
            self.thermal_coeff["laser"] = jnp.asarray(base * 2.0 * rate0 * n_over_nc**2)
        if "raman" in self.thermal_waves:
            if rate1 is None:
                raise ValueError("terms.iaw.thermal_filamentation.raman needs terms.light.absorption")
            nc1 = (self.w_raman / derived["w0"]) ** 2  # the Raman class's critical density (n_env combined)
            self.thermal_coeff["raman"] = jnp.asarray(base * 2.0 * rate1 * (n_over_nc / nc1) ** 2)
        if "lw" in self.thermal_waves:
            nu_coll = float(derived.get("nu_coll", 0.0))
            if nu_coll <= 0.0:
                raise ValueError("terms.iaw.thermal_filamentation.lw needs terms.epw.damping.collisions")
            self.thermal_coeff["lw"] = jnp.asarray(base * 2.0 * nu_coll * n_over_nc / float(units["envelope density"]))
        self.thermal_nonlocal = bool(tf.get("nonlocal", tf.get("nonlocal_", False)))
        if self.thermal_nonlocal:
            lambda_nl_cm = (
                30.0 * (kb_cgs * te_k) ** 2 / (4.0 * np.pi * e_cgs**4 * np.sqrt(z + 1.0) * log_lambda * n_cgs)
            )
            lambda_nl_um = lambda_nl_cm * 1.0e4
            k_mag = np.sqrt(self.k_sq)
            # (k lambda_nl(n))^(4/3): k^(4/3) in k-space, lambda^(4/3)(x) in x-space
            self.thermal_k_factor = jnp.asarray(k_mag ** (4.0 / 3.0))
            self.thermal_lambda_factor = jnp.asarray(lambda_nl_um ** (4.0 / 3.0))
        print(f"IAW thermal filamentation on ({', '.join(self.thermal_waves)}, nonlocal={self.thermal_nonlocal})")

    def thermal_filamentation_source(self, phi_k: Array, E0: Array, E1: Array) -> Array:
        """The heating source on the velocity divergence (1/ps^2), x-space; zero when off."""
        if not self.thermal_waves:
            return jnp.zeros((self.nx, self.ny))
        source = jnp.zeros((self.nx, self.ny))
        for wave, coeff in self.thermal_coeff.items():
            if wave == "laser":
                e_sq = jnp.sum(jnp.abs(E0) ** 2, axis=-1)
            elif wave == "raman":
                e_sq = jnp.sum(jnp.abs(E1) ** 2, axis=-1)
            else:
                ex, ey = self.epw_fields(phi_k)
                e_sq = jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2
            source = source + coeff * (e_sq - jnp.mean(e_sq))
        if self.thermal_nonlocal:
            nonlocal_part = jnp.real(jnp.fft.ifft2(self.thermal_k_factor * jnp.fft.fft2(source)))
            source = source + self.thermal_lambda_factor * nonlocal_part
        return source

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
        """The EPW/pump/Raman part of the ponderomotive potential (no acoustic term), times the
        square of the IAW source window (LPSE ``getPonderomotivePotential`` applies
        ``restrictRange`` squared; ``terms.iaw.source_window``, plan 2 I.3)."""
        # LPSE iaw.sourceTerm.{lw|laser|raman}.enable gate each term (getPonderomotivePotential)
        drive = 0.0
        if self.drive_on["epw"] and not self.combined:
            ex, ey = self.epw_fields(phi_k)
            drive = drive + (jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2) / self.wp0**2
        if self.drive_on["pump"]:
            drive = drive + jnp.sum(jnp.abs(E0) ** 2, axis=-1) / self.w0**2
        if self.drive_on["raman"]:
            # with the combined solver E1 is the whole field (Raman light + EPW, cross term included)
            drive = drive + jnp.sum(jnp.abs(E1) ** 2, axis=-1) / self.w_raman**2
        return jnp.zeros((self.nx, self.ny)) + self.ponderomotive_prefactor * drive * self.source_mask_sq

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
        if self.thermal_waves:
            velocity_divergence = velocity_divergence + self.dt * self.thermal_filamentation_source(
                y["epw"], y["E0"], y["E1"]
            )
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
        if self.thermal_waves:
            w_new = w_new + self.dt * jnp.fft.fft2(self.thermal_filamentation_source(y["epw"], y["E0"], y["E1"]))
        if self.noise_enabled:
            w_new = w_new + self.get_noise(t)

        n_new = n_new * self.filter
        w_new = w_new * self.filter
        density = jnp.real(jnp.fft.ifft2(n_new))
        velocity_divergence = jnp.real(jnp.fft.ifft2(w_new)) * self.boundary
        return density, velocity_divergence

    def _fd_step(self, t: float, y: dict[str, Array]) -> dict[str, Array]:
        """The FD solver on the (super-sampled) fine grid; returns the state entries it owns."""
        from adept._lpse2d.core.iaw_fd import FD_STATE_KEYS, downsample, upsample

        fd = self.fd
        fine = fd.s > 1
        n = y[FD_STATE_KEYS[0]] if fine else y["iaw_density"]
        w = y[FD_STATE_KEYS[1]] if fine else y["iaw_velocity_divergence"]
        # E2 = -lap(PP) = k^2 PP formed on the EPW grid (+ the thermal-filamentation source), then
        # interpolated to the fine grid (ZakharovSolver::getPonderomotivePotential(pp, false),
        # advanceNelfAndDivV_fd); the same k^2 drive as the spectral step
        drive_k = jnp.fft.fft2(self.ponderomotive_drive(y["epw"], y["E0"], y["E1"]))
        e2 = jnp.real(jnp.fft.ifft2(self.k_sq * drive_k))
        if self.thermal_waves:
            e2 = e2 + self.thermal_filamentation_source(y["epw"], y["E0"], y["E1"])
        n, w = fd.step(t, n, w, upsample(e2, fd.s))
        out = {"iaw_density": downsample(n, fd.s), "iaw_velocity_divergence": downsample(w, fd.s)}
        if fine:
            out[FD_STATE_KEYS[0]], out[FD_STATE_KEYS[1]] = n, w
        return out

    def __call__(self, y: dict[str, Array], t: float = 0.0) -> dict[str, Array]:
        """Advance one IAW step (of length ``stride * grid.dt``) and return the full state."""
        if self.solver == "fd":
            return {**y, **self._fd_step(t, y)}
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
