import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from adept._lpse2d.core.driver import Driver

LANDAU_FORMS = ("matlab", "lpse", "relativistic")
NOISE_MODELS = ("flat", "thermal")
TPD_FORMS = ("lpse", "matlab")


def landau_damping_rate(k_sq: Array, wp0: float, vte_sq: float, zero_mask: Array, form: str = "matlab") -> Array:
    """
    Non-relativistic Landau damping rate for each k mode (amplitude rate, 1/ps).

    ``form="matlab"`` is m201805_matlabLpse_v11.m line 913:

        gammaLandauEpw = sqrt(pi/8) * (1 + 3/2*k^2*vte^2/wp^2) * wp^4/(k^3*vte^3) * exp(...)

    ``form="lpse"`` is the LPSE C++ expression (ZakharovSolver.cpp landauDamping_nonRel):

        gamma = sqrt(pi/8) (kde/k)^3 w_k exp(-w_k^2 / (2 k^2 vte^2)),  w_k = wp sqrt(1 + 3 k^2/kde^2)

    The exponents are identical (w_k^2/(2 k^2 vte^2) = wp^2/(2 k^2 vte^2) + 3/2 exactly); the
    prefactors differ by sqrt(1 + 3 x^2) vs (1 + 1.5 x^2), x = k lambda_D -- 0.7% at x = 0.3,
    3.9% at x = 0.5.

    Module-level so the solver (`SpectralEPWSolver`), the HPE calibration and the dissipation
    diagnostic (`helpers.get_default_save_func`) use the *same* rates and can never drift apart.
    """
    k_sq_safe = jnp.where(k_sq > 0, k_sq, 1.0)

    if form == "lpse":
        kde_sq = wp0**2 / vte_sq
        w_k = wp0 * jnp.sqrt(1.0 + 3.0 * k_sq / kde_sq)
        damping = (
            jnp.sqrt(np.pi / 8.0) * (kde_sq / k_sq_safe) ** 1.5 * w_k * jnp.exp(-0.5 * w_k**2 / (k_sq_safe * vte_sq))
        )
    elif form == "matlab":
        damping = (
            jnp.sqrt(np.pi / 8.0)
            * (1.0 + 1.5 * k_sq * vte_sq / wp0**2)
            * wp0**4
            / (k_sq_safe**1.5 * vte_sq**1.5)
            * jnp.exp(-(1.5 + 0.5 * wp0**2 / (k_sq_safe * vte_sq)))
        )
    else:
        raise ValueError(f"landau_damping_rate: form must be 'matlab' or 'lpse', got {form!r}")

    return damping * zero_mask


def landau_damping_rate_relativistic(
    k_sq: np.ndarray, wp0: float, vte_sq: float, c: float, ndim: int = 2
) -> np.ndarray:
    """
    Relativistic (Maxwell-Juettner) Landau damping rate, ported from LPSE
    ``ZakharovSolver::landauDamping_rel_2D`` / ``landauDamping_rel_3D`` (Bessel-function
    forms after Vu / Russell). ``ndim`` selects the 2-D or 3-D velocity-space integral;
    modes whose phase velocity exceeds c (``arg <= 0``) are undamped, as in LPSE.

    Evaluated with numpy/scipy once at setup (the rate is a static k-array).
    """
    from scipy import special

    k_sq = np.asarray(k_sq, dtype=np.float64)
    vte = np.sqrt(vte_sq)
    kde = wp0 / vte
    k = np.sqrt(np.where(k_sq > 0.0, k_sq, 1.0))
    x = k / kde
    beta_sq = vte_sq / c**2  # (Ve/C)^2
    omega = 1.0 + 1.5 * x**2
    arg = 1.0 - beta_sq * omega**2 / x**2
    ok = (k_sq > 0.0) & (arg > 0.0)
    A = 1.0 / np.sqrt(np.where(ok, arg, 1.0))
    a = A / beta_sq  # (C/Ve)^2 A
    # e_bessel_kN in LPSE's Bessel.h are the exponentially scaled K_N: e^a K_N(a)
    if ndim == 2:
        gamma = (
            0.25
            * (kde / k) ** 3
            * wp0
            * omega
            * np.exp(1.0 / beta_sq - a)
            * (1.0 / np.sqrt(beta_sq))
            * A**3
            * 2.0
            * (special.k0e(a) + special.k1e(a) / a)
            / (1.0 + beta_sq)
        )
    elif ndim == 3:
        b = 1.0 / beta_sq
        gamma = (
            0.25
            * np.pi
            * (kde / k) ** 3
            * wp0
            * omega
            * np.exp(b - a)
            * b ** (-2.5)
            * (2.0 + a * (2.0 + a))
            / (special.k0e(b) + 2.0 * special.k1e(b) / b)
        )
    else:
        raise ValueError("landau_damping_rate_relativistic: ndim must be 2 or 3")
    return np.where(ok, gamma, 0.0)


def analytic_landau_rate(cfg: dict) -> Array:
    """
    The static Landau damping rate array (nx, ny) selected by ``terms.epw.damping``:

    - ``landau_form``: ``matlab`` (default), ``lpse`` or ``relativistic`` (LPSE 2-D Bessel form;
      ``relativistic_3d`` for the 3-D velocity-space integral)
    - ``landau_lower_threshold`` (1/ps): modes damped below it are treated as undamped
      (LPSE ``lw.landauDamping.lowerThreshold``)
    - ``landau_multiplier``: static scalar on the rate (LPSE ``LD_multiplier``)

    Returns zeros when ``terms.epw.damping.landau`` is off.
    """
    derived = cfg["units"]["derived"]
    damping = cfg["terms"]["epw"]["damping"]
    kx = jnp.asarray(cfg["grid"]["kx"])
    ky = jnp.asarray(cfg["grid"]["ky"])
    k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
    zero_mask = jnp.where(k_sq > 0, 1.0, 0.0)
    if not damping.get("landau", True):
        return jnp.zeros_like(k_sq)
    form = str(damping.get("landau_form", "matlab"))
    if form in ("relativistic", "relativistic_2d"):
        rate = jnp.asarray(
            landau_damping_rate_relativistic(np.asarray(k_sq), derived["wp0"], derived["vte_sq"], derived["c"], ndim=2)
        )
    elif form == "relativistic_3d":
        rate = jnp.asarray(
            landau_damping_rate_relativistic(np.asarray(k_sq), derived["wp0"], derived["vte_sq"], derived["c"], ndim=3)
        )
    elif form in ("matlab", "lpse"):
        rate = landau_damping_rate(k_sq, derived["wp0"], derived["vte_sq"], zero_mask, form=form)
    else:
        raise ValueError(
            f"terms.epw.damping.landau_form must be one of matlab, lpse, relativistic, relativistic_3d; got {form!r}"
        )
    threshold = float(damping.get("landau_lower_threshold", 0.0) or 0.0)
    if threshold > 0.0:
        rate = jnp.where(rate > threshold, rate, 0.0)
    multiplier = float(damping.get("landau_multiplier", 1.0) if damping.get("landau_multiplier") is not None else 1.0)
    return rate * multiplier


def noise_kick_spectrum(cfg: dict) -> np.ndarray:
    """
    Per-step, per-mode EPW noise kick amplitude ``D_k`` (k-space potential units), for both
    noise models. ``get_noise`` multiplies it by a random phase every step; the diagnostics
    (``diagnostics.expected_noise_energy``) integrate the same array, so the two never drift.

    ``noise_model: flat`` (default; the MATLAB source): ``D_k = dt * noise_amplitude`` on
    every retained mode.

    ``noise_model: thermal`` (LPSE ``lw.noise``, ZakharovSolver::addNoiseToPotential_fft):
    a fluctuation-dissipation source balanced against the *frozen* analytic Landau +
    collisional rate ``gamma_k``:

        D_k = N A / sqrt(1 + k^2 lambda_D^2) * sqrt(1 - exp(-2 gamma_k dt)) / |k|

    ``N = nx*ny`` converts an x-space amplitude to this code's unnormalized-FFT k-space
    convention, and the ``1/|k|`` converts field to potential, so the steady state of
    ``phi_k *= exp(-gamma_k dt); phi_k += D_k e^{i theta}`` is the Cerenkov spectrum
    ``<|E_k|^2> = A^2 / (1 + k^2 lambda_D^2)`` per mode (x-space envelope amplitude
    squared), independent of dt. With ``noise_calibrate: true`` the amplitude is set from
    the electron temperature by equipartition -- electric energy ``kT/2`` per mode over
    the box volume ``V`` (``Lz = Ly`` for the 2-D box, ``V = Lx^3`` when ny = 1, following
    LPSE's ``deltaK3``): ``A_thermal = sqrt(8 pi kT / V)``, and ``noise_amplitude`` then
    multiplies it (LPSE's ``isCalculated`` convention, amplitude 1 = thermal). Modes with
    no damping receive no noise; a retained band with ``gamma_k <= 0`` everywhere is refused,
    as in LPSE ("Cannot add LW noise without damping").
    """
    grid = cfg["grid"]
    source = cfg["terms"]["epw"]["source"]
    derived = cfg["units"]["derived"]
    kx = np.asarray(grid["kx"])
    ky = np.asarray(grid["ky"])
    k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
    zero_mask = np.where(k_sq > 0, 1.0, 0.0)
    band = np.asarray(grid["low_pass_filter_grid"]) * zero_mask
    max_k = source.get("noise_max_wavenumber")
    if max_k is not None:
        k0 = derived["w0"] / derived["c"]
        band = band * np.where(np.sqrt(k_sq) < float(max_k) * k0, 1.0, 0.0)

    model = str(source.get("noise_model", "flat"))
    amplitude = float(source.get("noise_amplitude", 1e-10))
    dt = grid["dt"]
    if model == "flat":
        return dt * amplitude * band
    if model != "thermal":
        raise ValueError(f"terms.epw.source.noise_model must be 'flat' or 'thermal', got {model!r}")

    gamma = np.asarray(analytic_landau_rate(cfg)) + derived.get("nu_coll", 0.0) * zero_mask
    retained = band > 0.0
    if not np.any(retained):
        return np.zeros_like(k_sq)
    if np.all(gamma[retained] <= 0.0):
        raise ValueError(
            "terms.epw.source.noise_model: thermal needs EPW damping (Landau and/or collisions) on the "
            "retained band -- LPSE refuses noise without damping and so does this model"
        )
    if source.get("noise_calibrate", False):
        # kT in this code's energy unit (massScale * spatialScale^2 / timeScale^2, cgs)
        from astropy.units import Quantity as _Q

        te_kev = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
        energy_scale = derived["massScale"] * derived["spatialScale"] ** 2 / derived["timeScale"] ** 2
        kT = te_kev * 1.602176634e-9 / energy_scale
        nx, ny = int(grid["nx"]), int(grid["ny"])
        lx = nx * grid["dx"]
        ly = ny * grid["dy"]
        volume = lx**3 if ny == 1 else lx * ly * ly
        amplitude = amplitude * np.sqrt(8.0 * np.pi * kT / volume)

    lambda_d_sq = derived["vte_sq"] / derived["wp0"] ** 2
    n_total = float(grid["nx"] * grid["ny"])
    k_safe = np.sqrt(np.where(k_sq > 0, k_sq, 1.0))
    two_g_dt = 2.0 * gamma * dt
    # LPSE switches to the small-argument form below 1e-4 to avoid a float32 cancellation;
    # in float64 -expm1 is exact enough everywhere
    fd_factor = np.sqrt(np.maximum(-np.expm1(-two_g_dt), 0.0))
    kick = n_total * amplitude / np.sqrt(1.0 + k_sq * lambda_d_sq) * fd_factor / k_safe
    return np.where(retained, kick, 0.0)


class SpectralEPWSolver:
    """
    Spectral solver for electrostatic plasma waves in k-space.

    Matches MATLAB's spectralEpwUpdate() function (lines 1966-2118), which is the same
    split-step as LPSE's default ``lw.solver = spectral`` path (ZakharovSolver.cpp).

    State variable: phi_k (electrostatic potential in k-space)
    - MATLAB convention: uses fftshift, so DC is in center
    - JAX convention: uses fftfreq, so DC is at [0,0]

    Key differences from original implementation:
    1. Filter applied at exactly 2 points per timestep
    2. Clear separation between operations (no combined expressions)
    3. Explicit comments matching MATLAB line numbers

    LPSE-parity options (all default to the MATLAB behaviour unless stated):

    - ``terms.epw.damping.landau_form / landau_lower_threshold / landau_multiplier``
      (see ``analytic_landau_rate``)
    - ``terms.epw.source.tpd_form``: ``lpse`` (default) keeps LPSE's exact coefficient
      ``i e/(4 me w0)`` and the ``(w0/wp0 - 1)`` factor on the charge-density term, and uses
      every pump component; ``matlab`` is the prototype's ``w0 -> 2 wp0`` form. The two are
      identical at envelope density n_c/4.
    - ``terms.epw.source.noise_model``: ``flat`` (default) or ``thermal`` (see
      ``noise_kick_spectrum``)
    - ``terms.epw.max_wavenumber``: LPSE ``lw.maxWavenumber`` (units of k0), folded into
      the retained band by ``helpers.get_solver_quantities``
    """

    def __init__(self, cfg: dict):
        """
        Initialize the spectral EPW solver.

        Args:
            cfg: Configuration dictionary with grid, units, and physics parameters
        """
        # Grid parameters
        self.nx = cfg["grid"]["nx"]
        self.ny = cfg["grid"]["ny"]
        self.dx = cfg["grid"]["dx"]
        self.dy = cfg["grid"]["dy"]
        self.dt = cfg["grid"]["dt"]

        # K-space grid (JAX uses fftfreq, not fftshift)
        self.kx = cfg["grid"]["kx"]  # Shape: (nx,)
        self.ky = cfg["grid"]["ky"]  # Shape: (ny,)

        # 2D k-space grids
        # Note: MATLAB uses [KX, KY] = meshgrid(kx, ky) with fftshift
        # JAX uses fftfreq which is already in correct order
        self.k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2

        # Avoid division by zero at k=0
        self.one_over_k_sq = jnp.where(self.k_sq > 0, 1.0 / self.k_sq, 0.0)
        self.zero_mask = jnp.where(self.k_sq > 0, 1.0, 0.0)

        # Physics parameters
        self.wp0 = cfg["units"]["derived"]["wp0"]  # Reference plasma frequency
        self.w0 = cfg["units"]["derived"]["w0"]  # Laser frequency
        self.vte_sq = cfg["units"]["derived"]["vte_sq"]  # Thermal velocity squared
        self.e = cfg["units"]["derived"]["e"]  # Elementary charge (normalized)
        self.me = cfg["units"]["derived"]["me"]  # Electron mass (normalized)
        self.nu_coll = cfg["units"]["derived"].get("nu_coll", 0.0)  # Collisional damping

        # Density profile
        self.envelope_density = cfg["units"]["envelope density"]
        self.background_density = cfg["grid"]["background_density"]

        # Boundaries
        self.boundary_envelope = cfg["grid"]["absorbing_boundaries"]

        # Low-pass filter
        # This should be binary (0 or 1) unless taper_fraction > 0
        self.low_pass_filter = cfg["grid"]["low_pass_filter_grid"]

        # TPD parameters
        source_cfg = cfg["terms"]["epw"]["source"]
        self.tpd_enabled = source_cfg["tpd"]
        self.tpd_form = str(source_cfg.get("tpd_form", "lpse"))
        if self.tpd_form not in TPD_FORMS:
            raise ValueError(f"terms.epw.source.tpd_form must be one of {TPD_FORMS}, got {self.tpd_form!r}")
        if self.tpd_enabled:
            if self.tpd_form == "lpse":
                # LPSE ZakharovSolver::updatePotentialWithTpdSource_fft:
                #   TPD_srcFactor = i (q/m)/(4 W0) exp(-i (W0 - 2 wpe) t)
                #   TPD_src = TPD1 - (1 - W0/wpe) i k.TPD2 / k^2
                self.tpd_prefactor = 1j * self.e / (4.0 * self.w0 * self.me)
                self.tpd_rho_factor = self.w0 / self.wp0 - 1.0
            else:
                # MATLAB line 2024: w0 -> 2 wp0 inside the coefficients
                self.tpd_prefactor = 1j * self.e / (8.0 * self.wp0 * self.me)
                self.tpd_rho_factor = 1.0

        # SRS parameters
        self.srs_enabled = source_cfg.get("srs", False)
        if self.srs_enabled:
            self.w1 = cfg["units"]["derived"]["w1"]
            self.c = cfg["units"]["derived"]["c"]
            # MATLAB line 2073: srsSourceTerm = 1i * e * wp0/(4*me*w0*w1) .* (1 + dn) .* E0_dot_E1
            self.srs_prefactor = 1j * self.e * self.wp0 / (4.0 * self.me * self.w0 * self.w1)
            # high-k filter for the light fields entering the source product
            # (MATLAB isSuppressHighKSource, lines 637-645): only wavevectors near the
            # light-wave envelope produce physically-realistic SRS
            max_source_k_multiplier = 1.2
            n_min = float(np.min(np.array(self.background_density)))
            max_k1_sq = max_source_k_multiplier**2 * max(1.0 - n_min * self.w0**2 / self.w1**2, 0.0)
            is_outside_max_k1 = self.k_sq * (self.c / self.w1) ** 2 > max_k1_sq
            self.E1_filter = jnp.where(is_outside_max_k1, 0.0, 1.0)[..., None]
            # when the pump is evolved (terms.light.pump_depletion) it is filtered too,
            # exactly as MATLAB's evaluate_E0_dot_E1 (lines 2302-2354) does on the
            # dynamic-laser path and skips on the static path (line 2307-2308)
            self.pump_depletion = cfg["terms"].get("light", {}).get("pump_depletion", False)
            if self.pump_depletion:
                max_k0_sq = max_source_k_multiplier**2 * max(1.0 - n_min, 0.0)
                is_outside_max_k0 = self.k_sq * (self.c / self.w0) ** 2 > max_k0_sq
                self.E0_filter = jnp.where(is_outside_max_k0, 0.0, 1.0)[..., None]

        # Noise parameters. Amplitude default matches MATLAB noiseAmp
        # (m201805_matlabLpse_v11.m:49). The seed is resolved (and written back into
        # the cfg, so MLflow logs it) in helpers.get_derived_quantities; the fallback
        # here only fires if that step was skipped.
        self.noise_enabled = source_cfg["noise"]
        self.noise_model = str(source_cfg.get("noise_model", "flat"))
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"terms.epw.source.noise_model must be one of {NOISE_MODELS}, got {self.noise_model!r}")
        self.noise_amplitude = float(source_cfg.get("noise_amplitude", 1e-10))
        cfg_seed = source_cfg.get("noise_seed")
        self.noise_seed = int(cfg_seed) if cfg_seed is not None else np.random.randint(2**20)
        # per-step keys are derived with fold_in rather than PRNGKey(step + seed):
        # additive seeds made nearby seeds share the same noise trajectory merely
        # time-shifted by a few steps, so a seed-sweep ensemble was one realization.
        # fold_in streams are still fully deterministic per (seed, step).
        self.noise_key = jax.random.PRNGKey(self.noise_seed)
        if self.noise_enabled:
            # per-mode kick amplitude, already including dt (flat) or the
            # fluctuation-dissipation factor (thermal) and the retained-band mask
            self.noise_kick = jnp.asarray(noise_kick_spectrum(cfg))
        else:
            self.noise_kick = None

        # Density gradient
        self.density_gradient_enabled = cfg["terms"]["epw"]["density_gradient"]
        self.iaw_enabled = cfg["terms"].get("iaw", {}).get("active", False)

        # Landau damping flag (previously ignored -- damping was unconditionally on)
        self.landau_enabled = bool(cfg["terms"]["epw"]["damping"].get("landau", True))
        # static analytic rate with the configured form / threshold / multiplier applied
        self.landau_rate = analytic_landau_rate(cfg)
        # HPE (Follett-style particle feedback): the damping rate is read from the
        # state (y["gamma_L"], written by HybridParticleEvolution) instead of the
        # static analytic array
        self.hpe_enabled = bool(cfg["terms"].get("hpe", {}).get("active", False))

        # direct EPW driver (drivers.E2), used by the validation/test configs
        self.driver = Driver(cfg)

        # Store config for reference
        self.cfg = cfg

    def calc_landau_damping_rate(self) -> Array:
        """
        Landau damping rate for each k mode with the configured form, threshold and
        multiplier (``analytic_landau_rate``).

        Returns:
            Landau damping rate array (shape: nx, ny)
        """
        return self.landau_rate

    def phi_k_to_e_fields(self, phi_k: Array) -> tuple[Array, Array]:
        """
        Convert phi_k to electric field components in real space.

        Matches MATLAB's calculateFieldsFromDivE() function.
        When isSolveForPotential=true, divE is actually phi_k.

        MATLAB (lines 2458-2502):
          phi = divE  (already in k-space)
          Ex_k = -1i * KX .* phi
          Ey_k = -1i * KY .* phi
          Ex = ifftn(ifftshift(Ex_k))
          Ey = ifftn(ifftshift(Ey_k))

        JAX equivalent (no fftshift needed with fftfreq):
          ex_k = -1j * kx * phi_k
          ey_k = -1j * ky * phi_k
          ex = ifft2(ex_k)
          ey = ifft2(ey_k)

        Args:
            phi_k: Potential in k-space (shape: nx, ny)

        Returns:
            Tuple of (ex, ey) in real space
        """
        # Gradient in k-space: E = -∇φ → E_k = -i*k*φ_k
        ex_k = -1j * self.kx[:, None] * phi_k
        ey_k = -1j * self.ky[None, :] * phi_k

        # Transform to real space
        ex = jnp.fft.ifft2(ex_k)
        ey = jnp.fft.ifft2(ey_k)

        return ex, ey

    def e_fields_to_phi_k(self, ex: Array, ey: Array) -> Array:
        """
        Convert electric field components to phi_k.

        Matches MATLAB's convertFieldsToDivE() function.

        MATLAB (lines 2506-2540):
          Ex_k = fftshift(fftn(Ex))
          Ey_k = fftshift(fftn(Ey))
          divE_k = 1i * (KX.*Ex_k + KY.*Ey_k)
          if isSuppressHighWavenumberModes
              divE_k(isHighWavenumberMode) = 0
          phi_k = divE_k ./ K_sq

        Args:
            ex: Electric field x-component in real space
            ey: Electric field y-component in real space

        Returns:
            phi_k in k-space
        """
        # Transform to k-space
        ex_k = jnp.fft.fft2(ex)
        ey_k = jnp.fft.fft2(ey)

        # Divergence in k-space: ∇·E → i*k·E_k
        div_e_k = 1j * (self.kx[:, None] * ex_k + self.ky[None, :] * ey_k)

        # Apply filter (MATLAB line 2523)
        div_e_k = div_e_k * self.low_pass_filter

        # Poisson equation: ∇²φ = -p → -k²φ = ∇·E → φ = -∇·E/k²
        phi_k = div_e_k * self.one_over_k_sq

        # Zero out k=0 mode (MATLAB line 2529)
        phi_k = phi_k * self.zero_mask

        return phi_k

    def calc_tpd_source(self, t: float, phi_k: Array, ex: Array, ey: Array, E0: Array) -> Array:
        """
        Calculate the Two Plasmon Decay source term.

        LPSE (ZakharovSolver.cpp, makeTpdSource / updatePotentialWithTpdSource_fft):

          S_k = i (q/m)/(4 w0) e^{-i (w0 - 2 wp0) t} [ F(E0 . E*) - (1 - w0/wp0) i k . F(E0 rho*) / k^2 ]

        with rho = div E = F^-1(k^2 phi_k) and every pump component in the dot product and
        the vector term. MATLAB lines 1996-2049 are the ``tpd_form: matlab`` special case
        (w0 -> 2 wp0 in the coefficients, y-polarized pump only in the prototype; here
        the x component is always carried, which is exact for the prototype's E0x = 0).

        Args:
            t: Current time
            phi_k: Potential in k-space
            ex, ey: Electric field components in real space
            E0: Pump field in real space, shape (nx, ny, 2)

        Returns:
            TPD source term in k-space
        """
        e0x, e0y = E0[..., 0], E0[..., 1]

        # Component 1: F(E0 . conj(E))  (MATLAB line 2011-2012 with E0x = 0)
        tpd1 = jnp.fft.fft2(e0x * jnp.conj(ex) + e0y * jnp.conj(ey))

        # Component 2: i k . F(E0 conj(rho)) / k^2  (MATLAB line 2014-2018)
        rho = jnp.fft.ifft2(self.k_sq * phi_k)
        tpd2_x = jnp.fft.fft2(e0x * jnp.conj(rho))
        tpd2_y = jnp.fft.fft2(e0y * jnp.conj(rho))
        tpd2 = 1j * (self.kx[:, None] * tpd2_x + self.ky[None, :] * tpd2_y) * self.one_over_k_sq

        # Combine with prefactor (MATLAB line 2024; LPSE TPD_srcFactor)
        phase = jnp.exp(-1j * (self.w0 - 2.0 * self.wp0) * t)
        source = self.tpd_prefactor * phase * (tpd1 + self.tpd_rho_factor * tpd2)

        # Apply filter to source (MATLAB line 2032-2033)
        source = source * self.low_pass_filter

        # Zero out k=0 (MATLAB line 2035)
        source = source * self.zero_mask

        return source

    def calc_srs_source(self, E0: Array, E1: Array) -> Array:
        """
        Calculate the SRS source term for the EPW potential.

        Matches MATLAB lines 2052-2078 for isSolveForPotential=true:
          E0_dot_E1 = E0 . conj(E1)  (E1 high-k filtered first, evaluate_E0_dot_E1 lines 2302-2354)
          srsSource = 1i * e * wp0/(4*me*w0*w1) * (1 + dn) * E0_dot_E1
          srsSource -> k-space

        The pump is static/prescribed here, so only E1 is filtered (in MATLAB the E0
        filter is skipped on the static-laser path, line 2308).

        Args:
            E0: Pump field (shape: nx, ny, 2)
            E1: Raman field (shape: nx, ny, 2)

        Returns:
            SRS source term in k-space
        """
        E1_filtered = jnp.fft.ifft2(jnp.fft.fft2(E1, axes=(0, 1)) * self.E1_filter, axes=(0, 1))
        if self.pump_depletion:
            E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * self.E0_filter, axes=(0, 1))
        E0_dot_E1 = E0[..., 0] * jnp.conj(E1_filtered[..., 0]) + E0[..., 1] * jnp.conj(E1_filtered[..., 1])

        # (1 + backgroundDensityPerturbation) = n / n_envelope
        source = self.srs_prefactor * self.background_density / self.envelope_density * E0_dot_E1

        return jnp.fft.fft2(source)

    def get_noise(self, t: float) -> Array:
        """
        Generate the random-phase noise kick for the plasma waves at this step.

        The returned array is added to phi_k directly: it already carries dt (flat model)
        or the fluctuation-dissipation factor (thermal model) from ``noise_kick_spectrum``.

        Args:
            t: Current time

        Returns:
            Random noise in k-space
        """
        # Per-step key: deterministic in (noise_seed, step), and statistically
        # independent across both steps and seeds (see __init__)
        step = (t / self.dt).astype(int)
        key = jax.random.fold_in(self.noise_key, step)

        # Random phases
        phases = 2.0 * np.pi * jax.random.uniform(key, (self.nx, self.ny))

        # per-mode amplitude with random phase; the kick spectrum is already masked to
        # the retained band (MATLAB epwNoise: phi_noise(isHighWavenumberMode) = 0) and k = 0
        return self.noise_kick * jnp.exp(1j * phases)

    def __call__(self, t: float, y, args) -> Array:
        """
        Advance EPW by one timestep using spectral method.

        This matches MATLAB's spectralEpwUpdate() lines 1966-2118.

        Order of operations (matching MATLAB exactly):
        1. Apply thermal dispersion in k-space (line 1975)
        2. Apply Landau damping in k-space (line 1981)
        3. FILTER (line 1976) ← Applied AFTER thermal/damping
        4. Add noise (line 1988)
        5. Calculate E fields from phi_k (line 1992)
        6. Calculate TPD source in k-space (lines 2011-2024)
        7. Apply density gradient to E fields (line 2081-2082)
        8. Apply absorbing boundaries to E fields (line 2088-2100)
        9. Convert E fields back to phi_k (line 2103) ← FILTER applied here too
        10. Add TPD source (line 2109)

        Args:
            t: Current time
            y: Dictionary containing:
                - "epw": Current EPW potential in k-space
                - "E0": Laser field (shape: nx, ny, 2) where E0[..., 1] is y-component
                - "E1": Raman field (shape: nx, ny, 2), optional for SRS
            args: Additional arguments (not used currently)

        Returns:
            Updated phi_k after one timestep
        """
        phi_k = y["epw"]
        E0 = y["E0"]
        background_density = self.background_density

        # ========================================================================
        # STEP 1-2: Thermal dispersion and Landau damping
        # ========================================================================
        # MATLAB line 1975: divE_k = divE_k .* exp(-1i*3/2*vte_sq/wp0 .* K_sq * DT)
        thermal_phase = jnp.exp(-1j * 1.5 * self.vte_sq / self.wp0 * self.k_sq * self.dt)
        phi_k = phi_k * thermal_phase

        # MATLAB line 1981: divE = divE .* exp(-(gammaLandau + nu_coll) * DT)
        if self.hpe_enabled:
            gamma_landau = y["gamma_L"]
        elif self.landau_enabled:
            gamma_landau = self.landau_rate
        else:
            gamma_landau = 0.0
        damping_factor = jnp.exp(-(gamma_landau + self.nu_coll) * self.dt)
        phi_k = phi_k * damping_factor

        # ========================================================================
        # STEP 3: Apply filter ONCE after thermal + damping
        # ========================================================================
        # MATLAB line 1976: divE_k(isHighWavenumberMode) = 0
        phi_k = phi_k * self.low_pass_filter

        # ========================================================================
        # STEP 4: Add noise (after the damping sub-step, as LPSE does for the
        # energetics of the fluctuation-dissipation balance)
        # ========================================================================
        if self.noise_enabled:
            # MATLAB line 1988: divE = divE + epwNoise * DT (dt is inside noise_kick)
            phi_k = phi_k + self.get_noise(t)

        # ========================================================================
        # STEP 5: Calculate electric fields
        # ========================================================================
        # MATLAB line 1992: [Ex, Ey] = calculateFieldsFromDivE(...)
        ex, ey = self.phi_k_to_e_fields(phi_k)

        # ========================================================================
        # STEP 6: Calculate TPD source (in k-space, before applying density gradient)
        # ========================================================================
        tpd_source = None
        if self.tpd_enabled:
            # MATLAB lines 1996-2035
            tpd_source = self.calc_tpd_source(t, phi_k, ex, ey, E0)

        srs_source = None
        if self.srs_enabled:
            # MATLAB lines 2052-2078
            srs_source = self.calc_srs_source(E0, y["E1"])

        # ========================================================================
        # STEP 7: Apply background and ion-acoustic density detuning (in REAL space)
        # ========================================================================
        if self.density_gradient_enabled or self.iaw_enabled:
            # MATLAB line 2081-2082:
            # Ex = Ex .* exp(-1i * wp0/2 * (n/n0 - 1 + Nelf) * DT)
            # Ey = Ey .* exp(-1i * wp0/2 * (n/n0 - 1 + Nelf) * DT)
            density_perturbation = (
                background_density / self.envelope_density - 1.0 if self.density_gradient_enabled else 0.0
            )
            if self.iaw_enabled:
                density_perturbation = density_perturbation + y["iaw_density"]
            density_phase = jnp.exp(-1j * self.wp0 / 2.0 * density_perturbation * self.dt)
            ex = ex * density_phase
            ey = ey * density_phase

        # ========================================================================
        # STEP 8: Apply absorbing boundaries to E fields (in REAL space)
        # ========================================================================
        # MATLAB line 2088-2100:
        # Ex = Ex .* exp(-DT * boundaryDampingRate)
        # Ey = Ey .* exp(-DT * boundaryDampingRate)
        ex = ex * self.boundary_envelope
        ey = ey * self.boundary_envelope

        # ========================================================================
        # STEP 9: Convert E fields back to phi_k
        # ========================================================================
        # MATLAB line 2103: divE = convertFieldsToDivE(Ex, Ey, ...)
        # This function applies filter at line 2523
        phi_k = self.e_fields_to_phi_k(ex, ey)

        # ========================================================================
        # STEP 10: Add TPD source
        # ========================================================================
        if self.tpd_enabled and tpd_source is not None:
            # MATLAB line 2109: divE = divE + tpdSourceTerm * DT
            phi_k = phi_k + self.dt * tpd_source

        # ========================================================================
        # STEP 11: Add SRS source
        # ========================================================================
        if self.srs_enabled and srs_source is not None:
            # MATLAB line 2113: divE = divE + srsSourceTerm * DT
            phi_k = phi_k + self.dt * srs_source

        return phi_k
