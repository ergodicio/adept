from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NoiseModel(BaseModel):
    """
    Noise model for the density profile
    """

    max: float
    min: float
    type: str


class DensityModel(BaseModel):
    """
    Density profile for the simulation. Which keys apply depends on ``basis``
    (see ``helpers.get_density_profile``):

    - ``uniform``: ``val`` (fraction of critical). Omitting it silently defaults
      to 1.0 -- at critical density -- so set it explicitly.
    - ``linear``: ``min``, ``max``, and ``gradient scale length`` (note the YAML
      key is spelled with spaces).
    """

    model_config = ConfigDict(populate_by_name=True)

    basis: str
    val: float | None = None
    gradient_scale_length: str | None = Field(default=None, alias="gradient scale length")
    max: float | None = None
    min: float | None = None
    # lpse-<shape> (linear, exp, gaussian, inverse-power, quadratic, qd, gd, file): the original
    # LPSE profiles between the N_min and N_max locations (helpers._lpse_density_profile)
    min_location: str | None = None
    max_location: str | None = None
    min_location_y: str | None = None
    max_location_y: str | None = None
    geometry: Literal["cartesian", "spherical"] = "cartesian"
    sg_order: float = 2.0  # LPSE sgOrder (gaussian, inverse-power, gd)
    central_density: float | None = None  # LPSE quadratic.centralDensity
    dip_depth: float | None = None  # LPSE dip.depth (qd, gd)
    dip_width: str | None = None  # LPSE dip.width (full width)
    dip_offset: str | None = None  # LPSE dip.offset from origin
    origin: str | None = None  # x of LPSE's box centre (default: the box centre)
    max_density: float = 1.25  # LPSE maxBackgroundDensity clip
    file: str | None = None  # lpse-file: .npy / text table / LPSE grid file
    noise: NoiseModel | None = None


class EnvelopeModel(BaseModel):
    """
    Envelope model for the driver

    """

    tw: str
    tr: str
    tc: str
    xr: str
    xw: str
    xc: str
    yr: str
    yw: str
    yc: str


class SpeckleModel(BaseModel):
    """
    LASY speckle profile configuration.

    Used to apply a y-dependent speckle envelope to the laser field.

    Supported smoothing types:
    - 'RPP': Random phase plates (static)
    - 'CPP': Continuous phase plates (static)
    - 'FM SSD': Frequency modulated smoothing by spectral dispersion (time-varying)
    - 'GP RPM SSD': Gaussian process randomly phase-modulated SSD (time-varying)
    - 'GP ISI': Gaussian process induced spatial incoherence (time-varying)
    """

    enabled: bool = False
    focal_length: str  # e.g. "3.5m"
    beam_aperture: list[str]  # [x, y] e.g. ["0.35m", "0.35m"]
    n_beamlets: list[int]  # [nx, ny]
    smoothing_type: str = "CPP"  # RPP, CPP, FM SSD, GP RPM SSD, GP ISI
    seed: int = 42
    # SSD-specific parameters (required for FM SSD, GP RPM SSD, GP ISI)
    relative_laser_bandwidth: float | None = None
    ssd_phase_modulation_amplitude: list[float] | None = None  # [x, y]
    ssd_number_color_cycles: list[float] | None = None  # [x, y]
    ssd_transverse_bandwidth_distribution: list[float] | None = None  # [x, y]


class E0DriverModel(BaseModel):
    """
    E0 driver model

    """

    amplitude_shape: str
    delta_omega_max: float
    num_colors: int
    envelope: EnvelopeModel
    speckle: SpeckleModel | None = None
    # in-plane angle of incidence from +x in degrees (LPSE laser.N.direction): the static pump
    # is the grid mode nearest to k0 (cos a, sin a); the spectral injector launches at the
    # y-snapped transverse wavenumber. Not with speckle or the FD injector.
    angle: float = Field(default=0.0, gt=-90.0, lt=90.0)
    # polarization angle in degrees about the beam axis, LPSE laser.N.polarization: 0 ("p") is
    # in-plane, 90 ("s") is along z, the out-of-plane component every light field carries (plan 2
    # F.1/F.2). The injectors launch cos(psi) in-plane + sin(psi) z
    polarization: float | Literal["p", "s"] = "p"
    # LPSE laser.N.* beams: [{intensity: fraction or W/cm^2 (normalised), angle: deg, phase: rad,
    # delta_omega: dW/W0, polarization: deg}]; every beam carries every color. Spectral injector
    # and static pump only.
    beams: list[dict] | None = None
    beam_width: str | None = None  # LPSE laser.N.width: transverse (y) standard deviation of injected beams
    beam_sg_order: float = 2.0  # LPSE laser.N.sgOrder of that transverse profile (2 = Gaussian)
    beam_offset: str | None = None  # y position of the beam centre (LPSE laser.N.offset)
    kap_bandwidth: float = Field(default=0.0, ge=0.0, lt=1.0)  # LPSE bandwidth.KAP.frequency (dW/W0)
    kap_seed: int = 0
    pulse_file: str | None = None  # LPSE laser.pulse.file: two-column (t_ps, relative amplitude) table


class E1DriverModel(BaseModel):
    """
    Raman seed driver.

    Injects a counter-propagating (-x) scattered-light wave at x = xmax - offset
    with the given (vacuum) intensity. Only used when terms.epw.source.srs is on.
    """

    intensity: str  # e.g. "1.0e+12W/cm^2"
    delta_omega: float = 0.0  # seed frequency shift relative to w1 = w0 - wp0 (fraction of w1)
    turn_on_time: str = "10fs"
    # distance of the injector from the right boundary; defaults to 1.6 * boundary_width,
    # which places it just inside the absorbing boundary's tanh skirt
    offset: str | None = None
    yw: str | None = None  # super-Gaussian width of the seed in y; omit for uniform in y
    # spectral light solver: Gaussian width of the smooth injector (default one local wavelength)
    injector_width: str | None = None
    # seed polarization angle in degrees (LPSE raman.N.polarization): 0 / "p" in-plane (y), 90 / "s" along z
    polarization: float | Literal["p", "s"] = "p"


class DriversModel(BaseModel):
    """
    Define the drivers for the simulation

    """

    E0: E0DriverModel
    E1: E1DriverModel | None = None


class GridModel(BaseModel):
    """
    Define the grid for the simulation

    """

    boundary_abs_coeff: float
    boundary_width: str
    # absorbing-layer profile: "tanh" (the MATLAB envelope, rate boundary_abs_coeff on the
    # plateau) or "exp" (LPSE absorbingBoundaries.cpp: rate = boundary_max_rate *
    # (exp(lambda s/L) - 1)/(exp(lambda) - 1) over the layer, per axis combined by max)
    boundary_profile: Literal["tanh", "exp"] = "tanh"
    boundary_max_rate: float = 200.0  # 1/ps, LPSE lw.abc.maxDampingRate default
    boundary_lambda: float = 7.0  # LPSE abc.lambda default
    low_pass_filter: float
    dealias: str = "isotropic"
    dt: str
    dx: str
    tmax: str
    tmin: str
    ymax: str
    ymin: str
    # number of dynamic-light sub-steps per EPW step; computed from the tightest
    # evolved-carrier stability limit if omitted
    light_substeps: int | None = None


class TimeSaveModel(BaseModel):
    dt: str
    tmax: str | None = None  # Optional: defaults to grid.tmax at runtime
    tmin: str | None = None  # Optional: defaults to grid.tmin at runtime


class XSaveModel(BaseModel):
    dx: str


class YSaveModel(BaseModel):
    dy: str


class SaveModel(BaseModel):
    t: TimeSaveModel
    x: XSaveModel
    y: YSaveModel
    # write the full solver state at grid.tmax to this path (or into the run's binary/ folder
    # when true); `restart.file` resumes from it (LPSE checkpoint / --restart)
    checkpoint: str | bool | None = None
    # Thomson-scattering probes (LPSE thomsonScattering.N): [{k: [kx, ky] in units of k0,
    # bandwidth: 0.1 (k0), field: epw | iaw}] -> series thomson_<i>_re/_im/_power
    thomson: list[dict] | None = None
    # save.fields.poynting: true adds s0_x, s0_y, s1_x, s1_y (LPSE laser/raman.save.S0)
    poynting: bool = False


class RestartModel(BaseModel):
    file: str


class BoundaryModel(BaseModel):
    x: str
    y: str


class DampingModel(BaseModel):
    """EPW damping. ``landau_form`` selects the static Landau rate: ``matlab`` (the
    prototype's prefactor), ``lpse`` (the C++ non-relativistic form) or
    ``relativistic`` / ``relativistic_3d`` (LPSE's Maxwell-Juettner Bessel forms).
    ``landau_lower_threshold`` (1/ps) zeroes the rate below it (LPSE
    ``lw.landauDamping.lowerThreshold``); ``landau_multiplier`` is LPSE's static
    ``LD_multiplier``."""

    collisions: bool | float
    landau: bool
    landau_form: Literal["matlab", "lpse", "relativistic", "relativistic_2d", "relativistic_3d"] = "matlab"
    landau_lower_threshold: float = 0.0
    landau_multiplier: float = 1.0


class SourceModel(BaseModel):
    """EPW sources. ``noise_model: flat`` is the MATLAB per-step source ``dt * amplitude``
    with a random phase on every retained mode; ``thermal`` is LPSE's fluctuation-
    dissipation source (see ``epw.noise_kick_spectrum``), optionally calibrated with
    ``noise_calibrate``: ``equipartition`` (``true``) sets the amplitude from the electron
    temperature, ``lpse`` uses LPSE's own ``lw.noise.calcNoiseAmp_K0`` constant (the
    ``isCalculated`` source). ``noise_max_wavenumber`` (units of k0)
    is LPSE ``lw.noise.maxWavenumber``. ``tpd_form: lpse`` keeps LPSE's exact TPD
    coefficient and its ``(w0/wp0 - 1)`` charge-density factor; ``matlab`` is the
    prototype's ``w0 -> 2 wp0`` form (identical at envelope density 0.25)."""

    noise: bool
    noise_model: Literal["flat", "thermal"] = "flat"
    noise_amplitude: float = 1e-10
    noise_seed: int | None = None
    noise_calibrate: bool | Literal["equipartition", "lpse"] = False
    noise_debye_factor: bool = True  # thermal model: keep the 1/sqrt(1 + k^2 lambda_D^2) spectrum shape
    noise_max_wavenumber: float | None = None
    tpd: bool
    tpd_form: Literal["lpse", "matlab"] = "lpse"
    srs: bool = False
    # high-k filter on the light entering the SRS source (LPSE lw.kFilter.enable / .scale).
    # The cutoff assumes zero detuning, so it removes the resonant mode in boxes below the
    # envelope density; LPSE leaves it off by default and translated decks follow suit.
    srs_k_filter: bool = True
    srs_k_filter_scale: float = Field(default=1.2, ge=1.0, le=10.0)


class EPWModel(BaseModel):
    boundary: BoundaryModel
    damping: DampingModel
    density_gradient: bool
    linear: bool
    source: SourceModel
    # LPSE lw.maxWavenumber: hard cap |k| < max_wavenumber * k0 on the retained EPW band
    max_wavenumber: float | None = None
    # per-operation EPW energy ledger accumulated in the state and reported in the default
    # series (epw_ledger_<channel>, epw_ledger_closure); LPSE asserts the same closure
    energy_ledger: bool = False
    # "separate": the four envelope equations (MATLAB / LPSE spectral path); "combined":
    # LPSE lw.solver = combined, one wp0-enveloped field carrying the Raman light
    # (transverse part) and the EPW (longitudinal part), required by LPSE when TPD and
    # SRS are both on (see core/combined.py)
    solver: Literal["separate", "combined"] = "separate"


class LightModel(BaseModel):
    """Light-wave evolution options. pump_depletion evolves E0 with the FD envelope
    solver and reciprocal coupling from every active TPD/SRS source. coupling selects
    how the SRS exchange between E0 and E1 is integrated inside the light sub-step:
    "explicit" (the MATLAB staggered update, which grows the light fields at a rate
    ~Omega^2 dt_l/4 once the EPW is finite -- see CoupledLight) or "rotation" (exact,
    action-conserving local rotation, Strang-split around the propagation)."""

    pump_depletion: bool = False
    coupling: Literal["explicit", "rotation"] = "explicit"
    # optional isotropic low-pass filter on E0/E1 once per EPW step, as a fraction of
    # the grid Nyquist wavenumber pi/dx (None = off)
    filter: float | None = None
    # TPD pump depletion: transverse (divergence-free) projection of E_h div(E_h) in
    # k-space (LPSE LwSolver::makeExyzDivE) and the optional LPSE lw.kFilter
    # (|k| < 1.2 k0 sqrt(1 - n_min)) on the same term
    tpd_projection: bool = True
    tpd_k_filter: bool = False
    # light propagator: "fd" (MATLAB staggered scheme, sub-cycled to its CFL limit) or
    # "spectral" (LPSE exact k-space propagator with L/T projection, no CFL limit)
    solver: Literal["fd", "spectral"] = "fd"
    # retained light band cap |k| < max_wavenumber * k0 (spectral solver; LPSE maxWavenumber)
    max_wavenumber: float | None = None
    # project the SRS light sources onto their transverse part every sub-step (LPSE
    # takeTransversePartOfSourceTerms; for the coupled spectral solver the fields are kept
    # transverse instead). No light propagator moves the longitudinal part of E1, so
    # without this it accumulates the source and pairs with the EPW in a spurious
    # two-wave instability (41/ps energy growth against LPSE's 5.8/ps in test_006)
    transverse_source: bool = True
    # fd solver: project the evolved light fields (E0 with pump depletion, E1) onto their
    # transverse part once per EPW step. The FD curl-curl propagator's discrete divergence
    # is not zero, so 2-D-structured transverse fields acquire a longitudinal part every
    # step (percent level at k0 dx ~ 1-2) that no propagator moves and that the EPW sources
    # see while the projected TPD depletion term cannot return energy from it (plan 2 N.4)
    transverse_fields: bool = True
    # collisional (inverse-bremsstrahlung) absorption: false, true (NRL formula as in LPSE)
    # or the amplitude rate at nc in 1/ps
    absorption: bool | float = False
    # peak amplitude damping rate (1/ps) of the light fields' absorbing layers with the exp
    # profile (LPSE {laser|raman}.evolution.abc.maxDampingRate, default 5e3); None = 5e3 for the
    # exp profile and the EPW absorber (boundary_abs_coeff) for the tanh profile
    boundary_max_rate: float | None = None


class IAWDampingModel(BaseModel):
    """Ion-acoustic damping. ``landau_form: simplified`` is ``gamma = landau * cs * |k|``
    (LPSE ``isSimplified``); ``full`` is the Z-generalized Krall-Trivelpiece rate with its
    ``k lambda_D`` and ``Z Te/Ti`` dependence (``iaw.ion_landau_rate``), in which case
    ``landau`` is ignored. ``collisions`` (1/ps) damps ``n`` as ``(1 - nu dt)`` in the
    explicit solver and ``div v`` as ``exp(-2 nu dt)`` in the spectral one (LPSE)."""

    collisions: float = 1.0e-5  # density damping rate, 1/ps
    landau: float = 0.1  # gamma_iaw = landau * cs * |k|
    landau_form: Literal["simplified", "full"] = "simplified"


class ThermalFilamentationModel(BaseModel):
    """LPSE thermalFil.*: inverse-bremsstrahlung heating of the spatially varying part of each
    wave's intensity, balanced by Spitzer conduction, driving the ion flow through the electron
    pressure (see IonAcousticWave._init_thermal_filamentation)."""

    model_config = ConfigDict(populate_by_name=True)

    laser: bool = False
    raman: bool = False
    lw: bool = False
    nonlocal_: bool = Field(default=False, alias="nonlocal")  # k^(4/3) correction
    conductivity_multiplier: float = 1.0


class IAWModel(BaseModel):
    """Ion-acoustic density/velocity-divergence evolution and ponderomotive drive.

    ``solver: explicit`` is the MATLAB kick/drift split step (stable for
    ``omega_max dt < 2``); ``spectral`` is LPSE's exact per-mode damped-oscillator
    propagator (unconditionally stable), which also supports a uniform background
    ``flow`` ([Mach_x, Mach_y], Doppler phase), advancing the IAW only every ``stride``
    EPW steps, and the LPSE fluctuation-dissipation ``noise`` source on ``div v``."""

    active: bool = False
    solver: Literal["explicit", "spectral"] = "explicit"
    boundary: BoundaryModel | None = None  # defaults to terms.epw.boundary
    boundary_max_rate: float | None = None  # exp absorber peak rate (1/ps); default half the EPW one
    damping: IAWDampingModel = IAWDampingModel()
    max_density_perturbation: float | None = None
    flow: list[float] | None = None
    stride: int = 1
    noise: bool = False
    noise_amplitude: float = 1.0
    noise_seed: int | None = None
    thermal_filamentation: ThermalFilamentationModel | None = None


class HPEModel(BaseModel):
    """Hybrid particle evolution (Follett et al. 2017): test electrons pushed in the
    de-enveloped EPW field feed an evolving Landau damping rate back to the wave
    solver (kinetic inflation + hot electrons). The tracker is 1D1V for ny == 1
    and 2D2V otherwise; both use one box-averaged ensemble. Requires
    terms.epw.damping.landau: true."""

    active: bool = False
    n_particles: int = 500000
    v_min: float = 2.5  # tail cutoff, units of vte
    v_max: float = 1.0  # histogram half-span, units of c
    v_blend_buffer: float = 0.5  # analytic/HPE blend buffer above v_min, units of vte
    nv: int = 512  # velocity bins spanning (-v_max, v_max)
    n_angles: int = 32  # oriented velocity projections spanning 2pi in 2-D
    gather_refine: int = 4  # spectral upsampling of Ex/Ey before the particle gather
    substep_courant: float = 0.05  # wp0 * particle substep
    tau_damping: str = "100fs"  # EMA window for the velocity histogram
    t_start: str = "0ps"  # push/feedback disabled before this time
    feedback: bool = True  # False = control run: particles evolve, damping stays analytic
    seed: int = 42
    omega_res: str = "bohm_gross"  # resonance v_phi convention: "bohm_gross" or "wp0"
    # ---- LPSE HPE controls and instruments (ElectronTracker.cu readParameters)
    gamma_limit_damping: float = 1500.0  # hpe.gammaLimit.damping, 1/ps: applied rate <= this
    gamma_limit_growth: float = 1500.0  # hpe.gammaLimit.growth, 1/ps: applied rate >= -this when allow_growth
    allow_growth: bool = False  # hpe.allowGrowth: negative (inverse Landau) rates allowed
    # hpe.thermalizationProbability (x, y walls): every crossing is counted by the instruments, then the
    # particle is thermalized with this probability or passes through periodically (LPSE particle walls
    # are independent of the field boundaries). None = 1 at absorbing field boundaries, 0 at periodic
    thermalization_probability: list[float] | None = None
    # hpe.magneticField, tesla: a number is B_z (out of plane); [B_x, B_y, B_z] adds in-plane
    # components, with which the 2-D tracker carries p_z (plan 2 F.4). 2-D push only
    magnetic_field: float | list[float] = 0.0
    energy_conservation: bool = False  # hpe.enforceEnergyConservation: LD multiplier
    energy_conservation_steps: float = 1.0  # hpe.numStepsToAverageEnergyChange
    flux_bins: list[float] | None = None  # keV edges of the wall-flux instrument (default 0, 50, 100, inf)
    cone_angle: float | None = None  # hpe.metrics.power: acceptance half-angle in degrees (None = off)
    cone_direction: list[float] = [1.0, 0.0]  # hpe.metrics.power direction


class TermsModel(BaseModel):
    epw: EPWModel
    light: LightModel = LightModel()
    iaw: IAWModel | None = None
    hpe: HPEModel | None = None
    zero_mask: bool


class UnitsModel(BaseModel):
    atomic_number: int
    envelope_density: float
    ionization_state: int
    laser_intensity: str
    laser_wavelength: str
    reference_electron_temperature: str
    reference_ion_temperature: str


class MLFlowModel(BaseModel):
    experiment: str
    run: str


class ConfigModel(BaseModel):
    density: DensityModel
    drivers: DriversModel
    grid: GridModel
    mlflow: MLFlowModel
    save: SaveModel
    solver: str
    terms: TermsModel
    units: UnitsModel
    restart: RestartModel | None = None
