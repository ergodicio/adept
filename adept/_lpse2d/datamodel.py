from pydantic import BaseModel


class NoiseModel(BaseModel):
    """
    Noise model for the density profile
    """

    max: float
    min: float
    type: str


class DensityModel(BaseModel):
    """
    Density profile for the simulation

    """

    basis: str
    gradient_scale_length: str
    max: float
    min: float
    noise: NoiseModel


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
    low_pass_filter: float
    dt: str
    dx: str
    tmax: str
    tmin: str
    ymax: str
    ymin: str
    # number of Raman-light sub-steps per EPW step (SRS only); computed from the
    # stability limit if omitted
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


class BoundaryModel(BaseModel):
    x: str
    y: str


class DampingModel(BaseModel):
    collisions: bool
    landau: bool


class SourceModel(BaseModel):
    noise: bool
    noise_amplitude: float = 1e-10
    noise_seed: int | None = None
    tpd: bool
    srs: bool = False


class EPWModel(BaseModel):
    boundary: BoundaryModel
    damping: DampingModel
    density_gradient: bool
    linear: bool
    source: SourceModel


class LightModel(BaseModel):
    """Light-wave evolution options. pump_depletion evolves E0 with the FD envelope
    solver (boundary injector + EPW coupling) instead of prescribing it analytically."""

    pump_depletion: bool = False


class TermsModel(BaseModel):
    epw: EPWModel
    light: LightModel = LightModel()
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
