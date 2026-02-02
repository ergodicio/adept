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


class DriversModel(BaseModel):
    """
    Define the drivers for the simulation

    """

    E0: E0DriverModel


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


class TimeSaveModel(BaseModel):
    dt: str
    tmax: str
    tmin: str


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
    tpd: bool


class EPWModel(BaseModel):
    boundary: BoundaryModel
    damping: DampingModel
    density_gradient: bool
    linear: bool
    source: SourceModel


class TermsModel(BaseModel):
    epw: EPWModel
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
