from pydantic import BaseModel, ConfigDict


class SpaceProfileModel(BaseModel):
    baseline: float
    bump_or_trough: str
    center: float
    rise: float
    slope: float
    bump_height: float
    width: float


class SpeciesBackgroundModel(BaseModel):
    noise_seed: int
    noise_type: str
    noise_val: float
    v0: float
    T0: float
    m: float
    basis: str
    space_profile: SpaceProfileModel


class DensityModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    quasineutrality: bool
    # Note: Allow arbitrary species-* fields to be defined dynamically
    # e.g., species_background, species_beam, etc.
    # These are validated as SpeciesBackgroundModel when accessed


class UnitsModel(BaseModel):
    laser_wavelength: str
    normalizing_temperature: str
    normalizing_density: str
    Z: int
    Zp: int


class GridModel(BaseModel):
    dt: float
    nv: int | None = None  # Optional: for backward compatibility with single-species config files
    nx: int
    tmin: float
    tmax: float
    vmax: float | None = None  # Optional: for backward compatibility with single-species config files
    xmax: float
    xmin: float


class TimeSaveModel(BaseModel):
    tmin: float | None = None  # Optional: defaults to grid.tmin at runtime
    tmax: float | None = None  # Optional: defaults to grid.tmax at runtime
    nt: int


class SaveModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    fields: dict[str, TimeSaveModel]
    # Note: Allow arbitrary species save sections (electron, ion, etc.)
    # These are validated as dict[str, TimeSaveModel] when accessed


class ExDriverModel(BaseModel):
    a0: float
    k0: float
    t_center: float
    t_rise: float
    t_width: float
    w0: float
    dw0: float
    x_center: float
    x_rise: float
    x_width: float


class EyDriverModel(BaseModel):
    pass


class DriversModel(BaseModel):
    ex: dict[str, ExDriverModel]
    ey: EyDriverModel


class TimeTermModel(BaseModel):
    baseline: float
    bump_or_trough: str
    center: float
    rise: float
    slope: float
    bump_height: float
    width: float


class SpaceTermModel(BaseModel):
    baseline: float
    bump_or_trough: str
    center: float
    rise: float
    slope: float
    bump_height: float
    width: float


class FokkerPlanckModel(BaseModel):
    is_on: bool
    type: str
    time: TimeTermModel
    space: SpaceTermModel


class KrookModel(BaseModel):
    is_on: bool
    time: TimeTermModel
    space: SpaceTermModel


class SpeciesConfig(BaseModel):
    """Configuration for a physical species in multi-species simulations.

    Each species has its own charge-to-mass ratio and velocity grid, allowing
    for electron-ion physics and other multi-species interactions.

    Attributes:
        name: Species name (e.g., 'electron', 'ion')
        charge: Charge in units of fundamental charge (e.g., -1.0 for electrons, 10.0 for Z=10 ions)
        mass: Mass in units of electron mass (e.g., 1.0 for electrons, 1836.0 for protons)
        vmax: Velocity grid maximum for this species
        nv: Number of velocity grid points for this species
        density_components: List of density component names from the 'density:' section
                          that contribute to this species' distribution function
    """

    name: str
    charge: float
    mass: float
    vmax: float
    nv: int
    density_components: list[str]


class TermsModel(BaseModel):
    field: str
    edfdv: str
    time: str
    species: list[SpeciesConfig] | None = None
    fokker_planck: FokkerPlanckModel
    krook: KrookModel


class MLFlowModel(BaseModel):
    experiment: str
    run: str


class ConfigModel(BaseModel):
    units: UnitsModel
    density: DensityModel
    grid: GridModel
    save: SaveModel
    solver: str
    mlflow: MLFlowModel
    drivers: DriversModel
    terms: TermsModel
