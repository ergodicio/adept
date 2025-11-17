from typing import Union
from pydantic import BaseModel


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
    v0: Union[str, float]  # Can be dimensionless or um/ps
    T0: Union[str, float]  # Can be dimensionless or eV
    m: float  # Mass ratio (dimensionless)
    basis: str
    space_profile: SpaceProfileModel


class DensityModel(BaseModel):
    quasineutrality: bool
    species_background: SpeciesBackgroundModel


class UnitsModel(BaseModel):
    laser_wavelength: str
    normalizing_temperature: str
    normalizing_density: str
    Z: int
    Zp: int


class GridModel(BaseModel):
    dt: Union[str, float]
    nv: int
    nx: int
    tmin: Union[str, float]
    tmax: Union[str, float]
    vmax: Union[str, float]
    xmax: Union[str, float]
    xmin: Union[str, float]


class TimeSaveModel(BaseModel):
    tmin: Union[str, float]
    tmax: Union[str, float]
    nt: int


class SaveModel(BaseModel):
    fields: dict[str, TimeSaveModel]
    electron: dict[str, TimeSaveModel]


class ExDriverModel(BaseModel):
    a0: Union[str, float]  # Can be dimensionless or intensity (W/cm^2)
    k0: Union[str, float]  # Can be dimensionless or 1/um
    t_center: Union[str, float]  # Can be dimensionless or ps
    t_rise: Union[str, float]  # Can be dimensionless or ps
    t_width: Union[str, float]  # Can be dimensionless or ps
    w0: Union[str, float]  # Can be dimensionless or rad/ps
    dw0: Union[str, float]  # Can be dimensionless or rad/ps
    x_center: Union[str, float]  # Can be dimensionless or um
    x_rise: Union[str, float]  # Can be dimensionless or um
    x_width: Union[str, float]  # Can be dimensionless or um


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


class TermsModel(BaseModel):
    field: str
    edfdv: str
    time: str
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
