from pydantic import BaseModel
from typing import Dict


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
    quasineutrality: bool
    species_background: SpeciesBackgroundModel


class UnitsModel(BaseModel):
    laser_wavelength: str
    normalizing_temperature: str
    normalizing_density: str
    Z: int
    Zp: int


class GridModel(BaseModel):
    dt: float
    nv: int
    nx: int
    tmin: float
    tmax: float
    vmax: float
    xmax: float
    xmin: float


class TimeSaveModel(BaseModel):
    tmin: float
    tmax: float
    nt: int


class SaveModel(BaseModel):
    fields: Dict[str, TimeSaveModel]
    electron: Dict[str, TimeSaveModel]


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
    ex: Dict[str, ExDriverModel]
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
