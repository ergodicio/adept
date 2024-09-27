from typing import Dict, Union
from pydantic import BaseModel


class MLFlowModel(BaseModel):
    experiment: str
    run: str


class UnitsModel(BaseModel):
    laser_wavelength: str
    normalizing_temperature: str
    normalizing_density: str
    Z: int
    Zp: int


class GridModel(BaseModel):
    nx: int
    xmin: float
    xmax: float
    tmin: float
    tmax: float


class TimeSaveModel(BaseModel):
    tmin: float
    tmax: float
    nt: int


class SpaceSaveModel(BaseModel):
    xmin: float
    xmax: float
    nx: int


class KxSaveModel(BaseModel):
    kxmin: float
    kxmax: float
    nkx: int


class SaveModel(BaseModel):
    t: TimeSaveModel
    x: SpaceSaveModel
    kx: KxSaveModel


class TrappingModel(BaseModel):
    is_on: bool
    kld: float


class IonModel(BaseModel):
    is_on: bool
    landau_damping: bool
    mass: float
    T0: float
    charge: float
    gamma: Union[int, str]
    trapping: TrappingModel


class ElectronModel(BaseModel):
    is_on: bool
    landau_damping: bool
    T0: float
    mass: float
    charge: float
    gamma: Union[int, str]
    trapping: TrappingModel


class PhysicsModel(BaseModel):
    ion: IonModel
    electron: ElectronModel


class ExDriverModel(BaseModel):
    k0: float
    w0: float
    dw0: float
    t_c: float
    t_w: float
    t_r: float
    x_c: float
    x_w: float
    x_r: float
    a0: float


class DriversModel(BaseModel):
    ex: Dict[str, ExDriverModel]


class ConfigModel(BaseModel):
    solver: str
    mlflow: MLFlowModel
    adjoint: bool
    units: UnitsModel
    grid: GridModel
    save: SaveModel
    physics: PhysicsModel
    drivers: DriversModel
