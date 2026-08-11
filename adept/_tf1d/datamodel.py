"""Pydantic configuration models for the two-fluid 1D (`tf-1d`) solver."""

from typing import Literal

from pydantic import BaseModel


class MLFlowModel(BaseModel):
    experiment: str
    run: str


class UnitsModel(BaseModel):
    normalizing_temperature: str
    normalizing_density: str


class GridModel(BaseModel):
    nx: int
    xmin: float
    xmax: float
    tmin: float
    tmax: float


class TimeSaveModel(BaseModel):
    # All three are required: `BaseTwoFluid1D.init_diffeqsolve` reads them
    # unconditionally. The tmin/tmax defaulting in `get_derived_quantities`
    # targets a nested `save.<name>.t` layout, which tf-1d does not use.
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
    kx: KxSaveModel | None = None  # Optional: omit to skip the k-space diagnostic


class TrappingModel(BaseModel):
    is_on: bool
    kld: float
    # Damping-reduction model applied by `VelocityStepper.landau_damping_term`.
    # Only read when `is_on` is True.
    model: Literal["none", "zk", "delta"] = "none"
    # Electron-electron collision frequency used by the trapping models.
    # Only read when `is_on` is True.
    nuee: float | None = None
    # Legacy neural-network shape hint (e.g. "8|8"). Present in the shipped
    # configs but not currently read by the solver.
    nn: str | None = None


class IonModel(BaseModel):
    is_on: bool
    landau_damping: bool
    mass: float
    T0: float
    charge: float
    gamma: int | float | str
    trapping: TrappingModel


class ElectronModel(BaseModel):
    is_on: bool
    landau_damping: bool
    T0: float
    mass: float
    charge: float
    gamma: int | float | str
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
    ex: dict[str, ExDriverModel]


class NNModel(BaseModel):
    """An `equinox.nn.MLP` specification for a learned closure term."""

    in_size: int
    out_size: int
    width_size: int
    depth: int
    activation: str
    final_activation: str | None = None


class ModelsModel(BaseModel):
    """Learned-closure models. Set `models: false` to disable them entirely."""

    file: str | bool = False  # path to serialized weights, or false for untrained
    nu_g: NNModel | None = None
    nu_d: NNModel | None = None


class ConfigModel(BaseModel):
    solver: str
    mlflow: MLFlowModel
    units: UnitsModel
    grid: GridModel
    save: SaveModel
    physics: PhysicsModel
    drivers: DriversModel
    adjoint: bool | str | None = None
    models: ModelsModel | bool | None = None
