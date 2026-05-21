"""Pydantic configuration models for the Vlasov-2D solver.

2D2V Vlasov-Maxwell: f(x, y, vx, vy) coupled to (Ex, Ey, Bz) in the TE
polarization. Velocity grids are square (nvx == nvy, vmax_x == vmax_y are
sensible defaults but each species owns its own (nvx, nvy, vmax)).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adept.functions import EnvelopeConfig, SpaceTimeEnvelopeConfig


class Space2DTimeEnvelopeConfig(BaseModel):
    """Time * space_x * space_y envelope for 2D drivers."""

    time: EnvelopeConfig
    space_x: EnvelopeConfig = Field(alias="space-x")
    space_y: EnvelopeConfig = Field(alias="space-y")

    model_config = ConfigDict(populate_by_name=True)


class SpeciesComponentConfig(BaseModel):
    """One additive component (subspecies) of a species' density."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    noise_seed: int = 42
    noise_type: str = "uniform"
    noise_val: float = 0.0
    v0x: float = 0.0
    v0y: float = 0.0
    T0: float = 1.0
    m: float = 2.0
    basis: str = "uniform"
    baseline: float | None = None
    amplitude: float | None = None
    wavenumber_x: float | None = Field(default=None, alias="wavenumber-x")
    wavenumber_y: float | None = Field(default=None, alias="wavenumber-y")
    # Optional masks (tanh envelopes applied multiplicatively)
    space_x: EnvelopeConfig | None = Field(default=None, alias="space-x")
    space_y: EnvelopeConfig | None = Field(default=None, alias="space-y")


class DensityConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    quasineutrality: bool = True

    def get_component(self, name: str) -> SpeciesComponentConfig:
        return SpeciesComponentConfig.model_validate(self.model_extra[name])


class UnitsConfig(BaseModel):
    normalizing_temperature: str
    normalizing_density: str
    laser_wavelength: str | None = None
    Z: float | None = None


class GridConfig(BaseModel):
    dt: float | str
    nx: int
    ny: int
    tmin: float | str = 0.0
    tmax: float | str
    xmin: float | str
    xmax: float | str
    ymin: float | str
    ymax: float | str
    nvx: int | None = None
    nvy: int | None = None
    vmax: float | None = None
    parallel: tuple[str, ...] | bool = False

    @field_validator("parallel", mode="before")
    @classmethod
    def validate_parallel(cls, v):
        if v is False or v is None:
            return False
        if isinstance(v, (list, tuple)):
            valid = {"x", "y"}
            invalid = set(v) - valid
            if invalid:
                raise ValueError(f"parallel axes must be 'x' and/or 'y', got unknown: {invalid}")
            return tuple(v)
        raise ValueError(f"parallel must be False or a sequence of axes ('x','y'), got: {v!r}")


class TimeSaveConfig(BaseModel):
    tmin: float | None = None
    tmax: float | None = None
    nt: int


class SaveConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    fields: dict[str, TimeSaveConfig]


class AKWDriverConfig(BaseModel):
    a0: float
    k0x: float = 0.0
    k0y: float = 0.0
    w0: float
    dw0: float = 0.0


class EMDriverConfig(BaseModel):
    """A single externally-prescribed driver entering Maxwell as a current source."""

    params: AKWDriverConfig
    envelope: Space2DTimeEnvelopeConfig
    polarization: Literal["x", "y"] = "y"


class EMDriverSetConfig(BaseModel):
    ex: dict[str, EMDriverConfig] = {}
    ey: dict[str, EMDriverConfig] = {}


class HouLiFilterConfig(BaseModel):
    is_on: bool = False
    alpha: float = 36.0
    order: int = 36
    dimensions: list[str] = ["x", "y", "vx", "vy"]


class FokkerPlanckConfig(BaseModel):
    is_on: bool = False
    type: str = "dougherty"
    time: EnvelopeConfig | None = None
    space_x: EnvelopeConfig | None = Field(default=None, alias="space-x")
    space_y: EnvelopeConfig | None = Field(default=None, alias="space-y")


class KrookConfig(BaseModel):
    is_on: bool = False
    time: EnvelopeConfig | None = None
    space_x: EnvelopeConfig | None = Field(default=None, alias="space-x")
    space_y: EnvelopeConfig | None = Field(default=None, alias="space-y")


class SpeciesConfig(BaseModel):
    name: str
    charge: float
    mass: float
    vmax: float
    nvx: int
    nvy: int
    density_components: list[str]


class TermsConfig(BaseModel):
    """Time-integration / physics knobs.

    edfdv: 'exponential' uses spectral shift in (kvx, kvy). 'sl' uses a 2D
        semi-Lagrangian (cubic interp). The magnetic rotation is always SL.
    vdfdx: only 'exponential' is implemented (spectral shift in (kx, ky)).
    skip_maxwell: if True, freeze (Ex, Ey, Bz) at their initial values for the
        whole run. Useful for kinetic-only tests (e.g. measuring a pure
        gyrofrequency under a prescribed uniform Bz).
    """

    edfdv: Literal["exponential", "sl"] = "exponential"
    vdfdx: Literal["exponential"] = "exponential"
    skip_maxwell: bool = False
    species: list[SpeciesConfig] | None = None
    fokker_planck: FokkerPlanckConfig = FokkerPlanckConfig()
    krook: KrookConfig = KrookConfig()
    hou_li_filter: HouLiFilterConfig = HouLiFilterConfig()


class MLFlowConfig(BaseModel):
    experiment: str
    run: str


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    diag_dfdt: bool = Field(default=False, alias="diag-dfdt")


class Vlasov2DConfig(BaseModel):
    units: UnitsConfig
    density: DensityConfig
    grid: GridConfig
    save: SaveConfig
    solver: str
    mlflow: MLFlowConfig
    drivers: EMDriverSetConfig = EMDriverSetConfig()
    terms: TermsConfig = TermsConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
