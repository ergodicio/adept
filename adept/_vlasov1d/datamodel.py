"""Pydantic configuration models for Vlasov-1D simulations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adept.functions import EnvelopeConfig, SpaceTimeEnvelopeConfig


# TODO(gh-250): refactor to use a nested SpaceTimeEnvelopeConfig instead of inlining
# envelope fields. See https://github.com/ergodicio/adept/issues/250
class SpeciesComponentConfig(BaseModel):
    """Density and velocity-space parameters for one species density component."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    noise_seed: int
    noise_type: str
    noise_val: float
    v0: float
    T0: float
    m: float = 2.0
    basis: str
    # Sine basis
    baseline: float | None = None
    amplitude: float | None = None
    wavenumber: float | None = None
    # Gradient basis (linear/exponential)
    center: float | str | None = None
    gradient_scale_length: float | str | None = Field(default=None, alias="gradient scale length")
    val_at_center: float | str | None = Field(default=None, alias="val at center")
    rise: float | str | None = None
    width: float | str | None = None
    # Envelope fields (tanh basis, or uniform top-level envelope)
    bump_or_trough: str | None = None
    bump_height: float | None = None
    slope: float | None = None
    # Space profile (uniform basis with nested envelope)
    space_profile: EnvelopeConfig | None = Field(default=None, alias="space-profile")


class DensityConfig(BaseModel):
    """Collection of species density components and quasineutrality setting."""

    model_config = ConfigDict(extra="allow")

    quasineutrality: bool

    def get_component(self, name: str) -> SpeciesComponentConfig:
        """Return a named species density component as a validated config model."""
        return SpeciesComponentConfig.model_validate(self.model_extra[name])


class UnitsConfig(BaseModel):
    """Physical units used to construct the plasma normalization."""

    normalizing_temperature: str
    normalizing_density: str


class GridConfig(BaseModel):
    """Configuration-space and velocity-space grid parameters."""

    dt: float | str
    nx: int
    tmin: float | str = 0.0
    tmax: float | str
    xmax: float | str
    xmin: float | str
    nv: int | None = None
    vmax: float | None = None
    vmin: float | None = None
    parallel: tuple[str, ...] | bool = False
    c_light: float | None = None

    @field_validator("parallel", mode="before")
    @classmethod
    def validate_parallel(cls, v):
        """Normalize and validate the optional parallelization axis list."""
        if v is False or v is None:
            return False
        if isinstance(v, (list, tuple)):
            valid = {"x", "v"}
            invalid = set(v) - valid
            if invalid:
                raise ValueError(f"parallel axes must be 'x' and/or 'v', got unknown: {invalid}")
            return tuple(v)
        raise ValueError(f"parallel must be False or a sequence of axes ('x', 'v'), got: {v!r}")


class TimeSaveConfig(BaseModel):
    """Temporal sampling configuration for saved quantities."""

    tmin: float | None = None
    tmax: float | None = None
    nt: int


class SaveConfig(BaseModel):
    """Top-level save configuration keyed by field, diagnostic, or species name."""

    model_config = ConfigDict(extra="allow")

    fields: dict[str, TimeSaveConfig]


class IntensityWavelengthDriverConfig(BaseModel):
    """Laser driver parameters specified by physical intensity and wavelength."""

    intensity: str
    wavelength: str
    leftgoing: bool = False


class AKWDriverConfig(BaseModel):
    """Laser driver parameters specified directly as amplitude, wavenumber, and frequency."""

    a0: float
    k0: float | None = None
    w0: float | None = None
    dw0: float

    @model_validator(mode="after")
    def check_w_or_k(self) -> "AKWDriverConfig":
        """Require enough information to infer both driver wavenumber and frequency."""
        if self.k0 is None and self.w0 is None:
            raise ValueError("You must specify at least one of k0 or w0.")
        return self


class EMDriverConfig(BaseModel):
    """One electromagnetic driver with parameters, envelope, and source geometry."""

    params: IntensityWavelengthDriverConfig | AKWDriverConfig
    envelope: SpaceTimeEnvelopeConfig
    source_type: Literal["extended", "point"] = "extended"


class EMDriverSetConfig(BaseModel):
    """Longitudinal and transverse electromagnetic driver collections."""

    ex: dict[str, EMDriverConfig]
    ey: dict[str, EMDriverConfig]


class HouLiFilterConfig(BaseModel):
    """Configuration for optional Hou-Li spectral filtering."""

    is_on: bool
    alpha: float = 36.0
    order: int = 36
    dimensions: list[str] = ["x", "v"]


class FokkerPlanckConfig(BaseModel):
    """Configuration for optional Fokker-Planck collisions."""

    is_on: bool
    type: str
    time: EnvelopeConfig
    space: EnvelopeConfig


class KrookConfig(BaseModel):
    """Configuration for optional Krook relaxation."""

    is_on: bool
    time: EnvelopeConfig
    space: EnvelopeConfig


class SpeciesConfig(BaseModel):
    """Configuration for a physical species in multi-species simulations."""

    name: str
    charge: float
    mass: float
    vmax: float
    vmin: float | None = None
    nv: int
    density_components: list[str]


class TermsConfig(BaseModel):
    """Numerical term selections and optional physics operators."""

    field: str
    edfdv: str
    time: str
    species: list[SpeciesConfig] | None = None
    fokker_planck: FokkerPlanckConfig
    krook: KrookConfig
    hou_li_filter: HouLiFilterConfig = HouLiFilterConfig(is_on=False)
    diags: bool | None = None


class MLFlowConfig(BaseModel):
    """MLflow experiment and run naming configuration."""

    experiment: str
    run: str


class DiagnosticsConfig(BaseModel):
    """Optional diagnostic distribution-function save toggles."""

    model_config = ConfigDict(populate_by_name=True)

    diag_vlasov_dfdt: bool = Field(default=False, alias="diag-vlasov-dfdt")
    diag_fp_dfdt: bool = Field(default=False, alias="diag-fp-dfdt")


class Vlasov1DConfig(BaseModel):
    """Validated top-level configuration for a Vlasov-1D ADEPT run."""

    units: UnitsConfig
    density: DensityConfig
    grid: GridConfig
    save: SaveConfig
    solver: str
    mlflow: MLFlowConfig
    drivers: EMDriverSetConfig
    terms: TermsConfig
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
