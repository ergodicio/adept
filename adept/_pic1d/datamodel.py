from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adept._vlasov1d.datamodel import (
    DensityConfig,
    EMDriverSetConfig,
    MLFlowConfig,
    SaveConfig,
    UnitsConfig,
)


class PICGridConfig(BaseModel):
    """Spatial/time grid + particle-resolution knobs for PIC-1D."""

    dt: float | str
    nx: int
    tmin: float | str = 0.0
    tmax: float | str
    xmax: float | str
    xmin: float | str
    ppc: int = 256
    particle_shape: Literal["linear", "tsc", "cubic"] = "tsc"


class PICSpeciesConfig(BaseModel):
    """Configuration for a physical species in PIC-1D simulations.

    Mirrors :class:`SpeciesConfig` from Vlasov-1D but carries PIC-specific
    fields (loading method, max loading velocity) instead of (vmax, nv).
    """

    name: str = "electron"
    charge: float = -1.0
    mass: float = 1.0
    density_components: list[str] | None = None
    loading: Literal["quiet", "random"] = "quiet"
    vmax_load: float = 8.0  # cutoff for inverse-CDF / rejection sampling in v


class PICTermsConfig(BaseModel):
    field: Literal["poisson"] = "poisson"
    time: Literal["leapfrog", "yoshida4"] = "leapfrog"
    species: list[PICSpeciesConfig] | None = None


class PICDiagnosticsConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Reserved for future PIC-specific diagnostics
    diag_dummy: bool = Field(default=False, alias="diag-dummy")


class PIC1DConfig(BaseModel):
    units: UnitsConfig
    density: DensityConfig
    grid: PICGridConfig
    save: SaveConfig
    solver: str
    mlflow: MLFlowConfig
    drivers: EMDriverSetConfig
    terms: PICTermsConfig
    diagnostics: PICDiagnosticsConfig = PICDiagnosticsConfig()
