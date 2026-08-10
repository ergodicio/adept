"""Pydantic configuration models for Vlasov-1D2V (cylindrical velocity) simulations."""

from pydantic import BaseModel, ConfigDict, Field

from adept._vlasov1d.datamodel import GridConfig, Vlasov1DConfig


class Grid2VConfig(GridConfig):
    """Grid configuration with the perpendicular-velocity axis.

    The parallel axis (nv, vmax, vmin) is inherited from the 1D grid config.
    The perpendicular axis is cylindrical: v_perp in (0, vperp_max) with
    cell-centered points and integration weight w_perp = 2*pi*v_perp*dv_perp,
    so that int f d3v = int f w_perp dv_perp dv_par.
    """

    nvperp: int
    vperp_max: float


class Diagnostics2VConfig(BaseModel):
    """Toggles for the accumulated (time-integrated) marginal diagnostics.

    These are integrals, not sampled rates: post-processing differences them
    between save points to recover exact interval-averaged rates without
    aliasing the 2*omega wave-particle exchange. See solvers/vector_field.py.
    """

    model_config = ConfigDict(populate_by_name=True)

    diag_vlasov_cumulative: bool = Field(default=False, alias="diag-vlasov-cumulative")
    diag_fp_cumulative: bool = Field(default=False, alias="diag-fp-cumulative")


class Vlasov1D2VConfig(Vlasov1DConfig):
    """Validated top-level configuration for a Vlasov-1D2V ADEPT run."""

    grid: Grid2VConfig
    diagnostics: Diagnostics2VConfig = Diagnostics2VConfig()
