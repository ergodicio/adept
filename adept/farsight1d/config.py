"""Strict host-side configuration for normalized 1D1V FARSIGHT simulations."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class GridConfig(_StrictConfig):
    """Uniform reference nodes; nx and nv count intervals, not nodes."""

    nx: int = Field(ge=4)
    nv: int = Field(ge=2)
    xmin: float = 0.0
    xmax: float
    vmin: float
    vmax: float

    @model_validator(mode="after")
    def validate_grid(self):
        if self.nx % 2 or self.nv % 2:
            raise ValueError("nx and nv must be even for the biquadratic panel mesh")
        if self.xmax <= self.xmin or self.vmax <= self.vmin:
            raise ValueError("xmax > xmin and vmax > vmin are required")
        if not math.isfinite(self.xmax - self.xmin) or not math.isfinite(self.vmax - self.vmin):
            raise ValueError("spatial and velocity domain lengths must be finite")
        return self


class TimeConfig(_StrictConfig):
    tmin: float = 0.0
    tmax: float
    dt: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_time(self):
        if self.tmax <= self.tmin:
            raise ValueError("tmax must be greater than tmin")
        steps = (self.tmax - self.tmin) / self.dt
        if not math.isfinite(steps) or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
            raise ValueError("(tmax - tmin) / dt must be an integer")
        if round(steps) < 1:
            raise ValueError("the simulation must contain at least one complete step")
        return self

    @property
    def num_steps(self) -> int:
        return round((self.tmax - self.tmin) / self.dt)


class InitialConfig(_StrictConfig):
    """One Maxwellian or equal counterstreaming Maxwellians, with a density mode."""

    kind: Literal["maxwellian", "two-stream"] = "maxwellian"
    thermal_speed: float = Field(default=1.0, gt=0)
    drift: float = 0.0
    amplitude: float = Field(default=0.01, ge=-1, le=1)
    mode: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_drift(self):
        if self.kind == "two-stream" and self.drift <= 0:
            raise ValueError("two-stream initialization requires drift > 0")
        return self


class NumericalConfig(_StrictConfig):
    """Direct regularized field evaluation and panel remeshing controls."""

    epsilon: float = Field(gt=0)
    quadrature: Literal["trapezoid", "simpson"] = "trapezoid"
    remesh_every: int = Field(default=1, ge=0)
    chunk_size: int = Field(default=64, ge=1)


class AMRConfig(_StrictConfig):
    """Bounded quadtree refinement using the nodal distribution range.

    Each remesh rebuilds leaves from the root panels, permitting both
    refinement and coarsening. Static candidate and active capacities keep
    array shapes fixed under JAX transformations.
    """

    enabled: bool = False
    max_level: int = Field(default=1, ge=0, le=4)
    min_level: int = Field(default=0, ge=0, le=4)
    max_panels: int = Field(default=256, ge=1)
    atol: float = Field(default=0.05, ge=0)
    rtol: float = Field(default=0.0, ge=0)
    max_gap_fraction: float = Field(default=0.01, gt=0)

    @model_validator(mode="after")
    def validate_levels(self):
        if self.min_level > self.max_level:
            raise ValueError("amr.min_level must not exceed amr.max_level")
        return self


class SaveCadence(_StrictConfig):
    """Save at this step interval, always including initial and final states."""

    every_steps: int = Field(default=1, ge=1)


class SaveConfig(_StrictConfig):
    scalars: SaveCadence = Field(default_factory=SaveCadence)
    fields: SaveCadence | None = Field(default_factory=lambda: SaveCadence(every_steps=10))
    distribution: SaveCadence | None = None


class Farsight1DConfig(_StrictConfig):
    """Normalized electrons (q=-1, m=1) in a periodic, neutralizing ion background."""

    grid: GridConfig
    time: TimeConfig
    initial: InitialConfig = Field(default_factory=InitialConfig)
    numerical: NumericalConfig
    amr: AMRConfig = Field(default_factory=AMRConfig)
    save: SaveConfig = Field(default_factory=SaveConfig)

    @model_validator(mode="after")
    def validate_mode(self):
        if self.initial.mode >= self.grid.nx // 2:
            raise ValueError("initial mode must lie strictly below the spatial Nyquist mode (nx / 2)")
        if self.amr.enabled:
            roots = (self.grid.nx // 2) * (self.grid.nv // 2)
            minimum = roots * 4**self.amr.min_level
            if self.amr.max_panels < minimum:
                raise ValueError(f"amr.max_panels must be at least {minimum} to cover all root panels at amr.min_level")
            candidates = roots * sum(4**level for level in range(self.amr.max_level + 1))
            if candidates > 32768:
                raise ValueError(
                    f"AMR requires {candidates} candidate panels, exceeding the static scratch limit of 32768; "
                    "reduce grid.nx/grid.nv or amr.max_level"
                )
        return self


__all__ = ["AMRConfig", "Farsight1DConfig"]
