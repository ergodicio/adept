"""Shared, normalized physical cases for the phase-space comparison examples."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ComparisonCase:
    """One collisionless, unit-density electron problem with uniform ions.

    ``nx`` and ``nv`` are Eulerian cells or FARSIGHT reference intervals.
    They do not imply equal degrees of freedom between the two methods.
    """

    name: str
    wave_number: float = 0.3
    thermal_speed: float = 1.0
    drift: float = 0.0
    amplitude: float = 0.1
    vmin: float = -6.0
    vmax: float = 6.0
    tmax: float = 40.0
    dt: float = 0.05
    frame_dt: float = 0.5
    nx: int = 32
    nv: int = 128
    epsilon: float = 1.5

    def __post_init__(self):
        if self.name not in {"two-stream", "nlepw"}:
            raise ValueError("Supported comparison cases are 'two-stream' and 'nlepw'")
        for name in ("wave_number", "thermal_speed", "tmax", "dt", "frame_dt", "epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not all(math.isfinite(v) for v in (self.vmin, self.vmax, self.amplitude, self.drift)):
            raise ValueError("Velocity bounds and initial parameters must be finite")
        if self.vmin >= self.vmax or abs(self.amplitude) > 1:
            raise ValueError("Require vmin < vmax and |amplitude| <= 1")
        if self.name == "two-stream" and self.drift <= 0:
            raise ValueError("Two-stream drift must be positive")
        if self.nx < 4 or self.nv < 4 or self.nx % 2 or self.nv % 2:
            raise ValueError("Comparison nx and nv must be even and at least four")
        for label, ratio in (("tmax/dt", self.tmax / self.dt), ("frame_dt/dt", self.frame_dt / self.dt)):
            if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8) or round(ratio) < 1:
                raise ValueError(f"{label} must be a positive integer so saves lie on solver steps")
        # Use one floating-point representation for the endpoint and its save
        # time (e.g. 3 * .05 differs from literal .15), as Diffrax checks bounds.
        object.__setattr__(self, "tmax", round(self.tmax / self.dt) * self.dt)

    @property
    def length(self) -> float:
        return 2.0 * math.pi / self.wave_number

    @property
    def num_steps(self) -> int:
        return round(self.tmax / self.dt)

    @property
    def frame_steps(self) -> np.ndarray:
        steps = np.arange(0, self.num_steps + 1, round(self.frame_dt / self.dt), dtype=int)
        if steps[-1] != self.num_steps:
            steps = np.append(steps, self.num_steps)
        return steps

    @property
    def frame_times(self) -> np.ndarray:
        return self.frame_steps * self.dt

    @property
    def step_times(self) -> np.ndarray:
        return np.arange(self.num_steps + 1) * self.dt

    def to_dict(self) -> dict:
        return asdict(self)

    def initial_distribution(self, x, v):
        """Sample the analytic IC; each solver performs its own initial quadrature normalization."""
        x, v = np.asarray(x), np.asarray(v)
        velocity = np.exp(-0.5 * ((v - self.drift) / self.thermal_speed) ** 2)
        if self.name == "two-stream":
            velocity = 0.5 * (velocity + np.exp(-0.5 * ((v + self.drift) / self.thermal_speed) ** 2))
        velocity /= self.thermal_speed * np.sqrt(2 * np.pi)
        return velocity * (1.0 + self.amplitude * np.cos(self.wave_number * x))

    def farsight_initial(self) -> dict:
        return {
            "kind": "two-stream" if self.name == "two-stream" else "maxwellian",
            "thermal_speed": self.thermal_speed,
            "drift": self.drift,
            "amplitude": self.amplitude,
            "mode": 1,
        }

    def to_farsight_config(self, *, field_solver: str = "direct", chunk_size: int = 64) -> dict:
        """Build the fixed-panel FARSIGHT half of the matched physical problem."""
        if field_solver not in {"direct", "treecode"}:
            raise ValueError("FARSIGHT field_solver must be 'direct' or 'treecode'")
        return {
            "solver": "farsight-1d",
            "grid": {
                "nx": self.nx,
                "nv": self.nv,
                "xmin": 0.0,
                "xmax": self.length,
                "vmin": self.vmin,
                "vmax": self.vmax,
            },
            "time": {"tmin": 0.0, "tmax": self.tmax, "dt": self.dt},
            "initial": self.farsight_initial(),
            "numerical": {
                "epsilon": self.epsilon,
                "quadrature": "trapezoid",
                "remesh_every": 1,
                "chunk_size": chunk_size,
                "field_solver": field_solver,
            },
            "save": {
                "scalars": {"every_steps": 1},
                "fields": {"every_steps": round(self.frame_dt / self.dt)},
                "distribution": {"every_steps": round(self.frame_dt / self.dt)},
            },
        }


def get_case(name: str, **overrides) -> ComparisonCase:
    """Construct a case, accepting explicit resolution/physics/time overrides."""
    defaults = {
        "two-stream": ComparisonCase("two-stream", thermal_speed=0.3, drift=2.0, amplitude=0.01),
        "nlepw": ComparisonCase("nlepw"),
    }
    if name not in defaults:
        raise ValueError(f"Unknown case {name!r}; choose one of {tuple(defaults)}")
    return replace(defaults[name], **overrides)


__all__ = ["ComparisonCase", "get_case"]
