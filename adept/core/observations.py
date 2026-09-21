"""Host-side observation planning contracts.

The objects in this module describe what a numerical program retains without
importing JAX or a solver backend. Concrete program adapters validate the declared
schemas against JAX abstract evaluation before an expensive solve.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable


class ScheduleKind(StrEnum):
    """Coordinate used by an observation schedule."""

    TIME = "time"
    STEP = "step"


class ObservationReduction(StrEnum):
    """Reduction applied independently to every observation leaf."""

    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


class ObservationPlacement(StrEnum):
    """Expected logical placement of an observation result."""

    DEVICE = "device"
    REPLICATED = "replicated"
    SHARDED = "sharded"


class ObservationCollective(StrEnum):
    """Collective communication expected while producing an observation."""

    NONE = "none"
    ALL_REDUCE = "all-reduce"
    ALL_GATHER = "all-gather"


class ObservationRetention(StrEnum):
    """Samples retained by the numerical program."""

    ALL = "all"
    LAST = "last"


class MaterializationTarget(StrEnum):
    """Hosts that receive materialized result arrays."""

    RANK_ZERO = "rank-zero"
    ALL_HOSTS = "all-hosts"


@dataclass(frozen=True, slots=True)
class ObservationSchedule:
    """Finite, strictly increasing observation points."""

    kind: ScheduleKind
    points: tuple[int | float, ...]

    def __init__(self, kind: ScheduleKind, points: Sequence[int | float]) -> None:
        normalized_kind = ScheduleKind(kind)
        normalized: list[int | float] = []
        for point in points:
            if normalized_kind is ScheduleKind.STEP:
                if isinstance(point, bool) or not isinstance(point, Integral):
                    raise TypeError("step observation points must be integers")
                if point < 0:
                    raise ValueError("step observation points must be non-negative")
                normalized.append(int(point))
            else:
                if isinstance(point, bool) or not isinstance(point, Real):
                    raise TypeError("time observation points must be real numbers")
                value = float(point)
                if not math.isfinite(value):
                    raise ValueError("time observation points must be finite")
                normalized.append(value)

        if not normalized:
            raise ValueError("an observation schedule must contain at least one point")
        if any(right <= left for left, right in pairwise(normalized)):
            raise ValueError("observation points must be strictly increasing")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "points", tuple(normalized))

    @classmethod
    def at_times(cls, points: Sequence[int | float]) -> ObservationSchedule:
        return cls(ScheduleKind.TIME, points)

    @classmethod
    def at_steps(cls, points: Sequence[int]) -> ObservationSchedule:
        return cls(ScheduleKind.STEP, points)

    @classmethod
    def every_steps(
        cls,
        every: int,
        *,
        stop: int | None,
        start: int = 0,
    ) -> ObservationSchedule:
        """Create a bounded cadence, including ``stop`` only when aligned."""

        if isinstance(every, bool) or not isinstance(every, Integral) or every <= 0:
            raise ValueError("observation cadence must be a positive integer")
        if stop is None:
            raise ValueError("an observation cadence requires a finite stop step")
        if isinstance(start, bool) or not isinstance(start, Integral) or start < 0:
            raise ValueError("observation cadence start must be a non-negative integer")
        if isinstance(stop, bool) or not isinstance(stop, Integral) or stop < start:
            raise ValueError("observation cadence stop must be an integer at or after start")
        return cls.at_steps(tuple(range(int(start), int(stop) + 1, int(every))))

    @classmethod
    def from_legacy_time_config(cls, config: Mapping[str, Any]) -> ObservationSchedule:
        """Adapt ``{tmin, tmax, nt}`` or an existing ``ax`` time configuration."""

        if {"tmin", "tmax", "nt"}.issubset(config):
            start = float(config["tmin"])
            stop = float(config["tmax"])
            raw_count = config["nt"]
            if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
                raise TypeError("legacy time schedule nt must be an integer")
            count = int(raw_count)
        elif "ax" in config:
            return cls.at_times(tuple(float(value) for value in config["ax"]))
        else:
            missing = sorted({"tmin", "tmax", "nt"}.difference(config))
            raise ValueError(f"legacy time schedule is missing fields: {missing!r}")
        if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
            raise ValueError("legacy time schedule bounds must be finite and ordered")
        if count <= 0:
            raise ValueError("legacy time schedule nt must be positive")
        if count == 1:
            return cls.at_times((start,))
        spacing = (stop - start) / (count - 1)
        return cls.at_times(tuple(start + index * spacing for index in range(count)))

    def as_steps(
        self,
        *,
        t0: float,
        dt: float,
        num_steps: int,
        tolerance: float = 1e-7,
    ) -> ObservationSchedule:
        """Map time points to exact discrete steps, rejecting off-grid schedules."""

        if self.kind is ScheduleKind.STEP:
            steps = tuple(int(point) for point in self.points)
        else:
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("dt must be finite and positive")
            steps_list = []
            for point in self.points:
                coordinate = (float(point) - t0) / dt
                step = round(coordinate)
                if not math.isclose(coordinate, step, rel_tol=0.0, abs_tol=tolerance):
                    raise ValueError(f"observation time {point} does not align with the discrete step grid")
                steps_list.append(step)
            steps = tuple(steps_list)
        if steps[-1] > num_steps:
            raise ValueError(f"observation step {steps[-1]} exceeds the program's final step {num_steps}")
        return ObservationSchedule.at_steps(steps)

    def retained_points(self, retention: ObservationRetention) -> tuple[int | float, ...]:
        retention = ObservationRetention(retention)
        return self.points[-1:] if retention is ObservationRetention.LAST else self.points

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "points": list(self.points)}


@dataclass(frozen=True, slots=True)
class ObservationLeaf:
    """Expected array leaf in one observation sample."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    itemsize: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("observation leaf path must be non-empty")
        if not self.dtype:
            raise ValueError("observation leaf dtype must be non-empty")
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in self.shape):
            raise ValueError("observation leaf dimensions must be non-negative integers")
        if isinstance(self.itemsize, bool) or not isinstance(self.itemsize, int) or self.itemsize <= 0:
            raise ValueError("observation leaf itemsize must be a positive integer")

    @property
    def sample_bytes(self) -> int:
        elements = math.prod(self.shape)
        return elements * self.itemsize

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "itemsize": self.itemsize,
        }


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    """Flattened PyTree schema for one observation sample."""

    leaves: tuple[ObservationLeaf, ...]

    def __init__(self, leaves: Sequence[ObservationLeaf]) -> None:
        normalized = tuple(leaves)
        if not normalized:
            raise ValueError("an observation schema must contain at least one array leaf")
        paths = [leaf.path for leaf in normalized]
        if len(paths) != len(set(paths)):
            raise ValueError("observation schema leaf paths must be unique")
        object.__setattr__(self, "leaves", normalized)

    @property
    def sample_bytes(self) -> int:
        return sum(leaf.sample_bytes for leaf in self.leaves)

    def to_dict(self) -> dict[str, Any]:
        return {"leaves": [leaf.to_dict() for leaf in self.leaves]}


@runtime_checkable
class ObservationFunction(Protocol):
    """Pure numerical observation evaluated within the JAX boundary."""

    def __call__(self, t: Any, state: Any, inputs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """One named, scheduled numerical observation."""

    name: str
    function: Callable[[Any, Any, Any], Any] = field(repr=False, compare=False)
    schedule: ObservationSchedule
    schema: ObservationSchema
    reduction: ObservationReduction = ObservationReduction.NONE
    placement: ObservationPlacement = ObservationPlacement.DEVICE
    collective: ObservationCollective = ObservationCollective.NONE
    retention: ObservationRetention = ObservationRetention.ALL

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name or "/" in name or name == "__final_state__":
            raise ValueError("observation names must be non-empty path components")
        if not callable(self.function):
            raise TypeError("observation function must be callable")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "reduction", ObservationReduction(self.reduction))
        object.__setattr__(self, "placement", ObservationPlacement(self.placement))
        object.__setattr__(self, "collective", ObservationCollective(self.collective))
        object.__setattr__(self, "retention", ObservationRetention(self.retention))

    @property
    def retained_samples(self) -> int:
        return len(self.schedule.retained_points(self.retention))

    @property
    def retained_bytes(self) -> int:
        return self.retained_samples * self.schema.sample_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schedule": self.schedule.to_dict(),
            "schema": self.schema.to_dict(),
            "reduction": self.reduction.value,
            "placement": self.placement.value,
            "collective": self.collective.value,
            "retention": self.retention.value,
            "retained_bytes": self.retained_bytes,
        }


@dataclass(frozen=True, slots=True)
class ObservationPlan:
    """Validated observation set with a pre-execution memory budget."""

    observations: tuple[ObservationSpec, ...]
    materialization: MaterializationTarget = MaterializationTarget.ALL_HOSTS
    max_retained_bytes: int | None = 1024**3

    def __init__(
        self,
        observations: Sequence[ObservationSpec] = (),
        *,
        materialization: MaterializationTarget = MaterializationTarget.ALL_HOSTS,
        max_retained_bytes: int | None = 1024**3,
    ) -> None:
        normalized = tuple(observations)
        names = [observation.name for observation in normalized]
        if len(names) != len(set(names)):
            raise ValueError("observation names must be unique within a plan")
        if max_retained_bytes is not None and (
            isinstance(max_retained_bytes, bool) or not isinstance(max_retained_bytes, int) or max_retained_bytes < 0
        ):
            raise ValueError("max_retained_bytes must be a non-negative integer or None")
        object.__setattr__(self, "observations", normalized)
        object.__setattr__(self, "materialization", MaterializationTarget(materialization))
        object.__setattr__(self, "max_retained_bytes", max_retained_bytes)
        if max_retained_bytes is not None and self.estimated_retained_bytes > max_retained_bytes:
            raise ValueError(
                "observation plan retains "
                f"{self.estimated_retained_bytes} bytes, exceeding its {max_retained_bytes}-byte budget"
            )

    @property
    def estimated_retained_bytes(self) -> int:
        return sum(observation.retained_bytes for observation in self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [observation.to_dict() for observation in self.observations],
            "materialization": self.materialization.value,
            "estimated_retained_bytes": self.estimated_retained_bytes,
            "max_retained_bytes": self.max_retained_bytes,
        }


__all__ = [
    "MaterializationTarget",
    "ObservationCollective",
    "ObservationFunction",
    "ObservationLeaf",
    "ObservationPlacement",
    "ObservationPlan",
    "ObservationReduction",
    "ObservationRetention",
    "ObservationSchedule",
    "ObservationSchema",
    "ObservationSpec",
    "ScheduleKind",
]
