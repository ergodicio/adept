"""Host-side contracts shared by ADEPT builders and numerical programs.

This module must remain importable without JAX, Equinox, MLflow, or a configured
runtime. Array and PRNG-key types are generic so concrete builders can use JAX types
without making specification and registry consumers import JAX themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Generic, NamedTuple, Protocol, TypeVar, runtime_checkable


class ExecutionKind(StrEnum):
    """Kind of state transition implemented by a solver program."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    EXTERNAL = "external"


class Precision(StrEnum):
    """Numerical precision requirement advertised by a solver."""

    DEFAULT = "default"
    X64 = "x64"


class Placement(StrEnum):
    """Execution placements a prepared solver can support."""

    SINGLE_DEVICE = "single-device"
    MULTI_DEVICE = "multi-device"
    MULTI_HOST = "multi-host"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class SolverCapabilities:
    """Capabilities used later by executors for compatibility preflight."""

    execution_kind: ExecutionKind
    precision: Precision = Precision.DEFAULT
    differentiable: bool = False
    batchable: bool = False
    placements: frozenset[Placement] = field(default_factory=lambda: frozenset({Placement.SINGLE_DEVICE}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_kind", ExecutionKind(self.execution_kind))
        object.__setattr__(self, "precision", Precision(self.precision))
        object.__setattr__(self, "placements", frozenset(Placement(value) for value in self.placements))
        if not self.placements:
            raise ValueError("A solver must support at least one execution placement")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "execution_kind": self.execution_kind.value,
            "precision": self.precision.value,
            "differentiable": self.differentiable,
            "batchable": self.batchable,
            "placements": sorted(placement.value for placement in self.placements),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolverCapabilities:
        """Restore capabilities from :meth:`to_dict` output."""

        copied = _copy_mapping(value, name="capabilities")
        try:
            execution_kind = copied.pop("execution_kind")
        except KeyError as exc:
            raise ValueError("Capabilities are missing required 'execution_kind'") from exc
        placements = copied.pop("placements", [Placement.SINGLE_DEVICE])
        try:
            capabilities = cls(execution_kind=execution_kind, placements=frozenset(placements), **copied)
        except TypeError as exc:
            raise ValueError(f"Invalid capabilities fields: {exc}") from exc
        return capabilities


def _copy_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    copied = deepcopy(dict(value))
    non_string_keys = [key for key in copied if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"{name} keys must be strings; received {non_string_keys!r}")
    return copied


@dataclass(frozen=True, slots=True, init=False)
class SimulationSpec:
    """Validated solver intent, independent of tracking and runtime services.

    The stored configuration is defensive-copied. ``config`` and ``to_dict`` return
    fresh copies, so a caller can reuse or mutate its original YAML dictionary without
    changing an existing specification.
    """

    solver: str
    schema_version: str
    _config: dict[str, Any] = field(repr=False)

    def __init__(self, solver: str, config: Mapping[str, Any], schema_version: str = "1") -> None:
        solver = solver.strip()
        schema_version = str(schema_version).strip()
        if not solver:
            raise ValueError("solver must be a non-empty stable registry name")
        if not schema_version:
            raise ValueError("schema_version must be non-empty")

        copied = _copy_mapping(config, name="config")
        reserved = sorted({"solver", "mlflow"}.intersection(copied))
        if reserved:
            joined = ", ".join(reserved)
            raise ValueError(f"SimulationSpec config contains reserved host-side keys: {joined}")

        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "_config", copied)

    @classmethod
    def from_legacy_config(cls, config: Mapping[str, Any], *, schema_version: str = "1") -> SimulationSpec:
        """Adapt an existing ADEPT YAML mapping without retaining MLflow settings."""

        copied = _copy_mapping(config, name="legacy config")
        try:
            solver = copied.pop("solver")
        except KeyError as exc:
            raise ValueError("Legacy ADEPT config is missing required 'solver' key") from exc
        copied.pop("mlflow", None)
        return cls(str(solver), copied, schema_version)

    @property
    def config(self) -> Mapping[str, Any]:
        """Return a read-only defensive copy of the solver configuration."""

        return MappingProxyType(deepcopy(self._config))

    def config_dict(self) -> dict[str, Any]:
        """Return a mutable defensive copy for a builder to validate or adapt."""

        return deepcopy(self._config)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation of the specification."""

        return {
            "solver": self.solver,
            "schema_version": self.schema_version,
            "config": self.config_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SimulationSpec:
        """Restore a specification from :meth:`to_dict` output."""

        copied = _copy_mapping(value, name="specification")
        try:
            solver = copied.pop("solver")
            config = copied.pop("config")
        except KeyError as exc:
            raise ValueError(f"Serialized specification is missing required {exc.args[0]!r} field") from exc
        schema_version = copied.pop("schema_version", "1")
        if copied:
            raise ValueError(f"Serialized specification contains unknown fields: {sorted(copied)!r}")
        if not isinstance(config, Mapping):
            raise TypeError("Serialized specification 'config' must be a mapping")
        return cls(str(solver), config, str(schema_version))


@dataclass(frozen=True, slots=True, init=False)
class RunManifest:
    """Reproducibility metadata produced during deterministic preparation."""

    structural_fingerprint: str
    seed: int | None
    key_provenance: str | None
    _raw_config: dict[str, Any] = field(repr=False)
    _resolved_config: dict[str, Any] = field(repr=False)
    _units: dict[str, Any] = field(repr=False)
    _code: dict[str, Any] = field(repr=False)
    _dependencies: dict[str, Any] = field(repr=False)

    def __init__(
        self,
        *,
        raw_config: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        structural_fingerprint: str,
        units: Mapping[str, Any] | None = None,
        seed: int | None = None,
        key_provenance: str | None = None,
        code: Mapping[str, Any] | None = None,
        dependencies: Mapping[str, Any] | None = None,
    ) -> None:
        structural_fingerprint = structural_fingerprint.strip()
        if not structural_fingerprint:
            raise ValueError("structural_fingerprint must be non-empty")

        object.__setattr__(self, "structural_fingerprint", structural_fingerprint)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "key_provenance", key_provenance)
        object.__setattr__(self, "_raw_config", _copy_mapping(raw_config, name="raw_config"))
        object.__setattr__(self, "_resolved_config", _copy_mapping(resolved_config, name="resolved_config"))
        object.__setattr__(self, "_units", _copy_mapping(units or {}, name="units"))
        object.__setattr__(self, "_code", _copy_mapping(code or {}, name="code"))
        object.__setattr__(self, "_dependencies", _copy_mapping(dependencies or {}, name="dependencies"))

    @staticmethod
    def _view(value: dict[str, Any]) -> Mapping[str, Any]:
        return MappingProxyType(deepcopy(value))

    @property
    def raw_config(self) -> Mapping[str, Any]:
        return self._view(self._raw_config)

    @property
    def resolved_config(self) -> Mapping[str, Any]:
        return self._view(self._resolved_config)

    @property
    def units(self) -> Mapping[str, Any]:
        return self._view(self._units)

    @property
    def code(self) -> Mapping[str, Any]:
        return self._view(self._code)

    @property
    def dependencies(self) -> Mapping[str, Any]:
        return self._view(self._dependencies)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of the manifest."""

        return {
            "raw_config": deepcopy(self._raw_config),
            "resolved_config": deepcopy(self._resolved_config),
            "units": deepcopy(self._units),
            "seed": self.seed,
            "key_provenance": self.key_provenance,
            "code": deepcopy(self._code),
            "dependencies": deepcopy(self._dependencies),
            "structural_fingerprint": self.structural_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        """Restore a manifest from :meth:`to_dict` output."""

        copied = _copy_mapping(value, name="manifest")
        try:
            return cls(**copied)
        except TypeError as exc:
            raise ValueError(f"Invalid manifest fields: {exc}") from exc


class RawResult(NamedTuple):
    """Stable PyTree-shaped result returned by every numerical program."""

    final_state: Any
    observations: Any
    times: Any
    status: Any
    stats: Any


@dataclass(frozen=True, slots=True)
class PassthroughAnalyzer:
    """Side-effect-free analyzer used until a solver has a structured report."""

    def analyze(self, result: RawResult, manifest: RunManifest) -> RawResult:
        del manifest
        return result


ProgramT = TypeVar("ProgramT")
ParamsT = TypeVar("ParamsT")
StateT = TypeVar("StateT")
InputsT = TypeVar("InputsT")
AnalyzerT = TypeVar("AnalyzerT")
ProgramParamsT = TypeVar("ProgramParamsT", contravariant=True)
ProgramStateT = TypeVar("ProgramStateT", contravariant=True)
ProgramInputsT = TypeVar("ProgramInputsT", contravariant=True)
KeyT = TypeVar("KeyT", contravariant=True)
RawResultT = TypeVar("RawResultT", covariant=True)
AnalyzedResultT = TypeVar("AnalyzedResultT", contravariant=True)
ReportT = TypeVar("ReportT", covariant=True)


@dataclass(frozen=True, slots=True)
class PreparedSimulation(Generic[ProgramT, ParamsT, StateT, InputsT, AnalyzerT]):
    """Explicit output of solver preparation.

    The container cannot be mutated. Numerical PyTrees held by it are governed by the
    same no-in-place-mutation rule as JAX inputs and can be replaced with
    ``dataclasses.replace`` when constructing a modified simulation.
    """

    program: ProgramT
    params: ParamsT
    state: StateT
    inputs: InputsT
    manifest: RunManifest
    analyzer: AnalyzerT
    capabilities: SolverCapabilities


@runtime_checkable
class JaxProgram(Protocol[ProgramParamsT, ProgramStateT, ProgramInputsT, KeyT, RawResultT]):
    """Pure numerical call boundary; concrete implementations are JAX PyTrees."""

    def __call__(
        self, params: ProgramParamsT, state: ProgramStateT, inputs: ProgramInputsT, key: KeyT
    ) -> RawResultT: ...


@runtime_checkable
class ContinuousSystem(Protocol):
    """A true continuous system returning a state time derivative."""

    def rhs(self, t: Any, state: Any, params: Any, inputs: Any) -> Any: ...


@runtime_checkable
class DiscreteSystem(Protocol):
    """A discrete system returning the complete state for the next step."""

    def step(self, step: Any, state: Any, params: Any, inputs: Any, key: Any) -> Any: ...


@runtime_checkable
class Analyzer(Protocol[AnalyzedResultT, ReportT]):
    """Host-side conversion of a numerical result into metrics and artifacts."""

    def analyze(self, result: AnalyzedResultT, manifest: RunManifest) -> ReportT: ...


@runtime_checkable
class SolverBuilder(Protocol[KeyT]):
    """Logging-free factory for an explicit prepared simulation."""

    def prepare(self, spec: SimulationSpec, *, key: KeyT) -> PreparedSimulation[Any, Any, Any, Any, Any]: ...
