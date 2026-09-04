"""Pluggable execution contracts and the initial local executor adapter."""

from __future__ import annotations

import importlib
import os
import re
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock, RLock
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from .contracts import ExecutionKind, Placement, Precision, PreparedSimulation, SolverCapabilities
from .registry import SolverRegistry, solver_registry
from .run_plans import AcceleratorKind, ExecutionFeature, RunPlan, ServiceReference
from .runtime import HostRunResult, run_prepared
from .tracking import (
    ArtifactSink,
    DirectoryArtifactSink,
    NullArtifactSink,
    NullTracker,
    Tracker,
)

_STABLE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_JAX_BOOTSTRAP_LOCK = Lock()


class CapabilityMismatchError(ValueError):
    """Raised when a run plan cannot execute on a selected executor or solver."""


class ExecutionState(StrEnum):
    """Executor-owned lifecycle state, independent of tracker status."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    """Serializable identity returned by an executor submission."""

    execution_id: str
    executor: str

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or not isinstance(self.executor, str):
            raise TypeError("execution_id and executor must be strings")
        execution_id = self.execution_id.strip()
        executor = self.executor.strip()
        if not execution_id:
            raise ValueError("execution_id must be non-empty")
        if _STABLE_NAME.fullmatch(executor) is None:
            raise ValueError("executor must be a non-empty lowercase kebab-case name")
        object.__setattr__(self, "execution_id", execution_id)
        object.__setattr__(self, "executor", executor)

    def to_dict(self) -> dict[str, str]:
        return {"execution_id": self.execution_id, "executor": self.executor}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionHandle:
        copied = dict(value)
        try:
            execution_id = copied.pop("execution_id")
            executor = copied.pop("executor")
        except KeyError as exc:
            raise ValueError(f"Serialized execution handle is missing required {exc.args[0]!r} field") from exc
        if copied:
            raise ValueError(f"Serialized execution handle contains unknown fields: {sorted(copied)!r}")
        return cls(execution_id=str(execution_id), executor=str(executor))


@dataclass(frozen=True, slots=True)
class ExecutorCapabilities:
    """Resources and services an executor can provide after preflight."""

    placements: frozenset[Placement]
    precisions: frozenset[Precision]
    accelerators: frozenset[AcceleratorKind]
    features: frozenset[ExecutionFeature] = field(default_factory=frozenset)
    tracker_kinds: frozenset[str] = field(default_factory=lambda: frozenset({"null"}))
    artifact_sink_kinds: frozenset[str] = field(default_factory=lambda: frozenset({"null"}))
    max_hosts: int | None = None
    max_devices_per_host: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", frozenset(Placement(value) for value in self.placements))
        object.__setattr__(self, "precisions", frozenset(Precision(value) for value in self.precisions))
        object.__setattr__(self, "accelerators", frozenset(AcceleratorKind(value) for value in self.accelerators))
        object.__setattr__(self, "features", frozenset(ExecutionFeature(value) for value in self.features))
        if any(not isinstance(kind, str) for kind in self.tracker_kinds):
            raise TypeError("tracker_kinds must contain strings")
        if any(not isinstance(kind, str) for kind in self.artifact_sink_kinds):
            raise TypeError("artifact_sink_kinds must contain strings")
        object.__setattr__(self, "tracker_kinds", frozenset(self.tracker_kinds))
        object.__setattr__(self, "artifact_sink_kinds", frozenset(self.artifact_sink_kinds))
        if not self.placements:
            raise ValueError("executor capabilities require at least one placement")
        if not self.precisions:
            raise ValueError("executor capabilities require at least one precision")
        if not self.accelerators:
            raise ValueError("executor capabilities require at least one accelerator")
        for kinds_name, kinds in (
            ("tracker_kinds", self.tracker_kinds),
            ("artifact_sink_kinds", self.artifact_sink_kinds),
        ):
            if not kinds or any(_STABLE_NAME.fullmatch(kind) is None for kind in kinds):
                raise ValueError(f"{kinds_name} must contain stable lowercase kebab-case names")
        for name, value in (
            ("max_hosts", self.max_hosts),
            ("max_devices_per_host", self.max_devices_per_host),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "placements": sorted(value.value for value in self.placements),
            "precisions": sorted(value.value for value in self.precisions),
            "accelerators": sorted(value.value for value in self.accelerators),
            "features": sorted(value.value for value in self.features),
            "tracker_kinds": sorted(self.tracker_kinds),
            "artifact_sink_kinds": sorted(self.artifact_sink_kinds),
            "max_hosts": self.max_hosts,
            "max_devices_per_host": self.max_devices_per_host,
        }


@runtime_checkable
class Executor(Protocol):
    """Execution backend with preflight, submission, status, cancellation, and retrieval."""

    name: str
    capabilities: ExecutorCapabilities

    def validate(self, plan: RunPlan) -> None: ...

    def execute(self, plan: RunPlan) -> HostRunResult: ...

    def submit(self, plan: RunPlan) -> ExecutionHandle: ...

    def status(self, handle: ExecutionHandle) -> ExecutionState: ...

    def cancel(self, handle: ExecutionHandle) -> bool: ...

    def result(self, handle: ExecutionHandle, *, timeout: float | None = None) -> HostRunResult: ...


def _unexpected_config(config: Mapping[str, Any], allowed: set[str], *, service: str) -> None:
    unknown = sorted(set(config).difference(allowed))
    if unknown:
        raise CapabilityMismatchError(f"{service} reference contains unsupported settings: {unknown!r}")


def _validate_service_reference(reference: ServiceReference, *, role: str) -> None:
    config = reference.config
    if reference.kind == "null":
        _unexpected_config(config, set(), service=f"null {role}")
        return
    if role == "artifact sink" and reference.kind == "directory":
        _unexpected_config(config, {"root"}, service="directory artifact sink")
        root = config.get("root")
        if not isinstance(root, str) or not root.strip():
            raise CapabilityMismatchError("directory artifact sink requires a non-empty string 'root'")
        return
    if reference.kind == "mlflow":
        allowed = {"tracking_uri", "registry_uri", "rest_api_path_prefix"}
        if role == "tracker":
            allowed.add("experiment_create_retries")
        _unexpected_config(config, allowed, service=f"MLflow {role}")
        for name in allowed.difference({"experiment_create_retries"}):
            value = config.get(name)
            if value is not None and not isinstance(value, str):
                raise CapabilityMismatchError(f"MLflow {role} setting {name!r} must be a string or null")
        retries = config.get("experiment_create_retries")
        if retries is not None and (isinstance(retries, bool) or not isinstance(retries, int) or retries < 1):
            raise CapabilityMismatchError("MLflow tracker experiment_create_retries must be a positive integer")


def _resolve_tracker(reference: ServiceReference) -> Tracker:
    config = reference.config_dict()
    if reference.kind == "null":
        return NullTracker()
    if reference.kind == "mlflow":
        module = importlib.import_module("adept.core.tracking_mlflow")
        return module.MLflowTracker(**config)
    raise CapabilityMismatchError(f"no local tracker adapter is registered for {reference.kind!r}")


def _resolve_artifact_sink(reference: ServiceReference) -> ArtifactSink:
    config = reference.config_dict()
    if reference.kind == "null":
        return NullArtifactSink()
    if reference.kind == "directory":
        return DirectoryArtifactSink(config["root"])
    if reference.kind == "mlflow":
        module = importlib.import_module("adept.core.tracking_mlflow")
        return module.MLflowArtifactSink(**config)
    raise CapabilityMismatchError(f"no local artifact-sink adapter is registered for {reference.kind!r}")


class LocalExecutor:
    """Thread-backed local execution with pre-JAX bootstrap and typed preflight."""

    name = "local"

    def __init__(
        self,
        *,
        registry: SolverRegistry | None = None,
        capabilities: ExecutorCapabilities | None = None,
        max_workers: int = 1,
        execute_prepared: Callable[[PreparedSimulation, Any], Any] | None = None,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self.registry = registry if registry is not None else solver_registry
        self.capabilities = capabilities or ExecutorCapabilities(
            placements=frozenset({Placement.SINGLE_DEVICE}),
            precisions=frozenset({Precision.DEFAULT, Precision.X64}),
            accelerators=frozenset({AcceleratorKind.CPU, AcceleratorKind.GPU, AcceleratorKind.TPU}),
            features=frozenset(
                {
                    ExecutionFeature.RANK_ZERO_IO,
                    ExecutionFeature.DIFFERENTIABLE,
                    ExecutionFeature.BATCHING,
                    ExecutionFeature.ARTIFACT_ACCESS,
                }
            ),
            tracker_kinds=frozenset({"null", "mlflow"}),
            artifact_sink_kinds=frozenset({"null", "directory", "mlflow"}),
            max_hosts=1,
        )
        self._execute_prepared = execute_prepared
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="adept-local")
        self._futures: dict[str, Future[HostRunResult]] = {}
        self._lock = RLock()

    def _compatibility_errors(
        self,
        plan: RunPlan,
        solver_capabilities: SolverCapabilities | None,
    ) -> list[str]:
        resources = plan.resources
        executor = self.capabilities
        errors = []
        if resources.placement not in executor.placements:
            errors.append(f"placement {resources.placement.value!r} is not supported")
        if resources.precision not in executor.precisions:
            errors.append(f"precision {resources.precision.value!r} is not supported")
        if (
            resources.accelerator is not AcceleratorKind.ANY
            and resources.accelerator not in executor.accelerators
        ):
            errors.append(f"accelerator {resources.accelerator.value!r} is not supported")
        if executor.max_hosts is not None and resources.hosts > executor.max_hosts:
            errors.append(f"requested {resources.hosts} hosts but the executor supports at most {executor.max_hosts}")
        if (
            executor.max_devices_per_host is not None
            and resources.devices_per_host > executor.max_devices_per_host
        ):
            errors.append(
                f"requested {resources.devices_per_host} devices per host but the executor supports at most "
                f"{executor.max_devices_per_host}"
            )
        missing_features = plan.required_features.difference(executor.features)
        if missing_features:
            errors.append(
                "missing executor features: " + ", ".join(sorted(feature.value for feature in missing_features))
            )
        if plan.tracker.kind not in executor.tracker_kinds:
            errors.append(f"tracker kind {plan.tracker.kind!r} is not supported")
        if plan.artifact_sink.kind not in executor.artifact_sink_kinds:
            errors.append(f"artifact sink kind {plan.artifact_sink.kind!r} is not supported")
        if plan.artifact_sink.kind == "mlflow" and plan.tracker.kind != "mlflow":
            errors.append("an MLflow artifact sink requires an MLflow tracker handle")

        if solver_capabilities is not None:
            if resources.placement not in solver_capabilities.placements:
                errors.append(f"solver does not support placement {resources.placement.value!r}")
            if solver_capabilities.precision is Precision.X64 and Precision.X64 not in executor.precisions:
                errors.append("solver requires x64 but the executor does not support it")
            if (
                ExecutionFeature.DIFFERENTIABLE in plan.required_features
                and not solver_capabilities.differentiable
            ):
                errors.append("solver is not differentiable")
            if ExecutionFeature.BATCHING in plan.required_features and not solver_capabilities.batchable:
                errors.append("solver is not batchable")
            external_solver = solver_capabilities.execution_kind is ExecutionKind.EXTERNAL
            external_placement = resources.placement is Placement.EXTERNAL
            if external_solver != external_placement:
                errors.append("external solvers and external placement must be selected together")
        return errors

    def validate(self, plan: RunPlan) -> None:
        """Validate serializable intent without importing JAX or loading a builder."""

        if not isinstance(plan, RunPlan):
            raise TypeError("plan must be a RunPlan")
        declared = self.registry.capabilities(plan.simulation.solver)
        errors = self._compatibility_errors(plan, declared)
        try:
            _validate_service_reference(plan.tracker, role="tracker")
        except CapabilityMismatchError as error:
            errors.append(str(error))
        try:
            _validate_service_reference(plan.artifact_sink, role="artifact sink")
        except CapabilityMismatchError as error:
            errors.append(str(error))
        if errors:
            details = "\n".join(f"- {error}" for error in errors)
            raise CapabilityMismatchError(f"RunPlan is incompatible with local execution:\n{details}")

    def _required_precision(self, plan: RunPlan) -> Precision:
        declared = self.registry.capabilities(plan.simulation.solver)
        if plan.resources.precision is Precision.X64 or (
            declared is not None and declared.precision is Precision.X64
        ):
            return Precision.X64
        return Precision.DEFAULT

    def _bootstrap_jax(self, precision: Precision) -> Any:
        with _JAX_BOOTSTRAP_LOCK:
            if precision is Precision.X64 and "jax" not in sys.modules:
                os.environ["JAX_ENABLE_X64"] = "true"
            jax = importlib.import_module("jax")
            if precision is Precision.X64 and not bool(jax.config.read("jax_enable_x64")):
                raise CapabilityMismatchError(
                    "x64 is required, but JAX was already initialized with x64 disabled; "
                    "execute this RunPlan in a fresh worker or enable JAX x64 before importing JAX"
                )
            return jax

    @staticmethod
    def _validate_actual_devices(plan: RunPlan, jax: Any) -> None:
        resources = plan.resources
        devices = tuple(jax.local_devices())
        if resources.accelerator is AcceleratorKind.ANY:
            matching = devices
        else:
            matching = tuple(
                device for device in devices if str(device.platform).lower() == resources.accelerator.value
            )
        if len(matching) < resources.devices_per_host:
            platform = resources.accelerator.value
            raise CapabilityMismatchError(
                f"RunPlan requests {resources.devices_per_host} {platform} device(s) per host, "
                f"but JAX discovered {len(matching)}"
            )
        if int(jax.process_count()) != resources.hosts:
            raise CapabilityMismatchError(
                f"RunPlan requests {resources.hosts} host process(es), but JAX initialized "
                f"{jax.process_count()}"
            )

    def _execute_plan(self, plan: RunPlan) -> HostRunResult:
        precision = self._required_precision(plan)
        jax = self._bootstrap_jax(precision)
        self._validate_actual_devices(plan, jax)
        key = jax.random.key(plan.seed)
        prepared = self.registry.prepare(plan.simulation, key=key)
        errors = self._compatibility_errors(plan, prepared.capabilities)
        declared = self.registry.capabilities(plan.simulation.solver)
        if declared is not None and prepared.capabilities != declared:
            errors.append("prepared solver capabilities differ from its import-light registry declaration")
        if prepared.capabilities.precision is Precision.X64 and precision is not Precision.X64:
            errors.append("solver requires x64 but did not declare that requirement before JAX bootstrap")
        if errors:
            details = "\n".join(f"- {error}" for error in errors)
            raise CapabilityMismatchError(f"Prepared solver is incompatible with the RunPlan:\n{details}")

        tracker = _resolve_tracker(plan.tracker)
        artifact_sink = _resolve_artifact_sink(plan.artifact_sink)
        return run_prepared(
            prepared,
            key=key,
            request=plan.run,
            tracker=tracker,
            artifact_sink=artifact_sink,
            tracking_failure_policy=plan.tracking_failure_policy,
            execute=self._execute_prepared,
        )

    def submit(self, plan: RunPlan) -> ExecutionHandle:
        """Validate and enqueue a defensive wire-format copy of one plan."""

        self.validate(plan)
        wire_plan = RunPlan.from_json(plan.to_json())
        handle = ExecutionHandle(uuid4().hex, self.name)
        future = self._pool.submit(self._execute_plan, wire_plan)
        with self._lock:
            self._futures[handle.execution_id] = future
        return handle

    def execute(self, plan: RunPlan) -> HostRunResult:
        """Submit one local run and wait for its result."""

        return self.result(self.submit(plan))

    def _future(self, handle: ExecutionHandle) -> Future[HostRunResult]:
        if handle.executor != self.name:
            raise ValueError(f"execution handle belongs to {handle.executor!r}, not {self.name!r}")
        with self._lock:
            try:
                return self._futures[handle.execution_id]
            except KeyError as exc:
                raise LookupError(f"unknown local execution {handle.execution_id!r}") from exc

    def status(self, handle: ExecutionHandle) -> ExecutionState:
        future = self._future(handle)
        if future.cancelled():
            return ExecutionState.CANCELLED
        if future.running():
            return ExecutionState.RUNNING
        if not future.done():
            return ExecutionState.PENDING
        return ExecutionState.FAILED if future.exception() is not None else ExecutionState.SUCCEEDED

    def cancel(self, handle: ExecutionHandle) -> bool:
        """Cancel an execution that has not begun; running JAX calls are not interruptible."""

        return self._future(handle).cancel()

    def result(self, handle: ExecutionHandle, *, timeout: float | None = None) -> HostRunResult:
        return self._future(handle).result(timeout=timeout)

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> LocalExecutor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.shutdown()


__all__ = [
    "CapabilityMismatchError",
    "ExecutionHandle",
    "ExecutionState",
    "Executor",
    "ExecutorCapabilities",
    "LocalExecutor",
]
