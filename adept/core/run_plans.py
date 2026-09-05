"""Serializable run intent that is safe to inspect before importing JAX."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .contracts import Placement, Precision, SimulationSpec
from .tracking import FailurePolicy, RunRequest

_STABLE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_RUN_PLAN_SCHEMA_VERSION = "2"
_SENSITIVE_FIELD_STEMS = frozenset(
    {
        "accesskey",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "secretkey",
        "token",
    }
)
_SENSITIVE_FIELD_SUFFIXES = ("", "id", "value")


def _compact_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _looks_like_credential_field(value: str) -> bool:
    compact = _compact_field_name(value)
    return any(
        compact.endswith(f"{stem}{suffix}") for stem in _SENSITIVE_FIELD_STEMS for suffix in _SENSITIVE_FIELD_SUFFIXES
    )


def _validate_no_embedded_secrets(value: Any, *, path: str = "plan") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if _looks_like_credential_field(key):
                raise ValueError(
                    f"{path}.{key} looks like embedded credential material; "
                    "store credentials outside the RunPlan and reference a configured profile instead"
                )
            _validate_no_embedded_secrets(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_no_embedded_secrets(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        sensitive_query_fields = [
            key for key, _ in parse_qsl(parsed.query, keep_blank_values=True) if _looks_like_credential_field(key)
        ]
        if parsed.username is not None or parsed.password is not None or sensitive_query_fields:
            raise ValueError(f"{path} contains credential material in a URI; store credentials outside the RunPlan")


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    copied = deepcopy(dict(value))
    _validate_no_embedded_secrets(copied, path=name)
    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only finite JSON values") from exc
    return copied


class AcceleratorKind(StrEnum):
    """Accelerator platform requested by a run plan."""

    ANY = "any"
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


class ExecutionFeature(StrEnum):
    """Optional behavior negotiated between a plan, solver, and executor."""

    DISTRIBUTED_JAX = "distributed-jax"
    SHARED_DURABLE_STORAGE = "shared-durable-storage"
    RANK_ZERO_IO = "rank-zero-io"
    CHECKPOINTING = "checkpointing"
    DIFFERENTIABLE = "differentiable"
    BATCHING = "batching"
    ARTIFACT_ACCESS = "artifact-access"


@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    """Typed topology and runtime capabilities required by one execution."""

    placement: Placement = Placement.SINGLE_DEVICE
    precision: Precision = Precision.DEFAULT
    accelerator: AcceleratorKind = AcceleratorKind.ANY
    hosts: int = 1
    devices_per_host: int = 1
    features: frozenset[ExecutionFeature] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "placement", Placement(self.placement))
        object.__setattr__(self, "precision", Precision(self.precision))
        object.__setattr__(self, "accelerator", AcceleratorKind(self.accelerator))
        object.__setattr__(self, "features", frozenset(ExecutionFeature(value) for value in self.features))
        for name, value in (("hosts", self.hosts), ("devices_per_host", self.devices_per_host)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if self.placement is Placement.SINGLE_DEVICE and (self.hosts != 1 or self.devices_per_host != 1):
            raise ValueError("single-device placement requires exactly one host and one device")
        if self.placement is Placement.MULTI_DEVICE and (self.hosts != 1 or self.devices_per_host < 2):
            raise ValueError("multi-device placement requires one host and at least two devices")
        if self.placement is Placement.MULTI_HOST and self.hosts < 2:
            raise ValueError("multi-host placement requires at least two hosts")

    @property
    def required_features(self) -> frozenset[ExecutionFeature]:
        """Return explicit features plus those implied by the topology."""

        required = set(self.features)
        if self.placement is Placement.MULTI_HOST:
            required.add(ExecutionFeature.DISTRIBUTED_JAX)
        return frozenset(required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement": self.placement.value,
            "precision": self.precision.value,
            "accelerator": self.accelerator.value,
            "hosts": self.hosts,
            "devices_per_host": self.devices_per_host,
            "features": sorted(feature.value for feature in self.features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceRequirements:
        copied = dict(value)
        known = {"placement", "precision", "accelerator", "hosts", "devices_per_host", "features"}
        unknown = sorted(set(copied).difference(known))
        if unknown:
            raise ValueError(f"Serialized resource requirements contain unknown fields: {unknown!r}")
        return cls(**copied)


@dataclass(frozen=True, slots=True, init=False)
class ServiceReference:
    """Serializable reference to an executor-configured runtime service.

    The reference contains configuration such as a directory or service URI, never a
    live client or credential. Executors decide which reference kinds they support.
    """

    kind: str
    _config: dict[str, Any] = field(repr=False)

    def __init__(self, kind: str, config: Mapping[str, Any] | None = None) -> None:
        if not isinstance(kind, str):
            raise TypeError("service reference kind must be a string")
        kind = kind.strip()
        if _STABLE_NAME.fullmatch(kind) is None:
            raise ValueError("service reference kind must be a non-empty lowercase kebab-case name")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "_config", _json_mapping(config or {}, name=f"{kind} service config"))

    @property
    def config(self) -> Mapping[str, Any]:
        return MappingProxyType(deepcopy(self._config))

    def config_dict(self) -> dict[str, Any]:
        return deepcopy(self._config)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "config": self.config_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServiceReference:
        copied = dict(value)
        try:
            kind = copied.pop("kind")
        except KeyError as exc:
            raise ValueError("Serialized service reference is missing required 'kind' field") from exc
        config = copied.pop("config", {})
        if copied:
            raise ValueError(f"Serialized service reference contains unknown fields: {sorted(copied)!r}")
        if not isinstance(config, Mapping):
            raise TypeError("Serialized service reference config must be a mapping")
        return cls(str(kind), config)


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Serializable checkpoint cadence and restore intent.

    Executors that accept an enabled policy must implement it completely. The policy
    deliberately contains no live store, callbacks, or checkpoint state.
    """

    every_steps: int | None = None
    save_on_completion: bool = False
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.every_steps is not None and (
            isinstance(self.every_steps, bool) or not isinstance(self.every_steps, int) or self.every_steps < 1
        ):
            raise ValueError("checkpoint every_steps must be a positive integer or None")
        if not isinstance(self.save_on_completion, bool):
            raise TypeError("checkpoint save_on_completion must be a boolean")
        if self.resume_from is not None:
            if not isinstance(self.resume_from, str):
                raise TypeError("checkpoint resume_from must be a string or None")
            resume_from = self.resume_from.strip()
            if not resume_from:
                raise ValueError("checkpoint resume_from must be non-empty when provided")
            object.__setattr__(self, "resume_from", resume_from)

    @property
    def enabled(self) -> bool:
        return self.every_steps is not None or self.save_on_completion or self.resume_from is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "every_steps": self.every_steps,
            "save_on_completion": self.save_on_completion,
            "resume_from": self.resume_from,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointPolicy:
        copied = dict(value)
        known = {"every_steps", "save_on_completion", "resume_from"}
        unknown = sorted(set(copied).difference(known))
        if unknown:
            raise ValueError(f"Serialized checkpoint policy contains unknown fields: {unknown!r}")
        return cls(**copied)


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Complete serializable intent for one executor-owned solver run."""

    simulation: SimulationSpec
    seed: int = 0
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    run: RunRequest = field(default_factory=RunRequest)
    tracker: ServiceReference = field(default_factory=lambda: ServiceReference("null"))
    artifact_sink: ServiceReference = field(default_factory=lambda: ServiceReference("null"))
    checkpoint_store: ServiceReference = field(default_factory=lambda: ServiceReference("null"))
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    tracking_failure_policy: FailurePolicy = FailurePolicy.STRICT
    schema_version: str = _RUN_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationSpec):
            raise TypeError("simulation must be a SimulationSpec")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in the unsigned 32-bit range")
        if not isinstance(self.resources, ResourceRequirements):
            raise TypeError("resources must be ResourceRequirements")
        if not isinstance(self.run, RunRequest):
            raise TypeError("run must be RunRequest")
        if not isinstance(self.tracker, ServiceReference):
            raise TypeError("tracker must be ServiceReference")
        if not isinstance(self.artifact_sink, ServiceReference):
            raise TypeError("artifact_sink must be ServiceReference")
        if not isinstance(self.checkpoint_store, ServiceReference):
            raise TypeError("checkpoint_store must be ServiceReference")
        if not isinstance(self.checkpoint_policy, CheckpointPolicy):
            raise TypeError("checkpoint_policy must be CheckpointPolicy")
        if self.checkpoint_policy.enabled and self.checkpoint_store.kind == "null":
            raise ValueError("an enabled checkpoint policy requires a non-null checkpoint_store")
        if not self.checkpoint_policy.enabled and self.checkpoint_store.kind != "null":
            raise ValueError("a non-null checkpoint_store requires an enabled checkpoint policy")
        object.__setattr__(self, "tracking_failure_policy", FailurePolicy(self.tracking_failure_policy))
        schema_version = str(self.schema_version).strip()
        if not schema_version:
            raise ValueError("schema_version must be non-empty")
        if schema_version != _RUN_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RunPlan schema_version {schema_version!r}; "
                f"this ADEPT version supports {_RUN_PLAN_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "schema_version", schema_version)

        serialized = self.to_dict()
        _validate_no_embedded_secrets(serialized)
        try:
            json.dumps(serialized, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("RunPlan must contain only finite JSON values") from exc

    @property
    def required_features(self) -> frozenset[ExecutionFeature]:
        required = set(self.resources.required_features)
        if self.artifact_sink.kind != "null":
            required.add(ExecutionFeature.ARTIFACT_ACCESS)
        if self.checkpoint_policy.enabled:
            required.add(ExecutionFeature.CHECKPOINTING)
            if self.resources.placement is Placement.MULTI_HOST:
                required.update(
                    {
                        ExecutionFeature.RANK_ZERO_IO,
                        ExecutionFeature.SHARED_DURABLE_STORAGE,
                    }
                )
        return frozenset(required)

    @property
    def fingerprint(self) -> str:
        """Stable digest of the canonical serialized plan."""

        payload = self.to_json().encode()
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "simulation": self.simulation.to_dict(),
            "seed": self.seed,
            "resources": self.resources.to_dict(),
            "run": self.run.to_dict(),
            "tracker": self.tracker.to_dict(),
            "artifact_sink": self.artifact_sink.to_dict(),
            "checkpoint_store": self.checkpoint_store.to_dict(),
            "checkpoint_policy": self.checkpoint_policy.to_dict(),
            "tracking_failure_policy": self.tracking_failure_policy.value,
        }

    def to_json(self) -> str:
        """Serialize using a stable canonical JSON representation."""

        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunPlan:
        copied = dict(value)
        try:
            simulation = copied.pop("simulation")
        except KeyError as exc:
            raise ValueError("Serialized RunPlan is missing required 'simulation' field") from exc
        resources = copied.pop("resources", {})
        run = copied.pop("run", {})
        tracker = copied.pop("tracker", {"kind": "null"})
        artifact_sink = copied.pop("artifact_sink", {"kind": "null"})
        checkpoint_store = copied.pop("checkpoint_store", {"kind": "null"})
        checkpoint_policy = copied.pop("checkpoint_policy", {})
        seed = copied.pop("seed", 0)
        tracking_failure_policy = copied.pop("tracking_failure_policy", FailurePolicy.STRICT)
        schema_version = copied.pop("schema_version", "1")
        if str(schema_version) == "1":
            schema_version = _RUN_PLAN_SCHEMA_VERSION
        if copied:
            raise ValueError(f"Serialized RunPlan contains unknown fields: {sorted(copied)!r}")
        for name, item in (
            ("simulation", simulation),
            ("resources", resources),
            ("run", run),
            ("tracker", tracker),
            ("artifact_sink", artifact_sink),
            ("checkpoint_store", checkpoint_store),
            ("checkpoint_policy", checkpoint_policy),
        ):
            if not isinstance(item, Mapping):
                raise TypeError(f"Serialized RunPlan {name} must be a mapping")
        return cls(
            simulation=SimulationSpec.from_dict(simulation),
            seed=seed,
            resources=ResourceRequirements.from_dict(resources),
            run=RunRequest.from_dict(run),
            tracker=ServiceReference.from_dict(tracker),
            artifact_sink=ServiceReference.from_dict(artifact_sink),
            checkpoint_store=ServiceReference.from_dict(checkpoint_store),
            checkpoint_policy=CheckpointPolicy.from_dict(checkpoint_policy),
            tracking_failure_policy=tracking_failure_policy,
            schema_version=str(schema_version),
        )

    @classmethod
    def from_json(cls, value: str) -> RunPlan:
        """Restore a plan from JSON without importing numerical dependencies."""

        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("Serialized RunPlan JSON must contain an object")
        return cls.from_dict(decoded)


__all__ = [
    "AcceleratorKind",
    "CheckpointPolicy",
    "ExecutionFeature",
    "ResourceRequirements",
    "RunPlan",
    "ServiceReference",
]
