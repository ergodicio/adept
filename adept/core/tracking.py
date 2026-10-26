"""Host-side tracking, reporting, and artifact storage services.

This module deliberately depends only on the Python standard library.  Numerical
programs and submitters can import the public contracts without importing JAX or
MLflow; the MLflow-backed implementations live in :mod:`adept.core.tracking_mlflow`
and load their dependency only when used.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4


class RunStatus(StrEnum):
    """Terminal state reported by a host-side tracker."""

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class FailurePolicy(StrEnum):
    """Whether telemetry failures fail an otherwise successful computation."""

    STRICT = "strict"
    BEST_EFFORT = "best-effort"


@dataclass(frozen=True, slots=True)
class RunHandle:
    """Explicit tracker identity passed to every tracking and artifact operation."""

    run_id: str
    backend: str
    parent_run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not isinstance(self.backend, str):
            raise TypeError("run_id and backend must be strings")
        if self.parent_run_id is not None and not isinstance(self.parent_run_id, str):
            raise TypeError("parent_run_id must be a string or None")
        run_id = self.run_id.strip()
        backend = self.backend.strip()
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not backend:
            raise ValueError("backend must be non-empty")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "backend", backend)

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-friendly tracker handle."""

        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "parent_run_id": self.parent_run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunHandle:
        """Restore a tracker handle from :meth:`to_dict` output."""

        copied = dict(value)
        try:
            run_id = copied.pop("run_id")
            backend = copied.pop("backend")
        except KeyError as exc:
            raise ValueError(f"Serialized run handle is missing required {exc.args[0]!r} field") from exc
        parent_run_id = copied.pop("parent_run_id", None)
        if copied:
            raise ValueError(f"Serialized run handle contains unknown fields: {sorted(copied)!r}")
        return cls(run_id=str(run_id), backend=str(backend), parent_run_id=parent_run_id)


@dataclass(frozen=True, slots=True, init=False)
class RunRequest:
    """Serializable intent for starting or resuming a tracked run."""

    experiment: str
    name: str | None
    run_id: str | None
    parent: RunHandle | None
    _tags: dict[str, str] = field(repr=False)

    def __init__(
        self,
        *,
        experiment: str = "adept",
        name: str | None = None,
        run_id: str | None = None,
        parent: RunHandle | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(experiment, str):
            raise TypeError("experiment must be a string")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string or None")
        if run_id is not None and not isinstance(run_id, str):
            raise TypeError("run_id must be a string or None")
        if parent is not None and not isinstance(parent, RunHandle):
            raise TypeError("parent must be a RunHandle or None")
        experiment = experiment.strip()
        name = name.strip() if name is not None else None
        run_id = run_id.strip() if run_id is not None else None
        if not experiment:
            raise ValueError("experiment must be non-empty")
        if name == "":
            raise ValueError("name must be non-empty when provided")
        if run_id == "":
            raise ValueError("run_id must be non-empty when provided")

        copied_tags = deepcopy(dict(tags or {}))
        invalid_tags = [
            key for key, value in copied_tags.items() if not isinstance(key, str) or not isinstance(value, str)
        ]
        if invalid_tags:
            raise TypeError("run tags must map strings to strings")

        object.__setattr__(self, "experiment", experiment)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "_tags", copied_tags)

    @property
    def tags(self) -> Mapping[str, str]:
        return MappingProxyType(deepcopy(self._tags))

    def with_tags(self, tags: Mapping[str, str]) -> RunRequest:
        """Return a copy with additional or replaced tags."""

        combined = dict(self._tags)
        combined.update(tags)
        return RunRequest(
            experiment=self.experiment,
            name=self.name,
            run_id=self.run_id,
            parent=self.parent,
            tags=combined,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly tracked-run request."""

        return {
            "experiment": self.experiment,
            "name": self.name,
            "run_id": self.run_id,
            "parent": self.parent.to_dict() if self.parent is not None else None,
            "tags": dict(self._tags),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunRequest:
        """Restore a tracked-run request from :meth:`to_dict` output."""

        copied = dict(value)
        experiment = copied.pop("experiment", "adept")
        name = copied.pop("name", None)
        run_id = copied.pop("run_id", None)
        parent = copied.pop("parent", None)
        tags = copied.pop("tags", None)
        if copied:
            raise ValueError(f"Serialized run request contains unknown fields: {sorted(copied)!r}")
        if parent is not None and not isinstance(parent, Mapping):
            raise TypeError("Serialized run request parent must be a mapping or null")
        if tags is not None and not isinstance(tags, Mapping):
            raise TypeError("Serialized run request tags must be a mapping or null")
        return cls(
            experiment=str(experiment),
            name=name,
            run_id=run_id,
            parent=RunHandle.from_dict(parent) if parent is not None else None,
            tags=tags,
        )


@dataclass(frozen=True, slots=True, init=False)
class MetricEvent:
    """One timestamped, stepped batch of scalar metrics."""

    step: int
    timestamp_ms: int | None
    _values: dict[str, float] = field(repr=False)

    def __init__(
        self,
        values: Mapping[str, float],
        *,
        step: int = 0,
        timestamp_ms: int | None = None,
    ) -> None:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("metric step must be an integer")
        if timestamp_ms is not None and (isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int)):
            raise TypeError("metric timestamp_ms must be an integer when provided")
        if not values:
            raise ValueError("a metric event must contain at least one value")

        copied: dict[str, float] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError("metric names must be non-empty strings")
            if isinstance(value, bool):
                raise TypeError(f"metric {key!r} must be numeric, not bool")
            try:
                copied[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"metric {key!r} must be scalar and numeric") from exc

        object.__setattr__(self, "step", step)
        object.__setattr__(self, "timestamp_ms", timestamp_ms)
        object.__setattr__(self, "_values", copied)

    @property
    def values(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._values))


def _artifact_parent(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if value != path.as_posix() or path.is_absolute() or ".." in path.parts or "\\" in value or windows_path.drive:
        raise ValueError("artifact_path must be a normalized relative POSIX path")
    return str(path)


@dataclass(frozen=True, slots=True)
class Artifact:
    """A local file or directory to place below an artifact-store prefix."""

    source: str | os.PathLike[str]
    artifact_path: str | None = None

    def __post_init__(self) -> None:
        source = os.fspath(self.source)
        if not source:
            raise ValueError("artifact source must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "artifact_path", _artifact_parent(self.artifact_path))


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Backend receipt used to verify a completed artifact upload."""

    path: str
    uri: str
    is_directory: bool
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Report:
    """Host-side analysis output with explicit metrics and artifact requests."""

    result: Any = None
    metrics: tuple[MetricEvent, ...] = ()
    artifacts: tuple[Artifact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if not all(isinstance(event, MetricEvent) for event in self.metrics):
            raise TypeError("Report.metrics must contain MetricEvent values")
        if not all(isinstance(artifact, Artifact) for artifact in self.artifacts):
            raise TypeError("Report.artifacts must contain Artifact values")


@runtime_checkable
class Tracker(Protocol):
    """Host-side run tracker with no process-global active-run state."""

    def preflight(self, request: RunRequest) -> None: ...

    def start(self, request: RunRequest) -> RunHandle: ...

    def log_metrics(self, handle: RunHandle, events: Sequence[MetricEvent]) -> None: ...

    def finish(self, handle: RunHandle, status: RunStatus, *, error: str | None = None) -> None: ...


@runtime_checkable
class ArtifactSink(Protocol):
    """Host-side durable artifact destination."""

    def preflight(self) -> None: ...

    def validate(self, handle: RunHandle) -> None: ...

    def put(self, handle: RunHandle, artifact: Artifact) -> ArtifactReceipt: ...

    def verify(self, handle: RunHandle, receipt: ArtifactReceipt) -> None: ...


class NullTracker:
    """No-op tracker for untracked runs and tests."""

    def preflight(self, request: RunRequest) -> None:
        del request

    def start(self, request: RunRequest) -> RunHandle:
        run_id = request.run_id or uuid4().hex
        parent_run_id = request.parent.run_id if request.parent is not None else None
        return RunHandle(run_id=run_id, backend="null", parent_run_id=parent_run_id)

    def log_metrics(self, handle: RunHandle, events: Sequence[MetricEvent]) -> None:
        del handle, events

    def finish(self, handle: RunHandle, status: RunStatus, *, error: str | None = None) -> None:
        del handle, status, error


class NullArtifactSink:
    """No-op artifact sink for callers that want reports but no persistence."""

    def preflight(self) -> None:
        pass

    def validate(self, handle: RunHandle) -> None:
        del handle

    def put(self, handle: RunHandle, artifact: Artifact) -> ArtifactReceipt:
        name = Path(artifact.source).name
        path = str(PurePosixPath(artifact.artifact_path or "") / name)
        return ArtifactReceipt(
            path=path,
            uri=f"null://{handle.run_id}/{path}",
            is_directory=Path(artifact.source).is_dir(),
        )

    def verify(self, handle: RunHandle, receipt: ArtifactReceipt) -> None:
        del handle, receipt


_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _validate_source(source: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"artifact source must not be a symbolic link: {source}")
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file() and not source.is_dir():
        raise ValueError(f"artifact source must be a regular file or directory: {source}")
    if source.is_dir():
        symlink = next((path for path in source.rglob("*") if path.is_symlink()), None)
        if symlink is not None:
            raise ValueError(f"artifact directories must not contain symbolic links: {symlink}")


def _hash_path(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        if path.is_dir():
            relative = item.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    return size, digest.hexdigest()


class DirectoryArtifactSink:
    """Verified local artifact storage rooted below one directory per run."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()

    def preflight(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        probe = self.root / f".adept-write-test-{uuid4().hex}"
        try:
            probe.write_bytes(b"")
        finally:
            probe.unlink(missing_ok=True)

    def _run_root(self, handle: RunHandle) -> Path:
        if _SAFE_RUN_ID.fullmatch(handle.run_id) is None:
            raise ValueError("directory artifact sinks require a filesystem-safe run_id")
        return self.root / handle.run_id

    def validate(self, handle: RunHandle) -> None:
        self._run_root(handle)

    def _destination(self, handle: RunHandle, artifact: Artifact) -> Path:
        run_root = self._run_root(handle).resolve()
        if not run_root.is_relative_to(self.root):
            raise ValueError("artifact run directory resolves outside the configured root")

        relative = PurePosixPath(artifact.artifact_path or "") / Path(artifact.source).name
        _artifact_parent(relative.as_posix())
        destination = run_root.joinpath(*relative.parts).resolve()
        if not destination.is_relative_to(run_root):
            raise ValueError("artifact destination resolves outside its run directory")
        return destination

    def put(self, handle: RunHandle, artifact: Artifact) -> ArtifactReceipt:
        source = Path(artifact.source).expanduser()
        _validate_source(source)
        destination = self._destination(handle, artifact)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"artifact destination already exists: {destination}")

        staging = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            if source.is_dir():
                shutil.copytree(source, staging)
            else:
                shutil.copy2(source, staging)
            os.replace(staging, destination)
        except Exception:
            if staging.is_dir():
                shutil.rmtree(staging)
            else:
                staging.unlink(missing_ok=True)
            raise

        size, digest = _hash_path(destination)
        relative = destination.relative_to(self._run_root(handle)).as_posix()
        return ArtifactReceipt(
            path=relative,
            uri=destination.as_uri(),
            is_directory=destination.is_dir(),
            size_bytes=size,
            sha256=digest,
        )

    def verify(self, handle: RunHandle, receipt: ArtifactReceipt) -> None:
        try:
            normalized = _artifact_parent(receipt.path)
        except ValueError as exc:
            raise ValueError("artifact receipt contains an unsafe path") from exc
        if normalized is None:
            raise ValueError("artifact receipt contains an unsafe path")
        relative = PurePosixPath(normalized)
        run_root = self._run_root(handle).resolve()
        if not run_root.is_relative_to(self.root):
            raise ValueError("artifact run directory resolves outside the configured root")
        destination = run_root.joinpath(*relative.parts)
        resolved_destination = destination.resolve()
        if not resolved_destination.is_relative_to(run_root):
            raise ValueError("artifact receipt resolves outside its run directory")
        destination = resolved_destination
        if not destination.exists():
            raise FileNotFoundError(f"artifact upload is missing: {destination}")
        if destination.is_dir() is not receipt.is_directory:
            raise OSError(f"artifact type changed after upload: {destination}")
        size, digest = _hash_path(destination)
        if receipt.size_bytes != size or receipt.sha256 != digest:
            raise OSError(f"artifact verification failed: {destination}")


__all__ = [
    "Artifact",
    "ArtifactReceipt",
    "ArtifactSink",
    "DirectoryArtifactSink",
    "FailurePolicy",
    "MetricEvent",
    "NullArtifactSink",
    "NullTracker",
    "Report",
    "RunHandle",
    "RunRequest",
    "RunStatus",
    "Tracker",
]
