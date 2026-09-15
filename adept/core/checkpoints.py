"""Versioned checkpoint contracts and an atomic local filesystem store."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .contracts import RunManifest

_CHECKPOINT_SCHEMA_VERSION = "1"
_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_METADATA_FILE = "metadata.json"
_STATE_FILE = "state.npz"
_COMMIT_MARKER = "COMMITTED"
_LATEST_FILE = "latest.json"
_LOCK_FILE = ".adept-checkpoint.lock"
_NUMPY_ENCODING = "numpy"
_RAW_BYTES_ENCODING = "raw-bytes-v1"
_EXTENDED_NUMERIC_DTYPE_SIZES = {
    "bfloat16": 2,
    "float4_e2m1fn": 1,
    "float6_e2m3fn": 1,
    "float6_e3m2fn": 1,
    "float8_e3m4": 1,
    "float8_e4m3": 1,
    "float8_e4m3b11fnuz": 1,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1,
    "int2": 1,
    "int4": 1,
    "uint2": 1,
    "uint4": 1,
}


class CheckpointError(RuntimeError):
    """Base class for checkpoint persistence and compatibility failures."""


class CheckpointPreflightError(CheckpointError):
    """Raised when a checkpoint store cannot safely persist data."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when committed checkpoint contents fail validation."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when checkpoint state cannot restore into the requested program."""


class UnsupportedCheckpointVersionError(CheckpointError):
    """Raised when no reader or migration exists for a metadata version."""


def _non_empty(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _checkpoint_id(value: str) -> str:
    normalized = _non_empty(value, name="checkpoint_id")
    if _CHECKPOINT_ID.fullmatch(normalized) is None:
        raise ValueError("checkpoint_id must contain only letters, numbers, '.', '_', and '-'")
    if normalized.lower() == _LATEST_FILE:
        raise ValueError(f"checkpoint_id {_LATEST_FILE!r} is reserved by the local store")
    return normalized


def _json_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    copied = deepcopy(dict(value or {}))
    if any(not isinstance(key, str) for key in copied):
        raise TypeError(f"{name} keys must be strings")
    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only finite JSON values") from exc
    return copied


def _json_fingerprint(value: Mapping[str, Any], *, name: str) -> str:
    copied = _json_mapping(value, name=name)
    encoded = json.dumps(copied, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointLeaf:
    """Stored schema and integrity information for one PyTree leaf."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    checksum: str
    logical_sharding: tuple[str, ...] | None = None
    encoding: str = _NUMPY_ENCODING

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _non_empty(self.path, name="checkpoint leaf path"))
        shape = tuple(self.shape)
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
            raise ValueError("checkpoint leaf shape entries must be non-negative integers")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _non_empty(self.dtype, name="checkpoint leaf dtype"))
        checksum = _non_empty(self.checksum, name="checkpoint leaf checksum")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", checksum) is None:
            raise ValueError("checkpoint leaf checksum must be a sha256 digest")
        object.__setattr__(self, "checksum", checksum)
        if self.logical_sharding is not None:
            sharding = tuple(str(axis) for axis in self.logical_sharding)
            object.__setattr__(self, "logical_sharding", sharding)
        encoding = _non_empty(self.encoding, name="checkpoint leaf encoding")
        expected_encoding = _RAW_BYTES_ENCODING if self.dtype in _EXTENDED_NUMERIC_DTYPE_SIZES else _NUMPY_ENCODING
        if encoding != expected_encoding:
            raise ValueError(f"checkpoint leaf dtype {self.dtype!r} requires {expected_encoding!r} encoding")
        object.__setattr__(self, "encoding", encoding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "checksum": self.checksum,
            "logical_sharding": list(self.logical_sharding) if self.logical_sharding is not None else None,
            "encoding": self.encoding,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointLeaf:
        copied = dict(value)
        known = {"path", "shape", "dtype", "checksum", "logical_sharding", "encoding"}
        unknown = sorted(set(copied).difference(known))
        if unknown:
            raise ValueError(f"Checkpoint leaf contains unknown fields: {unknown!r}")
        try:
            path = copied["path"]
            shape = copied["shape"]
            dtype = copied["dtype"]
            checksum = copied["checksum"]
        except KeyError as exc:
            raise ValueError(f"Checkpoint leaf is missing required {exc.args[0]!r} field") from exc
        logical_sharding = copied.get("logical_sharding")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise TypeError("Checkpoint leaf shape must be a sequence")
        if logical_sharding is not None and (
            not isinstance(logical_sharding, Sequence) or isinstance(logical_sharding, (str, bytes))
        ):
            raise TypeError("Checkpoint leaf logical_sharding must be a sequence or null")
        return cls(
            path=str(path),
            shape=tuple(shape),
            dtype=str(dtype),
            checksum=str(checksum),
            logical_sharding=None if logical_sharding is None else tuple(str(axis) for axis in logical_sharding),
            encoding=str(copied.get("encoding", _NUMPY_ENCODING)),
        )


@dataclass(frozen=True, slots=True, init=False)
class CheckpointMetadata:
    """Versioned execution identity, position, and state schema for a checkpoint."""

    checkpoint_id: str
    program: str
    program_version: str
    config_fingerprint: str
    provenance_fingerprint: str
    structural_fingerprint: str
    simulation_time: float
    step: int
    chunk_id: str | None
    leaves: tuple[CheckpointLeaf, ...]
    schema_version: str
    _code: dict[str, Any] = field(repr=False)
    _dependencies: dict[str, Any] = field(repr=False)

    def __init__(
        self,
        *,
        checkpoint_id: str,
        program: str,
        program_version: str,
        config_fingerprint: str,
        provenance_fingerprint: str,
        structural_fingerprint: str,
        simulation_time: float,
        step: int,
        chunk_id: str | None = None,
        leaves: Sequence[CheckpointLeaf] = (),
        code: Mapping[str, Any] | None = None,
        dependencies: Mapping[str, Any] | None = None,
        schema_version: str = _CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        schema_version = _non_empty(str(schema_version), name="checkpoint schema_version")
        if schema_version != _CHECKPOINT_SCHEMA_VERSION:
            raise UnsupportedCheckpointVersionError(
                f"unsupported checkpoint schema version {schema_version!r}; "
                f"this ADEPT version supports {_CHECKPOINT_SCHEMA_VERSION!r}"
            )
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        if isinstance(simulation_time, bool) or not isinstance(simulation_time, (int, float)):
            raise TypeError("checkpoint simulation_time must be a finite number")
        simulation_time = float(simulation_time)
        if not math.isfinite(simulation_time) or simulation_time < 0:
            raise ValueError("checkpoint simulation_time must be a finite non-negative number")
        if chunk_id is not None:
            chunk_id = _non_empty(chunk_id, name="checkpoint chunk_id")
        normalized_leaves = tuple(leaves)
        if any(not isinstance(leaf, CheckpointLeaf) for leaf in normalized_leaves):
            raise TypeError("checkpoint leaves must contain CheckpointLeaf values")
        paths = [leaf.path for leaf in normalized_leaves]
        if len(paths) != len(set(paths)):
            raise ValueError("checkpoint leaf paths must be unique")

        object.__setattr__(self, "checkpoint_id", _checkpoint_id(checkpoint_id))
        object.__setattr__(self, "program", _non_empty(program, name="checkpoint program"))
        object.__setattr__(self, "program_version", _non_empty(program_version, name="checkpoint program_version"))
        object.__setattr__(
            self,
            "config_fingerprint",
            _non_empty(config_fingerprint, name="checkpoint config_fingerprint"),
        )
        object.__setattr__(
            self,
            "provenance_fingerprint",
            _non_empty(provenance_fingerprint, name="checkpoint provenance_fingerprint"),
        )
        object.__setattr__(
            self,
            "structural_fingerprint",
            _non_empty(structural_fingerprint, name="checkpoint structural_fingerprint"),
        )
        object.__setattr__(self, "simulation_time", simulation_time)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "leaves", normalized_leaves)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "_code", _json_mapping(code, name="checkpoint code versions"))
        object.__setattr__(
            self,
            "_dependencies",
            _json_mapping(dependencies, name="checkpoint dependency versions"),
        )

    @property
    def code(self) -> Mapping[str, Any]:
        return MappingProxyType(deepcopy(self._code))

    @property
    def dependencies(self) -> Mapping[str, Any]:
        return MappingProxyType(deepcopy(self._dependencies))

    def with_leaves(self, leaves: Sequence[CheckpointLeaf]) -> CheckpointMetadata:
        payload = self.to_dict()
        payload["leaves"] = [leaf.to_dict() for leaf in leaves]
        return CheckpointMetadata.from_dict(payload)

    @classmethod
    def from_manifest(
        cls,
        manifest: RunManifest,
        *,
        checkpoint_id: str,
        program: str,
        program_version: str,
        simulation_time: float,
        step: int,
        chunk_id: str | None = None,
    ) -> CheckpointMetadata:
        """Create metadata from preparation provenance before state leaves are captured."""

        if not isinstance(manifest, RunManifest):
            raise TypeError("manifest must be RunManifest")
        provenance = {
            "raw_config": dict(manifest.raw_config),
            "resolved_config": dict(manifest.resolved_config),
            "units": dict(manifest.units),
            "seed": manifest.seed,
            "key_provenance": manifest.key_provenance,
        }
        return cls(
            checkpoint_id=checkpoint_id,
            program=program,
            program_version=program_version,
            config_fingerprint=_json_fingerprint(
                manifest.resolved_config,
                name="checkpoint resolved configuration",
            ),
            provenance_fingerprint=_json_fingerprint(
                provenance,
                name="checkpoint provenance",
            ),
            structural_fingerprint=manifest.structural_fingerprint,
            simulation_time=simulation_time,
            step=step,
            chunk_id=chunk_id,
            code=manifest.code,
            dependencies=manifest.dependencies,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "program": self.program,
            "program_version": self.program_version,
            "config_fingerprint": self.config_fingerprint,
            "provenance_fingerprint": self.provenance_fingerprint,
            "structural_fingerprint": self.structural_fingerprint,
            "simulation_time": self.simulation_time,
            "step": self.step,
            "chunk_id": self.chunk_id,
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "code": deepcopy(self._code),
            "dependencies": deepcopy(self._dependencies),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointMetadata:
        copied = dict(value)
        schema_version = str(copied.get("schema_version", ""))
        if schema_version != _CHECKPOINT_SCHEMA_VERSION:
            raise UnsupportedCheckpointVersionError(
                f"unsupported checkpoint schema version {schema_version!r}; "
                f"this ADEPT version supports {_CHECKPOINT_SCHEMA_VERSION!r}"
            )
        known = {
            "schema_version",
            "checkpoint_id",
            "program",
            "program_version",
            "config_fingerprint",
            "provenance_fingerprint",
            "structural_fingerprint",
            "simulation_time",
            "step",
            "chunk_id",
            "leaves",
            "code",
            "dependencies",
        }
        unknown = sorted(set(copied).difference(known))
        if unknown:
            raise ValueError(f"Checkpoint metadata contains unknown fields: {unknown!r}")
        required = known.difference({"chunk_id", "leaves", "code", "dependencies"})
        missing = sorted(required.difference(copied))
        if missing:
            raise ValueError(f"Checkpoint metadata is missing required fields: {missing!r}")
        leaves = copied.get("leaves", ())
        if not isinstance(leaves, Sequence) or isinstance(leaves, (str, bytes)):
            raise TypeError("Checkpoint metadata leaves must be a sequence")
        return cls(
            checkpoint_id=str(copied["checkpoint_id"]),
            program=str(copied["program"]),
            program_version=str(copied["program_version"]),
            config_fingerprint=str(copied["config_fingerprint"]),
            provenance_fingerprint=str(copied["provenance_fingerprint"]),
            structural_fingerprint=str(copied["structural_fingerprint"]),
            simulation_time=copied["simulation_time"],
            step=copied["step"],
            chunk_id=copied.get("chunk_id"),
            leaves=tuple(CheckpointLeaf.from_dict(leaf) for leaf in leaves),
            code=copied.get("code"),
            dependencies=copied.get("dependencies"),
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class CheckpointCompatibility:
    """Program identity and fingerprints required for a compatible restore."""

    program: str
    program_version: str
    config_fingerprint: str
    structural_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("program", "program_version", "config_fingerprint", "structural_fingerprint"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name=f"compatibility {name}"))

    @classmethod
    def from_metadata(cls, metadata: CheckpointMetadata) -> CheckpointCompatibility:
        return cls(
            program=metadata.program,
            program_version=metadata.program_version,
            config_fingerprint=metadata.config_fingerprint,
            structural_fingerprint=metadata.structural_fingerprint,
        )

    def validate(self, metadata: CheckpointMetadata) -> None:
        mismatches = []
        for name in ("program", "program_version", "config_fingerprint", "structural_fingerprint"):
            expected = getattr(self, name)
            actual = getattr(metadata, name)
            if actual != expected:
                mismatches.append(f"{name}: checkpoint has {actual!r}, target requires {expected!r}")
        if mismatches:
            details = "\n".join(f"- {message}" for message in mismatches)
            raise IncompatibleCheckpointError(f"checkpoint is incompatible with the restore target:\n{details}")


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Identity and validated metadata returned by a checkpoint store."""

    checkpoint_id: str
    metadata: CheckpointMetadata

    def __post_init__(self) -> None:
        checkpoint_id = _checkpoint_id(self.checkpoint_id)
        if not isinstance(self.metadata, CheckpointMetadata):
            raise TypeError("checkpoint reference metadata must be CheckpointMetadata")
        if self.metadata.checkpoint_id != checkpoint_id:
            raise ValueError("checkpoint reference identity does not match its metadata")
        object.__setattr__(self, "checkpoint_id", checkpoint_id)


@runtime_checkable
class CheckpointSave(Protocol):
    """Completion handle for stores that may persist state asynchronously."""

    def wait(self, *, timeout: float | None = None) -> CheckpointRef | None: ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Durable solver-state storage independent of observations and artifacts."""

    def preflight(self) -> None: ...

    def save(self, state: Any, metadata: CheckpointMetadata) -> CheckpointSave: ...

    def restore(
        self,
        checkpoint: str | CheckpointRef,
        target: Any,
        *,
        compatibility: CheckpointCompatibility | None = None,
    ) -> Any: ...

    def validate(
        self,
        checkpoint: str | CheckpointRef,
        *,
        target: Any | None = None,
        compatibility: CheckpointCompatibility | None = None,
    ) -> CheckpointMetadata: ...

    def list(self) -> tuple[CheckpointRef, ...]: ...

    def latest(self) -> CheckpointRef | None: ...


@dataclass(frozen=True, slots=True)
class _CompletedCheckpointSave:
    reference: CheckpointRef | None

    def wait(self, *, timeout: float | None = None) -> CheckpointRef | None:
        del timeout
        return self.reference


class NullCheckpointStore:
    """Checkpoint store that explicitly disables persistence."""

    def preflight(self) -> None:
        return None

    def save(self, state: Any, metadata: CheckpointMetadata) -> CheckpointSave:
        del state, metadata
        return _CompletedCheckpointSave(None)

    def restore(
        self,
        checkpoint: str | CheckpointRef,
        target: Any,
        *,
        compatibility: CheckpointCompatibility | None = None,
    ) -> Any:
        del checkpoint, target, compatibility
        raise LookupError("null checkpoint store contains no checkpoints")

    def validate(
        self,
        checkpoint: str | CheckpointRef,
        *,
        target: Any | None = None,
        compatibility: CheckpointCompatibility | None = None,
    ) -> CheckpointMetadata:
        del checkpoint, target, compatibility
        raise LookupError("null checkpoint store contains no checkpoints")

    def list(self) -> tuple[CheckpointRef, ...]:
        return ()

    def latest(self) -> CheckpointRef | None:
        return None


def _partition_axis_name(axis: Any) -> str:
    if axis is None:
        return ""
    if isinstance(axis, tuple):
        return ",".join(str(part) for part in axis)
    return str(axis)


def _logical_sharding(leaf: Any) -> tuple[str, ...] | None:
    spec = getattr(getattr(leaf, "sharding", None), "spec", None)
    if spec is None:
        return None
    return tuple(_partition_axis_name(axis) for axis in spec)


def _array_checksum(array: Any) -> str:
    return "sha256:" + hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _encode_array(array: Any) -> Any:
    """Return a pickle-free NumPy payload for one logical checkpoint array."""

    import numpy as np

    if str(array.dtype) not in _EXTENDED_NUMERIC_DTYPE_SIZES:
        return array
    return np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).copy()


def _decode_array(array: Any, schema: CheckpointLeaf) -> Any:
    """Reconstruct a logical array from its declared on-disk encoding."""

    import numpy as np

    if schema.encoding == _NUMPY_ENCODING:
        return array

    import ml_dtypes

    dtype = np.dtype(getattr(ml_dtypes, schema.dtype))
    itemsize = _EXTENDED_NUMERIC_DTYPE_SIZES[schema.dtype]
    if str(dtype) != schema.dtype or dtype.itemsize != itemsize:
        raise CheckpointCorruptionError(f"checkpoint dtype {schema.dtype!r} has an unsupported runtime definition")
    expected_size = math.prod(schema.shape) * itemsize
    if array.dtype != np.dtype(np.uint8) or array.shape != (expected_size,):
        raise CheckpointCorruptionError(
            f"checkpoint leaf {schema.path} does not match its {schema.encoding!r} storage encoding"
        )
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(schema.shape).copy()


def _flatten_state(state: Any) -> tuple[list[Any], tuple[CheckpointLeaf, ...], Any, list[Any]]:
    import jax.tree_util as tree_util
    import numpy as np

    paths_and_leaves, tree = tree_util.tree_flatten_with_path(state)
    arrays = []
    schemas = []
    targets = []
    for index, (path, leaf) in enumerate(paths_and_leaves):
        try:
            array = np.array(leaf, copy=True, order="C")
        except Exception as exc:
            raise TypeError(f"checkpoint state leaf {tree_util.keystr(path) or '<root>'} is not array-like") from exc
        dtype = str(array.dtype)
        extended_itemsize = _EXTENDED_NUMERIC_DTYPE_SIZES.get(dtype)
        if array.dtype.kind not in "biufc" and extended_itemsize is None:
            raise TypeError(
                f"checkpoint state leaf {tree_util.keystr(path) or '<root>'} has unsupported dtype {array.dtype}"
            )
        if extended_itemsize is not None and array.dtype.itemsize != extended_itemsize:
            raise TypeError(f"checkpoint state leaf {tree_util.keystr(path) or '<root>'} has invalid dtype {dtype}")
        leaf_path = tree_util.keystr(path) or "<root>"
        schemas.append(
            CheckpointLeaf(
                path=leaf_path,
                shape=tuple(int(size) for size in array.shape),
                dtype=dtype,
                checksum=_array_checksum(array),
                logical_sharding=_logical_sharding(leaf),
                encoding=_RAW_BYTES_ENCODING if extended_itemsize is not None else _NUMPY_ENCODING,
            )
        )
        arrays.append(array)
        targets.append(leaf)
        if schemas[-1].path != leaf_path:
            raise AssertionError(f"failed to capture checkpoint path at leaf {index}")
    return arrays, tuple(schemas), tree, targets


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LocalCheckpointStore:
    """Filesystem checkpoint store using commit markers and atomic renames."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._lock = RLock()

    def preflight(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self.root.is_dir():
                raise NotADirectoryError(self.root)
            descriptor, probe = tempfile.mkstemp(prefix=".adept-checkpoint-probe-", dir=self.root)
            os.close(descriptor)
            Path(probe).unlink()
            with self._filesystem_lock():
                pass
        except OSError as exc:
            raise CheckpointPreflightError(f"checkpoint directory {self.root} is not writable: {exc}") from exc

    @contextmanager
    def _filesystem_lock(self, *, shared: bool = False) -> Iterator[None]:
        try:
            import fcntl
        except ImportError as exc:
            raise CheckpointPreflightError("LocalCheckpointStore requires POSIX advisory file locking") from exc

        path = self.root / _LOCK_FILE
        if path.is_symlink():
            raise CheckpointPreflightError(f"checkpoint lock {path} must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        locked = False
        try:
            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(descriptor, operation)
            locked = True
            yield
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _write_arrays(self, path: Path, arrays: Sequence[Any]) -> None:
        import numpy as np

        np.savez(path, **{f"leaf_{index:06d}": _encode_array(array) for index, array in enumerate(arrays)})
        _fsync_file(path)

    @staticmethod
    def _write_metadata(path: Path, metadata: CheckpointMetadata) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(metadata.to_dict(), stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())

    def _write_latest(self, checkpoint_id: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".latest-", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"checkpoint_id": checkpoint_id}, stream, separators=(",", ":"), sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / _LATEST_FILE)
            _fsync_directory(self.root)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _latest_snapshot(self) -> bytes | None:
        path = self.root / _LATEST_FILE
        if not path.exists():
            return None
        if path.is_symlink():
            raise CheckpointCorruptionError("latest checkpoint pointer must not be a symlink")
        return path.read_bytes()

    def _restore_latest_snapshot(self, snapshot: bytes | None) -> None:
        path = self.root / _LATEST_FILE
        if snapshot is None:
            path.unlink(missing_ok=True)
            _fsync_directory(self.root)
            return

        descriptor, temporary_name = tempfile.mkstemp(prefix=".latest-rollback-", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(snapshot)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(self.root)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _commit_checkpoint(self, arrays: Sequence[Any], metadata: CheckpointMetadata) -> CheckpointRef:
        final_directory = self.root / metadata.checkpoint_id
        if final_directory.exists():
            raise FileExistsError(f"checkpoint {metadata.checkpoint_id!r} already exists")
        previous_latest = self._latest_snapshot()
        temporary_directory = Path(tempfile.mkdtemp(prefix=f".{metadata.checkpoint_id}-", dir=self.root))
        committed = False
        latest_update_started = False
        try:
            self._write_arrays(temporary_directory / _STATE_FILE, arrays)
            self._write_metadata(temporary_directory / _METADATA_FILE, metadata)
            marker = temporary_directory / _COMMIT_MARKER
            marker.touch()
            _fsync_file(marker)
            _fsync_directory(temporary_directory)
            os.replace(temporary_directory, final_directory)
            committed = True
            _fsync_directory(self.root)
            latest_update_started = True
            self._write_latest(metadata.checkpoint_id)
        except Exception as error:
            rollback_errors = []
            checkpoint_path = final_directory if committed else temporary_directory
            try:
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                _fsync_directory(self.root)
            except Exception as rollback_error:
                rollback_errors.append(f"checkpoint data rollback failed: {rollback_error}")
            if latest_update_started:
                try:
                    self._restore_latest_snapshot(previous_latest)
                except Exception as rollback_error:
                    rollback_errors.append(f"latest pointer rollback failed: {rollback_error}")
            for message in rollback_errors:
                error.add_note(message)
            raise
        return CheckpointRef(metadata.checkpoint_id, metadata)

    def save(self, state: Any, metadata: CheckpointMetadata) -> CheckpointSave:
        if not isinstance(metadata, CheckpointMetadata):
            raise TypeError("metadata must be CheckpointMetadata")

        with self._lock:
            self.preflight()
            arrays, captured_leaves, _, _ = _flatten_state(state)
            if metadata.leaves and metadata.leaves != captured_leaves:
                raise IncompatibleCheckpointError("checkpoint metadata leaves do not match the supplied state")
            committed_metadata = metadata.with_leaves(captured_leaves)
            with self._filesystem_lock():
                reference = self._commit_checkpoint(arrays, committed_metadata)
        return _CompletedCheckpointSave(reference)

    def _checkpoint_directory(self, checkpoint_id: str) -> Path:
        checkpoint_id = _checkpoint_id(checkpoint_id)
        directory = self.root / checkpoint_id
        if directory.is_symlink():
            raise CheckpointCorruptionError(f"checkpoint {checkpoint_id!r} directory must not be a symlink")
        if not directory.is_dir() or not (directory / _COMMIT_MARKER).is_file():
            raise LookupError(f"checkpoint {checkpoint_id!r} is not committed in {self.root}")
        return directory

    def _read_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        directory = self._checkpoint_directory(checkpoint_id)
        path = directory / _METADATA_FILE
        if path.is_symlink():
            raise CheckpointCorruptionError(f"checkpoint {checkpoint_id!r} metadata must not be a symlink")
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, Mapping):
                raise TypeError("metadata document must contain an object")
            metadata = CheckpointMetadata.from_dict(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptionError(f"checkpoint {checkpoint_id!r} has invalid metadata: {exc}") from exc
        if metadata.checkpoint_id != checkpoint_id:
            raise CheckpointCorruptionError(
                f"checkpoint directory {checkpoint_id!r} contains metadata for {metadata.checkpoint_id!r}"
            )
        return metadata

    def _load_arrays(self, metadata: CheckpointMetadata) -> builtins.list[Any]:
        import numpy as np

        path = self._checkpoint_directory(metadata.checkpoint_id) / _STATE_FILE
        if path.is_symlink():
            raise CheckpointCorruptionError(f"checkpoint {metadata.checkpoint_id!r} state must not be a symlink")
        expected = [f"leaf_{index:06d}" for index in range(len(metadata.leaves))]
        try:
            with np.load(path, allow_pickle=False) as archive:
                if sorted(archive.files) != expected:
                    raise CheckpointCorruptionError(
                        f"checkpoint {metadata.checkpoint_id!r} state leaves do not match its metadata"
                    )
                stored_arrays = [np.array(archive[name], copy=True) for name in expected]
        except CheckpointCorruptionError:
            raise
        except Exception as exc:
            raise CheckpointCorruptionError(
                f"checkpoint {metadata.checkpoint_id!r} state archive cannot be read: {exc}"
            ) from exc
        arrays = [_decode_array(array, schema) for schema, array in zip(metadata.leaves, stored_arrays, strict=True)]
        for schema, array in zip(metadata.leaves, arrays, strict=True):
            if tuple(array.shape) != schema.shape or str(array.dtype) != schema.dtype:
                raise CheckpointCorruptionError(
                    f"checkpoint leaf {schema.path} does not match its recorded shape and dtype"
                )
            if _array_checksum(array) != schema.checksum:
                raise CheckpointCorruptionError(f"checkpoint leaf {schema.path} failed checksum validation")
        return arrays

    @staticmethod
    def _checkpoint_name(checkpoint: str | CheckpointRef) -> str:
        if isinstance(checkpoint, CheckpointRef):
            return checkpoint.checkpoint_id
        if isinstance(checkpoint, str):
            return checkpoint
        raise TypeError("checkpoint must be a checkpoint id or CheckpointRef")

    @staticmethod
    def _validate_target(metadata: CheckpointMetadata, target: Any) -> tuple[Any, builtins.list[Any]]:
        _, target_leaves, tree, target_values = _flatten_state(target)
        if len(metadata.leaves) != len(target_leaves):
            raise IncompatibleCheckpointError(
                f"checkpoint has {len(metadata.leaves)} leaves but restore target has {len(target_leaves)}"
            )
        mismatches = []
        for stored, expected in zip(metadata.leaves, target_leaves, strict=True):
            for name in ("path", "shape", "dtype"):
                if getattr(stored, name) != getattr(expected, name):
                    mismatches.append(
                        f"{stored.path} {name}: checkpoint has {getattr(stored, name)!r}, "
                        f"target requires {getattr(expected, name)!r}"
                    )
        if mismatches:
            details = "\n".join(f"- {message}" for message in mismatches)
            raise IncompatibleCheckpointError(
                f"checkpoint state tree is incompatible with the restore target:\n{details}"
            )
        return tree, target_values

    def _validated_contents(
        self,
        checkpoint: str | CheckpointRef,
        *,
        target: Any | None,
        compatibility: CheckpointCompatibility | None,
    ) -> tuple[CheckpointMetadata, builtins.list[Any], Any | None, builtins.list[Any] | None]:
        checkpoint_id = self._checkpoint_name(checkpoint)
        metadata = self._read_metadata(checkpoint_id)
        if compatibility is not None:
            if not isinstance(compatibility, CheckpointCompatibility):
                raise TypeError("compatibility must be CheckpointCompatibility or None")
            compatibility.validate(metadata)
        arrays = self._load_arrays(metadata)
        if target is None:
            return metadata, arrays, None, None
        tree, target_values = self._validate_target(metadata, target)
        return metadata, arrays, tree, target_values

    def validate(
        self,
        checkpoint: str | CheckpointRef,
        *,
        target: Any | None = None,
        compatibility: CheckpointCompatibility | None = None,
    ) -> CheckpointMetadata:
        if not self.root.exists():
            metadata, _, _, _ = self._validated_contents(
                checkpoint,
                target=target,
                compatibility=compatibility,
            )
            return metadata
        with self._lock, self._filesystem_lock(shared=True):
            metadata, _, _, _ = self._validated_contents(
                checkpoint,
                target=target,
                compatibility=compatibility,
            )
        return metadata

    @staticmethod
    def _restore_leaf(array: Any, target: Any) -> Any:
        import numpy as np

        sharding = getattr(target, "sharding", None)
        target_module = type(target).__module__.split(".", maxsplit=1)[0]
        if sharding is not None or target_module in {"jax", "jaxlib"}:
            import jax

            return jax.device_put(array, sharding) if sharding is not None else jax.device_put(array)
        if isinstance(target, np.ndarray):
            return np.asarray(array, dtype=target.dtype)
        if isinstance(target, np.generic):
            return np.asarray(array, dtype=target.dtype).reshape(())[()]
        if isinstance(target, (bool, int, float, complex)):
            return type(target)(array.reshape(()).item())
        return array

    def restore(
        self,
        checkpoint: str | CheckpointRef,
        target: Any,
        *,
        compatibility: CheckpointCompatibility | None = None,
    ) -> Any:
        if not self.root.exists():
            self._checkpoint_directory(self._checkpoint_name(checkpoint))
        with self._lock, self._filesystem_lock(shared=True):
            _, arrays, tree, target_values = self._validated_contents(
                checkpoint,
                target=target,
                compatibility=compatibility,
            )
        assert tree is not None and target_values is not None
        restored = [
            self._restore_leaf(array, target_leaf) for array, target_leaf in zip(arrays, target_values, strict=True)
        ]
        import jax.tree_util as tree_util

        return tree_util.tree_unflatten(tree, restored)

    def list(self) -> tuple[CheckpointRef, ...]:
        if not self.root.exists():
            return ()
        references = []
        with self._lock, self._filesystem_lock(shared=True):
            for path in self.root.iterdir():
                if path.name.startswith(".") or not path.is_dir() or not (path / _COMMIT_MARKER).is_file():
                    continue
                metadata, _, _, _ = self._validated_contents(
                    path.name,
                    target=None,
                    compatibility=None,
                )
                references.append(CheckpointRef(path.name, metadata))
        return tuple(
            sorted(
                references,
                key=lambda reference: (
                    reference.metadata.step,
                    reference.metadata.simulation_time,
                    reference.checkpoint_id,
                ),
            )
        )

    def latest(self) -> CheckpointRef | None:
        if not self.root.exists():
            return None
        with self._lock, self._filesystem_lock(shared=True):
            path = self.root / _LATEST_FILE
            if not path.exists():
                return None
            if path.is_symlink():
                raise CheckpointCorruptionError("latest checkpoint pointer must not be a symlink")
            try:
                with path.open(encoding="utf-8") as stream:
                    payload = json.load(stream)
                if not isinstance(payload, Mapping) or set(payload) != {"checkpoint_id"}:
                    raise ValueError("latest pointer must contain only checkpoint_id")
                checkpoint_id = _checkpoint_id(payload["checkpoint_id"])
                metadata, _, _, _ = self._validated_contents(
                    checkpoint_id,
                    target=None,
                    compatibility=None,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CheckpointCorruptionError(f"latest checkpoint pointer is invalid: {exc}") from exc
        return CheckpointRef(checkpoint_id, metadata)


__all__ = [
    "CheckpointCompatibility",
    "CheckpointCorruptionError",
    "CheckpointError",
    "CheckpointLeaf",
    "CheckpointMetadata",
    "CheckpointPreflightError",
    "CheckpointRef",
    "CheckpointSave",
    "CheckpointStore",
    "IncompatibleCheckpointError",
    "LocalCheckpointStore",
    "NullCheckpointStore",
    "UnsupportedCheckpointVersionError",
]
