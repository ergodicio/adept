"""Distributed initialization helpers for the Vlasov-2D distribution function."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

DISTRIBUTION_DIMS = ("x", "y", "vx", "vy")


@dataclass(frozen=True)
class DistributionSharding:
    """Named sharding metadata for f(x, y, vx, vy) arrays."""

    mesh: Mesh
    partition: PartitionSpec
    sharding: NamedSharding

    def with_unsharded_axis(self, axis: int | str) -> NamedSharding:
        """Return a NamedSharding with one logical distribution axis replicated."""
        axis_index = DISTRIBUTION_DIMS.index(axis) if isinstance(axis, str) else axis
        parts = list(self.partition)
        parts[axis_index] = None
        return NamedSharding(self.mesh, PartitionSpec(*parts))


def _normalize_raw_config(raw: Any) -> dict:
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    return dict(raw)


def _validate_partition(mesh_axes: Sequence[str], partition: Sequence[str | None]) -> None:
    if len(partition) != len(DISTRIBUTION_DIMS):
        raise ValueError(f"distribution partition must have four entries for {DISTRIBUTION_DIMS}, got {partition!r}")
    unknown = {axis for axis in partition if axis is not None and axis not in mesh_axes}
    if unknown:
        raise ValueError(f"distribution partition uses mesh axes not present in mesh_axes: {sorted(unknown)}")


def create_distribution_sharding(
    raw_cfg: Any, devices: Sequence[jax.Device] | None = None
) -> DistributionSharding | None:
    """Create the NamedSharding requested by grid.distribution-sharding.

    If mesh_shape is omitted, all visible devices are placed on the first mesh
    axis. A single-device host still returns a NamedSharding, which lets tests
    exercise the distributed initialization code path without multiple GPUs.
    """
    cfg = _normalize_raw_config(raw_cfg)
    if not cfg.get("enabled", False):
        return None

    mesh_axes = tuple(cfg.get("mesh_axes") or cfg.get("mesh-axes") or ("x",))
    mesh_shape = cfg.get("mesh_shape") or cfg.get("mesh-shape")
    partition = tuple(cfg.get("partition") or ("x", None, None, None))
    _validate_partition(mesh_axes, partition)

    devices = list(devices) if devices is not None else list(jax.devices())
    if not devices:
        raise ValueError("No JAX devices are available for Vlasov-2D distribution sharding.")

    if mesh_shape is None:
        mesh_shape = (len(devices),) + (1,) * (len(mesh_axes) - 1)
    else:
        mesh_shape = tuple(int(v) for v in mesh_shape)

    if len(mesh_shape) != len(mesh_axes):
        raise ValueError(f"mesh_shape {mesh_shape!r} must have one entry per mesh axis {mesh_axes!r}.")
    if int(np.prod(mesh_shape)) != len(devices):
        raise ValueError(f"mesh_shape {mesh_shape!r} requires {int(np.prod(mesh_shape))} devices, got {len(devices)}.")

    mesh = jax.make_mesh(mesh_shape, mesh_axes, devices=devices)
    pspec = PartitionSpec(*partition)
    return DistributionSharding(mesh=mesh, partition=pspec, sharding=NamedSharding(mesh, pspec))


def make_sharded_array(
    shape: tuple[int, ...],
    sharding: NamedSharding,
    callback: Callable[[tuple[slice, ...]], np.ndarray],
    dtype: Any,
):
    """Create a globally shaped sharded JAX array from per-shard callbacks."""

    def _callback(index):
        if index is None:
            index = tuple(slice(None) for _ in shape)
        return np.asarray(callback(index), dtype=dtype)

    return jax.make_array_from_callback(shape, sharding, _callback, dtype=dtype)


def reshard_for_global_axis_fft(array, dist: DistributionSharding, axis: int | str):
    """Reshard an array so a future global FFT has the transform axis local.

    This is intentionally a small building block: pusher code can call this
    before an FFT along x, y, vx, or vy. JAX will insert the necessary collective
    communication for the reshard on multi-device meshes.
    """
    return jax.device_put(array, dist.with_unsharded_axis(axis))
