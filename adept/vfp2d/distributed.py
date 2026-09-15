"""Named x-axis sharding for the VFP-2D spatial state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True)
class SpatialSharding:
    """Shard arrays whose leading dimension is the VFP-2D x axis."""

    mesh: Mesh
    nx: int

    def for_array(self, value: Any) -> NamedSharding:
        ndim = np.ndim(value)
        partition = P("x", *(None for _ in range(ndim - 1))) if ndim else P()
        return NamedSharding(self.mesh, partition)

    def put(self, value: Any):
        array = jax.numpy.asarray(value)
        if array.ndim and array.shape[0] == self.nx:
            return jax.device_put(array, self.for_array(array))
        return jax.device_put(array, NamedSharding(self.mesh, P(*(None for _ in range(array.ndim)))))

    def replicate(self, value: Any):
        """Replicate a saved snapshot so Diffrax can assemble its leading time axis."""

        array = jax.numpy.asarray(value)
        replicated = NamedSharding(self.mesh, P(*(None for _ in range(array.ndim))))
        return jax.device_put(array, replicated)


def create_spatial_sharding(
    raw_cfg: Any,
    nx: int,
    devices: Sequence[jax.Device] | None = None,
) -> SpatialSharding | None:
    """Create an all-visible-device x mesh when ``grid.sharding.enabled`` is true."""

    cfg = dict(raw_cfg or {})
    if not cfg.get("enabled", False):
        return None
    if cfg.get("axis", "x") != "x":
        raise ValueError("VFP-2D currently supports sharding only along the x axis")

    devices = tuple(jax.devices() if devices is None else devices)
    if not devices:
        raise ValueError("No JAX devices are available for VFP-2D sharding")
    if nx % len(devices):
        raise ValueError(f"VFP-2D nx={nx} must be divisible by the {len(devices)} visible devices")
    mesh = jax.make_mesh((len(devices),), ("x",), devices=devices)
    return SpatialSharding(mesh=mesh, nx=nx)
