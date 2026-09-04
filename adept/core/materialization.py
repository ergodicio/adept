"""Explicit device-to-host materialization for numerical results."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from .contracts import MaterializedResult, RawResult
from .observations import MaterializationTarget


def _materialize_leaf(value: Any) -> Any:
    if not isinstance(value, jax.Array):
        return value
    if not value.is_fully_addressable:
        from jax.experimental import multihost_utils

        value = multihost_utils.process_allgather(value, tiled=True)
    return np.asarray(jax.device_get(value))


def _materialize_tree(value: Any) -> Any:
    return jax.tree.map(_materialize_leaf, value)


def materialize_result(
    result: RawResult,
    target: MaterializationTarget | str = MaterializationTarget.ALL_HOSTS,
    *,
    process_index: int | None = None,
) -> MaterializedResult | None:
    """Collect a result explicitly and return host arrays on the selected hosts.

    Every process must call this function for a non-fully-addressable result because
    those leaves require a collective. ``RANK_ZERO`` suppresses the returned host
    tree on other ranks only after the collective has completed.
    """

    target = MaterializationTarget(target)
    rank = jax.process_index() if process_index is None else process_index
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("process_index must be a non-negative integer")

    materialized = MaterializedResult(
        final_state=_materialize_tree(result.final_state),
        observations=_materialize_tree(result.observations),
        times=_materialize_tree(result.times),
        status=_materialize_tree(result.status),
        stats=_materialize_tree(result.stats),
    )
    if target is MaterializationTarget.RANK_ZERO and rank != 0:
        return None
    return materialized


__all__ = ["materialize_result"]
