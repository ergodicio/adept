"""JAX-side validation helpers for observation plans."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .observations import (
    ObservationLeaf,
    ObservationReduction,
    ObservationSchema,
    ObservationSpec,
)


def reduce_observation(value: Any, reduction: ObservationReduction) -> Any:
    """Apply a declared leaf-wise reduction inside the JAX transform boundary."""

    reduction = ObservationReduction(reduction)
    if reduction is ObservationReduction.NONE:
        return value
    operation = jnp.sum if reduction is ObservationReduction.SUM else jnp.mean
    return jax.tree.map(operation, value)


def infer_observation_schema(
    function: Any,
    *,
    t: Any,
    state: Any,
    inputs: Any,
    reduction: ObservationReduction = ObservationReduction.NONE,
) -> ObservationSchema:
    """Infer an output PyTree schema without computing observation values."""

    abstract = jax.eval_shape(
        lambda time, simulation_state, runtime_inputs: reduce_observation(
            function(time, simulation_state, runtime_inputs), reduction
        ),
        t,
        state,
        inputs,
    )
    path_leaves, _ = jax.tree.flatten_with_path(abstract)
    leaves = []
    for path, leaf in path_leaves:
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is None or dtype is None:
            raise TypeError(f"observation leaf {jax.tree_util.keystr(path) or '<root>'} is not array-like")
        numpy_dtype = np.dtype(dtype)
        leaves.append(
            ObservationLeaf(
                path=jax.tree_util.keystr(path) or "<root>",
                shape=tuple(int(size) for size in shape),
                dtype=str(numpy_dtype),
                itemsize=numpy_dtype.itemsize,
            )
        )
    return ObservationSchema(leaves)


def infer_observation_spec(
    name: str,
    function: Any,
    schedule: Any,
    *,
    t: Any,
    state: Any,
    inputs: Any,
    reduction: ObservationReduction = ObservationReduction.NONE,
    **metadata: Any,
) -> ObservationSpec:
    """Construct an observation specification from JAX abstract evaluation."""

    schema = infer_observation_schema(
        function,
        t=t,
        state=state,
        inputs=inputs,
        reduction=reduction,
    )
    return ObservationSpec(
        name=name,
        function=function,
        schedule=schedule,
        schema=schema,
        reduction=reduction,
        **metadata,
    )


def validate_observation_spec(spec: ObservationSpec, *, t: Any, state: Any, inputs: Any) -> None:
    """Check a declared schema against a function's abstract JAX output."""

    actual = infer_observation_schema(
        spec.function,
        t=t,
        state=state,
        inputs=inputs,
        reduction=spec.reduction,
    )
    if actual != spec.schema:
        raise ValueError(
            f"observation {spec.name!r} output schema does not match its declaration: "
            f"expected {spec.schema.to_dict()!r}, got {actual.to_dict()!r}"
        )


def with_step_schedule(
    spec: ObservationSpec,
    *,
    t0: float,
    dt: float,
    num_steps: int,
) -> ObservationSpec:
    """Return a spec whose time schedule has been validated against a step grid."""

    return replace(spec, schedule=spec.schedule.as_steps(t0=t0, dt=dt, num_steps=num_steps))


__all__ = [
    "infer_observation_schema",
    "infer_observation_spec",
    "reduce_observation",
    "validate_observation_spec",
    "with_step_schedule",
]
