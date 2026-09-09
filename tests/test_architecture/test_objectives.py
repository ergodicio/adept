from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adept import ObjectiveResult, RawResult
from adept.core.objectives import (
    CallableObjective,
    L2Penalty,
    LegacyVGAdapter,
    WeightedSumObjective,
    evaluate_objective,
    partition_parameters,
    value_and_grad,
)


class AffineProgram(eqx.Module):
    def __call__(self, params: Any, state: Any, inputs: Any, key: jax.Array) -> RawResult:
        del key
        values = eqx.combine(params, inputs)
        final_state = values["scale"] * state + values["bias"]
        return RawResult(
            final_state=final_state,
            observations=final_state,
            times=jnp.asarray([1.0]),
            status=jnp.asarray(0),
            stats={"num_steps": jnp.asarray(1)},
        )


class TargetObjective(eqx.Module):
    def __call__(self, result: RawResult, params: Any, inputs: Any) -> ObjectiveResult:
        del params
        residual = result.final_state - inputs["target"]
        loss = residual**2
        return ObjectiveResult(loss=loss, metrics={"residual": residual}, aux={"prediction": result.final_state})


def affine_partition():
    values = {
        "scale": jnp.asarray(2.0),
        "bias": jnp.asarray(1.0),
        "target": jnp.asarray(4.0),
    }
    selector = {"scale": True, "bias": False, "target": False}
    return values, partition_parameters(values, selector)


def test_partition_parameters_selects_only_explicit_inexact_leaves():
    values, partition = affine_partition()

    assert eqx.tree_equal(partition.trainable, {"scale": jnp.asarray(2.0), "bias": None, "target": None})
    assert eqx.tree_equal(partition.frozen, {"scale": None, "bias": jnp.asarray(1.0), "target": jnp.asarray(4.0)})
    assert eqx.tree_equal(eqx.combine(partition.trainable, partition.frozen), values)

    with pytest.raises(ValueError, match="did not include"):
        partition_parameters(values, False)
    with pytest.raises(TypeError, match="inexact JAX arrays"):
        partition_parameters({"count": 2}, True)


def test_objectives_compose_with_metrics_aux_and_l2_penalty():
    _, partition = affine_partition()
    objective = WeightedSumObjective(
        (TargetObjective(), L2Penalty()),
        weights=(1.0, 0.25),
        names=("fit", "regularization"),
    )

    run = value_and_grad(
        AffineProgram(),
        objective,
        partition.trainable,
        jnp.asarray(1.0),
        partition.frozen,
        jax.random.key(0),
    )

    np.testing.assert_allclose(run.objective.loss, 2.0)
    np.testing.assert_allclose(run.gradients["scale"], -1.0)
    assert run.gradients["bias"] is None
    assert run.gradients["target"] is None
    np.testing.assert_allclose(run.objective.metrics["fit/residual"], -1.0)
    np.testing.assert_allclose(run.objective.metrics["regularization/parameter_l2"], 4.0)
    np.testing.assert_allclose(run.objective.aux["fit"]["prediction"], 3.0)


def test_callable_objective_supports_forward_only_evaluation():
    _, partition = affine_partition()
    objective = CallableObjective(
        lambda result, params, inputs: jnp.abs(result.final_state - eqx.combine(params, inputs)["target"]),
        metric_name="absolute_error",
    )

    run = evaluate_objective(
        AffineProgram(),
        objective,
        partition.trainable,
        jnp.asarray(1.0),
        partition.frozen,
        jax.random.key(0),
    )

    np.testing.assert_allclose(run.objective.loss, 1.0)
    np.testing.assert_allclose(run.objective.metrics["absolute_error"], 1.0)
    np.testing.assert_allclose(run.simulation.final_state, 3.0)


def test_value_and_grad_reuses_compilation_for_changed_same_shape_inputs():
    _, partition = affine_partition()
    objective = TargetObjective()
    traces = []

    def run(params, inputs):
        traces.append(None)
        return value_and_grad(
            AffineProgram(),
            objective,
            params,
            jnp.asarray(1.0),
            inputs,
            jax.random.key(0),
        )

    compiled = eqx.filter_jit(run)
    first = compiled(partition.trainable, partition.frozen)
    changed_inputs = eqx.tree_at(lambda tree: tree["bias"], partition.frozen, jnp.asarray(2.0))
    second = compiled(partition.trainable, changed_inputs)

    assert len(traces) == 1
    np.testing.assert_allclose(first.simulation.final_state, 3.0)
    np.testing.assert_allclose(second.simulation.final_state, 4.0)


def test_objective_contract_rejects_unstructured_and_vector_losses():
    _, partition = affine_partition()
    args = (
        AffineProgram(),
        lambda result, params, inputs: result.final_state,
        partition.trainable,
        jnp.asarray(1.0),
        partition.frozen,
        jax.random.key(0),
    )
    with pytest.raises(TypeError, match="ObjectiveResult"):
        evaluate_objective(*args)

    vector_objective = CallableObjective(lambda result, params, inputs: jnp.stack([result.final_state] * 2))
    with pytest.raises(ValueError, match="must be scalar"):
        evaluate_objective(
            AffineProgram(),
            vector_objective,
            partition.trainable,
            jnp.asarray(1.0),
            partition.frozen,
            jax.random.key(0),
        )


def test_legacy_vg_adapter_preserves_the_old_numerical_call_shape_with_warning():
    _, partition = affine_partition()
    with pytest.warns(FutureWarning, match="migration bridge"):
        adapter = cast(
            LegacyVGAdapter,
            LegacyVGAdapter(
                AffineProgram(),
                TargetObjective(),
                jnp.asarray(1.0),
                partition.frozen,
                jax.random.key(0),
            ),
        )

    (loss, output), gradients = eqx.filter_jit(adapter.vg)(partition.trainable)

    np.testing.assert_allclose(loss, 1.0)
    np.testing.assert_allclose(gradients["scale"], -2.0)
    np.testing.assert_allclose(output["solver result"].final_state, 3.0)
    assert output["objective"].metrics.keys() == {"residual"}
