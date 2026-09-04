from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from adept.core import (
    MaterializationTarget,
    ObservationCollective,
    ObservationLeaf,
    ObservationPlacement,
    ObservationPlan,
    ObservationReduction,
    ObservationRetention,
    ObservationSchedule,
    ObservationSchema,
    RawResult,
)
from adept.core.observations_jax import infer_observation_spec
from adept.core.programs import ScanProgram


class IncrementSystem(eqx.Module):
    def step(self, step, state, params, inputs, key):
        del step, params, key
        return state + inputs["increment"]


class StateObservation(eqx.Module):
    def __call__(self, t, state, inputs):
        del t, inputs
        return state


class ParticleSummary(eqx.Module):
    def __call__(self, t, state, inputs):
        del t, inputs
        return {"centroid": jnp.mean(state), "edge": state[jnp.asarray([0, -1])]}


def test_schedule_requires_finite_bounded_normalized_points():
    assert ObservationSchedule.every_steps(2, start=1, stop=5).points == (1, 3, 5)
    assert ObservationSchedule.from_legacy_time_config({"tmin": 0.0, "tmax": 1.0, "nt": 3}).points == (
        0.0,
        0.5,
        1.0,
    )

    with pytest.raises(ValueError, match="finite stop"):
        ObservationSchedule.every_steps(1, stop=None)
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservationSchedule.at_times((0.0, 0.5, 0.5))
    with pytest.raises(ValueError, match="does not align"):
        ObservationSchedule.at_times((0.25,)).as_steps(t0=0.0, dt=1.0, num_steps=1)


def test_plan_reports_schema_and_rejects_excess_retention_before_execution():
    state = jnp.ones((4, 8), dtype=jnp.float32)
    spec = infer_observation_spec(
        "state",
        StateObservation(),
        ObservationSchedule.at_steps((0, 2, 4)),
        t=0.0,
        state=state,
        inputs={},
    )

    plan = ObservationPlan((spec,), max_retained_bytes=spec.retained_bytes)

    assert spec.schema.leaves[0].shape == (4, 8)
    assert spec.schema.leaves[0].dtype == "float32"
    assert plan.estimated_retained_bytes == 3 * 4 * 8 * 4
    assert "function" not in plan.to_dict()["observations"][0]
    with pytest.raises(ValueError, match="exceeding"):
        ObservationPlan((spec,), max_retained_bytes=spec.retained_bytes - 1)


def test_scan_program_captures_named_field_reduction_and_particle_observations():
    state = jnp.array([1.0, 2.0, 3.0])
    inputs = {"increment": jnp.ones(3)}
    schedule = ObservationSchedule.at_steps((0, 2, 4))
    fields = infer_observation_spec(
        "field",
        StateObservation(),
        schedule,
        t=0.0,
        state=state,
        inputs=inputs,
    )
    means = infer_observation_spec(
        "mean",
        StateObservation(),
        schedule,
        t=0.0,
        state=state,
        inputs=inputs,
        reduction=ObservationReduction.MEAN,
        placement=ObservationPlacement.SHARDED,
        collective=ObservationCollective.ALL_REDUCE,
    )
    particles = infer_observation_spec(
        "particles",
        ParticleSummary(),
        ObservationSchedule.at_steps((1, 4)),
        t=0.0,
        state=state,
        inputs=inputs,
        retention=ObservationRetention.LAST,
    )
    plan = ObservationPlan((fields, means, particles))
    program = ScanProgram.from_observation_plan(
        system=IncrementSystem(),
        plan=plan,
        state=state,
        inputs=inputs,
        t0=0.0,
        dt=0.5,
        num_steps=4,
    )

    result = eqx.filter_jit(lambda value: value({}, state, inputs, jax.random.key(0)))(program)

    np.testing.assert_allclose(
        result.observations["field"],
        np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]),
    )
    np.testing.assert_allclose(result.observations["mean"], np.asarray([2.0, 4.0, 6.0]))
    assert plan.to_dict()["observations"][1]["collective"] == "all-reduce"
    np.testing.assert_allclose(result.observations["particles"]["centroid"], np.asarray([6.0]))
    np.testing.assert_allclose(result.observations["particles"]["edge"], np.asarray([[5.0, 7.0]]))
    np.testing.assert_allclose(result.times["particles"], np.asarray([2.0]))


def test_program_rejects_a_declared_schema_mismatch_before_execution():
    state = jnp.ones((2,))
    inputs = {"increment": jnp.ones((2,))}
    spec = infer_observation_spec(
        "field",
        StateObservation(),
        ObservationSchedule.at_steps((0,)),
        t=0.0,
        state=state,
        inputs=inputs,
    )
    incorrect = replace(
        spec,
        schema=ObservationSchema((ObservationLeaf("<root>", (3,), "float32", 4),)),
    )

    with pytest.raises(ValueError, match="does not match its declaration"):
        ScanProgram.from_observation_plan(
            system=IncrementSystem(),
            plan=ObservationPlan((incorrect,)),
            state=state,
            inputs=inputs,
            t0=0.0,
            dt=1.0,
            num_steps=1,
        )


def test_sharded_result_materialization_is_explicit_and_host_resident():
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ("device",))
    sharding = NamedSharding(mesh, PartitionSpec("device"))
    device_value = jax.device_put(jnp.arange(max(1, len(devices)) * 2), sharding)
    result = RawResult(
        final_state={"field": device_value},
        observations={"reduced": jnp.mean(device_value)},
        times={"reduced": jnp.asarray([0.0])},
        status="ok",
        stats={"steps": jnp.asarray(1)},
    )

    materialized = result.materialize(MaterializationTarget.ALL_HOSTS)

    assert isinstance(materialized.final_state["field"], np.ndarray)
    assert isinstance(materialized.observations["reduced"], np.ndarray)
    assert result.materialize(MaterializationTarget.RANK_ZERO, process_index=1) is None
