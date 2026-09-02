from __future__ import annotations

from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from adept.core import RawResult
from adept.core.programs import DiffraxProgram, ScanProgram


class LinearSystem(eqx.Module):
    def rhs(self, t, state, params, inputs):
        del t
        return params["rate"] * state + inputs["forcing"]


class StateObservation(eqx.Module):
    def __call__(self, t, state, inputs):
        del t, inputs
        return state


class RandomWalk(eqx.Module):
    def step(self, step, state, params, inputs, key):
        del step
        noise = params["noise"] * jax.random.normal(key, state.shape)
        return state + inputs["increment"] + noise


def run_program(program, params, state, inputs, key):
    return program(params, state, inputs, key)


def _continuous_program() -> DiffraxProgram:
    return cast(
        DiffraxProgram,
        DiffraxProgram(
            system=LinearSystem(),
            solver=diffrax.Tsit5(),
            observation=StateObservation(),
            save_times=jnp.array([0.0, 0.5, 1.0]),
            t0=0.0,
            t1=1.0,
            dt0=0.05,
            max_steps=64,
        ),
    )


def test_raw_result_has_a_stable_pytree_schema():
    result = RawResult(
        final_state=jnp.array(1.0),
        observations=jnp.array([1.0]),
        times=jnp.array([0.0]),
        status=diffrax.RESULTS.successful,
        stats={"num_steps": jnp.array(1)},
    )

    assert RawResult._fields == ("final_state", "observations", "times", "status", "stats")
    assert jax.tree.structure(result) == jax.tree.structure(
        RawResult(jnp.array(0.0), jnp.array([0.0]), jnp.array([0.0]), result.status, {"num_steps": jnp.array(0)})
    )


def test_continuous_program_jit_grad_and_changed_runtime_inputs():
    program = _continuous_program()
    compiled = eqx.filter_jit(run_program)
    params = {"rate": jnp.array(0.2)}
    state = jnp.array(1.0)
    key = jax.random.key(0)
    first_inputs = {"forcing": jnp.array(0.25)}
    second_inputs = {"forcing": jnp.array(0.75)}

    first = compiled(program, params, state, first_inputs, key)
    second = compiled(program, params, state, second_inputs, key)
    derivative = jax.grad(
        lambda rate: compiled(program, {"rate": rate}, state, first_inputs, key).final_state
    )(params["rate"])

    expected_first = (state + first_inputs["forcing"] / params["rate"]) * jnp.exp(params["rate"])
    expected_first -= first_inputs["forcing"] / params["rate"]
    np.testing.assert_allclose(first.final_state, expected_first, rtol=2e-5)
    assert not np.isclose(first.final_state, second.final_state)
    assert jnp.isfinite(derivative)
    assert first_inputs == {"forcing": jnp.array(0.25)}
    assert second_inputs == {"forcing": jnp.array(0.75)}


def test_continuous_program_does_not_leak_tracers():
    program = _continuous_program()
    with jax.checking_leaks():
        result = eqx.filter_jit(run_program)(
            program,
            {"rate": jnp.array(0.2)},
            jnp.array(1.0),
            {"forcing": jnp.array(0.25)},
            jax.random.key(0),
        )
        jax.block_until_ready(result.final_state)


def test_discrete_program_is_keyed_deterministic_and_vmappable():
    program = ScanProgram(system=RandomWalk(), t0=0.0, dt=1.0, num_steps=4)
    compiled = eqx.filter_jit(run_program)
    params = {"noise": jnp.array(0.1)}
    state = jnp.zeros((2,))
    inputs = {"increment": jnp.array([1.0, -0.5])}
    key = jax.random.key(7)

    first = compiled(program, params, state, inputs, key)
    repeated = compiled(program, params, state, inputs, key)
    changed = compiled(program, params, state, inputs, jax.random.key(8))
    batched = eqx.filter_jit(
        eqx.filter_vmap(run_program, in_axes=(None, None, None, 0, 0))
    )(
        program,
        params,
        state,
        {"increment": jnp.array([[1.0, 0.0], [0.0, 1.0]])},
        jax.random.split(key, 2),
    )

    np.testing.assert_array_equal(first.final_state, repeated.final_state)
    assert not np.allclose(first.final_state, changed.final_state)
    assert batched.final_state.shape == (2, 2)
    assert batched.stats["num_steps"].shape == (2,)
