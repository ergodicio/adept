"""Differentiate an initial perturbation through the complete prepared solver."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from adept import SimulationSpec, solver_registry


def test_full_rollout_initial_distribution_gradient_matches_finite_difference():
    config = {
        "grid": {"nx": 8, "nv": 12, "xmax": 12.0, "vmin": -4.0, "vmax": 4.0},
        "time": {"dt": 0.02, "tmax": 0.06},
        "initial": {"amplitude": 0.0, "drift": 0.2},
        "numerical": {"epsilon": 1.5, "chunk_size": 16},
        "save": {"fields": None, "scalars": {"every_steps": 3}},
    }
    prepared = solver_registry.prepare(SimulationSpec("farsight-1d", config), key=0)

    def objective(amplitude, program, initial, params, inputs, key):
        # No numeric arrays are closed over: this is the explicit transform
        # boundary used by downstream inverse problems and initial-data fits.
        f = initial["f"] * (1 + amplitude * jnp.cos(2 * jnp.pi * initial["x"] / program.system.length))
        result = program(params, {**initial, "f": f}, inputs, key)
        return jnp.sum(program.system.weights * result.final_state["f"] ** 2)

    evaluate = eqx.filter_jit(objective)
    gradient = eqx.filter_jit(jax.grad(objective))
    args = (prepared.program, prepared.state, prepared.params, prepared.inputs, jax.random.key(0))
    amplitude, step = jnp.asarray(0.017), 1e-5
    derivative = gradient(amplitude, *args)
    difference = (evaluate(amplitude + step, *args) - evaluate(amplitude - step, *args)) / (2 * step)
    assert jnp.isfinite(derivative) and jnp.abs(derivative) > 1e-6
    np.testing.assert_allclose(derivative, difference, rtol=2e-5, atol=1e-8)
