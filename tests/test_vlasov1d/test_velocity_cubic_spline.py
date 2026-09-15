"""Tests for the uniform-grid cubic-spline velocity pusher."""

from functools import partial

import jax
import numpy as np
import pytest
from interpax import interp1d
from jax import numpy as jnp

from adept._vlasov1d.solvers.pushers.vlasov import VelocityCubicSpline, _uniform_cubic_interp


def _reference_interp(f: jnp.ndarray, shift: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    interp_rows = jax.vmap(partial(interp1d, extrap=1.0e-30), in_axes=0)
    v_repeated = jnp.broadcast_to(v, f.shape)
    return interp_rows(xq=v_repeated - shift[:, None], x=v_repeated, f=f)


@pytest.mark.parametrize("vmin, vmax", [(-6.4, 6.4), (-4.0, 8.0)])
def test_uniform_cubic_interp_matches_interpax(vmin: float, vmax: float):
    """The optimized stencil reproduces interpax on symmetric and asymmetric grids."""
    nx, nv = 8, 64
    dv = (vmax - vmin) / nv
    v = jnp.linspace(vmin + dv / 2.0, vmax - dv / 2.0, nv)
    f = jax.random.normal(jax.random.key(42), (nx, nv), dtype=jnp.float64)
    shift = dv * jnp.array([-70.0, -2.17, -0.37, 0.0, 0.25, 1.13, 3.4, 70.0])

    expected = _reference_interp(f, shift, v)
    actual = _uniform_cubic_interp(f, shift, dv)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_uniform_cubic_interp_handles_exact_cell_shifts_at_boundaries():
    """Exact integer-cell shifts retain endpoint knots without roundoff leakage."""
    nx, nv = 3, 16
    dv = 0.25
    f = jnp.arange(nx * nv, dtype=jnp.float64).reshape(nx, nv)
    actual = _uniform_cubic_interp(f, dv * jnp.array([1.0, -1.0, 0.0]), dv)

    expected = np.empty((nx, nv))
    expected[0] = np.concatenate(([1.0e-30], np.asarray(f[0, :-1])))
    expected[1] = np.concatenate((np.asarray(f[1, 1:]), [1.0e-30]))
    expected[2] = np.asarray(f[2])
    np.testing.assert_array_equal(actual, expected)


def test_uniform_cubic_interp_matches_interpax_gradients():
    """The optimized stencil preserves gradients with respect to f and the shift."""
    nx, nv = 5, 48
    vmin, vmax = -3.0, 7.0
    dv = (vmax - vmin) / nv
    v = jnp.linspace(vmin + dv / 2.0, vmax - dv / 2.0, nv)
    f = jax.random.normal(jax.random.key(1), (nx, nv), dtype=jnp.float64)
    weights = jax.random.normal(jax.random.key(2), (nx, nv), dtype=jnp.float64)
    shift = dv * jnp.array([-1.37, -0.22, 0.19, 0.83, 2.41])

    def reference_objective(values, offsets):
        return jnp.sum(_reference_interp(values, offsets, v) * weights)

    def optimized_objective(values, offsets):
        return jnp.sum(_uniform_cubic_interp(values, offsets, dv) * weights)

    expected_value, expected_grad = jax.value_and_grad(reference_objective, argnums=(0, 1))(f, shift)
    actual_value, actual_grad = jax.value_and_grad(optimized_objective, argnums=(0, 1))(f, shift)

    np.testing.assert_allclose(actual_value, expected_value, rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(actual_grad[0], expected_grad[0], rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(actual_grad[1], expected_grad[1], rtol=2.0e-11, atol=2.0e-11)


def test_velocity_cubic_spline_uses_each_species_grid_and_force():
    """Each species uses its own dv and q/m acceleration, including ponderomotive force."""
    nx = 4
    species_grids = {}
    species_params = {
        "electron": {"charge": -1.0, "mass": 1.0},
        "ion": {"charge": 1.0, "mass": 4.0},
    }
    f_dict = {}

    for seed, name, vmin, vmax, nv in [
        (3, "electron", -4.0, 8.0, 64),
        (4, "ion", -0.5, 1.0, 40),
    ]:
        dv = (vmax - vmin) / nv
        species_grids[name] = {
            "v": jnp.linspace(vmin + dv / 2.0, vmax - dv / 2.0, nv),
            "dv": dv,
        }
        f_dict[name] = jax.random.normal(jax.random.key(seed), (nx, nv), dtype=jnp.float64)

    e = jnp.array([-0.2, 0.1, 0.3, -0.4])
    pond = jnp.array([0.05, -0.03, 0.02, 0.01])
    dt = 0.17
    actual = VelocityCubicSpline(species_grids, species_params)(f_dict, e, pond, dt)

    for name, f in f_dict.items():
        q = species_params[name]["charge"]
        m = species_params[name]["mass"]
        shift = ((q * e + (q**2 / m) * pond) / m) * dt
        expected = _reference_interp(f, shift, species_grids[name]["v"])
        np.testing.assert_allclose(actual[name], expected, rtol=2.0e-12, atol=2.0e-12)


def test_velocity_cubic_spline_parallel_matches_unsharded():
    """The optimized stencil remains compatible with x-axis shard_map execution."""
    nx = 4 * jax.device_count()
    nv = 32
    dv = 0.2
    v = jnp.linspace(-3.2 + dv / 2.0, 3.2 - dv / 2.0, nv)
    species_grids = {"electron": {"v": v, "dv": dv}}
    species_params = {"electron": {"charge": -1.0, "mass": 1.0}}
    f_dict = {"electron": jax.random.normal(jax.random.key(5), (nx, nv), dtype=jnp.float64)}
    e = jnp.linspace(-0.3, 0.4, nx)
    pond = jnp.linspace(0.02, -0.01, nx)

    expected = VelocityCubicSpline(species_grids, species_params)(f_dict, e, pond, 0.1)
    actual = VelocityCubicSpline(species_grids, species_params, parallel=True)(f_dict, e, pond, 0.1)

    np.testing.assert_allclose(actual["electron"], expected["electron"], rtol=2.0e-12, atol=2.0e-12)
