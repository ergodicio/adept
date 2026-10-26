"""Focused checks for the uniform-grid degree-7 velocity remap."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from adept._vlasov1d.solvers.pushers.vlasov import (
    VelocityLagrange7,
    _uniform_cubic_interp,
    _uniform_lagrange7_interp,
)


@pytest.mark.parametrize("vmin, vmax", [(-4.0, 4.0), (-2.0, 6.0)])
def test_lagrange7_translates_degree7_polynomial(vmin, vmax):
    """Fractional shifts reproduce a degree-7 polynomial where the stencil fits."""
    nv = 64
    dv = (vmax - vmin) / nv
    v = jnp.linspace(vmin + dv / 2, vmax - dv / 2, nv)
    shifts = dv * jnp.array([-2.2, -0.4, 0.0, 0.3, 1.75])

    def polynomial(v):
        return 1 + 0.2 * v - 0.03 * v**3 + 0.0001 * v**7

    f = jnp.broadcast_to(polynomial(v), (len(shifts), nv))
    actual = jax.jit(_uniform_lagrange7_interp)(f, shifts, dv)
    expected = polynomial(v[None, :] - shifts[:, None])
    np.testing.assert_allclose(actual[:, 8:-8], expected[:, 8:-8], atol=2e-12, rtol=2e-12)


def test_lagrange7_integer_shifts_and_exterior_queries():
    """Knots are exact, outflow is discarded, and nothing wraps into the other tail."""
    nv, dv = 16, 0.25
    shifts = jnp.array([-1, 0, 2, -100, 100, 1e20])
    f = jnp.arange(len(shifts) * nv, dtype=jnp.float64).reshape(len(shifts), nv)
    actual = _uniform_lagrange7_interp(f, shifts * dv, dv)
    expected = np.full(f.shape, 1e-30)
    for row, shift in enumerate(np.asarray(shifts)):
        for cell in range(nv):
            source = cell - int(shift)
            if 0 <= source < nv:
                expected[row, cell] = f[row, source]
    np.testing.assert_array_equal(actual, expected)


def test_lagrange7_fractional_boundary_stencil_uses_exterior_floor():
    """A half-cell boundary query uses floor-filled ghosts, not wrapped or clamped data."""
    f = jnp.arange(1, 17, dtype=jnp.float64)[None, :]
    # Exact eight-point Lagrange weights at a half-cell departure point.
    weights = np.array([-5, 49, -245, 1225, 1225, -245, 49, -5]) / 2048
    actual = _uniform_lagrange7_interp(f, jnp.array([0.5]), 1.0)
    expected_first_interior = weights @ np.array([1e-30, 1e-30, 1e-30, 1, 2, 3, 4, 5])
    assert actual[0, 0] == 1e-30
    np.testing.assert_allclose(actual[0, 1], expected_first_interior, atol=1e-15)


def test_lagrange7_preserves_smooth_structure_better_than_local_cubic():
    """The new stencil has less error on a resolved, translated Fourier mode."""
    v = jnp.arange(128, dtype=jnp.float64)
    f = jnp.cos(2 * jnp.pi * v / 8)[None, :]
    expected = jnp.cos(2 * jnp.pi * (v - 0.5) / 8)[None, :]
    high_order = _uniform_lagrange7_interp(f, jnp.array([0.5]), 1.0)
    cubic = _uniform_cubic_interp(f, jnp.array([0.5]), 1.0)
    error = lambda values: jnp.linalg.norm((values - expected)[:, 8:-8])
    assert error(high_order) < error(cubic) / 50


def test_lagrange7_gradients_match_directional_finite_difference():
    """JIT/autodiff propagate changes in the distribution and displacement."""
    f, direction, weights = jax.random.normal(jax.random.key(42), (3, 4, 32), dtype=jnp.float64)
    shifts = jnp.array([-0.37, 0.18, 1.23, -1.15])
    shift_direction = jnp.array([0.1, -0.2, 0.05, 0.3])

    def objective(values, offsets):
        return jnp.sum(weights * _uniform_lagrange7_interp(values, offsets, 0.25))

    df, ds = jax.jit(jax.grad(objective, argnums=(0, 1)))(f, shifts)
    tangent = jnp.sum(df * direction) + jnp.sum(ds * shift_direction)
    eps = 1e-6
    finite_difference = (
        objective(f + eps * direction, shifts + eps * shift_direction)
        - objective(f - eps * direction, shifts - eps * shift_direction)
    ) / (2 * eps)
    np.testing.assert_allclose(tangent, finite_difference, atol=1e-7, rtol=1e-7)


def test_lagrange7_multispecies_force_and_sharding():
    """Each species uses its own grid and q/m, including the ponderomotive force."""
    nx = 4 * jax.device_count()
    grids = {}
    params = {"electron": {"charge": -1.0, "mass": 1.0}, "ion": {"charge": 2.0, "mass": 4.0}}
    f_dict = {}
    for name, vmin, vmax, nv in [("electron", -4.0, 8.0, 96), ("ion", -1.0, 2.0, 48)]:
        dv = (vmax - vmin) / nv
        v = jnp.linspace(vmin + dv / 2, vmax - dv / 2, nv)
        grids[name] = {"v": v, "dv": dv}
        f_dict[name] = jnp.broadcast_to(1 + 0.2 * v + 0.03 * v**2, (nx, nv))

    e, pond, dt = jnp.linspace(-0.2, 0.3, nx), jnp.linspace(0.01, 0.04, nx), 0.17
    actual = VelocityLagrange7(grids, params)(f_dict, e, pond, dt)
    parallel = VelocityLagrange7(grids, params, parallel=True)(f_dict, e, pond, dt)
    for name in params:
        q, m = params[name]["charge"], params[name]["mass"]
        departure = grids[name]["v"][None, :] - (q * e / m + q**2 * pond / m**2)[:, None] * dt
        expected = 1 + 0.2 * departure + 0.03 * departure**2
        np.testing.assert_allclose(actual[name][:, 8:-8], expected[:, 8:-8], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(parallel[name], actual[name], atol=2e-12, rtol=2e-12)


def test_lagrange7_rejects_undersized_grid():
    with pytest.raises(ValueError, match="at least eight"):
        _uniform_lagrange7_interp(jnp.ones((2, 7)), jnp.zeros(2), 1.0)
