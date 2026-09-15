"""Numerical guarantees for conservative cell-average Vlasov advection."""

from functools import partial

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from adept._vlasov1d.solvers.pushers.conservative import conservative_remap
from adept._vlasov1d.solvers.pushers.vlasov import SpacePFC3, SpaceSLWENO5, VelocityPFC3, VelocitySLWENO5

METHODS = ("pfc3", "sl-weno5")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("periodic", [True, False])
def test_discontinuous_vacuum_and_large_signed_shifts(method, dtype, periodic):
    """No negative mass or spurious mass source, including empty lines and tails."""
    rng = np.random.default_rng(32)
    values = rng.random((10, 64))
    values[:, 12:32] = 0
    values[0] = 0
    values[1, 1:] = 0
    values[2] *= 1e-20
    f = jnp.asarray(values, dtype=dtype)
    shifts = jnp.array([0, 0.23, -0.76, 1, -1, 2.46, -7.83, 64.37, -130.19, 1e10], dtype=dtype)
    remap = jax.jit(partial(conservative_remap, method=method, periodic=periodic))
    result = remap(f, shifts, jnp.array(1.0, dtype=jnp.float64))
    assert result.dtype == f.dtype
    assert bool(jnp.all(jnp.isfinite(result)))
    assert float(result.min()) >= 0
    mass_before, mass_after = np.sum(values, axis=1), np.asarray(result).sum(axis=1)
    tol = 2e-6 if dtype == jnp.float32 else 2e-14
    if periodic:
        np.testing.assert_allclose(mass_after, mass_before, rtol=tol, atol=tol)
    else:
        assert np.all(mass_after <= mass_before + tol * np.maximum(mass_before, 1))
        np.testing.assert_array_equal(result[7:], 0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("periodic", [True, False])
def test_integer_shifts_are_exact(method, periodic):
    """Whole-cell motion is an exact permutation or zero-inflow truncation."""
    n = 32
    offsets = [-65, -32, -3, 0, 2, 32, 65]
    f = np.random.default_rng(3).random((len(offsets), n))
    expected = np.zeros_like(f)
    for row, offset in enumerate(offsets):
        for i in range(n):
            donor = i - offset
            if periodic or 0 <= donor < n:
                expected[row, i] = f[row, donor % n]
    result = conservative_remap(jnp.asarray(f), jnp.array(offsets) * 0.25, 0.25, method=method, periodic=periodic)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("method", METHODS)
def test_velocity_outflow_does_not_reappear_at_opposite_boundary(method):
    """A translated right-edge packet leaves the domain for fractional CFL > 1."""
    f = jnp.zeros((2, 32)).at[0, -3:].set(1).at[1, :3].set(1)
    result = conservative_remap(f, jnp.array([2.4, -2.4]), 1, method=method, periodic=False)
    assert float(result.sum()) < float(f.sum())
    np.testing.assert_array_equal(result[0, :-1], 0)
    np.testing.assert_array_equal(result[1, 1:], 0)
    np.testing.assert_allclose(result[0, -1], result[1, 0], rtol=1e-14)


def _smooth_averages(n, shift=0):
    x = (jnp.arange(n) + 0.5) / n - shift
    return 1 + 0.3 * jnp.sinc(1 / n) * jnp.cos(2 * jnp.pi * x) + 0.1 * jnp.sinc(2 / n) * jnp.sin(4 * jnp.pi * x)


@pytest.mark.parametrize("method, minimum_order", [("pfc3", 2.8), ("sl-weno5", 4.7)])
@pytest.mark.parametrize("courant", [2.37, -2.37])
def test_spatial_convergence_after_many_large_courant_steps(method, minimum_order, courant):
    """Converge to analytic translated cell averages at a fixed physical time."""
    errors = []
    for n in [32, 64, 128]:
        remap = partial(conservative_remap, shift=jnp.array([courant / n]), spacing=1 / n, method=method, periodic=True)

        def advance(f, count=n, push=remap):
            return jax.lax.fori_loop(0, count, lambda i, y: push(y), f)

        result = jax.jit(advance)(_smooth_averages(n)[None, :])
        errors.append(float(jnp.mean(jnp.abs(result[0] - _smooth_averages(n, courant)))))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > minimum_order), (errors, orders)


@pytest.mark.parametrize("method", METHODS)
def test_repeated_cold_beam_advection_stays_positive_and_conservative(method):
    """Unresolved, nearly rectangular beams retain nonnegative cell averages."""
    n = 256
    v = (jnp.arange(n) + 0.5) / n
    f = ((abs(v - 0.3) < 0.018) | (abs(v - 0.7) < 0.018)).astype(jnp.float64)[None, :]
    shifts = jnp.array([2.46, -0.31, 0.67, -5.19])

    def body(i, y):
        return conservative_remap(y, shifts[i % 4], 1, method=method, periodic=True)

    result = jax.jit(lambda y: jax.lax.fori_loop(0, 300, body, y))(f)
    assert float(result.min()) >= 0
    np.testing.assert_allclose(result.sum(), f.sum(), rtol=3e-14)


@pytest.mark.parametrize("method", METHODS)
def test_gradients_match_directional_finite_difference(method):
    """AD differentiates through reconstruction and signed characteristic shifts."""
    f = jnp.stack([_smooth_averages(48), _smooth_averages(48, 0.1)])
    direction = jnp.sin(jnp.arange(f.size).reshape(f.shape)) * 0.03
    shifts = jnp.array([-2.37, 3.19])
    dshift = jnp.array([0.12, -0.08])
    weights = jnp.cos(jnp.arange(f.size).reshape(f.shape) * 0.31)

    def objective(values, displacement):
        return jnp.sum(weights * conservative_remap(values, displacement, 1, method=method, periodic=True))

    df, ds = jax.jit(jax.grad(objective, argnums=(0, 1)))(f, shifts)
    tangent = jnp.sum(df * direction) + jnp.sum(ds * dshift)
    eps = 1e-5
    reference = (
        objective(f + eps * direction, shifts + eps * dshift) - objective(f - eps * direction, shifts - eps * dshift)
    ) / (2 * eps)
    np.testing.assert_allclose(tangent, reference, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_reverse_mode_remains_finite_in_vacuum(method, dtype):
    """Vacuum normalization must not produce 0/0 in the backward pass."""
    f = jnp.zeros((1, 64), dtype=dtype).at[0, 20:25].set(1)
    weights = jnp.cos(jnp.arange(64, dtype=dtype))[None, :]

    def objective(values):
        return jnp.sum(weights * conservative_remap(values, 0.37, 1, method=method, periodic=True))

    gradient = jax.jit(jax.grad(objective))(f)
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize("velocity_type, space_type", [(VelocityPFC3, SpacePFC3), (VelocitySLWENO5, SpaceSLWENO5)])
def test_multispecies_force_spatial_translation_and_sharding(velocity_type, space_type):
    """Species grids and q/m are respected; either sharding orientation agrees."""
    nx = 8 * jax.device_count()
    x = jnp.arange(nx) * 0.2
    params = {"electron": {"charge": -1, "mass": 1}, "ion": {"charge": 2, "mass": 4}}
    grids, f = {}, {}
    for name, nv, dv in [("electron", 96, 0.25), ("ion", 48, 0.5)]:
        v = (jnp.arange(nv) + 0.5) * dv - 8
        grids[name] = {"v": v, "dv": dv}
        # Exact cell averages of a quadratic, with a periodic spatial modulation.
        f[name] = (1 + 0.1 * jnp.cos(2 * jnp.pi * x / (nx * 0.2)))[:, None] * (1 + 0.03 * (v**2 + dv**2 / 12))[None, :]
    e, pond, dt = jnp.linspace(-0.8, 0.5, nx), jnp.full(nx, 0.03), -0.37
    actual = jax.jit(velocity_type(grids, params))(f, e, pond, dt)
    parallel = jax.jit(velocity_type(grids, params, parallel=True))(f, e, pond, dt)
    for name, p in params.items():
        q, m = p["charge"], p["mass"]
        departure = grids[name]["v"][None, :] - (q / m * e + q**2 / m**2 * pond)[:, None] * dt
        expected = (1 + 0.1 * jnp.cos(2 * jnp.pi * x / (nx * 0.2)))[:, None] * (
            1 + 0.03 * (departure**2 + grids[name]["dv"] ** 2 / 12)
        )
        np.testing.assert_allclose(actual[name][:, 6:-6], expected[:, 6:-6], atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(parallel[name], actual[name], atol=2e-13, rtol=2e-13)
    serial_space = jax.jit(space_type(x, grids))(f, dt)
    parallel_space = jax.jit(space_type(x, grids, parallel=True))(f, dt)
    for name in f:
        np.testing.assert_allclose(parallel_space[name], serial_space[name], atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(serial_space[name].sum(axis=0), f[name].sum(axis=0), atol=2e-13, rtol=2e-13)
        assert float(serial_space[name].min()) >= 0


@pytest.mark.parametrize("method, n", [("pfc3", 2), ("sl-weno5", 4)])
def test_rejects_undersized_grid(method, n):
    with pytest.raises(ValueError, match="at least"):
        conservative_remap(jnp.ones((2, n)), jnp.zeros(2), 1, method=method, periodic=True)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.int32])
def test_rejects_unsupported_precision(method, dtype):
    """Half-precision nonlinear weights can overflow; fail before producing NaNs."""
    with pytest.raises(ValueError, match="float32 or float64"):
        conservative_remap(jnp.ones((1, 8), dtype=dtype), 0.37, 1, method=method, periodic=True)
