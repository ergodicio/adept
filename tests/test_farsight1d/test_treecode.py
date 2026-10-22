"""BLTC accuracy, periodic near fields, fixed-shape traversal and AD limits."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from adept.farsight1d.numerics import electric_field
from adept.farsight1d.treecode import _basis, _chebyshev, electric_field_treecode, electric_field_treecode_with_info


def _cloud(n=257):
    rng = np.random.default_rng(1208)
    sources = jnp.asarray(rng.uniform(-1.0, 2.0, n))
    charges = jnp.asarray(rng.normal(size=n) / n)
    targets = jnp.asarray(rng.uniform(-0.5, 1.5, (7, 9)))
    return targets, sources, charges


@pytest.mark.parametrize("degree", [1, 3, 8])
def test_barycentric_basis_reproduces_polynomials_and_node_derivatives(degree):
    nodes, weights = _chebyshev(degree, jnp.float64)
    samples = jnp.concatenate((nodes, jnp.linspace(-0.93, 0.97, 29)))
    coefficients = jnp.arange(1, degree + 2, dtype=jnp.float64) / (degree + 1)
    values = jnp.polyval(coefficients, nodes)
    interpolated = lambda x: _basis(x, nodes, weights) @ values
    np.testing.assert_allclose(interpolated(samples), jnp.polyval(coefficients, samples), atol=3e-13)
    derivative = jax.jvp(interpolated, (samples,), (jnp.ones_like(samples),))[1]
    np.testing.assert_allclose(derivative, jnp.polyval(jnp.polyder(coefficients), samples), atol=3e-12)


@pytest.mark.parametrize("chunk_size", [1, 13, 128])
def test_zero_opening_matches_exact_direct_and_preserves_shape(chunk_size):
    targets, sources, charges = _cloud()
    evaluate = jax.jit(lambda t, s, q: electric_field_treecode(t, s, q, 1.0, 0.03, theta=0.0, chunk_size=chunk_size))
    actual = evaluate(targets, sources, charges)
    assert actual.shape == targets.shape
    np.testing.assert_allclose(
        actual, electric_field(targets, sources, charges, 1.0, 0.03, chunk_size), atol=2e-17, rtol=2e-15
    )


@pytest.mark.parametrize("epsilon", [0.001, 0.03, 0.3])
def test_signed_source_field_matches_direct(epsilon):
    targets, sources, charges = _cloud()
    actual = jax.jit(lambda t, s, q: electric_field_treecode(t, s, q, 1.0, epsilon, degree=8, leaf_size=8))(
        targets, sources, charges
    )
    expected = electric_field(targets, sources, charges, 1.0, epsilon)
    # Signed charges can make individual field values cross zero; use a
    # charge-scaled absolute error and a field-norm relative error instead.
    assert float(jnp.max(jnp.abs(actual - expected)) / jnp.sum(jnp.abs(charges))) < 1e-7
    assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < 3e-6


def test_accuracy_improves_with_degree_and_tighter_opening():
    targets, sources, charges = _cloud(513)
    expected = electric_field(targets, sources, charges, 1.0, 0.025)
    errors = []
    for degree, theta in ((2, 0.7), (8, 0.7), (8, 0.35)):
        actual = jax.jit(
            lambda t, s, q, degree=degree, theta=theta: electric_field_treecode(
                t, s, q, 1.0, 0.025, degree=degree, theta=theta, leaf_size=8
            )
        )(targets, sources, charges)
        errors.append(float(jnp.max(jnp.abs(actual - expected))))
    assert errors[1] < errors[0] / 50
    assert errors[2] < errors[1] / 50
    assert errors[2] < 1e-9


def test_periodic_seams_antipodes_and_source_permutations():
    targets = jnp.array([[-1e-10, 0.0, 1e-10, 1.0 - 1e-10], [0.4999, 0.5, 0.5001, 1.0]])
    sources = jnp.concatenate((jnp.linspace(-0.02, 0.02, 49), jnp.linspace(0.48, 0.52, 49)))
    charges = jnp.cos(jnp.arange(sources.size) * 0.7) / sources.size
    evaluate = jax.jit(lambda t, s, q: electric_field_treecode(t, s, q, 1.0, 0.002, degree=8, leaf_size=4))
    actual = evaluate(targets, sources, charges)
    np.testing.assert_allclose(actual, electric_field(targets, sources, charges, 1.0, 0.002), atol=2e-9)
    np.testing.assert_allclose(evaluate(targets + 3.0, sources, charges), actual, atol=3e-14)
    order = np.random.default_rng(831).permutation(sources.size)
    np.testing.assert_allclose(evaluate(targets, sources[order], charges[order]), actual, atol=1e-15)


def test_duplicate_nodes_self_force_and_inactive_capacity():
    sources = jnp.full(133, 0.321)
    charges = jnp.arange(133, dtype=jnp.float64) / 133
    targets = jnp.array([0.0, 0.321, 0.499, 0.821, 1.321])
    actual = jax.jit(lambda s, q: electric_field_treecode(targets, s, q, 1.0, 0.01, leaf_size=4))(sources, charges)
    np.testing.assert_allclose(actual, electric_field(targets, sources, charges, 1.0, 0.01), atol=2e-12)
    assert abs(float(actual[1])) < 1e-14
    padded_sources = jnp.concatenate((sources, jnp.array([jnp.nan, jnp.inf, -1e30]), jnp.zeros(119)))
    padded_charges = jnp.pad(charges, (0, 122))
    padded = electric_field_treecode(targets, padded_sources, padded_charges, 1.0, 0.01, leaf_size=4)
    np.testing.assert_allclose(padded, actual, atol=2e-12)


def test_zero_and_empty_sources_targets():
    targets = jnp.linspace(-1.0, 2.0, 11)
    for theta in (0.0, 0.5):
        actual = electric_field_treecode(targets, jnp.full(53, jnp.nan), jnp.zeros(53), 1.0, 0.02, theta=theta)
        np.testing.assert_array_equal(actual, 0.0)
        empty_sources = electric_field_treecode(targets, jnp.zeros(0), jnp.zeros(0), 1.0, 0.02, theta=theta)
        np.testing.assert_array_equal(empty_sources, 0.0)
        empty_targets = electric_field_treecode(
            jnp.zeros((0, 2)), targets, jnp.ones_like(targets), 1.0, 0.02, theta=theta
        )
        assert empty_targets.shape == (0, 2)


def test_isolated_far_cluster_has_interpolation_charge_cancellation():
    sources = jnp.linspace(0.2, 0.3, 129)
    charges = jnp.cos(jnp.arange(129))
    charges -= charges.mean()
    targets = jnp.linspace(0.61, 0.65, 11)
    actual, info = jax.jit(lambda t: electric_field_treecode_with_info(t, sources, charges, 1.0, 0.015, leaf_size=8))(
        targets
    )
    np.testing.assert_allclose(actual, electric_field(targets, sources, charges, 1.0, 0.015), atol=2e-11)
    assert int(info["visited_nodes"]) == targets.size
    assert int(info["accepted_clusters"]) == targets.size
    assert int(info["direct_pairs"]) == 0


def test_kernel_work_is_subquadratic_on_resolved_uniform_cloud():
    work = []
    for n in (256, 1024):
        sources = (jnp.arange(n, dtype=jnp.float64) + 0.31) / n
        charges = (1.0 + 0.2 * jnp.sin(2 * jnp.pi * sources)) / n
        values, info = jax.jit(
            lambda s, q: electric_field_treecode_with_info(s, s, q, 1.0, 1e-5, degree=8, leaf_size=16)
        )(sources, charges)
        assert bool(jnp.all(jnp.isfinite(values)))
        work.append(int(info["kernel_evaluations"]))
        assert int(info["accepted_clusters"]) > 0
    assert work[1] < 7 * work[0]  # N quadruples; direct work would grow by 16.
    assert work[1] < 1024**2 / 3


def test_forward_derivative_matches_difference_with_fixed_tree():
    targets, sources, charges = _cloud(73)
    targets = targets.reshape(-1)[:5]
    directions = (jnp.full_like(targets, 0.13), jnp.cos(sources) * 0.02, jnp.ones_like(charges) / charges.size)
    function = jax.jit(lambda t, s, q: electric_field_treecode(t, s, q, 1.0, 0.03, degree=8, leaf_size=4))
    primals = (targets, sources, charges)
    actual = jax.jvp(function, primals, directions)[1]
    h = 1e-6
    plus = tuple(x + h * dx for x, dx in zip(primals, directions, strict=True))
    minus = tuple(x - h * dx for x, dx in zip(primals, directions, strict=True))
    expected = (function(*plus) - function(*minus)) / (2 * h)
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=2e-6)


def test_reverse_mode_is_explicitly_limited_to_direct_fallback():
    targets, sources, charges = _cloud(37)
    charges = charges.at[2:9].set(0.0)
    targets = targets[:1]
    accelerated = lambda q: jnp.sum(electric_field_treecode(targets, sources, q, 1.0, 0.03, leaf_size=4))
    with pytest.raises(ValueError, match="Reverse-mode differentiation"):
        jax.grad(accelerated)(charges)
    fallback = lambda q: jnp.sum(electric_field_treecode(targets, sources, q, 1.0, 0.03, theta=0.0))
    direct = lambda q: jnp.sum(electric_field(targets, sources, q, 1.0, 0.03))
    np.testing.assert_array_equal(jax.grad(fallback)(charges), jax.grad(direct)(charges))


@pytest.mark.parametrize(
    "options",
    [
        {"degree": 0},
        {"degree": True},
        {"theta": -0.1},
        {"theta": 1.0},
        {"theta": np.nan},
        {"leaf_size": 0},
        {"chunk_size": 1.5},
    ],
)
def test_invalid_options_fail_before_tracing(options):
    with pytest.raises(ValueError):
        electric_field_treecode(jnp.ones(3), jnp.ones(3), jnp.ones(3), 1.0, 0.03, **options)


def test_source_charge_size_mismatch_fails():
    with pytest.raises(ValueError, match="same number"):
        electric_field_treecode(jnp.ones(3), jnp.ones(3), jnp.ones(4), 1.0, 0.03)
