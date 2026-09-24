"""Independent numerical checks for the direct-sum FARSIGHT prototype."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from scipy.integrate import solve_ivp

from adept.farsight1d.numerics import (
    FarsightSystem,
    diagnose,
    electric_field,
    initial_state,
    make_mesh,
    remesh,
    rk4_push,
)


def _numpy_field(targets, sources, charges, length, epsilon):
    """Scalar-loop reference, independent of the JAX blocking implementation."""
    result = np.zeros(np.shape(targets))
    alpha = epsilon / length
    for index, target in np.ndenumerate(np.asarray(targets)):
        for source, charge in zip(np.ravel(sources), np.ravel(charges), strict=True):
            separation = ((target - source) / length + 0.5) % 1.0 - 0.5
            kernel = 0.5 * separation * np.sqrt(1.0 + 4.0 * alpha**2) / np.sqrt(separation**2 + alpha**2) - separation
            result[index] += charge * kernel
    return result


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_periodic_field_matches_independent_direct_sum(chunk_size):
    targets = jnp.array([[0.0, 0.23, 1.8], [2.7, -3.8, 8.6]])
    sources = jnp.array([0.11, 0.5, 1.42, 1.73, 2.29])
    charges = jnp.array([0.3, -0.2, 0.1, 0.6, -0.9])
    expected = _numpy_field(targets, sources, charges, 2.7, 0.13)
    field = jax.jit(lambda x: electric_field(x, sources, charges, 2.7, 0.13, chunk_size=chunk_size))
    np.testing.assert_allclose(field(targets), expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(field(targets + 3 * 2.7), expected, rtol=2e-14, atol=2e-14)


def test_field_sign_self_force_and_pairwise_momentum_balance():
    """Positive source repels positive test charge and exerts no force on itself."""
    targets = jnp.array([-0.2, 0.0, 0.2, 0.5])
    result = electric_field(targets, jnp.array([0.0]), jnp.array([1.0]), 1.0, 0.03)
    assert float(result[0]) < 0 < float(result[2])
    np.testing.assert_allclose(result[0], -result[2], atol=2e-15)
    np.testing.assert_allclose(result[jnp.array([1, 3])], 0.0, atol=2e-15)

    positions = jnp.array([0.03, 0.18, 0.39, 0.71, 0.92])
    charges = jnp.array([0.6, -0.7, 0.1, 0.3, -0.2])
    force = charges * electric_field(positions, positions, charges, 1.0, 0.04, chunk_size=3)
    np.testing.assert_allclose(force.sum(), 0.0, atol=3e-16)


def test_uniform_periodic_charge_has_zero_field_at_nodes():
    """Trapezoidal endpoint weights must count a periodic seam only once."""
    x, _, weights = make_mesh(12, 4, 0.0, 2.0, -1.0, 1.0)
    field = electric_field(x, x, weights, 2.0, 0.06, chunk_size=7)
    np.testing.assert_allclose(field, 0.0, atol=3e-15)


def test_resolved_kernel_controls_uniform_density_aliases_between_nodes():
    """A homogeneous continuum is field-free, including after a ballistic shear.

    Zero field at the original lattice nodes alone is insufficient: an
    underresolved softened kernel can generate large off-lattice self-fields.
    """
    length = 2 * np.pi / 0.3
    x, v, weights = make_mesh(32, 64, 0.0, length, -4.0, 4.0)
    f = jnp.exp(-0.5 * ((v - 2.0) / 0.3) ** 2) + jnp.exp(-0.5 * ((v + 2.0) / 0.3) ** 2)
    f *= length / jnp.sum(weights * f)
    shifted = x + 0.025 * v
    resolved = electric_field(shifted, shifted, -weights * f, length, 1.5)
    underresolved = electric_field(shifted, shifted, -weights * f, length, 0.1)
    # With epsilon/dx=2.29, this force error lies below one percent of the
    # physical Fourier field used by the 1e-4 two-stream perturbation test.
    assert float(jnp.max(jnp.abs(resolved))) < 1e-6
    assert float(jnp.max(jnp.abs(underresolved))) > 0.1


@pytest.mark.parametrize("quadrature", ["trapezoid", "simpson"])
def test_mesh_quadrature_integrates_domain_and_linear_moments(quadrature):
    x, v, weights = make_mesh(8, 6, -0.3, 1.7, -2.0, 3.0, quadrature=quadrature)
    assert x.shape == v.shape == weights.shape == (9, 7)
    assert bool(jnp.all(weights > 0))
    np.testing.assert_allclose(weights.sum(), 10.0, atol=2e-14)
    np.testing.assert_allclose(jnp.sum(weights * x), 7.0, atol=2e-14)
    np.testing.assert_allclose(jnp.sum(weights * v), 5.0, atol=2e-14)
    if quadrature == "simpson":
        np.testing.assert_allclose(
            jnp.sum(weights * x**2 * v**3), ((1.7**3 + 0.3**3) / 3) * ((3**4 - 2**4) / 4), rtol=3e-15
        )


def test_identity_remesh_preserves_all_nodal_values():
    x, v, _ = make_mesh(8, 6, 0.0, 2 * np.pi, -2.0, 2.0)
    f = (1 + 0.2 * jnp.cos(x)) * jnp.exp(-(v**2))
    actual, _ = jax.jit(remesh)(x, v, f, x, v)
    np.testing.assert_allclose(actual, f, rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(actual[0], actual[-1])


@pytest.mark.parametrize("failure", ["folded", "inverted", "collapsed", "nonfinite"])
def test_invalid_panels_are_reported_without_silent_interpolation_fallback(failure):
    x0, v0, _ = make_mesh(8, 4, 0.0, 2 * np.pi, -1.0, 1.0)
    f = (1 + 0.2 * jnp.cos(x0)) * jnp.exp(-(v0**2))
    if failure == "folded":
        x = x0.at[:3, 0].set(x0[:3, 0][::-1])
    elif failure == "inverted":
        x = x0.at[:3].set(x0[:3][::-1])
    elif failure == "collapsed":
        x = x0.at[:3].set(jnp.broadcast_to(x0[1], x0[:3].shape))
    else:
        x = x0.at[1, 1].set(jnp.nan)
    values, info = jax.jit(remesh)(x, v0, f, x0, v0)
    assert not bool(info["valid"])
    assert int(info["invalid_panels"]) > 0
    assert bool(jnp.all(jnp.isnan(values)))


def test_diagnostics_use_phase_space_quadrature_and_retain_negative_mass():
    x, v, weights = make_mesh(8, 4, 0.0, 2 * np.pi, -1.0, 1.0)
    f = v + 0.1 * jnp.cos(x)
    actual = diagnose(initial_state(x, v, f, weights), weights)
    vv, ff, ww = map(np.asarray, (v, f, weights))
    expected = {
        "mass": np.sum(ww * ff),
        "c2": np.sum(ww * ff**2),
        "momentum": np.sum(ww * ff * vv),
        "kinetic_energy": 0.5 * np.sum(ww * ff * vv**2),
        "min_f": np.min(ff),
        "negative_mass": np.sum(ww * np.maximum(-ff, 0.0)),
    }
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=2e-14, atol=2e-14)


def test_deformed_panel_reproduces_physical_biquadratic_and_zero_inflow():
    """Polynomial reconstruction uses advected coordinates, not reference indices."""
    x0, v0, _ = make_mesh(12, 8, 0.0, 2.0, -1.0, 1.0)
    x = x0 + 0.17 * v0 + 0.09
    v = v0 + 0.23

    def polynomial(xx, vv):
        return 1 + 0.13 * xx - 0.07 * vv + 0.11 * xx * vv + 0.09 * xx**2 * vv**2

    actual, _ = jax.jit(remesh)(x, v, polynomial(x, v), x0, v0)
    # Stay away from periodic images of this deliberately nonperiodic polynomial.
    interior = (x0 > 0.4) & (x0 < 1.6) & (v0 >= -0.75)
    np.testing.assert_allclose(actual[interior], polynomial(x0, v0)[interior], rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(actual[:, 0], 0.0)


def test_periodic_free_streaming_remesh_has_third_order_spatial_accuracy():
    """An analytic shear crosses the x seam and is smooth inside the v boundary."""
    errors = []
    for n in [8, 16, 32]:
        x0, v0, _ = make_mesh(n, n, 0.0, 2 * np.pi, -1.0, 1.0)
        f0 = (1 + 0.2 * jnp.cos(x0)) * (1 + 0.1 * v0**2)
        actual, _ = jax.jit(remesh)(x0 + 0.37 * v0, v0, f0, x0, v0)
        expected = (1 + 0.2 * jnp.cos(x0 - 0.37 * v0)) * (1 + 0.1 * v0**2)
        errors.append(float(jnp.sqrt(jnp.mean((actual[:-1] - expected[:-1]) ** 2))))
        np.testing.assert_allclose(actual[0], actual[-1], rtol=3e-12, atol=3e-12)
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert np.all(orders > 2.5), (errors, orders)


def test_remesh_reverse_mode_matches_directional_finite_difference():
    """Differentiate values and moving nodes while panel ownership stays fixed."""
    x0, v0, _ = make_mesh(8, 4, 0.0, 2 * np.pi, -1.0, 1.0)
    f0 = (1 + 0.2 * jnp.cos(x0)) * jnp.exp(-(v0**2))
    direction = 0.03 * jnp.cos(2 * x0) * (1 + v0)
    objective_weights = jnp.sin(0.31 * jnp.arange(x0.size).reshape(x0.shape))

    def objective(amplitude, shift):
        values, _ = remesh(x0 + shift + 0.03 * v0, v0, f0 + amplitude * direction, x0, v0)
        return jnp.sum(values * objective_weights)

    point = (jnp.array(0.1), jnp.array(0.17))
    gradient = jax.jit(jax.grad(objective, argnums=(0, 1)))(*point)
    eps = 1e-5
    for i in range(2):
        plus, minus = list(point), list(point)
        plus[i] += eps
        minus[i] -= eps
        reference = (objective(*plus) - objective(*minus)) / (2 * eps)
        assert np.isfinite(gradient[i])
        np.testing.assert_allclose(gradient[i], reference, rtol=3e-7, atol=3e-8)


def test_rk4_self_consistent_push_has_fourth_order_temporal_accuracy():
    """Compare the interacting characteristic ODE with independent SciPy DOP853."""
    x, v, weights = make_mesh(4, 2, 0.0, 2.0, -0.5, 0.5)
    f = (1 + 0.3 * jnp.cos(jnp.pi * x)) * jnp.exp(-2 * v**2)
    charges = -f * weights
    size = x.size

    def reference_rhs(_time, state):
        field = _numpy_field(state[:size], state[:size], charges, 2.0, 0.2)
        return np.concatenate((state[size:], -field))

    reference = solve_ivp(
        reference_rhs,
        (0.0, 0.4),
        np.concatenate((np.ravel(x), np.ravel(v))),
        method="DOP853",
        rtol=2e-13,
        atol=2e-14,
    )
    assert reference.success
    expected_x = reference.y[:size, -1].reshape(x.shape)
    expected_v = reference.y[size:, -1].reshape(v.shape)
    errors = []
    for steps in [4, 8, 16]:

        def advance(position, velocity, steps=steps):
            def body(_i, nodes):
                return rk4_push(*nodes, charges, 0.4 / steps, 2.0, 0.2, -1.0, chunk_size=4)

            return jax.lax.fori_loop(0, steps, body, (position, velocity))

        actual_x, actual_v = jax.jit(advance)(x, v)
        errors.append(float(jnp.sqrt(jnp.mean((actual_x - expected_x) ** 2 + (actual_v - expected_v) ** 2))))
        np.testing.assert_allclose(jnp.sum(weights * f * actual_v), jnp.sum(weights * f * v), atol=2e-14)
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert np.all(orders > 3.6), (errors, orders)


def test_characteristic_push_reverse_mode_matches_finite_difference():
    x, v, weights = make_mesh(4, 2, 0.0, 2.0, -0.5, 0.5)
    f = (1 + 0.2 * jnp.cos(jnp.pi * x)) * jnp.exp(-(v**2))
    probe = jnp.sin(jnp.arange(x.size).reshape(x.shape))

    def objective(amplitude):
        pushed_x, pushed_v = rk4_push(x, v, -weights * f * amplitude, 0.17, 2.0, 0.08, -1.0, chunk_size=4)
        return jnp.sum(probe * (pushed_x + 0.3 * pushed_v**2))

    actual = jax.jit(jax.grad(objective))(jnp.array(0.7))
    eps = 1e-5
    reference = (objective(0.7 + eps) - objective(0.7 - eps)) / (2 * eps)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, reference, rtol=3e-7, atol=3e-8)


def test_free_streaming_characteristics_preserve_material_casimirs():
    x, v, weights = make_mesh(8, 4, 0.0, 2 * np.pi, -1.0, 1.0)
    f = (1 + 0.2 * jnp.cos(x)) * jnp.exp(-(v**2))
    initial = initial_state(x, v, f, weights)
    system = FarsightSystem(x, v, weights, 2 * np.pi, 0.17, 0.03, charge=0.0, remesh_every=0)
    key = jax.random.PRNGKey(0)

    def advance(state):
        return jax.lax.fori_loop(0, 5, lambda i, y: system.step(i, y, {}, {}, key), state)

    actual = jax.jit(advance)(initial)
    np.testing.assert_allclose(actual["x"], x + 5 * 0.17 * v, atol=3e-15)
    np.testing.assert_array_equal(actual["v"], v)
    np.testing.assert_array_equal(actual["f"], f)
    before, after = diagnose(initial, weights), diagnose(actual, weights)
    for name in ["mass", "c2", "kinetic_energy", "momentum"]:
        np.testing.assert_allclose(after[name], before[name], atol=3e-15)
    assert int(after["remesh_count"]) == 0
    assert float(after["remap_c2_change"]) == 0.0


def test_remap_c2_budget_accounts_for_every_remesh_without_claiming_conservation():
    """Free streaming changes C2 only through interpolation in this representation."""
    x, v, weights = make_mesh(8, 6, 0.0, 2 * np.pi, -1.0, 1.0)
    f = (1 + 0.7 * jnp.cos(2 * x)) * jnp.exp(-(v**2))
    initial = initial_state(x, v, f, weights)
    system = FarsightSystem(x, v, weights, 2 * np.pi, 0.23, 0.03, charge=0.0, remesh_every=1)
    step = jax.jit(lambda i, state: system.step(i, state, {}, {}, jax.random.PRNGKey(0)))
    state = initial
    changes = []
    initial_diagnostics = diagnose(initial, weights)
    previous_c2 = float(initial_diagnostics["c2"])
    for i in range(4):
        state = step(i, state)
        current_c2 = float(diagnose(state, weights)["c2"])
        changes.append(current_c2 - previous_c2)
        previous_c2 = current_c2
    actual = diagnose(state, weights)
    np.testing.assert_allclose(actual["remap_c2_change"], actual["c2"] - initial_diagnostics["c2"], atol=2e-14)
    np.testing.assert_allclose(actual["remap_c2_abs_change"], np.abs(changes).sum(), atol=2e-14)
    np.testing.assert_allclose(actual["remap_mass_change"], actual["mass"] - initial_diagnostics["mass"], atol=2e-14)
    assert int(actual["remesh_count"]) == 4
    assert bool(actual["valid"])
    assert int(actual["invalid_panels"]) == 0
    # This deliberately coarse, high-wavenumber case must expose remap loss.
    assert float(actual["c2"]) < float(initial_diagnostics["c2"])
