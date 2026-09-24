"""Sign-resolved native quadrature diagnostics must not hide negative f."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from adept.farsight1d.numerics import diagnose, initial_state


def _signed_state():
    f = jnp.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    v = jnp.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    weights = jnp.array([[1.0, 0.5, 2.0], [0.25, 3.0, 1.5]])
    return initial_state(jnp.zeros_like(f), v, f, weights), weights


@pytest.mark.parametrize("compiled", [False, True])
def test_signed_quadrature_parts_sum_to_unclipped_mass_and_c2(compiled):
    state, weights = _signed_state()
    evaluate = jax.jit(diagnose) if compiled else diagnose
    actual = evaluate(state, weights)
    # Independent, exactly representable weighted sums for both signs.
    expected = {
        "mass": 8.25,
        "positive_mass": 10.75,
        "negative_mass": 2.5,
        "c2": 30.25,
        "c2_positive": 25.75,
        "c2_negative": 4.5,
        "negative_node_count": 2,
        "min_f": -2.0,
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    np.testing.assert_array_equal(actual["c2"], actual["c2_positive"] + actual["c2_negative"])
    np.testing.assert_array_equal(actual["mass"], actual["positive_mass"] - actual["negative_mass"])
    assert all(value.shape == () for value in actual.values())
    assert jnp.issubdtype(actual["negative_node_count"].dtype, jnp.integer)
    np.testing.assert_array_equal(state["f"], [[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])


def test_positive_only_distribution_preserves_legacy_diagnostics():
    state, weights = _signed_state()
    state = {**state, "f": jnp.abs(state["f"])}
    actual = diagnose(state, weights)
    f, v, w = map(np.asarray, (state["f"], state["v"], weights))
    expected = {
        "mass": np.sum(w * f),
        "c2": np.sum(w * f**2),
        "momentum": np.sum(w * f * v),
        "kinetic_energy": 0.5 * np.sum(w * f * v**2),
        "min_f": np.min(f),
        "negative_mass": 0.0,
        "c2_negative": 0.0,
        "negative_node_count": 0,
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    np.testing.assert_array_equal(actual["positive_mass"], actual["mass"])
    np.testing.assert_array_equal(actual["c2_positive"], actual["c2"])


@pytest.mark.parametrize("padding", [jnp.nan, jnp.inf, -jnp.inf, -1e20])
def test_inactive_padding_is_excluded_before_arithmetic(padding):
    state, weights = _signed_state()
    expected = diagnose(state, weights)

    def pad(values):
        return jnp.stack((values[0], jnp.full_like(values[0], padding), values[1]))

    padded = {
        **state,
        **{name: pad(state[name]) for name in ("x", "v", "f")},
        "weights": pad(weights),
        "active": jnp.array([True, False, True]),
        "panel_id": jnp.array([4, -1, 8]),
        "level": jnp.array([0, -1, 1]),
    }
    # State weights, not the unused fallback, define an AMR quadrature.
    actual = jax.jit(diagnose)(padded, jnp.full_like(weights, jnp.nan))
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].shape == ()
    assert int(actual["negative_node_count"]) == 2


def test_negative_node_count_is_unweighted_and_excludes_zero_values():
    state, weights = _signed_state()
    state = {**state, "active": jnp.array([True, False])}
    weights = weights.at[0, 0].set(0.0)
    actual = diagnose(state, weights)
    assert int(actual["negative_node_count"]) == 2
    np.testing.assert_array_equal(actual["negative_mass"], 0.5)
    np.testing.assert_array_equal(actual["c2_negative"], 0.5)


def test_active_nonfinite_values_remain_visible():
    state, weights = _signed_state()
    state = {**state, "active": jnp.array([True, False]), "f": state["f"].at[0, 0].set(jnp.nan)}
    actual = diagnose(state, weights)
    for key in ("mass", "c2", "c2_positive", "c2_negative", "positive_mass", "negative_mass", "min_f"):
        assert bool(jnp.isnan(actual[key]))


def test_positivity_stage_and_budget_state_scalars_pass_through():
    state, weights = _signed_state()
    names = (
        "initial_positivity_mass_change",
        "initial_positivity_c2_change",
        "initial_positivity_polynomial_c2_change",
        "initial_positivity_limited_panels",
        "initial_positivity_failed_panels",
        "initial_positivity_min_theta",
        "interpolation_mass_change",
        "interpolation_c2_change",
        "source_limiter_mass_change",
        "source_limiter_c2_change",
        "destination_limiter_mass_change",
        "destination_limiter_c2_change",
        "destination_limiter_polynomial_c2_change",
        "source_limiter_panels",
        "destination_limiter_panels",
        "positivity_failed_panels",
        "source_limiter_min_theta",
        "destination_limiter_min_theta",
        "min_bernstein_coefficient",
    )
    extras = {name: jnp.asarray(index + 0.25) for index, name in enumerate(names)}
    actual = jax.jit(diagnose)({**state, **extras}, weights)
    for name, value in extras.items():
        np.testing.assert_array_equal(actual[name], value)
    assert all(value.shape == () for value in actual.values())
