"""Algebraic and small implementation tests, not nonlinear science runs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adept.farsight1d.amr import AdaptiveFarsightSystem, initialize_amr, make_hierarchy, remesh_amr
from adept.farsight1d.positivity import (
    bernstein_coefficients,
    initialize_positivity,
    limit_rectangles,
    limit_source_samples,
    polynomial_c2,
)


def _weights(areas):
    axis = np.array([1, 4, 1]) / 6
    return jnp.asarray(np.asarray(areas)[:, None] * np.outer(axis, axis).reshape(1, 9))


def _nodal(coefficients):
    basis = np.array([[1, 0, 0], [0.25, 0.5, 0.25], [0, 0, 1]])
    return jnp.asarray(np.einsum("ai,pij,bj->pab", basis, coefficients, basis).reshape(-1, 9))


def _dense(f):
    x = np.linspace(0, 1, 101)
    basis = np.stack((2 * (x - 0.5) * (x - 1), 4 * x * (1 - x), 2 * x * (x - 0.5)), axis=1)
    return np.einsum("ai,pij,bj->pab", basis, np.asarray(f).reshape(-1, 3, 3), basis)


def _adversarial_state():
    """Positive Bernstein polynomials whose sheared physical fits undershoot."""
    hierarchy = make_hierarchy(4, 4, 0.0, 4.0, -1.0, 1.0, 1, "simpson")
    state = initialize_amr(hierarchy, jnp.ones_like(hierarchy.x), max_panels=4, atol=100.0)
    coefficients = np.array(
        [
            [[0.004, 0.045, 0.00006], [5.9, 0.017, 0.0003], [0.62, 1.58, 0.31]],
            [[6, 0.001, 0.035], [0.009, 0.096, 0.0013], [0.16, 0.012, 0.0045]],
            [[0.2, 0.0012, 0.0024], [7.2, 0.0049, 0.0026], [4.83, 3.47, 2.83]],
            [[0.00015, 0.707, 0.00057], [0.109, 1.316, 0.354], [0.00009, 0.057, 0.00035]],
        ]
    )
    return hierarchy, initialize_positivity({**state, "f": _nodal(coefficients)})


def _system(hierarchy, capacity, **kwargs):
    options = {
        "hierarchy": hierarchy,
        "max_panels": capacity,
        "min_level": 0,
        "atol": 100.0,
        "rtol": 0.0,
        "length": 4.0,
        "dt": 0.3,
        "epsilon": 0.1,
        "charge": 0.0,
        "chunk_size": 16,
        "positivity_limiter": "bernstein",
    }
    return AdaptiveFarsightSystem(**(options | kwargs))


def _moments(state):
    active = np.asarray(state["active"])
    weights, f = np.asarray(state["weights"])[active], np.asarray(state["f"])[active]
    return np.array([np.sum(weights * f), np.sum(weights * f**2)])


def test_bernstein_conversion_and_exact_mean():
    coefficients = np.arange(18, dtype=float).reshape(2, 3, 3) / 7
    f = _nodal(coefficients)
    np.testing.assert_allclose(bernstein_coefficients(f), coefficients, atol=2e-15)
    np.testing.assert_allclose(np.sum(np.asarray(_weights([1, 1])) * f, axis=1), coefficients.mean(axis=(1, 2)))


def test_destination_limiter_preserves_mass_and_exposes_c2_dissipation():
    # Positive nodal values do not imply a positive interpolating polynomial.
    f = jnp.asarray(np.tile([1.0, 0.01, 0.05], (2, 3)))
    assert _dense(f).min() < -0.09
    areas = jnp.asarray([2.0, 3.5])
    weights = _weights(areas)
    limited, info = jax.jit(limit_rectangles)(f, weights, jnp.array([True, True]))
    assert bool(info["valid"])
    assert int(info["limited_panels"]) == 2
    assert float(jnp.min(bernstein_coefficients(limited))) >= -1e-15
    assert _dense(limited).min() >= -1e-15
    np.testing.assert_allclose(jnp.sum(weights * limited, axis=1), jnp.sum(weights * f, axis=1), atol=2e-16)
    assert abs(float(info["mass_change"])) < 5e-16
    assert float(info["c2_change"]) < 0
    assert float(info["polynomial_c2_change"]) < 0
    means = jnp.sum(weights * f, axis=1) / areas
    initial = polynomial_c2(f, areas)
    expected = jnp.sum(areas * means**2) + info["theta"][0] ** 2 * (initial - jnp.sum(areas * means**2))
    np.testing.assert_allclose(polynomial_c2(limited, areas), expected, atol=3e-15)
    np.testing.assert_allclose(info["polynomial_c2_change"], polynomial_c2(limited, areas) - initial, atol=3e-15)


def test_certified_polynomial_is_bitwise_unchanged():
    f = _nodal(np.arange(1, 19, dtype=float).reshape(2, 3, 3) / 17)
    limited, info = limit_rectangles(f, _weights([1.0, 0.25]), jnp.array([True, True]))
    np.testing.assert_array_equal(limited, f)
    np.testing.assert_array_equal(info["theta"], 1.0)
    assert int(info["limited_panels"]) == 0
    assert float(info["mass_change"]) == float(info["c2_change"]) == float(info["polynomial_c2_change"]) == 0


def test_bernstein_certificate_can_damp_an_already_nonnegative_polynomial():
    # (2v-1)^2 is nonnegative but has Bernstein coefficients [1,-1,1].
    f = jnp.asarray(np.tile([1.0, 0.0, 1.0], (1, 3)))
    limited, info = limit_rectangles(f, _weights([1]), jnp.array([True]))
    assert _dense(f).min() >= -1e-15
    assert int(info["limited_panels"]) == 1
    assert not np.array_equal(limited, f)


@pytest.mark.parametrize("scale", [0.0, 1e-280, 1e-30, 1.0, 1e30])
def test_zero_and_tiny_tails_do_not_use_an_absolute_density_floor(scale):
    f = jnp.asarray(scale * np.tile([1.0, 0.0, 1.0], (1, 3)))
    limited, info = jax.jit(limit_rectangles)(f, _weights([1]), jnp.array([True]))
    assert bool(info["valid"])
    assert np.isfinite(np.asarray(limited)).all()
    assert np.min(np.asarray(limited)) >= 0
    np.testing.assert_allclose(jnp.sum(_weights([1]) * limited), scale / 3, rtol=5e-15, atol=0)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_negative_or_nonfinite_active_means_fail_without_mass_repair(bad):
    f = jnp.full((1, 9), bad)
    limited, info = limit_rectangles(f, _weights([1]), jnp.array([True]))
    assert not bool(info["valid"])
    assert int(info["failed_panels"]) == 1
    np.testing.assert_array_equal(limited, f)
    _, source = limit_source_samples(
        jnp.array([1.0]), jnp.array([0]), jnp.array([False]), f, _weights([1]), jnp.array([True])
    )
    assert not bool(source["valid"])


def test_trapezoid_is_rejected_by_destination_mean_contract():
    weights = jnp.asarray(np.outer([1, 2, 1], [1, 2, 1]).reshape(1, 9) / 16)
    _, info = limit_rectangles(jnp.ones((1, 9)), weights, jnp.array([True]))
    assert not bool(info["valid"])
    assert int(info["failed_panels"]) == 1


def test_inactive_nan_padding_is_ignored_before_all_limiter_arithmetic():
    f = jnp.array([[1.0] * 9, [jnp.nan] * 9])
    weights = _weights([1, 1]).at[1].set(jnp.nan)
    active = jnp.array([True, False])
    limited, info = limit_rectangles(f, weights, active)
    assert bool(info["valid"])
    np.testing.assert_array_equal(limited, [[1.0] * 9, [0.0] * 9])
    assert all(np.isfinite(np.asarray(value)).all() for value in info.values())
    values, source = limit_source_samples(jnp.array([-1.0]), jnp.array([0]), jnp.array([False]), f, weights, active)
    assert bool(source["valid"])
    assert values[0] >= 0
    assert all(np.isfinite(np.asarray(value)).all() for value in source.values())


def test_source_scales_all_queries_by_owner_and_preserves_exterior_zero():
    f = jnp.asarray(np.arange(1, 28, dtype=float).reshape(3, 9) / 10)
    weights = _weights([1, 2, 3])
    raw = jnp.array([-2.0, 3.0, -100.0, -0.1, 1e99])
    owners = jnp.array([0, 0, 1, 1, 2])
    exterior = jnp.array([False, False, True, False, True])
    limited, info = jax.jit(limit_source_samples)(raw, owners, exterior, f, weights, jnp.ones(3, dtype=bool))
    assert bool(info["valid"])
    assert int(info["limited_panels"]) == 2
    assert np.asarray(limited).min() >= 0
    np.testing.assert_array_equal(limited[exterior], 0.0)
    means = jnp.sum(weights * f, axis=1) / jnp.sum(weights, axis=1)
    expected = (1 - info["theta"][owners]) * means[owners] + info["theta"][owners] * raw
    np.testing.assert_allclose(limited[~exterior], expected[~exterior], atol=3e-16)
    assert float(info["theta"][2]) == 1.0  # No exterior value influences this owner.
    assert abs(float(info["material_mass_change"])) < 1e-15
    assert float(info["material_c2_change"]) < 0


def test_destination_locality_explicitly_allows_different_shared_edge_traces():
    first = np.repeat(np.array([1.0, 0.0, 1.0])[:, None], 3, axis=1)
    f = jnp.asarray(np.stack((first, np.ones((3, 3)))).reshape(2, 9))
    np.testing.assert_array_equal(f[0, 6:9], f[1, :3])
    limited, info = limit_rectangles(f, _weights([1, 1]), jnp.array([True, True]))
    assert bool(info["valid"])
    assert np.max(np.abs(np.asarray(limited[0, 6:9] - limited[1, :3]))) > 0.4
    np.testing.assert_allclose(jnp.sum(_weights([1, 1]) * limited, axis=1), [1 / 3, 1], atol=2e-16)


def test_initial_limiting_has_separate_budgets_and_never_renormalizes():
    hierarchy = make_hierarchy(4, 2, 0.0, 4.0, -1.0, 1.0, 0, "simpson")
    state = initialize_amr(hierarchy, jnp.ones_like(hierarchy.x), max_panels=3)
    state["f"] = state["f"] * jnp.tile(jnp.array([1.0, 0.01, 0.05]), 3)[None]
    limited = jax.jit(initialize_positivity)(state)
    assert bool(limited["valid"])
    assert int(limited["initial_positivity_limited_panels"]) == 2
    assert float(limited["initial_positivity_c2_change"]) < 0
    np.testing.assert_allclose(_moments(limited)[0], _moments(state)[0], atol=5e-16)
    for key, value in limited.items():
        if key.endswith("change") and not key.startswith("initial_positivity_"):
            assert float(value) == 0.0, key


def test_positive_source_nodes_can_undershoot_on_advected_candidate_queries():
    hierarchy, state = _adversarial_state()
    assert float(jnp.min(bernstein_coefficients(state["f"]))) > 0
    moved = {**state, "x": state["x"] + 0.3 * state["v"]}
    raw, _ = remesh_amr(moved, hierarchy, chunk_size=16)
    assert float(jnp.min(raw)) < -0.02
    limited, info = jax.jit(lambda y: remesh_amr(y, hierarchy, chunk_size=16, positivity_limiter="bernstein"))(moved)
    assert bool(info["valid"])
    assert int(info["source_limiter_panels"]) > 0
    assert float(jnp.min(limited)) >= -1e-15
    np.testing.assert_allclose(info["raw_candidate_f"], raw, atol=3e-15)


def test_full_limited_remesh_budgets_telescope_and_state_keys_are_static():
    hierarchy, initial = _adversarial_state()
    system = _system(hierarchy, 4)
    step = jax.jit(lambda i, y: system.step(i, y, {}, {}, jax.random.PRNGKey(0)))
    state = initial
    for index in range(3):
        state = step(index, state)
        assert set(state) == set(initial)
        assert bool(state["valid"])
        assert _dense(state["f"]).min() >= -2e-14
        for index_moment, moment in enumerate(("mass", "c2")):
            decomposition = sum(
                float(state[f"{stage}_{moment}_change"])
                for stage in ("interpolation", "source_limiter", "regrid", "destination_limiter")
            )
            np.testing.assert_allclose(decomposition, state[f"remap_{moment}_change"], atol=4e-14)
            np.testing.assert_allclose(
                _moments(state)[index_moment] - _moments(initial)[index_moment],
                state[f"remap_{moment}_change"],
                atol=4e-14,
            )
        assert float(state["destination_limiter_polynomial_c2_change"]) <= 0
        assert abs(float(state["destination_limiter_mass_change"])) < 4e-14
    assert int(state["source_limiter_panels"]) > 0
    assert int(state["destination_limiter_panels"]) > 0


def test_disabled_limiter_keeps_existing_result_and_info_bitwise():
    hierarchy, state = _adversarial_state()
    moved = {**state, "x": state["x"] + 0.3 * state["v"]}
    default = remesh_amr(moved, hierarchy, chunk_size=16)
    explicit = remesh_amr(moved, hierarchy, chunk_size=16, positivity_limiter="none")
    assert (
        set(default[1])
        == set(explicit[1])
        == {"uncovered_nodes", "gap_nodes", "max_gap_fraction", "invalid_panels", "max_panel_area_error", "valid"}
    )
    for left, right in zip(jax.tree.leaves(default), jax.tree.leaves(explicit), strict=True):
        np.testing.assert_array_equal(left, right)


def test_no_remesh_preserves_enabled_state_structure_and_nonnegative_markers():
    hierarchy, initial = _adversarial_state()
    system = _system(hierarchy, 4, remesh_every=2)
    step = jax.jit(lambda i, y: system.step(i, y, {}, {}, jax.random.PRNGKey(0)))
    skipped = step(0, initial)
    np.testing.assert_array_equal(skipped["f"], initial["f"])
    assert int(skipped["remesh_count"]) == 0
    remapped = step(1, skipped)
    assert set(remapped) == set(initial)
    assert int(remapped["remesh_count"]) == 1
    assert bool(remapped["valid"])


def test_permitted_gap_samples_are_limited_and_exterior_inflow_remains_zero():
    hierarchy = make_hierarchy(4, 4, 0.0, 2.0, -1.0, 1.0, 1, "simpson")
    state = initialize_amr(hierarchy, 1.0 + 2.0 * jnp.maximum(hierarchy.v, 0), max_panels=16, atol=1.0)
    coefficients = np.ones((16, 3, 3))
    # This fine leaf owns (x=.5,v=0) in the polygon crack. Its positive
    # Bernstein polynomial extrapolates negatively just below its lower edge.
    coefficients[4] = np.tile([1e-6, 0.01, 0.005], (3, 1))
    moved = {
        **state,
        "v": state["v"] + 0.002 * (jnp.sin(jnp.pi * state["x"]) - 0.5),
        "f": jnp.where(state["active"][:, None], _nodal(coefficients), 0.0),
    }
    values, info = remesh_amr(moved, hierarchy, chunk_size=16, positivity_limiter="bernstein")
    assert bool(info["valid"])
    assert int(info["gap_nodes"]) > 0
    crack = (hierarchy.x == 0.5) & (hierarchy.v == 0)
    assert float(jnp.min(info["raw_candidate_f"][crack])) < -1e-5
    assert float(jnp.min(values[crack])) >= -1e-15
    assert int(info["source_limiter_panels"]) > 0
    assert np.any(np.asarray(info["raw_candidate_f"]) == 0.0)
    np.testing.assert_array_equal(values[info["raw_candidate_f"] == 0], 0.0)


def test_changing_leaf_partition_preserves_exclusive_budget_closure():
    hierarchy = make_hierarchy(4, 4, 0.0, 4.0, -1.0, 1.0, 1, "simpson")
    state = initialize_amr(hierarchy, 1.0 + 0.3 * hierarchy.v**2, max_panels=16, min_level=1)
    initial = initialize_positivity(state)
    system = _system(hierarchy, 16, dt=0.0)
    final = jax.jit(lambda y: system.step(0, y, {}, {}, jax.random.PRNGKey(0)))(initial)
    assert bool(final["valid"])
    assert int(initial["active_panels"]) == 16
    assert int(final["active_panels"]) == 4
    assert abs(float(final["regrid_c2_change"])) > 1e-3
    for moment in ("mass", "c2"):
        total = sum(
            final[f"{stage}_{moment}_change"]
            for stage in ("interpolation", "source_limiter", "regrid", "destination_limiter")
        )
        np.testing.assert_allclose(total, final[f"remap_{moment}_change"], atol=3e-14)


def test_enabled_full_step_ignores_inactive_nan_sentinels():
    hierarchy = make_hierarchy(4, 4, 0.0, 4.0, -1.0, 1.0, 0, "simpson")
    initial = initialize_positivity(initialize_amr(hierarchy, jnp.ones_like(hierarchy.x), max_panels=6))
    padded = {
        **initial,
        **{key: jnp.where(initial["active"][:, None], initial[key], jnp.nan) for key in ("x", "v", "f", "weights")},
    }
    system = _system(hierarchy, 6, dt=0.0)
    step = jax.jit(lambda y: system.step(0, y, {}, {}, jax.random.PRNGKey(0)))
    expected, actual = step(initial), step(padded)
    assert bool(actual["valid"])
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=3e-14, err_msg=key)


def test_limited_step_amplitude_gradient_matches_fixed_topology_difference():
    hierarchy, initial = _adversarial_state()
    system = _system(hierarchy, 4)

    def objective(amplitude):
        state = {**initial, "f": amplitude * initial["f"]}
        final = system.step(0, state, {}, {}, jax.random.PRNGKey(0))
        return jnp.sum(final["weights"] * final["f"] ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    value, derivative = value_and_grad(1.0)
    expected = (objective(1.0 + 1e-5) - objective(1.0 - 1e-5)) / 2e-5
    assert np.isfinite(float(value)) and np.isfinite(float(derivative))
    np.testing.assert_allclose(derivative, expected, rtol=2e-9)
