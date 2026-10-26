"""Independent checks for adaptive panel quadrature and remeshing."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from adept.farsight1d.amr import AdaptiveFarsightSystem, initialize_amr, make_hierarchy, remesh_amr
from adept.farsight1d.numerics import electric_field, make_mesh, rk4_push


def _moments(state):
    """Use state weights: refinement changes the quadrature rule itself."""
    weights, values = np.asarray(state["weights"]), np.asarray(state["f"])
    return np.sum(weights * values), np.sum(weights * values**2)


def _system(hierarchy, max_panels, **kwargs):
    options = {
        "hierarchy": hierarchy,
        "max_panels": max_panels,
        "min_level": 0,
        "atol": 0.3,
        "rtol": 0.0,
        "length": 4.0,
        "dt": 0.07,
        "epsilon": 0.3,
        "charge": 0.0,
        "remesh_every": 1,
        "chunk_size": 16,
    }
    return AdaptiveFarsightSystem(**(options | kwargs))


@pytest.mark.parametrize("quadrature", ["trapezoid", "simpson"])
def test_zero_level_matches_shared_node_field_and_characteristic_push(quadrature):
    """Duplicated panel edges must assemble to the original source weights."""
    length, epsilon, dt = 4.0, 0.3, 0.035
    x, v, weights = make_mesh(8, 4, 0.0, length, -1.0, 1.0, quadrature)
    hierarchy = make_hierarchy(8, 4, 0.0, length, -1.0, 1.0, 0, quadrature)

    def distribution(xx, vv):
        return (1.0 + 0.2 * jnp.cos(2 * jnp.pi * xx / length)) * jnp.exp(-(vv**2))

    f = distribution(x, v)
    capacity = hierarchy.root_count + 3
    state = initialize_amr(hierarchy, distribution(hierarchy.x, hierarchy.v), max_panels=capacity)
    assert bool(state["valid"])
    np.testing.assert_array_equal(np.asarray(state["weights"])[~np.asarray(state["active"])], 0.0)
    np.testing.assert_allclose(_moments(state), [jnp.sum(weights * f), jnp.sum(weights * f**2)], atol=3e-14)

    probes = jnp.array([0.13, 0.59, 1.71, 2.91, 3.99])
    expected_field = electric_field(probes, x, -weights * f, length, epsilon)
    actual_field = electric_field(probes, state["x"], -state["weights"] * state["f"], length, epsilon)
    np.testing.assert_allclose(actual_field, expected_field, rtol=3e-13, atol=3e-14)

    expected_x, expected_v = rk4_push(x, v, -weights * f, dt, length, epsilon, -1.0)
    system = _system(hierarchy, capacity, length=length, dt=dt, epsilon=epsilon, charge=-1.0, remesh_every=0)
    pushed = jax.jit(lambda y: system.step(0, y, {}, {}, jax.random.PRNGKey(0)))(state)
    active = np.asarray(state["active"])
    # Recover the shared-grid node represented by each independent panel slot.
    ix = np.rint(np.asarray(state["x"])[active] / (length / 8)).astype(int)
    iv = np.rint((np.asarray(state["v"])[active] + 1.0) / 0.5).astype(int)
    np.testing.assert_allclose(np.asarray(pushed["x"])[active], np.asarray(expected_x)[ix, iv], atol=3e-14)
    np.testing.assert_allclose(np.asarray(pushed["v"])[active], np.asarray(expected_v)[ix, iv], atol=3e-14)


@pytest.mark.parametrize("quadrature", ["trapezoid", "simpson"])
def test_mixed_level_quadrature_partitions_the_domain(quadrature):
    hierarchy = make_hierarchy(8, 8, 0.0, 4.0, -2.0, 2.0, 2, quadrature)
    state = initialize_amr(hierarchy, 1.0 + 0.3 * hierarchy.v**2, max_panels=128, atol=0.5)
    assert bool(state["valid"])
    active = np.asarray(state["active"])
    assert np.unique(np.asarray(state["level"])[active]).size > 1
    assert np.all(np.asarray(state["weights"])[active] > 0.0)
    np.testing.assert_allclose(jnp.sum(state["weights"]), 16.0, atol=5e-14)
    np.testing.assert_allclose(jnp.sum(state["weights"] * state["x"]), 32.0, atol=5e-14)
    np.testing.assert_allclose(jnp.sum(state["weights"] * state["v"]), 0.0, atol=5e-14)
    if quadrature == "simpson":
        np.testing.assert_allclose(jnp.sum(state["weights"] * state["v"] ** 2), 64.0 / 3.0, atol=5e-14)


def test_mixed_level_remesh_reproduces_physical_biquadratic():
    hierarchy = make_hierarchy(8, 8, 0.0, 4.0, -2.0, 2.0, 1)
    state = initialize_amr(hierarchy, 1.0 + 0.3 * hierarchy.v**2, max_panels=64, atol=0.5)
    active = np.asarray(state["active"])
    assert np.unique(np.asarray(state["level"])[active]).size > 1
    xx, vv = state["x"] + 0.04 * state["v"] + 0.05, state["v"] + 0.08

    def polynomial(x, v):
        return 1.0 + 0.13 * x - 0.07 * v + 0.11 * x * v + 0.09 * x**2 * v**2

    moved = {**state, "x": xx, "v": vv, "f": jnp.where(state["active"][:, None], polynomial(xx, vv), 0.0)}
    values, info = jax.jit(lambda y: remesh_amr(y, hierarchy, chunk_size=16))(moved)
    assert bool(info["valid"])
    # Exclude periodic images of this deliberately nonperiodic polynomial.
    interior = (hierarchy.x > 0.5) & (hierarchy.x < 3.5) & (hierarchy.v > -1.75) & (hierarchy.v < 1.75)
    np.testing.assert_allclose(values[interior], polynomial(hierarchy.x, hierarchy.v)[interior], atol=3e-12, rtol=3e-12)
    np.testing.assert_array_equal(values[hierarchy.v == -2.0], 0.0)


def test_nonconforming_interface_kick_does_not_create_interior_vacuum():
    """A coarse edge chord and two fine chords separate under a nonlinear kick.

    This exact volume-preserving map creates a real polygon gap, even when
    the hierarchy is 2:1 balanced. All interior values of a constant density
    must remain one; zero inflow applies only at the outer velocity boundary.
    """
    hierarchy = make_hierarchy(4, 4, 0.0, 2.0, -1.0, 1.0, 1)
    selector = 1.0 + 2.0 * jnp.maximum(hierarchy.v, 0.0)
    state = initialize_amr(hierarchy, selector, max_panels=16, atol=1.0)
    assert bool(state["valid"])
    active = np.asarray(state["active"])
    assert np.unique(np.asarray(state["level"])[active]).size == 2
    moved = {
        **state,
        "v": state["v"] + 0.002 * (jnp.sin(jnp.pi * state["x"]) - 0.5),
        "f": jnp.where(state["active"][:, None], jnp.ones_like(state["f"]), 0.0),
    }
    values, info = jax.jit(lambda y: remesh_amr(y, hierarchy, chunk_size=16, max_gap_fraction=0.01))(moved)
    assert bool(info["valid"])
    assert int(info["gap_nodes"]) > 0
    interior = jnp.abs(hierarchy.v) < 0.5
    np.testing.assert_allclose(values[interior], 1.0, atol=3e-13)


def test_excessive_nonconforming_gap_is_reported_as_invalid():
    hierarchy = make_hierarchy(4, 4, 0.0, 2.0, -1.0, 1.0, 1)
    state = initialize_amr(hierarchy, 1.0 + 2.0 * jnp.maximum(hierarchy.v, 0.0), max_panels=16, atol=1.0)
    moved = {
        **state,
        "v": state["v"] + 0.1 * (jnp.sin(jnp.pi * state["x"]) - 0.5),
        "f": jnp.where(state["active"][:, None], jnp.ones_like(state["f"]), 0.0),
    }
    _, info = jax.jit(lambda y: remesh_amr(y, hierarchy, chunk_size=16))(moved)
    assert not bool(info["valid"])
    assert int(info["invalid_panels"]) == 0  # The failure is interface extension, not a folded panel.
    assert int(info["gap_nodes"]) > 0
    assert float(info["max_gap_fraction"]) > 0.01


def test_rebuild_coarsens_a_flattened_distribution():
    hierarchy = make_hierarchy(8, 4, 0.0, 4.0, -1.0, 1.0, 1)
    state = initialize_amr(hierarchy, 1.0 + hierarchy.v**2, max_panels=32, atol=0.2)
    assert int(state["active_panels"]) > hierarchy.root_count
    flattened = {**state, "f": jnp.where(state["active"][:, None], jnp.ones_like(state["f"]), 0.0)}
    system = _system(hierarchy, 32, atol=0.2, dt=0.0)
    coarsened = jax.jit(lambda y: system.step(0, y, {}, {}, jax.random.PRNGKey(0)))(flattened)
    assert bool(coarsened["valid"])
    assert int(coarsened["active_panels"]) == hierarchy.root_count
    np.testing.assert_array_equal(np.asarray(coarsened["level"])[np.asarray(coarsened["active"])], 0)
    np.testing.assert_allclose(_moments(coarsened), _moments(flattened), atol=3e-14)


def test_capacity_overflow_is_explicit_and_stays_invalid_after_a_step():
    hierarchy = make_hierarchy(4, 4, 0.0, 2.0, -1.0, 1.0, 1)
    capacity = hierarchy.root_count
    state = initialize_amr(hierarchy, 2.0 + hierarchy.v, max_panels=capacity, atol=0.1)
    assert bool(state["capacity_exceeded"])
    assert not bool(state["valid"])
    assert int(state["requested_panels"]) > capacity
    system = _system(hierarchy, capacity, length=2.0, dt=0.0, atol=0.1)
    stepped = jax.jit(lambda y: system.step(0, y, {}, {}, jax.random.PRNGKey(0)))(state)
    assert not bool(stepped["valid"])
    assert bool(stepped["capacity_exceeded"])


def test_maximum_level_saturation_is_distinct_from_capacity_overflow():
    hierarchy = make_hierarchy(4, 4, 0.0, 2.0, -1.0, 1.0, 1)
    state = initialize_amr(hierarchy, 2.0 + hierarchy.v, max_panels=16, atol=0.01)
    assert bool(state["valid"])
    assert not bool(state["capacity_exceeded"])
    assert int(state["refinement_limited_panels"]) > 0
    assert int(state["active_panels"]) == 16


@pytest.mark.parametrize("vmin,vmax,nv", [(-0.2, 0.4, 6), (-3.1, 4.7, 10), (0.1, 0.3, 6)])
def test_decimal_velocity_bounds_are_recognized_as_outer_boundary(vmin, vmax, nv):
    hierarchy = make_hierarchy(4, nv, 0.1, 1.3, vmin, vmax, 1)
    state = initialize_amr(hierarchy, jnp.ones_like(hierarchy.x), max_panels=4 * hierarchy.root_count, atol=1.0)
    values, info = jax.jit(lambda y: remesh_amr(y, hierarchy, chunk_size=16))(state)
    assert bool(info["valid"])
    np.testing.assert_allclose(values, 1.0, atol=3e-13)


def test_balancing_retests_new_children_for_the_refinement_indicator():
    """Balancing exposes a peak that was not sampled by an original leaf.

    The upper half drives refinement down to the interface. A narrow peak
    below it becomes visible after balancing subdivides the lower leaves,
    so those new leaves must be tested against the range criterion again.
    """
    hierarchy = make_hierarchy(4, 4, 0.0, 4.0, -2.0, 2.0, 3)
    candidate_f = 1.0 + jnp.maximum(hierarchy.v, 0.0) + 0.5 * jnp.exp(-(((hierarchy.v + 0.5) / 0.1) ** 2))
    state = initialize_amr(hierarchy, candidate_f, max_panels=256, atol=0.2)
    assert bool(state["valid"])
    unsaturated = np.asarray(state["active"] & (state["level"] < hierarchy.max_level))
    ranges = np.ptp(np.asarray(state["f"]), axis=1)
    assert np.all(ranges[unsaturated] <= 0.2 + 1e-12)


def test_leaf_neighbors_remain_two_to_one_balanced_including_periodic_seam():
    """Check geometric neighbors independently of the hierarchy neighbor table."""
    hierarchy = make_hierarchy(4, 4, 0.0, 4.0, -2.0, 2.0, 3)
    periodic_x = (hierarchy.x - 0.15 + 2.0) % 4.0 - 2.0
    candidate_f = 1.0 + jnp.exp(-((periodic_x / 0.4) ** 2) - (hierarchy.v / 0.4) ** 2)
    state = initialize_amr(hierarchy, candidate_f, max_panels=256, atol=0.2)
    assert bool(state["valid"])
    active = np.asarray(state["active"])
    levels = np.asarray(state["level"])[active]
    assert levels.max() >= 2
    xs, vs = np.asarray(state["x"])[active], np.asarray(state["v"])[active]
    boxes = list(zip(xs.min(axis=1), xs.max(axis=1), vs.min(axis=1), vs.max(axis=1), strict=True))
    checked = periodic_checked = 0
    for i, (ax0, ax1, av0, av1) in enumerate(boxes):
        for j, (bx0, bx1, bv0, bv1) in enumerate(boxes[i + 1 :], start=i + 1):
            overlap_x = min(ax1, bx1) - max(ax0, bx0) > 1e-12
            overlap_v = min(av1, bv1) - max(av0, bv0) > 1e-12
            face_x = min(abs(ax1 - bx0), abs(bx1 - ax0)) < 1e-12
            periodic_x = min(abs(ax0 + 4.0 - bx1), abs(bx0 + 4.0 - ax1)) < 1e-12
            face_v = min(abs(av1 - bv0), abs(bv1 - av0)) < 1e-12
            if ((face_x or periodic_x) and overlap_v) or (face_v and overlap_x):
                checked += 1
                periodic_checked += periodic_x and overlap_v
                assert abs(int(levels[i]) - int(levels[j])) <= 1, (boxes[i], boxes[j], levels[i], levels[j])
    assert checked > 0 and periodic_checked > 0


def test_remesh_budgets_include_changes_to_values_and_panel_weights():
    hierarchy = make_hierarchy(8, 4, 0.0, 4.0, -1.0, 1.0, 1)
    f = (1.0 + 0.4 * jnp.cos(0.5 * jnp.pi * hierarchy.x)) * jnp.exp(-(hierarchy.v**2))
    initial = initialize_amr(hierarchy, f, max_panels=32, atol=0.3)
    system = _system(hierarchy, 32, atol=0.3, dt=0.09)
    advance = jax.jit(lambda i, y: system.step(i, y, {}, {}, jax.random.PRNGKey(0)))
    state = initial
    before = np.asarray(_moments(initial))
    previous = before
    absolute_changes = np.zeros(2)
    for step in range(3):
        state = advance(step, state)
        assert bool(state["valid"])
        current = np.asarray(_moments(state))
        absolute_changes += np.abs(current - previous)
        previous = current
    np.testing.assert_allclose([state["remap_mass_change"], state["remap_c2_change"]], current - before, atol=5e-13)
    np.testing.assert_allclose(
        [state["remap_mass_abs_change"], state["remap_c2_abs_change"]], absolute_changes, atol=5e-13
    )
    assert int(state["remesh_count"]) == 3


def test_zero_time_coarsening_isolates_quadrature_change_in_regrid_budget():
    """An exactly reproduced quadratic still changes trapezoidal moments."""
    hierarchy = make_hierarchy(8, 4, 0.0, 4.0, -1.0, 1.0, 1)
    state = initialize_amr(hierarchy, 1.0 + 0.4 * hierarchy.v**2, max_panels=32, atol=0.01)
    assert int(state["active_panels"]) == 32
    system = _system(hierarchy, 32, atol=100.0, dt=0.0)
    coarsened = jax.jit(lambda y: system.step(0, y, {}, {}, jax.random.PRNGKey(0)))(state)
    assert bool(coarsened["valid"])
    assert int(coarsened["active_panels"]) == hierarchy.root_count
    reference_moments = []
    for nx, nv in [(16, 8), (8, 4)]:
        _, v, weights = make_mesh(nx, nv, 0.0, 4.0, -1.0, 1.0)
        f = 1.0 + 0.4 * v**2
        reference_moments.append(np.array([jnp.sum(weights * f), jnp.sum(weights * f**2)]))
    expected_delta = reference_moments[1] - reference_moments[0]
    assert np.all(expected_delta > 0.0)
    np.testing.assert_allclose(_moments(coarsened), reference_moments[1], atol=3e-14)
    np.testing.assert_allclose(
        [coarsened["remap_mass_change"], coarsened["remap_c2_change"]], expected_delta, atol=3e-14
    )
    np.testing.assert_allclose(
        [coarsened["regrid_mass_change"], coarsened["regrid_c2_change"]], expected_delta, atol=3e-14
    )


def test_adaptive_reverse_mode_matches_finite_difference_away_from_thresholds():
    """The integer topology is piecewise constant; selected values retain AD."""
    hierarchy = make_hierarchy(4, 8, 0.0, 2.0, -2.0, 2.0, 1)
    shape = (1.0 + 0.05 * jnp.cos(jnp.pi * hierarchy.x)) * (1.0 + 0.3 * hierarchy.v**2)
    system = _system(hierarchy, 32, length=2.0, dt=0.03, atol=0.5)

    def objective(amplitude):
        state = initialize_amr(hierarchy, amplitude * shape, max_panels=32, atol=0.5)
        after = system.step(0, state, {}, {}, jax.random.PRNGKey(0))
        return jnp.sum(after["weights"] * after["f"] ** 2)

    amplitude, epsilon = jnp.asarray(1.0), 1e-5
    gradient = jax.jit(jax.grad(objective))(amplitude)
    reference = (objective(amplitude + epsilon) - objective(amplitude - epsilon)) / (2 * epsilon)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, reference, rtol=2e-7, atol=2e-7)
