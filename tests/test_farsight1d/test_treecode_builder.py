"""Tree field selection through preparation, evolution, observations, and AD."""

from copy import deepcopy
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from adept import LocalExecutor, RunPlan, SimulationSpec, run_prepared, solver_registry
from adept.farsight1d.builder import FarsightFieldsObservation
from adept.farsight1d.config import Farsight1DConfig
from adept.farsight1d.numerics import electric_field
from adept.farsight1d.treecode import TreecodeField
from tests.test_farsight1d.test_builder import prepare, run_jit, small_config


def _config(adaptive=False, theta=0.5, degree=8):
    config = small_config()
    config["numerical"].update(field_solver="treecode", treecode={"theta": theta, "degree": degree, "leaf_size": 4})
    if adaptive:
        config["amr"] = {"enabled": True, "max_level": 1, "max_panels": 40, "atol": 0.05}
    return config


def _execute(prepared):
    return run_jit(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(7))


def test_default_direct_and_selected_tree_controls_are_explicit_and_deterministic():
    direct = prepare()
    assert direct.program.system.field_solver is None
    assert direct.manifest.resolved_config["numerical"]["field_solver"] == "direct"
    config = _config()
    original = deepcopy(config)
    tree, second = prepare(config), prepare(config)
    assert config == original
    assert tree.manifest.to_dict() == second.manifest.to_dict()
    assert tree.manifest.structural_fingerprint != direct.manifest.structural_fingerprint
    assert isinstance(tree.program.system.field_solver, TreecodeField)
    assert tree.program.system.field_solver.degree == 8
    assert tree.program.system.field_solver.leaf_size == 4
    assert not tree.capabilities.differentiable
    assert "reverse-mode" in tree.manifest.units["treecode_gradients"]
    assert "not pair-symmetric" in tree.manifest.units["treecode_conservation"]


@pytest.mark.parametrize("adaptive", [False, True])
def test_zero_theta_full_run_matches_direct(adaptive):
    config = _config(adaptive, theta=0.0)
    tree = prepare(config)
    config["numerical"]["field_solver"] = "direct"
    direct = prepare(config)
    assert tree.capabilities.differentiable
    actual, expected = _execute(tree), _execute(direct)
    assert actual.status == diffrax.RESULTS.successful
    for name in ("x", "v", "f", "remap_mass_change", "remap_c2_change"):
        np.testing.assert_allclose(actual.final_state[name], expected.final_state[name], atol=3e-13, rtol=3e-13)
    for name in actual.observations:
        for field in actual.observations[name]:
            np.testing.assert_allclose(
                actual.observations[name][field], expected.observations[name][field], atol=3e-12, rtol=3e-12
            )


@pytest.mark.parametrize("adaptive", [False, True])
def test_accelerated_fixed_and_adaptive_runs_preserve_direct_solution_to_field_tolerance(adaptive):
    config = _config(adaptive, theta=0.35, degree=10)
    tree = prepare(config)
    config["numerical"]["field_solver"] = "direct"
    actual, expected = _execute(tree), _execute(prepare(config))
    assert actual.status == diffrax.RESULTS.successful
    for name in ("x", "v", "f", "remap_mass_change", "remap_c2_change"):
        np.testing.assert_allclose(actual.final_state[name], expected.final_state[name], atol=3e-8, rtol=3e-8)
    np.testing.assert_allclose(
        actual.observations["fields"]["electric_field"], expected.observations["fields"]["electric_field"], atol=3e-8
    )
    report = tree.analyzer.analyze(actual.materialize(), tree.manifest)
    assert report.metrics[0].values["final_valid"] == 1.0
    if adaptive:
        inactive = ~np.asarray(actual.final_state["active"])
        assert np.any(inactive)
        np.testing.assert_array_equal(actual.final_state["weights"][inactive], 0.0)
        np.testing.assert_array_equal(actual.final_state["f"][inactive], 0.0)


def test_selected_tree_reaches_the_characteristics_and_saved_field_observations():
    config = _config(theta=0.8, degree=1)
    config["grid"].update(nx=8)
    config["numerical"].update(epsilon=0.6, remesh_every=0)
    config["initial"]["amplitude"] = 0.3
    tree = prepare(config)
    result = _execute(tree)
    config["numerical"]["field_solver"] = "direct"
    direct = _execute(prepare(config))
    # Deliberately coarse interpolation makes this a dispatch regression: a
    # push that silently continued to use direct summation would fail it.
    assert float(jnp.max(jnp.abs(result.final_state["v"] - direct.final_state["v"]))) > 1e-8
    state = result.final_state
    field_x = jnp.linspace(0.0, 12.0, 9)[:-1]
    expected_field = tree.program.system.field_solver(
        field_x, state["x"], -tree.program.system.weights * state["f"], 12.0, 0.6, 8
    )
    np.testing.assert_allclose(result.observations["fields"]["electric_field"][-1], expected_field, atol=2e-14)
    expected_energy = 0.5 * 12.0 * jnp.mean(expected_field**2)
    np.testing.assert_allclose(result.observations["scalars"]["electric_energy"][-1], expected_energy, atol=2e-14)


def test_tree_observation_uses_current_amr_weights_and_retains_constructor_compatibility():
    prepared = prepare(_config(adaptive=True))
    state = {**prepared.state, "weights": 1.3 * prepared.state["weights"]}
    x = jnp.array([0.7, 1.8, 9.4])
    solver = prepared.program.system.field_solver
    observe = FarsightFieldsObservation(x, jnp.zeros_like(state["weights"]), 12.0, 0.1, 8, solver)
    expected = solver(x, state["x"], -state["weights"] * state["f"], 12.0, 0.1, 8)
    np.testing.assert_allclose(observe(0.0, state, {})["electric_field"], expected, atol=1e-14)
    direct_observe = FarsightFieldsObservation(x, state["weights"], 12.0, 0.1, 8)
    expected_direct = electric_field(x, state["x"], -state["weights"] * state["f"], 12.0, 0.1, 8)
    np.testing.assert_allclose(direct_observe(0.0, state, {})["electric_field"], expected_direct, atol=1e-14)


def test_zero_theta_prepared_rollout_supports_reverse_mode():
    config = _config(theta=0.0)
    config["numerical"]["remesh_every"] = 0
    prepared = prepare(config)

    def objective(amplitude, program, initial):
        f = initial["f"] * (1.0 + amplitude * jnp.cos(initial["x"]))
        result = program({}, {**initial, "f": f}, {}, jax.random.key(7))
        return result.observations["scalars"]["electric_energy"][-1]

    evaluate = eqx.filter_jit(objective)
    gradient = eqx.filter_jit(jax.grad(objective))
    amplitude, delta = jnp.asarray(0.07), 1e-5
    arguments = (prepared.program, prepared.state)
    derivative = gradient(amplitude, *arguments)
    difference = (evaluate(amplitude + delta, *arguments) - evaluate(amplitude - delta, *arguments)) / (2 * delta)
    assert float(jnp.abs(derivative)) > 1e-8
    np.testing.assert_allclose(derivative, difference, atol=1e-9, rtol=2e-5)


def test_accelerated_tree_runs_through_public_host_execution():
    config = _config()
    config["numerical"]["remesh_every"] = 0
    completed = run_prepared(prepare(config), key=jax.random.key(7))
    with LocalExecutor() as executor:
        planned = executor.execute(RunPlan(SimulationSpec("farsight-1d", config), seed=7))
    np.testing.assert_allclose(planned.report.result["scalars"].c2, completed.report.result["scalars"].c2)
    assert planned.report.metrics[0].values["final_valid"] == 1.0


@pytest.mark.parametrize(
    "controls",
    [
        {"field_solver": "fft"},
        {"field_solver": True},
        {"treecode": {"degree": 0}},
        {"treecode": {"degree": 33}},
        {"treecode": {"degree": True}},
        {"treecode": {"theta": -0.1}},
        {"treecode": {"theta": 1.0}},
        {"treecode": {"theta": float("nan")}},
        {"treecode": {"leaf_size": 0}},
        {"treecode": {"unknown": 1}},
    ],
)
def test_tree_controls_are_strictly_validated(controls):
    config = small_config()
    config["numerical"].update(controls)
    with pytest.raises(ValidationError):
        Farsight1DConfig.model_validate(config)


def test_treecode_example_validates_and_initializes():
    path = Path(__file__).parents[2] / "configs" / "farsight-1d" / "two-stream-treecode.yaml"
    config = yaml.safe_load(path.read_text())
    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=7)
    assert bool(prepared.state["valid"])
    assert isinstance(prepared.program.system.field_solver, TreecodeField)
    assert not prepared.capabilities.differentiable
