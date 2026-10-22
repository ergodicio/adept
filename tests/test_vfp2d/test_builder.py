"""Parity and transform-boundary tests for the explicit VFP2D program."""

from copy import deepcopy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from adept import (
    CallableObjective,
    LocalExecutor,
    RunPlan,
    SimulationSpec,
    ergoExo,
    partition_parameters,
    run_prepared,
    solver_registry,
    value_and_grad,
)
from adept.core.programs import ScanProgram
from adept.vfp2d import BaseVFP2D
from tests.test_vfp2d.test_base import _config


def run(program, params, state, inputs, key):
    return program(params, state, inputs, key)


run_jit = eqx.filter_jit(run)


def assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=3e-11, atol=3e-13)


def legacy_setup(config):
    module = BaseVFP2D(deepcopy(config))
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module


def prepare(config):
    return solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=42)


def small_config(mode="maxwell"):
    config = _config(collisions=True)
    config["grid"].update(nx=4, ny=4, lmax=1, mmax=1)
    config["terms"]["field_solver"] = {"mode": mode}
    config["density"]["species-electron"]["T"] = {
        "basis": "gaussian_spots",
        "baseline": 1.0,
        "x_center": "1um",
        "x_radius": "0.5um",
        "y_centers": ["1um"],
        "y_radius": "0.5um",
    }
    if mode == "ampere":
        config["terms"]["field_solver"]["relative_permittivity"] = 100.0
    return config


@pytest.mark.parametrize(
    "mode", ["maxwell", "ampere", "oshun-implicit", "kinetic-ohm", "coupled", "reservoir", "sharded"]
)
def test_prepared_matches_legacy_fields_collisions_and_coupling(mode):
    config = small_config(mode if mode not in ("coupled", "reservoir", "sharded") else "kinetic-ohm")
    if mode in ("coupled", "reservoir"):
        config["terms"]["ion_fluid"] = {
            "active": True,
            "mass_ratio": 100.0,
            "initial_velocity": [1e-4, 0.0, 0.0],
            "momentum_relaxation_rate": 0.2,
            "temperature_relaxation_rate": 0.3,
        }
    if mode == "reservoir":
        config["drivers"]["reservoir"] = {"active": True, "relaxation_time": "0.1fs", "x_width": "0.75um"}
    if mode == "sharded":
        config["grid"]["sharding"] = {"enabled": True, "axis": "x"}
    prepared = prepare(config)
    legacy = legacy_setup(config)
    assert_tree_close(prepared.state, legacy.state)
    actual = run_jit(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(42))
    expected = legacy(None, None)["solver result"]
    assert_tree_close(actual.observations["state"], expected.ys)
    assert_tree_close(actual.final_state, jax.tree.map(lambda value: value[-1], expected.ys))
    np.testing.assert_allclose(actual.times["state"], expected.ts)
    assert int(actual.stats["num_steps"]) == legacy.nt
    report = prepared.analyzer.analyze(actual.materialize(), prepared.manifest)
    xr.testing.assert_allclose(report.result["vfp2d"], legacy.post_process({"solver result": expected}, "")["vfp2d"])


def test_off_grid_saves_match_legacy_and_final_state_is_independent():
    config = small_config()
    config["grid"]["tmin"] = "0.01fs"
    config["grid"]["tmax"] = "0.03fs"
    config["save"]["t"] = {"tmin": "0.013fs", "tmax": "0.027fs", "nt": 9}
    prepared = prepare(config)
    legacy = legacy_setup(config)
    actual = run_jit(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(0))
    expected = legacy(None, None)["solver result"]
    assert_tree_close(actual.observations["state"], expected.ys)
    final = legacy.state
    for step in range(legacy.nt):
        final = legacy.diffeqsolve_quants["terms"].vf(legacy.tmin + step * legacy.grid.dt, final, legacy.args)
    assert_tree_close(actual.final_state, final)
    assert not np.allclose(actual.final_state["flm"], actual.observations["state"]["flm"][-1], rtol=1e-12, atol=1e-15)


def test_preparation_is_deterministic_and_runtime_arrays_are_explicit():
    config = small_config()
    original = deepcopy(config)
    first = prepare(config)
    second = prepare(config)
    assert config == original
    assert first.manifest.to_dict() == second.manifest.to_dict()
    assert "mlflow" not in first.manifest.raw_config
    assert isinstance(first.program, ScanProgram)
    assert first.capabilities.differentiable
    assert_tree_close(first.state, second.state)
    assert_tree_close(eqx.filter(first.program, eqx.is_array), eqx.filter(second.program, eqx.is_array))
    paths = [
        jax.tree_util.keystr(path)
        for path, value in jax.tree.flatten_with_path(first.program)[0]
        if eqx.is_array(value)
    ]
    assert any("vlasov" in path and ".kx" in path for path in paths)
    assert any("collisions" in path and ".grid.v" in path for path in paths)
    forbidden = ("mlflow", "pint", "xarray", "matplotlib", "adept.vfp2d.preparation", "adept._base_")
    assert not any(type(leaf).__module__.startswith(forbidden) for leaf in jax.tree.leaves(first.program))


def test_selected_heating_control_has_finite_difference_gradient():
    config = _config(collisions=True)
    config["grid"].update(lmax=1, mmax=1)
    config["drivers"]["maxwellian_heating"] = {"D0": 0.01}
    prepared = prepare(config)
    partition = partition_parameters(prepared.inputs, {key: key == "D0_heating" for key in prepared.inputs})
    objective = CallableObjective(lambda result, params, inputs: jnp.sum(result.final_state["flm"] ** 2))
    evaluated = eqx.filter_jit(value_and_grad)(
        prepared.program, objective, partition.trainable, prepared.state, partition.frozen, jax.random.key(0)
    )
    derivative = jnp.sum(evaluated.gradients["D0_heating"])
    assert jnp.isfinite(derivative) and jnp.abs(derivative) > 1e-10
    eps = 1e-5
    losses = []
    for delta in (-eps, eps):
        params = {**partition.trainable, "D0_heating": partition.trainable["D0_heating"] + delta}
        result = run_jit(prepared.program, params, prepared.state, partition.frozen, jax.random.key(0))
        losses.append(jnp.sum(result.final_state["flm"] ** 2))
    np.testing.assert_allclose(derivative, (losses[1] - losses[0]) / (2 * eps), rtol=2e-5)


def test_host_runtime_and_ergoexo_preserve_output_contract(tmp_path):
    config = _config(collisions=False)
    prepared = prepare(config)
    completed = run_prepared(prepared, key=jax.random.key(0))
    assert "vfp2d" in completed.report.result
    assert completed.report.metrics
    with LocalExecutor() as executor:
        planned = executor.execute(RunPlan(simulation=SimulationSpec.from_legacy_config(config), seed=42))
    assert_tree_close(planned.raw_result.final_state, completed.raw_result.final_state)
    exo = ergoExo()
    modules = exo._setup_(deepcopy(config), str(tmp_path), log=False)
    assert exo.execution_backend == "prepared"
    output = exo._execute_simulation(modules, None)
    assert output["solver result"].ts.shape == (3,)
    assert_tree_close(output["solver result"].ys, completed.raw_result.observations["state"])
    exo.adept_module.state = {**exo.adept_module.state, "e": exo.adept_module.state["e"] + 1e-8}
    exo._execute_simulation(modules, None)
    assert exo.execution_backend == "legacy"
    assert "state replacement" in exo.compatibility_fallback_reason


def test_analyzer_uses_executed_inputs_after_runtime_control_changes():
    from dataclasses import replace

    config = _config(collisions=False)
    config["terms"]["field_solver"] = {
        "mode": "kinetic-ohm",
        "hidden_density_gradient": {"active": True, "scale_length": "2um"},
    }
    prepared = prepare(config)
    inputs = {**prepared.inputs, "hidden_dndz": 2.0 * prepared.inputs["hidden_dndz"]}
    completed = run_prepared(replace(prepared, inputs=inputs), key=jax.random.key(42))
    dataset = completed.report.result["vfp2d"]
    reconstructed = sum(
        dataset[f"ohm_{name}"]
        for name in (
            "resistive",
            "hall",
            "nernst",
            "scalar_pressure",
            "tensor_pressure",
        )
    )
    np.testing.assert_allclose(reconstructed[-1], dataset.e[-1], rtol=3e-12, atol=3e-13)
    np.testing.assert_array_equal(completed.raw_result.stats["runtime_inputs"]["hidden_dndz"], inputs["hidden_dndz"])
