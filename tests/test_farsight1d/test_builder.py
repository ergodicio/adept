"""Explicit host API and validation coverage for the independent FARSIGHT backend."""

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
from adept.core.programs import ScanProgram
from adept.farsight1d.config import Farsight1DConfig


def small_config():
    return {
        "grid": {"nx": 4, "nv": 8, "xmin": 0.0, "xmax": 12.0, "vmin": -4.0, "vmax": 4.0},
        "time": {"tmin": 0.0, "tmax": 0.02, "dt": 0.01},
        "initial": {"kind": "maxwellian", "amplitude": 0.01},
        "numerical": {"epsilon": 0.1, "chunk_size": 8},
        "save": {
            "scalars": {"every_steps": 1},
            "fields": {"every_steps": 7},
            "distribution": {"every_steps": 7},
        },
    }


def prepare(config=None):
    return solver_registry.prepare(SimulationSpec("farsight-1d", config or small_config()), key=7)


def run(program, params, state, inputs, key):
    return program(params, state, inputs, key)


run_jit = eqx.filter_jit(run)


def test_preparation_is_deterministic_normalized_and_has_a_pure_program():
    config = small_config()
    original = deepcopy(config)
    first, second = prepare(config), prepare(config)
    assert config == original
    assert first.manifest.to_dict() == second.manifest.to_dict()
    assert isinstance(first.program, ScanProgram)
    assert first.capabilities.differentiable and not first.capabilities.batchable
    assert first.params == first.inputs == {}
    assert first.state["f"].shape == (5, 9)
    for left, right in zip(jax.tree.leaves(first.state), jax.tree.leaves(second.state), strict=True):
        np.testing.assert_array_equal(left, right)
    mass = jnp.sum(first.program.system.weights * first.state["f"])
    np.testing.assert_allclose(mass, 12.0, rtol=1e-14)
    forbidden = ("mlflow", "pint", "pydantic", "xarray", "matplotlib", "adept._base_")
    assert not any(type(leaf).__module__.startswith(forbidden) for leaf in jax.tree.leaves(first.program))


def test_final_observation_is_retained_when_cadence_does_not_divide_final_step():
    prepared = prepare()
    result = run_jit(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(7))
    assert result.status == diffrax.RESULTS.successful
    np.testing.assert_allclose(result.times["distribution"], [0.0, 0.02])
    np.testing.assert_allclose(result.times["fields"], [0.0, 0.02])
    for name in ("x", "v", "f"):
        np.testing.assert_array_equal(result.observations["distribution"][name][-1], result.final_state[name])
    report = prepared.analyzer.analyze(result.materialize(), prepared.manifest)
    assert report.result["fields"].electric_field.shape == (2, 4)
    assert report.result["distribution"].f.shape == (2, 5, 9)
    assert report.result["scalars"].sizes["t"] == 3
    assert report.metrics[0].values["final_remesh_count"] == 2
    assert report.artifacts == ()


def test_run_prepared_and_local_executor_use_the_registered_backend():
    prepared = prepare()
    completed = run_prepared(prepared, key=jax.random.key(7))
    assert completed.report.metrics[0].values["final_valid"] == 1.0
    with LocalExecutor() as executor:
        planned = executor.execute(RunPlan(SimulationSpec("farsight-1d", small_config()), seed=7))
    np.testing.assert_allclose(planned.report.result["scalars"].c2, completed.report.result["scalars"].c2)


def test_failed_state_has_failed_status_and_is_rejected_by_host_analysis():
    prepared = prepare()
    invalid = {**prepared.state, "valid": jnp.asarray(False)}
    result = run_jit(prepared.program, prepared.params, invalid, prepared.inputs, jax.random.key(7))
    assert result.status == diffrax.RESULTS.nonfinite
    with pytest.raises(ArithmeticError, match="invalid panel geometry or nonfinite"):
        prepared.analyzer.analyze(result, prepared.manifest)
    with pytest.raises(ArithmeticError, match="invalid panel geometry or nonfinite"):
        run_prepared(prepared, key=jax.random.key(7), execute=lambda _prepared, _key: result)


def test_host_analysis_rejects_nonfinite_observations_even_with_valid_geometry():
    prepared = prepare()
    result = run_jit(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(7))
    observations = deepcopy(result.observations)
    observations["scalars"]["c2"] = observations["scalars"]["c2"].at[-1].set(jnp.nan)
    with pytest.raises(ArithmeticError, match="contains nonfinite"):
        prepared.analyzer.analyze(result._replace(observations=observations), prepared.manifest)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("grid", "nx", 5),
        ("grid", "nv", 3),
        ("grid", "nv", True),
        ("grid", "vmax", float("inf")),
        ("time", "dt", 0.003),
        ("time", "dt", float("nan")),
        ("time", "tmax", 1e-15),
        ("numerical", "epsilon", 0.0),
        ("numerical", "remesh_every", -1),
        ("numerical", "quadrature", "unknown"),
        ("initial", "kind", "two-stream"),
        ("initial", "amplitude", 1.1),
        ("initial", "mode", 2),
        ("initial", "thermal_speed", 0.0),
        ("initial", "unsupported", 1),
        ("save", "unknown", {}),
    ],
)
def test_config_rejects_invalid_or_unsupported_controls(section, field, value):
    config = small_config()
    config[section][field] = value
    with pytest.raises(ValidationError):
        Farsight1DConfig.model_validate(config)


def test_no_remesh_and_optional_observations_are_explicit():
    config = small_config()
    config["numerical"]["remesh_every"] = 0
    config["save"].update(fields=None, distribution=None)
    prepared = prepare(config)
    assert prepared.program.system.remesh_every == 0
    assert prepared.program.observation_names == ("scalars",)


def test_builder_rejects_other_schemas_and_requires_x64():
    builder = solver_registry.resolve("farsight-1d")
    with pytest.raises(ValueError, match="schema version"):
        builder.prepare(SimulationSpec("farsight-1d", small_config(), schema_version="2"), key=7)
    with pytest.raises(ValueError, match="cannot prepare solver"):
        builder.prepare(SimulationSpec("pic-1d", small_config()), key=7)
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="requires float64"):
        builder.prepare(SimulationSpec("farsight-1d", small_config()), key=7)


def test_program_rejects_unsupported_runtime_controls():
    prepared = prepare()
    with pytest.raises(ValueError, match="requires empty params and inputs"):
        run_jit(prepared.program, {"unsupported": jnp.asarray(1.0)}, prepared.state, {}, jax.random.key(7))


@pytest.mark.parametrize("filename", ["landau-damping.yaml", "two-stream.yaml"])
def test_example_configs_validate_and_initialize(filename):
    path = Path(__file__).parents[2] / "configs" / "farsight-1d" / filename
    config = yaml.safe_load(path.read_text())
    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=7)
    grid = config["grid"]
    mass = jnp.sum(prepared.program.system.weights * prepared.state["f"])
    np.testing.assert_allclose(mass, grid["xmax"] - grid["xmin"], rtol=1e-14)
