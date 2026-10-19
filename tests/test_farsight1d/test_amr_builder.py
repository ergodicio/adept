"""Adaptive FARSIGHT preparation, observations, and explicit failure contracts."""

from copy import deepcopy
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError

from adept import LocalExecutor, RunPlan, SimulationSpec, run_prepared, solver_registry
from adept.farsight1d.__main__ import FileAnalyzer
from adept.farsight1d.builder import FarsightFieldsObservation
from adept.farsight1d.config import Farsight1DConfig
from adept.farsight1d.numerics import electric_field
from tests.test_farsight1d.test_builder import prepare, run_jit, small_config


def adaptive_config(**amr):
    config = small_config()
    config["amr"] = {"enabled": True, "max_level": 1, "max_panels": 32, "atol": 0.05, **amr}
    return config


def execute(prepared):
    return run_jit(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(7))


def test_disabled_amr_preserves_fixed_panel_execution():
    default = prepare()
    config = small_config()
    config["amr"] = {"enabled": False}
    disabled = prepare(config)
    for left, right in zip(jax.tree.leaves(execute(default)), jax.tree.leaves(execute(disabled)), strict=True):
        np.testing.assert_array_equal(left, right)
    assert default.manifest.resolved_config["amr"]["enabled"] is False


@pytest.mark.parametrize("quadrature", ["trapezoid", "simpson"])
def test_zero_level_adaptive_panels_match_fixed_scalar_and_field_observations(quadrature):
    config = small_config()
    config["numerical"]["quadrature"] = quadrature
    fixed = execute(prepare(config))
    config["amr"] = {"enabled": True, "max_level": 0, "max_panels": 12}
    adaptive = execute(prepare(config))
    assert adaptive.status == diffrax.RESULTS.successful
    for name in fixed.observations["scalars"]:
        np.testing.assert_allclose(
            adaptive.observations["scalars"][name], fixed.observations["scalars"][name], atol=2e-12, rtol=2e-12
        )
    np.testing.assert_allclose(
        adaptive.observations["fields"]["electric_field"],
        fixed.observations["fields"]["electric_field"],
        atol=2e-12,
        rtol=2e-12,
    )


def test_adaptive_initialization_is_deterministic_normalized_and_packed():
    config = adaptive_config(max_panels=40)
    original = deepcopy(config)
    first, second = prepare(config), prepare(config)
    assert config == original
    assert first.state["f"].shape == (40, 9)
    assert first.state["active"].shape == (40,)
    for left, right in zip(jax.tree.leaves(first.state), jax.tree.leaves(second.state), strict=True):
        np.testing.assert_array_equal(left, right)
    active = np.asarray(first.state["active"])
    assert np.count_nonzero(active) == int(first.state["active_panels"])
    assert 8 < np.count_nonzero(active) <= 32
    assert np.all(np.asarray(first.state["weights"])[~active] == 0.0)
    assert np.any(np.asarray(first.state["level"])[active] == 1)
    np.testing.assert_allclose(jnp.sum(first.state["weights"] * first.state["f"]), 12.0, rtol=1e-14)
    assert first.capabilities.differentiable and not first.capabilities.batchable
    assert "piecewise" in first.manifest.units["amr_gradients"]


def test_field_observation_uses_current_adaptive_quadrature_weights():
    prepared = prepare(adaptive_config())
    observation = FarsightFieldsObservation(
        jnp.array([0.3, 2.1]), jnp.zeros_like(prepared.state["weights"]), 12.0, 0.1, 8
    )
    state = {**prepared.state, "weights": 1.7 * prepared.state["weights"]}
    actual = observation(0.0, state, {})["electric_field"]
    expected = electric_field(observation.x, state["x"], -state["weights"] * state["f"], 12.0, 0.1, 8)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert float(jnp.max(jnp.abs(actual))) > 0


def test_adaptive_observations_and_analyzer_preserve_masks_and_panel_dimensions():
    prepared = prepare(adaptive_config(max_panels=40))
    result = execute(prepared)
    assert result.status == diffrax.RESULTS.successful
    report = prepared.analyzer.analyze(result.materialize(), prepared.manifest)
    distribution = report.result["distribution"]
    assert distribution.f.dims == ("t", "panel", "node")
    assert distribution.f.shape == (2, 40, 9)
    assert distribution.active.dims == distribution.panel_id.dims == distribution.level.dims == ("t", "panel")
    for name in ("x", "v", "f", "weights", "active", "panel_id", "level"):
        np.testing.assert_array_equal(distribution[name][-1], result.final_state[name])
    assert "slot" in distribution.attrs["description"]
    scalars = report.result["scalars"]
    assert scalars.active_panels.dims == ("t",)
    np.testing.assert_allclose((distribution.weights * distribution.f).sum(("panel", "node")), scalars.mass[[0, -1]])
    assert not bool(scalars.capacity_exceeded.any())
    assert report.metrics[0].values["final_active_panels"] == int(result.final_state["active_panels"])


def test_adaptive_run_prepared_and_local_executor_use_registered_backend():
    config = adaptive_config(max_level=0, max_panels=8)
    completed = run_prepared(prepare(config), key=jax.random.key(7))
    with LocalExecutor() as executor:
        planned = executor.execute(RunPlan(SimulationSpec("farsight-1d", config), seed=7))
    np.testing.assert_allclose(planned.report.result["scalars"].c2, completed.report.result["scalars"].c2)
    assert completed.report.metrics[0].values["final_valid"] == 1.0


def test_adaptive_artifacts_preserve_panel_identity_and_weights(tmp_path):
    prepared = prepare(adaptive_config(max_level=0, max_panels=12))
    result = execute(prepared).materialize()
    report = FileAnalyzer(prepared.analyzer, tmp_path).analyze(result, prepared.manifest)
    assert {Path(artifact.source).name for artifact in report.artifacts} == {
        "distribution.nc",
        "fields.nc",
        "scalars.nc",
        "final_state.npz",
        "manifest.json",
        "metrics.json",
    }
    with np.load(tmp_path / "final_state.npz", allow_pickle=False) as final_state:
        assert final_state["active"].dtype == np.dtype("bool")
        with xr.open_dataset(tmp_path / "distribution.nc", engine="h5netcdf") as distribution:
            assert distribution.active.dtype == np.dtype("int8")
            assert distribution.f.dims == ("t", "panel", "node")
            for name in ("x", "v", "f", "weights", "active", "panel_id", "level"):
                np.testing.assert_array_equal(distribution[name][-1], final_state[name])
        with xr.open_dataset(tmp_path / "scalars.nc", engine="h5netcdf") as scalars:
            assert bool(scalars.valid.all())
            assert not bool(scalars.capacity_exceeded.any())
            np.testing.assert_allclose(scalars.mass[-1], np.sum(final_state["weights"] * final_state["f"]), rtol=1e-14)


def test_initial_refinement_capacity_exhaustion_is_a_host_error():
    with pytest.raises(ValueError, match=r"Initial AMR hierarchy requests.*increase amr.max_panels"):
        prepare(adaptive_config(max_panels=8, atol=0.0, rtol=0.0))


def test_invalid_adaptive_state_reports_failure_and_actionable_capacity_message():
    prepared = prepare(adaptive_config(max_level=0, max_panels=8))
    state = {**prepared.state, "valid": jnp.asarray(False), "capacity_exceeded": jnp.asarray(True)}
    result = run_jit(prepared.program, {}, state, {}, jax.random.key(7))
    assert result.status == diffrax.RESULTS.nonfinite
    with pytest.raises(ArithmeticError, match=r"capacity.*increase amr.max_panels"):
        prepared.analyzer.analyze(result, prepared.manifest)


@pytest.mark.parametrize(
    "amr",
    [
        {"max_level": -1},
        {"max_level": 5},
        {"min_level": 2, "max_level": 1},
        {"max_panels": 0},
        {"max_panels": 7},
        {"min_level": 1, "max_panels": 31},
        {"atol": -1.0},
        {"rtol": float("inf")},
        {"atol": float("nan")},
        {"max_gap_fraction": 0.0},
        {"max_gap_fraction": float("inf")},
        {"enabled": 1},
        {"unknown": 1},
    ],
)
def test_invalid_amr_controls_are_rejected(amr):
    with pytest.raises(ValidationError):
        Farsight1DConfig.model_validate(adaptive_config(**amr))


def test_candidate_scratch_limit_is_validated_before_arrays_are_built():
    config = adaptive_config(max_level=4, max_panels=1024)
    config["grid"].update(nx=32, nv=32)
    with pytest.raises(ValidationError, match=r"candidate panels.*32768.*reduce grid"):
        Farsight1DConfig.model_validate(config)


def test_adaptive_example_validates_and_initializes():
    path = Path(__file__).parents[2] / "configs" / "farsight-1d" / "two-stream-amr.yaml"
    config = yaml.safe_load(path.read_text())
    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=7)
    assert bool(prepared.state["valid"])
    grid = config["grid"]
    np.testing.assert_allclose(
        jnp.sum(prepared.state["weights"] * prepared.state["f"]), grid["xmax"] - grid["xmin"], rtol=1e-14
    )
