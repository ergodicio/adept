"""Host integration and scientific metadata for opt-in AMR positivity limiting."""

from copy import deepcopy

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from adept import SimulationSpec, solver_registry
from adept.farsight1d import builder
from adept.farsight1d.__main__ import FileAnalyzer
from adept.farsight1d.config import Farsight1DConfig
from adept.farsight1d.positivity import bernstein_coefficients


def _config(limiter="bernstein"):
    return {
        "grid": {"nx": 4, "nv": 8, "xmax": 12.0, "vmin": -4.0, "vmax": 4.0},
        "time": {"tmax": 0.002, "dt": 0.001},
        "initial": {"kind": "maxwellian", "amplitude": 0.01},
        "numerical": {
            "epsilon": 0.1,
            "quadrature": "simpson",
            "positivity_limiter": limiter,
            "chunk_size": 8,
        },
        "amr": {"enabled": True, "max_level": 0, "max_panels": 10},
        "save": {
            "scalars": {"every_steps": 1},
            "fields": {"every_steps": 1},
            "distribution": {"every_steps": 1},
        },
    }


def _prepare(config):
    return solver_registry.prepare(SimulationSpec("farsight-1d", config), key=7)


@eqx.filter_jit
def _run(program, params, state, inputs):
    return program(params, state, inputs, jax.random.key(7))


def _execute(prepared, state=None):
    return _run(prepared.program, prepared.params, prepared.state if state is None else state, prepared.inputs)


@pytest.fixture(scope="module")
def limited_run():
    prepared = _prepare(_config())
    return prepared, _execute(prepared)


def test_limiter_defaults_to_none_and_accepts_only_declared_methods():
    config = _config()
    del config["numerical"]["positivity_limiter"]
    assert Farsight1DConfig.model_validate(config).numerical.positivity_limiter == "none"
    config["numerical"]["positivity_limiter"] = "clip"
    with pytest.raises(ValidationError, match="positivity_limiter"):
        Farsight1DConfig.model_validate(config)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("amr", "enabled", False, "requires amr.enabled=true"),
        ("numerical", "quadrature", "trapezoid", "requires numerical.quadrature='simpson'"),
    ],
)
def test_bernstein_requires_independent_amr_panels_and_simpson(section, field, value, message):
    config = _config()
    config[section][field] = value
    with pytest.raises(ValidationError, match=message):
        Farsight1DConfig.model_validate(config)


def test_explicit_none_preserves_default_initialization_and_state_shape():
    explicit = _config("none")
    implicit = deepcopy(explicit)
    del implicit["numerical"]["positivity_limiter"]
    first, second = _prepare(explicit), _prepare(implicit)
    assert first.program.system.positivity_limiter == "none"
    assert first.manifest.resolved_config["numerical"]["positivity_limiter"] == "none"
    assert "initial_positivity_mass_change" not in first.state
    for left, right in zip(jax.tree.leaves(first.state), jax.tree.leaves(second.state), strict=True):
        np.testing.assert_array_equal(left, right)


def test_initial_limiting_preserves_normalized_mass_and_reports_separate_defects(limited_run):
    limited, _ = limited_run
    original = _prepare(_config("none"))
    weights = np.asarray(limited.state["weights"])
    before, after = np.asarray(original.state["f"]), np.asarray(limited.state["f"])
    np.testing.assert_allclose(np.sum(weights * after), 12.0, rtol=2e-15)
    assert int(limited.state["initial_positivity_limited_panels"]) > 0
    assert int(limited.state["initial_positivity_failed_panels"]) == 0
    np.testing.assert_allclose(
        limited.state["initial_positivity_mass_change"], np.sum(weights * (after - before)), atol=2e-15
    )
    np.testing.assert_allclose(
        limited.state["initial_positivity_c2_change"], np.sum(weights * (after**2 - before**2)), atol=2e-15
    )
    assert float(limited.state["initial_positivity_c2_change"]) < 0
    assert float(limited.state["initial_positivity_polynomial_c2_change"]) < 0
    active = np.asarray(limited.state["active"])
    assert np.min(np.asarray(bernstein_coefficients(limited.state["f"]))[active]) >= -1e-15
    for stage in ("remap", "interpolation", "source_limiter", "regrid", "destination_limiter"):
        for moment in ("mass", "c2"):
            assert float(limited.state[f"{stage}_{moment}_change"]) == 0.0
    assert limited.program.system.positivity_limiter == "bernstein"
    assert limited.manifest.resolved_config["numerical"]["positivity_limiter"] == "bernstein"
    assert "without renormalization" in limited.manifest.units["positivity_limiter"]


def test_builder_limits_after_normalization_without_rewriting_helper_output(monkeypatch):
    initialize = builder.initialize_positivity
    captured = {}

    def capture(state):
        captured["incoming_mass"] = np.sum(np.asarray(state["weights"] * state["f"]))
        state = initialize(state)
        # A deliberate synthetic defect makes a forbidden post-limiter
        # normalization observable; this is not a numerical limiter substitute.
        state = {**state, "f": state["f"] * (1.0 - 1e-8)}
        captured["limited_f"] = np.asarray(state["f"])
        return state

    monkeypatch.setattr(builder, "initialize_positivity", capture)
    prepared = _prepare(_config())
    np.testing.assert_allclose(captured["incoming_mass"], 12.0, rtol=2e-15)
    np.testing.assert_array_equal(prepared.state["f"], captured["limited_f"])


def test_initial_limiter_invalidity_is_actionable_host_error(monkeypatch):
    initialize = builder.initialize_positivity

    def fail(state):
        return {
            **initialize(state),
            "valid": jnp.asarray(False),
            "initial_positivity_failed_panels": jnp.asarray(2, dtype=jnp.int32),
        }

    monkeypatch.setattr(builder, "initialize_positivity", fail)
    with pytest.raises(ValueError, match=r"Initial Bernstein positivity limiter failed on 2 panels.*panel means"):
        _prepare(_config())


def test_run_preserves_budget_identity_and_exposes_scientific_metadata(limited_run):
    prepared, result = limited_run
    assert result.status == diffrax.RESULTS.successful
    report = prepared.analyzer.analyze(result.materialize(), prepared.manifest)
    scalars = report.result["scalars"]
    for moment in ("mass", "c2"):
        parts = sum(
            scalars[f"{stage}_{moment}_change"]
            for stage in ("interpolation", "source_limiter", "regrid", "destination_limiter")
        )
        np.testing.assert_allclose(scalars[f"remap_{moment}_change"], parts, atol=1e-13, rtol=1e-12)
        initial = scalars[f"initial_positivity_{moment}_change"]
        np.testing.assert_array_equal(initial, np.full(initial.shape, initial[0]))
    np.testing.assert_allclose(scalars.c2, scalars.c2_positive + scalars.c2_negative, atol=1e-14)
    assert "Native nodal" in scalars.c2_negative.attrs["description"]
    assert "not a lower bound" in scalars.min_f.attrs["description"]
    assert "previous leaf layout" in scalars.source_limiter_mass_change.attrs["description"]
    assert "not a term" in scalars.destination_limiter_polynomial_c2_change.attrs["description"]
    assert "initial_positivity_c2_change" in scalars
    assert report.metrics[0].values["final_initial_positivity_c2_change"] == float(
        scalars.initial_positivity_c2_change[-1]
    )
    for dataset in report.result.values():
        assert dataset.attrs["positivity_limiter"] == "bernstein"
        assert "Not a conservative remap" in dataset.attrs["positivity_scope"]
        assert "independent panel traces" in dataset.attrs["positivity_scope"]


def test_invalid_limiter_result_remains_failed_and_analysis_explains_next_checks(limited_run):
    prepared, _ = limited_run
    invalid = {
        **prepared.state,
        "valid": jnp.asarray(False),
        "positivity_failed_panels": jnp.asarray(1, dtype=jnp.int32),
    }
    result = _execute(prepared, invalid)
    assert result.status == diffrax.RESULTS.nonfinite
    with pytest.raises(ArithmeticError, match=r"positivity_failed_panels.*negative panel mean"):
        prepared.analyzer.analyze(result, prepared.manifest)


def test_limiter_metadata_and_budgets_survive_saved_artifacts(limited_run, tmp_path):
    prepared, result = limited_run
    FileAnalyzer(prepared.analyzer, tmp_path).analyze(result.materialize(), prepared.manifest)
    with xr.open_dataset(tmp_path / "scalars.nc", engine="h5netcdf") as scalars:
        assert scalars.attrs["positivity_limiter"] == "bernstein"
        assert "not a term" in scalars.destination_limiter_polynomial_c2_change.attrs["description"]
        np.testing.assert_array_equal(
            scalars.initial_positivity_c2_change, result.observations["scalars"]["initial_positivity_c2_change"]
        )
    with xr.open_dataset(tmp_path / "distribution.nc", engine="h5netcdf") as distribution:
        assert "independent panel traces" in distribution.attrs["positivity_scope"]
        np.testing.assert_array_equal(distribution.active[-1], result.final_state["active"])
