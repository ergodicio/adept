"""Legacy parity and transform-boundary tests for the explicit Vlasov-1D program."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

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
from adept._vlasov1d.modules import BaseVlasov1D
from adept.core.programs import ScanProgram


def small_config(mode="electrostatic"):
    config = yaml.safe_load((Path(__file__).parent / "configs" / "resonance.yaml").read_text())
    config["grid"].update(nx=8, nv=32, xmax=2 * np.pi, dt=0.125, tmax=0.3)
    config["save"] = {
        "fields": {"t": {"nt": 4}},
        "electron": {"raw": {"t": {"nt": 3}}},
    }
    config["terms"]["fokker_planck"]["is_on"] = False
    config["terms"]["krook"]["is_on"] = False
    component = config["density"]["species-background"]
    component.update(basis="sine", baseline=1.0, amplitude=0.01, wavenumber=1.0)
    driver = config["drivers"]["ex"]["0"]
    driver["params"].update(a0=0.03, k0=1.0, w0=1.0)
    driver["envelope"]["time"].update(center=0.0, rise=0.1, width=10.0)
    if mode == "collisions":
        component["m"] = 3.0
        config["terms"]["edfdv"] = "cubic-spline"
        config["terms"]["fokker_planck"]["is_on"] = True
        config["terms"]["fokker_planck"]["space"]["baseline"] = 0.02
        config["terms"]["fokker_planck"]["self_consistent_beta"] = {"enabled": True, "max_steps": 3}
        config["terms"]["krook"]["is_on"] = True
        config["terms"]["krook"]["space"]["baseline"] = 0.01
        config["diagnostics"] = {"diag-vlasov-dfdt": True, "diag-fp-dfdt": True}
        config["save"].update({name: {"t": {"nt": 4}} for name in config["diagnostics"]})
    elif mode == "multispecies":
        config["density"]["species-ion"] = {**component, "T0": 0.1}
        config["terms"]["time"] = "sixth"
        config["terms"]["species"] = [
            {
                "name": "electron",
                "charge": -1.0,
                "mass": 1.0,
                "vmax": 6.4,
                "nv": 32,
                "density_components": ["species-background"],
            },
            {
                "name": "ion",
                "charge": 1.0,
                "mass": 100.0,
                "vmax": 0.8,
                "nv": 24,
                "density_components": ["species-ion"],
            },
        ]
        config["save"]["ion"] = {"raw": {"t": {"nt": 4}}}
    elif mode in ("electromagnetic", "point-source"):
        config["drivers"]["ey"] = {"0": deepcopy(driver)}
        config["drivers"]["ey"]["0"]["params"].update(a0=0.01, k0=0.3, w0=2.0)
        if mode == "point-source":
            config["drivers"]["ey"]["0"]["source_type"] = "point"
            config["drivers"]["ey"]["0"]["envelope"]["space"]["center"] = np.pi
    elif mode == "stochastic":
        config["drivers"]["ex_stochastic"] = {"modes": [1, 2], "amplitude": 0.03, "tau": 0.2, "seed": 17}
    elif mode == "strang":
        config["terms"]["time"] = "strang"
    elif mode == "hampere":
        config["terms"]["field"] = "hampere"
    elif mode == "nonbinary":
        config["grid"].update(dt=0.1, tmax=1.0)
    return config


def prepare(config):
    return solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=42)


def legacy_setup(config):
    module = BaseVlasov1D(deepcopy(config))
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module


def assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=3e-11, atol=3e-13)


@eqx.filter_jit
def run(program, params, state, inputs, key):
    return program(params, state, inputs, key)


@pytest.mark.parametrize(
    "mode",
    [
        "electrostatic",
        "collisions",
        "multispecies",
        "electromagnetic",
        "point-source",
        "stochastic",
        "strang",
        "hampere",
        "nonbinary",
    ],
)
def test_prepared_matches_legacy_observations_and_complete_final_state(mode):
    config = small_config(mode)
    prepared = prepare(config)
    legacy = legacy_setup(config)
    assert_tree_close(prepared.state, legacy.state)
    actual = run(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(42))
    expected = legacy(None, None)["solver result"]
    assert_tree_close(actual.observations, expected.ys)
    assert_tree_close(actual.times, expected.ts)
    final = legacy.state
    for step in range(legacy.simulation.grid.nt):
        final = legacy.diffeqsolve_quants["terms"].vf(step * legacy.simulation.grid.dt, final, legacy.args)
    assert_tree_close(actual.final_state, final)
    assert int(actual.stats["num_steps"]) == legacy.simulation.grid.nt
    assert int(expected.stats["num_accepted_steps"]) == legacy.simulation.grid.nt
    if mode == "collisions":
        for name in config["diagnostics"]:
            assert np.max(np.abs(actual.observations[name][1:])) > 0.0
    elif mode in ("electromagnetic", "point-source"):
        assert np.max(np.abs(actual.final_state["a"])) > 0.0


def test_off_grid_saves_interpolate_and_do_not_replace_final_state():
    config = small_config()
    config["save"] = {
        "fields": {"t": {"tmin": 0.03, "tmax": 0.29, "nt": 9}},
        "electron": {"early": {"t": {"tmin": 0.01, "tmax": 0.24, "nt": 7}}},
    }
    prepared = prepare(config)
    legacy = legacy_setup(config)
    result = run(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(42))
    expected = legacy(None, None)["solver result"]
    assert_tree_close(result.observations, expected.ys)
    assert_tree_close(result.times, expected.ts)
    final = legacy.state
    for step in range(legacy.simulation.grid.nt):
        final = legacy.diffeqsolve_quants["terms"].vf(step * legacy.simulation.grid.dt, final, legacy.args)
    assert_tree_close(result.final_state, final)
    assert not np.allclose(result.final_state["electron"], result.observations["electron.early"][-1], atol=1e-12)


def test_multiple_distribution_saves_preserve_interpolation():
    config = small_config()
    config["save"]["electron"]["sampled"] = {
        "t": {"nt": 2},
        "x": {"xmin": 0.5, "xmax": 5.5, "nx": 4},
        "v": {"vmin": -3.0, "vmax": 3.0, "nv": 12},
    }
    prepared = prepare(config)
    legacy = legacy_setup(config)
    result = run(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(42))
    expected = legacy(None, None)["solver result"]
    assert_tree_close(result.observations, expected.ys)
    assert result.observations["electron.raw"].shape == (3, 8, 32)
    assert result.observations["electron.sampled"].shape == (2, 4, 12)


def test_fourier_distribution_save_has_the_known_mode_amplitude_and_coordinates():
    config = small_config()
    config["save"]["electron"]["spectrum"] = {
        "t": {"nt": 2},
        "kx": {"kxmin": 0.0, "kxmax": 4.0, "nkx": 5},
        # These endpoints are the centers of the full velocity-grid cells.
        "v": {"vmin": -6.2, "vmax": 6.2, "nv": 32},
    }
    prepared = prepare(config)
    completed = run_prepared(prepared, key=jax.random.key(42))
    spectrum = completed.raw_result.observations["electron.spectrum"]
    # f(x, v) = (1 + 0.01 sin(x)) * f_bar(v), on a box of length 2 pi.
    # The unnormalized real FFT therefore has amplitudes N and 0.01 N / 2
    # in modes zero and one, with all other resolved modes initially zero.
    f_bar = np.asarray(prepared.state["electron"]).mean(axis=0)
    expected = np.zeros((5, 32))
    expected[0] = 8 * f_bar
    expected[1] = 0.01 * 8 / 2 * f_bar
    np.testing.assert_allclose(spectrum[0], expected, rtol=2e-12, atol=2e-15)
    assert spectrum.shape == (2, 5, 32)
    dataset = completed.report.result["dists"]["electron.spectrum"]
    assert dataset["electron.spectrum"].dims == ("t", "kx", "v_electron")
    np.testing.assert_allclose(dataset.kx, np.arange(5))
    np.testing.assert_allclose(dataset.v_electron, np.linspace(-6.2, 6.2, 32))
    np.testing.assert_allclose(dataset.t, completed.raw_result.times["electron.spectrum"])


@pytest.mark.parametrize("spatial_dim", ["kx", "x"])
def test_ergoexo_exports_distribution_plots_and_netcdf(spatial_dim, tmp_path, monkeypatch):
    import mlflow
    import xarray

    config = small_config()
    spatial_bounds = (
        {"kxmin": 0.0, "kxmax": 4.0, "nkx": 5} if spatial_dim == "kx" else {"xmin": 0.5, "xmax": 5.5, "nx": 5}
    )
    config["save"]["electron"] = {
        "sampled": {
            # Four times exercise the initial snapshot and difference panels.
            "t": {"nt": 4},
            spatial_dim: spatial_bounds,
            "v": {"vmin": -6.2, "vmax": 6.2, "nv": 32},
        }
    }
    config["mlflow"] = {"experiment": "vlasov1d-distribution-export", "run": spatial_dim}
    original_tracking_uri = mlflow.get_tracking_uri()
    # Isolate both MLflow's process state and its on-disk tracking/artifact store.
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    try:
        exo = ergoExo()
        modules = exo.setup(config)
        output, datasets, run_id = exo(modules)
        assert exo.execution_backend == "prepared"
        dataset = datasets["dists"]["electron.sampled"]
        assert dataset["electron.sampled"].dims == ("t", spatial_dim, "v_electron")
        expected_spatial = np.arange(5) if spatial_dim == "kx" else np.linspace(1.0, 5.0, 5)
        np.testing.assert_allclose(dataset[spatial_dim], expected_spatial)
        np.testing.assert_allclose(dataset.v_electron, np.linspace(-6.2, 6.2, 32))
        np.testing.assert_allclose(dataset.t, output["solver result"].ts["electron.sampled"])

        # Read from the run's artifact destination after ergoExo removes its
        # temporary export directory, proving that export reached MLflow.
        client = mlflow.tracking.MlflowClient()
        artifact_root = Path(client.download_artifacts(run_id, ""))
        with xarray.open_dataset(artifact_root / "binary" / "dist-electron.sampled.nc") as exported:
            xarray.testing.assert_identical(exported, dataset)
        plot = artifact_root / "plots" / "dists" / "electron.sampled" / "phase_space.png"
        assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)


def test_explicit_null_save_bounds_default_to_the_resolved_grid():
    config = small_config()
    config["grid"]["tmin"] = 0.05
    for time_config in (config["save"]["fields"]["t"], config["save"]["electron"]["raw"]["t"]):
        time_config.update(tmin=None, tmax=None)
    prepared = prepare(config)
    result = run(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(42))
    for name, count in (("fields", 4), ("electron.raw", 3)):
        np.testing.assert_allclose(result.times[name], np.linspace(0.05, 0.375, count))
    assert config["save"]["fields"]["t"]["tmin"] is None
    assert config["save"]["electron"]["raw"]["t"]["tmax"] is None


def test_builder_import_does_not_load_tracking_or_plotting_facades():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import adept._vlasov1d.builder; "
            "forbidden = {'mlflow', 'adept._base_', 'matplotlib.pyplot'}; "
            "loaded = forbidden.intersection(sys.modules); assert not loaded, loaded",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("bounds", [("tmin", -0.01), ("tmax", 1.0)])
def test_preparation_rejects_observations_outside_the_solve_interval(bounds):
    config = small_config()
    name, value = bounds
    config["save"]["fields"]["t"][name] = value
    with pytest.raises(ValueError, match="outside the simulation interval"):
        prepare(config)


def test_builder_rejects_incompatible_schema_and_precision():
    builder = solver_registry.resolve("vlasov-1d")
    spec = SimulationSpec.from_legacy_config(small_config())
    with pytest.raises(ValueError, match="schema version"):
        builder.prepare(SimulationSpec("vlasov-1d", spec.config_dict(), schema_version="2"), key=42)
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="requires float64"):
        builder.prepare(spec, key=42)


def test_preparation_is_deterministic_and_runtime_arrays_are_explicit():
    config = small_config("collisions")
    original = deepcopy(config)
    first, second = prepare(config), prepare(config)
    assert config == original
    assert eqx.tree_equal(first.manifest.to_dict(), second.manifest.to_dict())
    assert "mlflow" not in first.manifest.raw_config
    assert isinstance(first.program, ScanProgram)
    assert first.capabilities.differentiable
    assert_tree_close(first.state, second.state)
    assert_tree_close(eqx.filter(first.program, eqx.is_array), eqx.filter(second.program, eqx.is_array))
    forbidden = ("mlflow", "pint", "pydantic", "xarray", "matplotlib", "adept._base_")
    leaves = jax.tree.leaves(first.program)
    assert not any(type(leaf).__module__.startswith(forbidden) for leaf in leaves)
    # Opaque legacy operators hide their arrays from transformations. Every leaf
    # should be a numerical runtime value or a small immutable static value.
    assert all(eqx.is_array(leaf) or isinstance(leaf, (str, bool, int, float, complex)) for leaf in leaves)
    assert eqx.is_inexact_array(first.inputs["drivers"].ex[0].a0)


def test_selected_driver_amplitude_has_finite_difference_gradient():
    prepared = prepare(small_config())
    selection = jax.tree.map(lambda _: False, prepared.inputs)
    selection = eqx.tree_at(lambda tree: tree["drivers"].ex[0].a0, selection, True)
    partition = partition_parameters(prepared.inputs, selection)
    objective = CallableObjective(lambda result, params, inputs: jnp.sum(result.final_state["e"] ** 2))
    evaluated = eqx.filter_jit(value_and_grad)(
        prepared.program, objective, partition.trainable, prepared.state, partition.frozen, jax.random.key(42)
    )
    derivative = evaluated.gradients["drivers"].ex[0].a0
    assert jnp.isfinite(derivative) and jnp.abs(derivative) > 1e-8
    eps = 1e-5
    losses = []
    for delta in (-eps, eps):
        params = eqx.tree_at(
            lambda tree: tree["drivers"].ex[0].a0,
            partition.trainable,
            partition.trainable["drivers"].ex[0].a0 + delta,
        )
        result = run(prepared.program, params, prepared.state, partition.frozen, jax.random.key(42))
        losses.append(jnp.sum(result.final_state["e"] ** 2))
    np.testing.assert_allclose(derivative, (losses[1] - losses[0]) / (2 * eps), rtol=2e-7)


@pytest.mark.parametrize("mode", ["point-source", "stochastic"])
def test_runtime_driver_changes_match_rebuilding_the_configuration(mode):
    config = small_config(mode)
    prepared = prepare(config)
    if mode == "point-source":
        driver = config["drivers"]["ey"]["0"]
        driver["params"].update(a0=0.017, w0=3.0)
        driver["envelope"]["space"]["center"] = 1.1
        inputs = eqx.tree_at(
            lambda tree: (
                tree["drivers"].ey[0].a0,
                tree["drivers"].ey[0].w0,
                tree["drivers"].ey[0].envelope.space_envelope.center,
            ),
            prepared.inputs,
            tuple(jnp.asarray(value) for value in (0.017, 3.0, 1.1)),
        )
    else:
        config["drivers"]["ex_stochastic"]["amplitude"] *= 2
        inputs = eqx.tree_at(
            lambda tree: (tree["drivers"].ex_stochastic.amp_real, tree["drivers"].ex_stochastic.amp_imag),
            prepared.inputs,
            (
                2 * prepared.inputs["drivers"].ex_stochastic.amp_real,
                2 * prepared.inputs["drivers"].ex_stochastic.amp_imag,
            ),
        )
    actual = run(prepared.program, prepared.params, prepared.state, inputs, jax.random.key(42))
    expected = legacy_setup(config)(None, None)["solver result"]
    assert_tree_close(actual.observations, expected.ys)


def test_host_analysis_and_executor_do_not_write_or_start_tracking(tmp_path, monkeypatch):
    import mlflow

    def forbidden(*args, **kwargs):
        pytest.fail("pure preparation and analysis must not start tracking or write artifacts")

    monkeypatch.chdir(tmp_path)
    for name in ("start_run", "log_metrics", "log_artifacts"):
        monkeypatch.setattr(mlflow, name, forbidden)
    config = small_config()
    prepared = prepare(config)
    completed = run_prepared(prepared, key=jax.random.key(42))
    assert {"fields", "dists", "scalars"} <= completed.report.result.keys()
    assert "electron.raw" in completed.report.result["dists"]
    with LocalExecutor() as executor:
        planned = executor.execute(RunPlan(simulation=SimulationSpec.from_legacy_config(config), seed=42))
    assert_tree_close(planned.raw_result.final_state, completed.raw_result.final_state)
    assert list(tmp_path.iterdir()) == []


def test_ergoexo_uses_prepared_backend_and_falls_back_for_state_replacement(tmp_path):
    config = small_config()
    prepared = prepare(config)
    expected = run(prepared.program, prepared.params, prepared.state, prepared.inputs, jax.random.key(42))
    exo = ergoExo()
    modules = exo._setup_(deepcopy(config), str(tmp_path), log=False)
    assert exo.execution_backend == "prepared"
    output = exo._execute_simulation(modules, None)
    assert_tree_close(output["solver result"].ys, expected.observations)
    assert_tree_close(output["solver result"].ts, expected.times)
    exo.adept_module.state = {**exo.adept_module.state, "e": exo.adept_module.state["e"] + 1e-8}
    exo._execute_simulation(modules, None)
    assert exo.execution_backend == "legacy"
    assert "state replacement" in exo.compatibility_fallback_reason


def test_ergoexo_fallback_honors_disabled_collision_terms(tmp_path):
    config = small_config("collisions")
    exo = ergoExo()
    modules = exo._setup_(deepcopy(config), str(tmp_path), log=False)
    assert exo.execution_backend == "prepared"
    for term in ("fokker_planck", "krook"):
        exo.adept_module.args["terms"][term]["is_on"] = False
        config["terms"][term]["is_on"] = False
    output = exo._execute_simulation(modules, None)
    expected = legacy_setup(config)(None, None)["solver result"]
    assert exo.execution_backend == "legacy"
    assert_tree_close(output["solver result"].ys, expected.ys)
    np.testing.assert_array_equal(output["solver result"].ys["diag-fp-dfdt"], 0.0)
