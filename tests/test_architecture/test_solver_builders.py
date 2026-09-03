from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from adept.core import (
    CallableObjective,
    DirectoryArtifactSink,
    ExecutionKind,
    NullTracker,
    Report,
    SimulationSpec,
    partition_parameters,
    run_prepared,
    solver_registry,
)


def run_program(program, params, state, inputs, key):
    return program(params, state, inputs, key)


def assert_trees_allclose(actual, expected, *, rtol=1e-6, atol=1e-7):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=rtol, atol=atol)


def assert_transform_boundary_is_numerical(prepared) -> None:
    forbidden_modules = ("mlflow", "pint", "pydantic", "xarray", "matplotlib")
    transformed_values = (prepared.program, prepared.params, prepared.state, prepared.inputs)
    leaves = jax.tree.leaves(transformed_values)

    assert any(eqx.is_array(leaf) for leaf in jax.tree.leaves(prepared.program))
    assert not [leaf for leaf in leaves if type(leaf).__module__.split(".", maxsplit=1)[0] in forbidden_modules]


def tf1d_config() -> dict:
    path = Path(__file__).parents[1] / "test_tf1d" / "configs" / "resonance.yaml"
    with path.open() as config_file:
        config = yaml.safe_load(config_file)
    config["grid"].update(nx=8, xmax=4.0, tmax=0.1)
    config["save"]["t"].update(tmax=0.125, nt=6)
    config["save"]["x"].update(xmax=4.0, nx=8)
    config["save"].pop("kx")
    config["drivers"]["ex"]["0"].update(
        a0=0.01,
        k0=np.pi / 2.0,
        w0=1.0,
        t_c=0.05,
        t_w=0.1,
        t_r=0.01,
        x_c=2.0,
        x_w=4.0,
        x_r=0.1,
    )
    return config


def pic1d_config(*, loading: str = "quiet") -> dict:
    return {
        "units": {"normalizing_temperature": "1eV", "normalizing_density": "1e21/cc"},
        "density": {
            "quasineutrality": True,
            "species-background": {
                "noise_seed": 42,
                "noise_type": "gaussian",
                "noise_val": 0.0,
                "v0": 0.0,
                "T0": 1.0,
                "m": 2.0,
                "basis": "uniform",
                "baseline": 1.0,
            },
        },
        "grid": {
            "dt": 0.05,
            "nx": 8,
            "tmin": 0.0,
            "tmax": 0.1,
            "xmin": 0.0,
            "xmax": 2.0 * np.pi,
            "ppc": 8,
            "particle_shape": "tsc",
        },
        "save": {"fields": {"t": {"tmin": 0.0, "tmax": 0.1, "nt": 3}}},
        "solver": "pic-1d",
        "mlflow": {"experiment": "unused", "run": "unused"},
        "drivers": {"ex": {}, "ey": {}},
        "diagnostics": {},
        "terms": {
            "field": "poisson",
            "time": "leapfrog",
            "species": [
                {
                    "name": "electron",
                    "charge": -1.0,
                    "mass": 1.0,
                    "density_components": ["species-background"],
                    "loading": loading,
                    "vmax_load": 8.0,
                }
            ],
        },
    }


def add_pic1d_ex_driver(config: dict) -> None:
    config["drivers"]["ex"]["drive"] = {
        "params": {"a0": 0.01, "k0": 1.0, "w0": 1.0, "dw0": 0.0},
        "source_type": "extended",
        "envelope": {
            "time": {"center": 0.05, "rise": 0.01, "width": 0.1},
            "space": {"center": np.pi, "rise": 0.1, "width": 2.0 * np.pi},
        },
    }


def test_builtin_builders_remain_lazy_until_resolved():
    assert {"tf-1d", "pic-1d"}.issubset(solver_registry.names())


def test_tf1d_builder_matches_the_legacy_continuous_path():
    from adept.tf1d import BaseTwoFluid1D

    config = tf1d_config()
    original = deepcopy(config)
    legacy = BaseTwoFluid1D(deepcopy(config))
    legacy.write_units()
    legacy.get_derived_quantities()
    legacy.get_solver_quantities()
    legacy.init_state_and_args()
    legacy.init_diffeqsolve()
    legacy_result = legacy({}, legacy.args)["solver result"]

    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=0)
    result = eqx.filter_jit(run_program)(
        prepared.program,
        prepared.params,
        prepared.state,
        prepared.inputs,
        jax.random.key(0),
    )

    assert config == original
    assert prepared.capabilities.execution_kind is ExecutionKind.CONTINUOUS
    assert_transform_boundary_is_numerical(prepared)
    assert_trees_allclose(result.final_state, jax.tree.map(lambda leaf: leaf[-1], legacy_result.ys["x"]))
    assert_trees_allclose(result.observations, legacy_result.ys)
    np.testing.assert_allclose(result.times, legacy_result.ts)
    assert not np.allclose(result.final_state["electron"]["u"], prepared.state["electron"]["u"])


def test_tf1d_objective_value_and_gradient_match_the_legacy_path():
    from adept import value_and_grad
    from adept.tf1d import BaseTwoFluid1D

    config = tf1d_config()
    legacy = BaseTwoFluid1D(deepcopy(config))
    legacy.write_units()
    legacy.get_derived_quantities()
    legacy.get_solver_quantities()
    legacy.init_state_and_args()
    legacy.init_diffeqsolve()

    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=0)
    runtime_values = eqx.combine(prepared.params, prepared.inputs)
    selector = jax.tree.map(lambda _: False, runtime_values)
    selector = eqx.tree_at(lambda tree: tree["drivers"]["ex"]["0"]["a0"], selector, True)
    partition = partition_parameters(runtime_values, selector)
    objective = CallableObjective(
        lambda result, params, inputs: jnp.mean(result.observations["x"]["electron"]["u"][-1] ** 2),
        metric_name="electron_flow_energy",
    )

    new_run = eqx.filter_jit(value_and_grad)(
        prepared.program,
        objective,
        partition.trainable,
        prepared.state,
        partition.frozen,
        jax.random.key(0),
    )

    def legacy_loss(amplitude):
        args = eqx.tree_at(lambda tree: tree["drivers"]["ex"]["0"]["a0"], legacy.args, amplitude)
        result = legacy({}, args)["solver result"]
        return jnp.mean(result.ys["x"]["electron"]["u"][-1] ** 2)

    amplitude = runtime_values["drivers"]["ex"]["0"]["a0"]
    legacy_value, legacy_gradient = eqx.filter_jit(eqx.filter_value_and_grad(legacy_loss))(amplitude)
    new_gradient = new_run.gradients["drivers"]["ex"]["0"]["a0"]

    np.testing.assert_allclose(new_run.objective.loss, legacy_value, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(new_gradient, legacy_gradient, rtol=2e-5, atol=1e-9)
    assert eqx.tree_equal(eqx.combine(partition.trainable, partition.frozen), runtime_values)


def test_pic1d_builder_matches_the_legacy_discrete_map():
    from adept.pic1d import BasePIC1D

    config = pic1d_config()
    original = deepcopy(config)
    legacy = BasePIC1D(deepcopy(config))
    legacy.write_units()
    legacy.get_derived_quantities()
    legacy.get_solver_quantities()
    legacy.init_state_and_args()
    legacy.init_diffeqsolve()
    legacy_map = legacy.diffeqsolve_quants["terms"].vf
    expected_state = deepcopy(legacy.state)
    for step in range(legacy.simulation.grid.nt):
        expected_state = legacy_map(step * legacy.simulation.grid.dt, expected_state, legacy.args)

    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=42)
    result = eqx.filter_jit(run_program)(
        prepared.program,
        prepared.params,
        prepared.state,
        prepared.inputs,
        jax.random.key(42),
    )

    assert config == original
    assert prepared.capabilities.execution_kind is ExecutionKind.DISCRETE
    assert_transform_boundary_is_numerical(prepared)
    assert_trees_allclose(prepared.state, legacy.state)
    assert_trees_allclose(result.final_state, expected_state)


def test_pic1d_prepared_run_completes_without_mlflow(tmp_path):
    prepared = solver_registry.prepare(
        SimulationSpec.from_legacy_config(pic1d_config()),
        key=42,
    )

    result = run_prepared(
        prepared,
        key=jax.random.key(42),
        tracker=NullTracker(),
        artifact_sink=DirectoryArtifactSink(tmp_path / "artifacts"),
    )

    assert isinstance(result.report, Report)
    assert result.report.result is result.raw_result
    assert result.handle.backend == "null"
    assert result.tracking_errors == ()
    assert_trees_allclose(result.raw_result.final_state, result.report.result.final_state)


def test_pic1d_preparation_is_seeded_and_structurally_reproducible():
    config = pic1d_config(loading="random")
    spec = SimulationSpec.from_legacy_config(config)

    first = solver_registry.prepare(spec, key=jax.random.key(12))
    repeated = solver_registry.prepare(spec, key=jax.random.key(12))
    changed = solver_registry.prepare(spec, key=jax.random.key(13))

    assert first.manifest.seed == 12
    assert first.manifest.key_provenance == "jax-key:00000000:0000000c"
    assert first.manifest.structural_fingerprint == repeated.manifest.structural_fingerprint
    assert_trees_allclose(first.state, repeated.state)
    assert not np.allclose(first.state["x_electron"], changed.state["x_electron"])


def test_pic1d_seed_streams_are_unique_for_variable_component_counts():
    from adept._pic1d.helpers import _particle_rng

    first_component = _particle_rng(
        configured_seed=42,
        seed=12,
        species_index=0,
        subspecies_index=1,
    ).random(16)
    second_species = _particle_rng(
        configured_seed=42,
        seed=12,
        species_index=1,
        subspecies_index=0,
    ).random(16)
    repeated = _particle_rng(
        configured_seed=99,
        seed=12,
        species_index=0,
        subspecies_index=1,
    ).random(16)
    legacy_primary = np.random.default_rng(12).random(16)
    primary = _particle_rng(
        configured_seed=99,
        seed=12,
        species_index=0,
        subspecies_index=0,
    ).random(16)

    assert not np.array_equal(first_component, second_species)
    np.testing.assert_array_equal(first_component, repeated)
    np.testing.assert_array_equal(primary, legacy_primary)


def test_pic1d_runtime_driver_is_not_a_stale_program_constant():
    config = pic1d_config()
    add_pic1d_ex_driver(config)
    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=42)
    changed_inputs = eqx.tree_at(
        lambda inputs: inputs["drivers"].ex[0].a0,
        prepared.inputs,
        jnp.array(0.02),
    )
    compiled = eqx.filter_jit(run_program)

    baseline = compiled(
        prepared.program,
        prepared.params,
        prepared.state,
        prepared.inputs,
        jax.random.key(42),
    )
    changed = compiled(
        prepared.program,
        prepared.params,
        prepared.state,
        changed_inputs,
        jax.random.key(42),
    )

    assert not np.allclose(baseline.final_state["v_electron"], changed.final_state["v_electron"])
    np.testing.assert_allclose(prepared.inputs["drivers"].ex[0].a0, 0.01)


def test_pic1d_builder_rejects_transverse_drivers_with_legacy_fallback():
    config = pic1d_config()
    config["drivers"]["ey"] = {
        "laser": {
            "params": {"a0": 0.1, "k0": 1.0, "w0": 1.0, "dw0": 0.0},
            "source_type": "extended",
            "envelope": {
                "time": {"center": 0.05, "rise": 0.01, "width": 0.1},
                "space": {"center": np.pi, "rise": 0.1, "width": 2.0 * np.pi},
            },
        }
    }

    with pytest.raises(ValueError, match="legacy ergoExo path"):
        solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=0)


def test_tf1d_builder_rejects_learned_trapping_with_legacy_fallback():
    config = tf1d_config()
    config["physics"]["electron"]["trapping"]["is_on"] = True

    with pytest.raises(ValueError, match="legacy ergoExo path"):
        solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=0)
