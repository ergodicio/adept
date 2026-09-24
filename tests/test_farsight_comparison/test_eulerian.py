"""Small setup/numerical tests; these deliberately do not create MLflow research runs."""

from copy import deepcopy

import equinox as eqx
import jax
import numpy as np
import pytest

from adept._vlasov1d.solvers.pushers.vlasov import SpaceExponential, VelocityCubicSpline
from adept._vlasov1d.solvers.vector_field import StrangIntegrator
from adept.farsight1d.config import Farsight1DConfig
from adept.farsight1d.numerics import electric_field
from examples.farsight_comparison.cases import get_case
from examples.farsight_comparison.eulerian import (
    BenchmarkVlasov1D,
    build_eulerian_config,
    extract_eulerian,
    softened_fourier_multiplier,
)


def _prepare(name="two-stream", field_model="poisson", output_dir=None):
    case = get_case(name, nx=16, nv=64, tmax=0.15, dt=0.05, frame_dt=0.1)
    config = build_eulerian_config(case, field_model=field_model, output_dir=output_dir)
    module = BenchmarkVlasov1D(config)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return case, module


@pytest.mark.parametrize("name", ["two-stream", "nlepw"])
def test_case_has_exact_step_aligned_frames_and_valid_farsight_config(name):
    case = get_case(name, tmax=0.15, frame_dt=0.1)
    np.testing.assert_array_equal(case.frame_steps, [0, 2, 3])
    np.testing.assert_allclose(case.frame_times, [0, 0.1, 0.15], atol=2e-17)
    config = case.to_farsight_config()
    config.pop("solver")
    validated = Farsight1DConfig.model_validate(config)
    assert validated.time.num_steps == case.num_steps
    assert validated.initial.kind == ("two-stream" if name == "two-stream" else "maxwellian")


@pytest.mark.parametrize("overrides", [{"dt": 0.03}, {"frame_dt": 0.123}, {"nx": 7}, {"epsilon": 0.0}])
def test_case_rejects_unmatched_steps_or_invalid_grid(overrides):
    with pytest.raises(ValueError):
        get_case("two-stream", **overrides)


@pytest.mark.parametrize("name", ["two-stream", "nlepw"])
@pytest.mark.parametrize("field_model", ["poisson", "farsight-softened"])
def test_uniform_ions_cosine_initial_density_and_correct_initial_field(name, field_model):
    case, module = _prepare(name, field_model)
    grid = module.simulation.grid
    electron_grid = module.cfg["grid"]["species_grids"]["electron"]
    f = np.asarray(module.state["electron"])
    n = f.sum(axis=1) * electron_grid["dv"]
    np.testing.assert_allclose(module.cfg["grid"]["ion_charge"], 1.0, atol=0)
    np.testing.assert_allclose(n, 1 + case.amplitude * np.cos(case.wave_number * grid.x), atol=5e-15)
    assert abs(f.sum() * grid.dx * electron_grid["dv"] / case.length - 1) < 1e-14
    mu = module.field_multiplier[1]
    expected_e = -mu * case.amplitude / case.wave_number * np.sin(case.wave_number * grid.x)
    np.testing.assert_allclose(module.state["e"], expected_e, atol=5e-15)
    assert np.max(np.abs(module.state["e"])) > 1e-3


def test_requested_pushers_and_exact_time_bounds_are_retained():
    case, module = _prepare()
    integrator = module.diffeqsolve_quants["terms"].vector_field.vpfp.vlasov_poisson
    assert isinstance(integrator, StrangIntegrator)
    assert isinstance(integrator.vdfdx, SpaceExponential)
    assert isinstance(integrator.edfdv, VelocityCubicSpline)
    assert integrator.field_solve.es_field_solver is module.benchmark_field_solver
    assert module.time_quantities["t1"] == case.tmax
    assert module.simulation.grid.nt == case.num_steps + 1
    subs = module.diffeqsolve_quants["saveat"]["subs"]
    np.testing.assert_array_equal(subs["electron.main"].ts, case.frame_times)
    np.testing.assert_array_equal(subs["fields"].ts, case.frame_times)
    np.testing.assert_array_equal(subs["default"].ts, case.step_times)
    assert not module.cfg["terms"]["fokker_planck"]["is_on"]
    assert not module.cfg["terms"]["krook"]["is_on"]


def test_softened_multiplier_matches_direct_kernel_fourier_field_and_poisson_limit():
    case = get_case("nlepw")
    nx = 64
    mu = softened_fourier_multiplier(nx, case.length, case.epsilon)
    assert mu[0] == 0
    assert 0 < mu[1] < 1
    np.testing.assert_array_equal(mu[1 : nx // 2], mu[: nx // 2 : -1])
    dx = case.length / 2048
    sources = np.arange(2048) * dx
    targets = (np.arange(nx) + 0.31) * case.length / nx
    rho = -case.amplitude * np.cos(case.wave_number * sources)
    actual = electric_field(targets, sources, rho * dx, case.length, case.epsilon)
    expected = -case.amplitude * mu[1] / case.wave_number * np.sin(case.wave_number * targets)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=3e-12)
    small_epsilon = softened_fourier_multiplier(16, case.length, 1e-5)
    np.testing.assert_allclose(small_epsilon[1:], 1.0, atol=5e-8)


def test_explicit_softening_override_updates_logged_model():
    cfg = build_eulerian_config(get_case("nlepw", nx=16, nv=64, tmax=0.1))
    module = BenchmarkVlasov1D(deepcopy(cfg), epsilon=0.9)
    assert module.epsilon == 0.9
    assert module.field_model == "farsight-softened"
    assert module.cfg["benchmark"]["case"]["epsilon"] == 0.9


def test_small_solve_outputs_integrated_invariants_and_artifacts(tmp_path):
    case, module = _prepare("nlepw", "farsight-softened", output_dir=tmp_path / "local")
    # Explicit adept-run exception: three-step numerical unit test, no research tracking.
    raw = eqx.filter_jit(module)(None)
    data = extract_eulerian(raw, module)
    assert data["distribution"].f.dims == ("t", "x", "v")
    assert data["distribution"].sizes["t"] == 3
    assert data["scalars"].sizes["t"] == 4
    np.testing.assert_allclose(data["scalars"].mass[0], case.length, rtol=2e-15)
    f0 = np.asarray(module.state["electron"])
    weight = (case.length / case.nx) * ((case.vmax - case.vmin) / case.nv)
    np.testing.assert_allclose(data["scalars"].c2[0], weight * np.sum(f0**2), rtol=2e-15)
    np.testing.assert_allclose(data["distribution"].t, case.frame_times, atol=0)
    result = module.post_process(raw, tmp_path / "tracking")
    assert result["metrics"]["final_c2"] > 0
    for directory in (tmp_path / "local", tmp_path / "tracking" / "comparison"):
        assert (directory / "distribution.nc").exists()
        assert (directory / "scalars.nc").exists()
        assert (directory / "benchmark.json").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.post_process(raw, tmp_path / "tracking2")


def test_softened_field_solver_jits():
    _, module = _prepare("nlepw", "farsight-softened")
    solve = module.benchmark_field_solver
    actual = jax.jit(solve)({"electron": module.state["electron"]}, None, None)
    np.testing.assert_allclose(actual, module.state["e"], atol=1e-14)
