"""Native solves retain ADEPT's callbacks and Python outputs at actual save times."""

import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
import yaml
from diffrax import SubSaveAt

from adept import ergoExo
from adept._vlasov1d.storage import store_f, store_fields
from adept.vlasov1d import BaseVlasov1D, NativeVlasov1D, native_backend


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    configured = os.environ.get("ADEPT_NATIVE_TEST_LIBRARY")
    if configured:
        return Path(configured)
    if not all(shutil.which(name) for name in ["cmake", "mpicxx", "pkg-config"]):
        pytest.skip("native integration tests require CMake, MPI and FFTW or ADEPT_NATIVE_TEST_LIBRARY")
    if subprocess.run(["pkg-config", "--exists", "fftw3"]).returncode:
        pytest.skip("FFTW validation backend is not installed")
    build = tmp_path_factory.mktemp("native-build")
    source = Path(__file__).resolve().parents[2] / "adept/_vlasov1d/native"
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DVLASOV_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cmake", "--build", str(build), "-j2"], check=True, capture_output=True)
    return next(path for path in [build / "libadept_vlasov.so", build / "libadept_vlasov.dylib"] if path.exists())


@pytest.fixture
def config(tmp_path):
    cfg = yaml.safe_load(Path("tests/test_base/configs/example.yaml").read_text())
    cfg["grid"].update(nx=32, nv=64, dt=0.02, tmax=0.08, vmax=8.0)
    cfg["drivers"] = {"ex": {}, "ey": {}}
    cfg["terms"]["time"] = "sixth"
    cfg["terms"]["fokker_planck"].update(is_on=True, type="chang_cooper_dougherty")
    cfg["terms"]["fokker_planck"]["time"]["baseline"] = 0.001
    cfg["terms"]["fokker_planck"]["space"]["baseline"] = 1.0
    cfg["terms"]["krook"]["is_on"] = False
    cfg["save"] = {
        "fields": {"t": {"nt": 5}},
        "electron": {
            "main": {"t": {"nt": 3}},
            "sampled": {
                "t": {"nt": 4},
                "x": {"xmin": 0.0, "xmax": 20.94, "nx": 8},
                "v": {"vmin": -7.0, "vmax": 7.0, "nv": 16},
            },
        },
    }
    beam = cfg["density"]["species-background"]
    beam.update(v0=3.0, T0=0.2, noise_val=0.001)
    cfg["density"]["species-left"] = deepcopy(beam)
    cfg["density"]["species-left"].update(v0=-3.0, noise_seed=43)
    cfg["native"] = {
        "device": os.environ.get("ADEPT_NATIVE_TEST_DEVICE", "cpu"),
        "workdir": str(tmp_path),
        "timeout": 120,
    }
    return cfg


def setup(cls, config):
    module = cls(deepcopy(config))
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module


def compare_results(original, native):
    assert set(original.ts) == set(native.ts)
    for key in original.ts:
        np.testing.assert_allclose(original.ts[key], native.ts[key], atol=1e-14)
        assert jax.tree.structure(original.ys[key]) == jax.tree.structure(native.ys[key])
        for a, b in zip(jax.tree.leaves(original.ys[key]), jax.tree.leaves(native.ys[key]), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11, equal_nan=True, err_msg=key)


def reference_at_native_times(original, native):
    subs = original.diffeqsolve_quants["saveat"]["subs"]
    for key, save in native.diffeqsolve_quants["saveat"]["subs"].items():
        subs[key] = SubSaveAt(ts=save.ts, fn=subs[key].fn)
    return original({})["solver result"]


@pytest.mark.parametrize("ranks", [1, 2])
def test_native_dense_saves_and_xarray(config, native_library, tmp_path, ranks):
    config["native"]["library"] = str(native_library)
    config["native"]["ranks"] = ranks
    original, native = (setup(cls, config) for cls in [BaseVlasov1D, NativeVlasov1D])
    a, b = reference_at_native_times(original, native), native({})["solver result"]
    compare_results(a, b)
    for key, times in b.ts.items():
        np.testing.assert_allclose(np.asarray(times) / 0.02, np.rint(np.asarray(times) / 0.02), atol=1e-14)
    assert len(b.ts["default"]) == 6
    output = tmp_path / "output"
    (output / "binary").mkdir(parents=True)
    fields = store_fields(native.cfg, str(output / "binary"), b.ys["fields"], b.ts["fields"], "fields")
    dists = store_f(native.cfg, b.ts, str(output), b.ys)
    assert isinstance(fields["electron"], xr.Dataset)
    assert dists["electron.sampled"].sizes == {"t": 4, "x": 8, "v_electron": 16}
    with xr.open_dataset(output / "binary/dist-electron.sampled.nc") as saved:
        xr.testing.assert_allclose(saved, dists["electron.sampled"])


def test_native_step_schedules(config):
    config["native"].update(scalar_stride=3, custom_save_times="steps")
    native = setup(native_backend(CustomTwoStream), config)
    subs = native.diffeqsolve_quants["saveat"]["subs"]
    np.testing.assert_allclose(subs["default"].ts, [0.0, 0.06, 0.1])
    np.testing.assert_allclose(subs["f0"].ts, [0.0, 0.04, 0.1])


@pytest.mark.parametrize("axis", ["time", "space"])
def test_native_rejects_omitted_collision_pulse_default(config, axis):
    config["terms"]["fokker_planck"][axis].pop("bump_height")
    with pytest.raises(NotImplementedError, match="constant frequency"):
        NativeVlasov1D(config)


def test_native_constant_collision_defaults(config):
    native = NativeVlasov1D(deepcopy(config))
    assert native.native_collision_frequency == 0.001
    config["terms"]["fokker_planck"]["time"].pop("baseline")
    assert NativeVlasov1D(config).native_collision_frequency == 0.0


class CustomTwoStream(BaseVlasov1D):
    def init_diffeqsolve(self):
        super().init_diffeqsolve()
        self.diffeqsolve_quants["saveat"]["subs"]["f0"] = SubSaveAt(
            ts=np.array([0.0, 0.037, self.time_quantities["t1"]]), fn=lambda t, y, args: jnp.mean(y["electron"], axis=0)
        )
        k = jnp.abs(jnp.asarray(self.cfg["grid"]["kx"]))
        self.diffeqsolve_quants["saveat"]["subs"]["peak"] = SubSaveAt(
            ts=np.array([0.0, 0.04, self.time_quantities["t1"]]),
            fn=lambda t, y, args: k[jnp.argmax(jnp.abs(jnp.fft.fft(y["e"])))],
        )

    def post_process(self, run_output, td):
        sol = run_output["solver result"]
        result = xr.Dataset(
            {"f0": (("t", "v"), sol.ys["f0"])},
            coords={"t": sol.ts["f0"], "v": self.cfg["grid"]["species_grids"]["electron"]["v"]},
        )
        result.to_netcdf(Path(td) / "custom.nc")
        return {"custom": result}


def test_ergoexo_custom_module_preserves_callbacks(config, native_library, tmp_path, monkeypatch):
    from adept import patched_mlflow as mlflow

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    config["native"]["library"] = str(native_library)
    original = setup(CustomTwoStream, config)
    config["backend"] = "native"
    config["mlflow"] = {"experiment": "native-integration-test", "run": "two-stream"}
    exo = ergoExo()
    modules = exo.setup(config, adept_module=CustomTwoStream)
    result, post, run_id = exo(modules)
    compare_results(reference_at_native_times(original, exo.adept_module), result["solver result"])
    np.testing.assert_allclose(result["solver result"].ts["f0"], [0.0, 0.037, 0.1])
    assert isinstance(exo.adept_module, CustomTwoStream)
    assert not exo.adept_module.jit_compile
    assert post["custom"].sizes == {"t": 3, "v": 64}
    artifacts = Path(mlflow.get_run(run_id).info.artifact_uri.removeprefix("file://"))
    assert (artifacts / "custom.nc").exists()
    assert (artifacts / "native/request.json").exists()
    assert (artifacts / "native/library.sha256").exists()


def test_backend_dispatch_and_guards(config):
    assert type(ergoExo()._get_adept_module_(config)) is BaseVlasov1D
    config["backend"] = "native"
    assert isinstance(ergoExo()._get_adept_module_(config), NativeVlasov1D)
    config["terms"]["fokker_planck"]["self_consistent_beta"] = {"enabled": True}
    with pytest.raises(NotImplementedError, match="analytic beta"):
        NativeVlasov1D(config)
    with pytest.raises(TypeError, match="BaseVlasov1D"):
        native_backend(object)
    from adept import SimulationSpec, solver_registry

    with pytest.raises(NotImplementedError, match="ergoExo"):
        solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=0)
    config["terms"]["fokker_planck"]["self_consistent_beta"] = None
    config["solver"] = "vlasov-1d-iaw"
    assert not ergoExo()._get_adept_module_(config).jit_compile


@pytest.mark.parametrize("ranks,collisions,mass,charge", [(1, False, 25.0, 1.0), (2, True, 1836.0, 2.0)])
def test_native_kinetic_ions(config, native_library, tmp_path, ranks, collisions, mass, charge):
    """Ion charge feeds back on electrons; all species move on independent grids."""
    config["native"].update(library=str(native_library), ranks=ranks)
    config["terms"]["fokker_planck"]["is_on"] = collisions
    electron = deepcopy(config["density"]["species-background"])
    electron.update(v0=0.0, T0=1.0, noise_val=0.0)
    ion = deepcopy(electron)
    ion.update(T0=0.1, basis="sine", baseline=0.8 / charge, amplitude=0.03, wavenumber=2 * np.pi / 20.94)
    alpha = deepcopy(electron)
    alpha.update(T0=0.2, **{"space-profile": {**electron["space-profile"], "baseline": 0.1}})
    config["density"] = {
        "quasineutrality": True,
        "species-electrons": electron,
        "species-ions": ion,
        "species-alpha": alpha,
    }
    # Put the electron last to exercise reference-species selection, rather than
    # accidentally applying its collision operator to the first configured ion.
    config["terms"]["species"] = [
        dict(
            name="ion",
            charge=charge,
            mass=mass,
            nv=32,
            vmax=6 * np.sqrt(0.1 / mass),
            density_components=["species-ions"],
        ),
        dict(
            name="alpha",
            charge=2.0,
            mass=4 * mass,
            nv=48,
            vmax=6 * np.sqrt(0.2 / (4 * mass)),
            density_components=["species-alpha"],
        ),
        dict(name="electron", charge=-1.0, mass=1.0, nv=64, vmax=8.0, density_components=["species-electrons"]),
    ]
    config["save"].update(ion={"main": {"t": {"nt": 3}}}, alpha={"main": {"t": {"nt": 3}}})
    original, native = (setup(cls, config) for cls in [BaseVlasov1D, NativeVlasov1D])
    a, output = reference_at_native_times(original, native), native({})
    b = output["solver result"]
    compare_results(a, b)
    for species in ["electron", "ion", "alpha"]:
        values = b.ys[f"{species}.main"]
        assert np.max(np.abs(values[-1] - values[0])) > 1e-9, species
    assert np.max(np.abs(b.ys["fields"]["e"][-1])) > 1e-4
    assert {s["name"] for s in output["native stats"]["species"]} == {"electron", "ion", "alpha"}
    directory = tmp_path / "multispecies"
    (directory / "binary").mkdir(parents=True)
    fields = store_fields(native.cfg, str(directory / "binary"), b.ys["fields"], b.ts["fields"], "fields")
    dists = store_f(native.cfg, b.ts, str(directory), b.ys)
    assert set(fields) >= {"electron", "ion", "alpha"}
    for species, nv in [("ion", 32), ("alpha", 48)]:
        with xr.open_dataset(directory / f"binary/dist-{species}.main.nc") as saved:
            xr.testing.assert_allclose(saved, dists[f"{species}.main"])
            assert saved.sizes[f"v_{species}"] == nv


@pytest.fixture(scope="module")
def conservative_benchmark(tmp_path_factory):
    """Exercise the canonical C++/CUDA kernels through the standalone harness."""
    if not all(shutil.which(name) for name in ["cmake", "mpicxx", "pkg-config"]):
        pytest.skip("native spline tests require CMake, MPI and FFTW")
    if subprocess.run(["pkg-config", "--exists", "fftw3"]).returncode:
        pytest.skip("FFTW validation backend is not installed")
    build = tmp_path_factory.mktemp("conservative-build")
    source = Path(__file__).resolve().parents[2] / "benchmarks/vlasov_cuda"
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DVLASOV_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cmake", "--build", str(build), "-j2"], check=True, capture_output=True)
    return build / "vlasov_bench"


def spline_translate(executable, tmp_path, edges, initial, shift, *, ranks=1, steps=1, mode="accelerate"):
    import json

    nv, nx = initial.shape
    paths = {key: tmp_path / f"{key}.bin" for key in ["edges", "initial", "final"]}
    edges.astype(np.float64).tofile(paths["edges"])
    initial.astype(np.float64).tofile(paths["initial"])
    command = [
        str(executable),
        "--nx",
        str(nx),
        "--nv",
        str(nv),
        "--cells",
        "--velocity",
        "conservative",
        "--layout",
        "transpose",
        "--mode",
        mode,
        "--dt",
        "1",
        "--steps",
        str(steps),
        "--warmup",
        "0",
        "--acceleration",
        str(shift),
        "--input",
        str(paths["initial"]),
        "--v-edges",
        str(paths["edges"]),
        "--dump",
        str(paths["final"]),
        "--host-mpi",
    ]
    if ranks > 1:
        command = ["mpiexec", "-n", str(ranks), *command]
    run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return np.fromfile(paths["final"], dtype=np.float64).reshape(initial.shape), json.loads(run.stdout)


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("shift", [0.0, 0.137, -0.281, 21.137, -21.281, 7.0])
def test_conservative_spline_matches_periodic_scipy(conservative_benchmark, tmp_path, ranks, shift):
    from scipy.interpolate import CubicSpline

    # Non-unit period, asymmetric domain, arbitrary abrupt changes in width.
    rng = np.random.default_rng(723)
    widths = rng.uniform(0.01, 1, 32)
    edges = np.r_[0, np.cumsum(widths / widths.sum() * 7)] - 2
    h = np.diff(edges)
    initial = rng.uniform(0.2, 1.2, (32, 8))
    mean = np.sum(initial * h[:, None], axis=0) / (edges[-1] - edges[0])
    primitive = np.vstack([np.zeros(8), np.cumsum((initial - mean) * h[:, None], axis=0)])
    primitive[-1] = 0
    interpolator = CubicSpline(edges, primitive, axis=0, bc_type="periodic")
    shifted = interpolator(edges - shift, extrapolate="periodic")
    expected = np.diff(shifted, axis=0) / h[:, None] + mean
    actual, stats = spline_translate(conservative_benchmark, tmp_path, edges, initial, shift, ranks=ranks)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.sum(actual * h[:, None], axis=0), np.sum(initial * h[:, None], axis=0), atol=2e-14)
    assert abs(stats["mass_relative_change"]) < 1e-14


def test_conservative_spline_convergence_and_repeated_mass(conservative_benchmark, tmp_path):
    errors = []
    for nv in [32, 64, 128]:
        # Smooth nonuniform mesh; exact cell averages of a translated sinusoid.
        s = np.linspace(0, 1, nv + 1)
        edges = 4 * (s + 0.2 * np.sin(2 * np.pi * s) / (2 * np.pi)) - 1.5
        h = np.diff(edges)
        k, shift = 2 * np.pi / 4, 0.271

        def averages(displacement, edges=edges, k=k, h=h):
            return 1 + 0.2 * (np.sin(k * (edges[1:] - displacement)) - np.sin(k * (edges[:-1] - displacement))) / (
                k * h
            )

        initial = np.repeat(averages(0)[:, None], 8, axis=1)
        actual, _ = spline_translate(conservative_benchmark, tmp_path, edges, initial, shift)
        errors.append(np.sqrt(np.sum((actual[:, 0] - averages(shift)) ** 2 * h)))
    # A cubic primitive gives third-order cell averages on a nonuniform grid.
    assert errors[0] / errors[1] > 6, errors
    assert errors[1] / errors[2] > 6, errors
    actual, stats = spline_translate(conservative_benchmark, tmp_path, edges, initial, -0.317, steps=200)
    np.testing.assert_allclose(np.sum(actual * h[:, None], axis=0), np.sum(initial * h[:, None], axis=0), atol=1e-12)
    assert abs(stats["mass_relative_change"]) < 1e-12
    constant = np.full_like(initial, 2.3)
    actual, _ = spline_translate(conservative_benchmark, tmp_path, edges, constant, 123.75, steps=20)
    np.testing.assert_allclose(actual, constant, rtol=0, atol=2e-13)


def test_native_velocity_grid_construction():
    from types import SimpleNamespace

    from adept._vlasov1d.datamodel import NativeVelocityGridConfig
    from adept._vlasov1d.native.velocity_grid import velocity_edges

    species = SimpleNamespace(nv=512, vmin=-6, vmax=6)
    cfg = NativeVelocityGridConfig(refined_min=1.2, refined_max=1.6, ratio=32)
    edges = velocity_edges(species, cfg)
    assert len(edges) == 513
    assert edges[0] == -6 and edges[-1] == 6
    h = np.diff(edges)
    np.testing.assert_allclose(h.max() / h.min(), 32, rtol=1e-11)
    fine = np.flatnonzero(h < h.max() / 2)
    assert abs(edges[fine[0]] - 1.2) <= h.max()
    assert abs(edges[fine[-1] + 1] - 1.6) <= 2 * h.max()
    np.testing.assert_array_equal(velocity_edges(species, NativeVelocityGridConfig(edges=edges.tolist())), edges)
    for invalid in [edges[:-1], edges[::-1], np.r_[edges[:-1], np.nan], edges + 1]:
        with pytest.raises(ValueError):
            velocity_edges(species, NativeVelocityGridConfig(edges=invalid.tolist()))
    with pytest.raises(ValueError, match="inside"):
        velocity_edges(species, NativeVelocityGridConfig(refined_min=-7, refined_max=1, ratio=4))
    with pytest.raises(ValueError, match="unresolved"):
        velocity_edges(SimpleNamespace(nv=4, vmin=-6, vmax=6), cfg)


@pytest.mark.parametrize("ranks", [1, 2])
def test_native_nonuniform_solve_and_saved_moments(config, native_library, tmp_path, ranks):
    config["native"].update(
        library=str(native_library),
        ranks=ranks,
        velocity_grids={"electron": dict(refined_min=1.0, refined_max=4.0, ratio=4)},
    )
    config["terms"]["edfdv"] = "conservative-cubic-spline"
    config["terms"]["fokker_planck"]["is_on"] = False
    native = setup(NativeVlasov1D, config)
    grid = native.cfg["grid"]["species_grids"]["electron"]
    edges, h = np.asarray(grid["v_edges"]), np.asarray(grid["dv"])
    np.testing.assert_allclose(
        np.sum(native.state["electron"] * h, axis=1), native.cfg["grid"]["n_prof_total"], atol=2e-14
    )
    result = native({})["solver result"]
    density = result.ys["default"]["mean_n_electron"]
    np.testing.assert_allclose(density, density[0], rtol=0, atol=2e-13)
    frames = result.ys["electron.main"]
    expected_pressure = np.mean(np.sum(frames[-1] * np.asarray(grid["v"]) ** 2 * h, axis=-1))
    np.testing.assert_allclose(result.ys["default"]["mean_P_electron"][-1], expected_pressure, atol=1e-13)
    np.testing.assert_allclose(result.ys["fields"]["electron"]["n"][-1], np.sum(frames[-1] * h, axis=-1), atol=1e-13)
    assert np.max(np.abs(frames[-1] - frames[0])) > 1e-8
    output = tmp_path / "nonuniform-output"
    (output / "binary").mkdir(parents=True)
    store_f(native.cfg, result.ts, str(output), result.ys)
    with xr.open_dataset(output / "binary/dist-electron.main.nc") as saved:
        np.testing.assert_array_equal(saved.v_electron_bounds[:, 0], edges[:-1])
        np.testing.assert_array_equal(saved.v_electron_bounds[:, 1], edges[1:])
        assert saved["electron.main"].attrs["velocity_representation"] == "cell_average"
    if ranks == 2:
        config["native"]["ranks"] = 1
        serial = setup(NativeVlasov1D, config)({})["solver result"]
        compare_results(serial, result)


def test_native_conservative_configuration_guards(config):
    config["native"]["velocity_grids"] = {"electron": dict(refined_min=0, refined_max=1, ratio=2)}
    with pytest.raises(ValueError, match=r"requires terms\.edfdv"):
        NativeVlasov1D(deepcopy(config))
    config["terms"]["edfdv"] = "conservative-cubic-spline"
    with pytest.raises(NotImplementedError, match="collisions"):
        NativeVlasov1D(deepcopy(config))
    config["terms"]["fokker_planck"]["is_on"] = False
    config["native"]["velocity_grids"]["typo"] = config["native"]["velocity_grids"]["electron"]
    with pytest.raises(ValueError, match="unknown species"):
        NativeVlasov1D(deepcopy(config))


@pytest.mark.parametrize("ranks", [1, 2])
def test_nonuniform_streaming_and_poisson(conservative_benchmark, tmp_path, ranks):
    edges = np.r_[np.linspace(-3, -1, 9), np.linspace(-1, 2, 25)[1:]]
    h, v = np.diff(edges), 0.5 * (edges[:-1] + edges[1:])
    nx, length, amplitude = 16, 4 * np.pi, 0.2
    k = 2 * np.pi / length
    x = np.arange(nx) * length / nx
    profile = np.exp(-(v**2))
    initial = profile[:, None] * (1 + amplitude * np.cos(k * x))
    actual, stats = spline_translate(conservative_benchmark, tmp_path, edges, initial, 0, ranks=ranks, mode="stream")
    expected = profile[:, None] * (1 + amplitude * np.cos(k * (x[None, :] - v[:, None])))
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    density = np.sum(profile * h)
    expected_energy = length * (amplitude * density / k) ** 2 / 4
    np.testing.assert_allclose(stats["initial_field_energy"], expected_energy, rtol=2e-14)


def test_native_conservative_multispecies(config, native_library):
    config["native"].update(library=str(native_library), ranks=2)
    config["terms"]["edfdv"] = "conservative-cubic-spline"
    config["terms"]["fokker_planck"]["is_on"] = False
    config["terms"]["species"] = [
        dict(name="electron", charge=-1, mass=1, nv=64, vmin=-7, vmax=9, density_components=["species-background"]),
        dict(name="ion", charge=1, mass=25, nv=32, vmax=2, density_components=["species-left"]),
    ]
    config["density"]["species-left"].update(v0=0, T0=0.5, noise_val=0.002)
    # Explicit nonuniform electron cells; omitted ion grid defaults to uniform.
    s = np.linspace(0, 1, 65)
    edges = -7 + 16 * (s + 0.3 * np.sin(2 * np.pi * s) / (2 * np.pi))
    config["native"]["velocity_grids"] = {"electron": {"edges": edges.tolist()}}
    config["save"]["ion"] = {"main": {"t": {"nt": 3}}}
    native = setup(NativeVlasov1D, config)
    result = native({})["solver result"]
    for name in ["electron", "ion"]:
        grid = native.cfg["grid"]["species_grids"][name]
        h = np.asarray(grid["dv"])
        frames = result.ys[f"{name}.main"]
        masses = np.mean(np.sum(frames * h, axis=-1), axis=-1)
        np.testing.assert_allclose(masses, masses[0], rtol=0, atol=2e-13)
        np.testing.assert_allclose(result.ys["fields"][name]["n"][-1], np.sum(frames[-1] * h, axis=-1), atol=2e-14)
        assert np.max(np.abs(frames[-1] - frames[0])) > 1e-8
    assert np.max(np.abs(result.ys["fields"]["e"][-1])) > 1e-5
    config["native"]["ranks"] = 1
    compare_results(setup(NativeVlasov1D, config)({})["solver result"], result)


@pytest.mark.parametrize("order", [2.0, 4.0])
def test_conservative_initial_cell_averages(order):
    from itertools import pairwise

    from scipy.integrate import quad
    from scipy.special import gamma

    from adept._vlasov1d.helpers import _initialize_supergaussian_distribution_

    edges = np.array([-4, -1.5, -0.5, 0.0, 0.1, 0.3, 0.7, 1.5, 4.0])
    width = np.sqrt(3 * gamma(3 / order) / gamma(5 / order)) * np.sqrt(0.8 / 2)
    masses = np.array([quad(lambda v: np.exp(-(abs((v - 0.2) / width) ** order)), a, b)[0] for a, b in pairwise(edges)])
    densities = np.linspace(0.5, 1.5, 8)
    actual, centers = _initialize_supergaussian_distribution_(
        8,
        8,
        v0=0.2,
        T0=0.8,
        mass=2,
        supergaussian_order=order,
        n_prof=densities,
        velocity_edges=edges,
    )
    expected = densities[:, None] * (masses / masses.sum() / np.diff(edges))[None, :]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(centers, (edges[:-1] + edges[1:]) / 2)


@pytest.mark.parametrize("invalid", ["duplicate", "nan", "short"])
def test_cpp_rejects_invalid_velocity_edges(conservative_benchmark, tmp_path, invalid):
    edges = np.linspace(-6, 6, 33)
    if invalid == "duplicate":
        edges[10] = edges[9]
    elif invalid == "nan":
        edges[-1] = np.nan
    else:
        edges = edges[:-1]
    with pytest.raises(subprocess.CalledProcessError) as error:
        spline_translate(conservative_benchmark, tmp_path, edges, np.ones((32, 8)), 0.1)
    assert "velocity edges" in error.value.stderr or "wrong binary size" in error.value.stderr
