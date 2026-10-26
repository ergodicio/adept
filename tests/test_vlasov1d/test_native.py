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
