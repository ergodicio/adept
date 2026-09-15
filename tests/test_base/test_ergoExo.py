from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from diffrax import Solution

from adept import ergoExo


def _tf1d_config() -> dict:
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


def _pic1d_config() -> dict:
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
                    "loading": "quiet",
                    "vmax_load": 8.0,
                }
            ],
        },
    }


def _setup_without_logging(exo: ergoExo, config: dict, tmp_path: Path) -> dict:
    return exo._setup_(deepcopy(config), str(tmp_path), log=False)


def test_reuse_config_dict():
    with open("tests/test_base/configs/example.yaml") as file:
        cfg = yaml.safe_load(file)
    original = deepcopy(cfg)

    exo = ergoExo()
    exo.setup(cfg)
    assert cfg == original
    assert exo.execution_backend == "legacy"
    assert "not yet enabled" in exo.compatibility_fallback_reason

    exo = ergoExo()
    exo.setup(cfg)
    assert cfg == original
    assert exo.execution_backend == "legacy"


def test_tf1d_forward_run_uses_prepared_compatibility_path(tmp_path):
    exo = ergoExo()
    modules = _setup_without_logging(exo, _tf1d_config(), tmp_path)

    assert exo.execution_backend == "prepared"
    assert exo.compatibility_fallback_reason is None

    output = exo._execute_simulation(modules, None)
    solution = output["solver result"]

    assert isinstance(solution, Solution)
    assert solution.ts.shape == (6,)
    assert set(solution.ys) == {"x"}
    assert exo.execution_backend == "prepared"


def test_pic1d_forward_run_preserves_named_solution_shape(tmp_path):
    exo = ergoExo()
    modules = _setup_without_logging(exo, _pic1d_config(), tmp_path)

    output = exo._execute_simulation(modules, None)
    solution = output["solver result"]

    assert isinstance(solution, Solution)
    assert set(solution.ts) == {"fields", "default"}
    assert set(solution.ys) == {"fields", "default"}
    assert set(solution.stats) == {"max_steps", "num_accepted_steps", "num_rejected_steps", "num_steps"}
    assert exo.execution_backend == "prepared"


def test_replaced_legacy_state_falls_back_before_execution(tmp_path):
    exo = ergoExo()
    modules = _setup_without_logging(exo, _tf1d_config(), tmp_path)
    exo.adept_module.state = deepcopy(exo.adept_module.state)

    output = exo._execute_simulation(modules, None)

    assert isinstance(output["solver result"], Solution)
    assert exo.execution_backend == "legacy"
    assert exo.compatibility_fallback_reason == "legacy state replacement requires the legacy execution path"


def test_unsupported_pic1d_feature_selects_legacy_fallback(tmp_path):
    config = _pic1d_config()
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
    exo = ergoExo()

    _setup_without_logging(exo, config, tmp_path)

    assert exo.execution_backend == "legacy"
    assert "transverse ey drivers" in exo.compatibility_fallback_reason


def test_custom_module_injection_selects_legacy_fallback(tmp_path):
    from adept.tf1d import BaseTwoFluid1D

    exo = ergoExo()
    exo._setup_(deepcopy(_tf1d_config()), str(tmp_path), adept_module=BaseTwoFluid1D, log=False)

    assert exo.execution_backend == "legacy"
    assert exo.compatibility_fallback_reason == "custom ADEPTModule injection requires the legacy execution path"
