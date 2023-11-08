#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import mlflow

from theory import electrostatic
from utils.runner import run


def _modify_defaults_(defaults, rng):
    rand_k0 = np.round(rng.uniform(0.25, 0.4), 3)

    wepw = np.sqrt(1.0 + 3.0 * rand_k0**2.0)
    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)
    print(rand_k0, wepw, root)

    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(wepw)
    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["fields"]["x"]["xmax"] = xmax - 0.5
    defaults["mlflow"]["experiment"] = "test-landau-damping"

    return defaults, float(np.imag(root))


def test_single_resonance():
    with open("tests/test_vlasov2d/configs/damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    # modify config
    rng = np.random.default_rng()
    mod_defaults, actual_damping_rate = _modify_defaults_(defaults, rng)

    # run
    mlflow.set_experiment(mod_defaults["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"]) as mlflow_run:
        result, datasets = run(mod_defaults)

    xax = datasets["fields"].coords["x"].data
    tax = datasets["fields"].coords["t"].data
    dt = tax[1] - tax[0]

    ex = datasets["fields"]["fields-e"].data[:, :, :, 0]
    ek1 = (2.0 / xax.size * np.abs(np.fft.fft(ex, axis=1)[:, 1, 0])) ** 2.0
    frslc = slice(-28, -1)
    measured_damping_rate = np.mean(np.gradient(ek1[frslc], dt) / ek1[frslc])
    print(
        f"Landau Damping rate check \n"
        f"measured: {np.round(measured_damping_rate, 5)}, "
        f"actual: {np.round(2*actual_damping_rate, 5)}, "
    )
    mlflow.log_metrics(
        {"actual damping rate": float(actual_damping_rate), "measured damping rate": float(measured_damping_rate)}
    )

    np.testing.assert_almost_equal(measured_damping_rate, 2 * actual_damping_rate, decimal=2)


if __name__ == "__main__":
    test_single_resonance()
