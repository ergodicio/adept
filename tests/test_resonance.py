#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
from jax import numpy as jnp
import mlflow

from theory import electrostatic
from utils.runner import run


def _modify_defaults_(defaults, rng):
    rand_k0 = np.round(rng.uniform(0.25, 0.4), 3)
    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    root = np.sqrt(1.0 + 3.0 * rand_k0**2.0)  # electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)
    # defaults["save"]["field"]["xmax_to_store"] = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = float(2.0 * np.pi / rand_k0)

    return defaults, float(np.real(root))


def test_single_resonance():
    with open("./tests/configs/resonance.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    # modify config
    rng = np.random.default_rng()
    mod_defaults, actual_resonance = _modify_defaults_(defaults, rng)

    # run
    mlflow.set_experiment(mod_defaults["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"]) as mlflow_run:
        result = run(mod_defaults)

    efs = jnp.real(
        jnp.fft.ifft(
            1j * mod_defaults["grid"]["one_over_kx"][None, :] * jnp.fft.fft(1 - result.ys["electron"]["n"][:, :])
        )
    )
    ek1 = np.fft.fft(efs, axis=1)[:, 1]
    env, freq = electrostatic.get_nlfs(ek1, result.ts[1] - result.ts[0])
    frslc = slice(400, -400)
    print(
        f"Frequency check \n"
        f"measured: {np.round(np.mean(freq[frslc]), 5)}, "
        f"bohm-gross: {np.round(actual_resonance, 5)}, "
    )
    measured_resonance = np.mean(freq[frslc])
    np.testing.assert_almost_equal(measured_resonance, actual_resonance, decimal=2)


if __name__ == "__main__":
    test_single_resonance()
