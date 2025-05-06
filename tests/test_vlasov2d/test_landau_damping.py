#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import yaml, pytest, os

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import mlflow
import pytest
from jax import devices

from adept import electrostatic


def _modify_defaults_(defaults, rng, real_or_imag):
    if real_or_imag == "real":
        rand_k0 = np.round(rng.uniform(0.2, 0.28), 3)
    else:
        rand_k0 = np.round(rng.uniform(0.28, 0.35), 3)

    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)

    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(np.real(root))
    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["mlflow"]["experiment"] = "vlasov2d-test"

    return defaults, root


@pytest.mark.parametrize("real_or_imag", ["real", "imag"])
def test_single_resonance(real_or_imag):

    if not any(["gpu" == device.platform for device in devices()]):
        pytest.skip("Skipping test on CPU")

    else:
        with open("tests/test_vlasov2d/configs/damping.yaml", "r") as file:
            defaults = yaml.safe_load(file)

        # modify config
        rng = np.random.default_rng()
        mod_defaults, root = _modify_defaults_(defaults, rng, real_or_imag)

        actual_damping_rate = np.imag(root)
        actual_resonance = np.real(root)
        # run
        mlflow.set_experiment(mod_defaults["mlflow"]["experiment"])
        # modify config
        with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"], log_system_metrics=True) as mlflow_run:
            result, datasets = run(mod_defaults)
            efs = result.ys["fields"]["ex"]
            ek1 = 2.0 / mod_defaults["grid"]["nx"] * np.fft.fft2(efs, axes=(1, 2))[:, 1, 0]
            ek1_mag = np.abs(ek1)
            if real_or_imag == "imag":
                frslc = slice(-64, -4)
                measured_damping_rate = np.mean(
                    np.gradient(ek1_mag[frslc], (result.ts["fields"][1] - result.ts["fields"][0])) / ek1_mag[frslc]
                )
                print(
                    f"Landau Damping rate check \n"
                    f"measured: {np.round(measured_damping_rate, 5)}, "
                    f"actual: {np.round(actual_damping_rate, 5)}, "
                )
                mlflow.log_metrics(
                    {
                        "actual damping rate": float(actual_damping_rate),
                        "measured damping rate": float(measured_damping_rate),
                    }
                )

                np.testing.assert_almost_equal(measured_damping_rate, actual_damping_rate, decimal=2)
            else:
                env, freq = electrostatic.get_nlfs(ek1, result.ts["fields"][1] - result.ts["fields"][0])
                frslc = slice(-200, -100)
                print(
                    f"Frequency check \n"
                    f"measured: {np.round(np.mean(freq[frslc]), 5)}, "
                    f"desired: {np.round(actual_resonance, 5)}, "
                )
                measured_resonance = np.mean(freq[frslc])
                mlflow.log_metrics(
                    {
                        "actual frequency": float(actual_resonance),
                        "measured frequency": float(measured_resonance),
                    }
                )
                np.testing.assert_almost_equal(measured_resonance, actual_resonance, decimal=2)


if __name__ == "__main__":
    test_single_resonance(real_or_imag="real")
