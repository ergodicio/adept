#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import itertools

import yaml, pytest

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import mlflow
from astropy import units as u

from theory import electrostatic
from utils.runner import run


def _modify_defaults_(defaults, rng, real_or_imag, time, field, edfdv):
    defaults["terms"]["time"] = time
    defaults["terms"]["field"] = field
    defaults["terms"]["edfdv"] = edfdv
    if field == "ampere":
        defaults["grid"]["dt"] = 0.025

    if real_or_imag == "real":
        rand_k0 = np.round(rng.uniform(0.2, 0.28), 3)
    else:
        rand_k0 = np.round(rng.uniform(0.28, 0.35), 3)

    ne_n0 = u.Quantity(defaults["units"]["reference electron density"]) / u.Quantity("9.0663e21/cm^3")
    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)

    defaults["drivers"]["ex"]["0"]["k0"] = str(rand_k0) + "kl_D"
    defaults["drivers"]["ex"]["0"]["w0"] = str(np.real(root)) + "wp"
    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = str(xmax) + "l_D"
    defaults["mlflow"]["experiment"] = "vlasov1d-test-resonance"
    root *= np.sqrt(ne_n0.to("").value)

    return defaults, root


@pytest.mark.parametrize(
    "real_or_imag, time, field, edfdv",
    itertools.product(
        ["real", "imag"], ["sixth", "leapfrog"], ["poisson", "ampere", "hampere"], ["exponential", "cubic-spline"]
    ),
)
def test_single_resonance(real_or_imag, time, field, edfdv):
    if (time == "sixth") and (field == "ampere"):
        print("not implemented - skipping test")
    elif (time == "sixth") and (field == "hampere"):
        print("not implemented - skipping test")
    else:
        with open("tests/test_vlasov1d2v/configs/resonance.yaml", "r") as file:
            defaults = yaml.safe_load(file)

        # modify config
        rng = np.random.default_rng()
        mod_defaults, root = _modify_defaults_(defaults, rng, real_or_imag, time, field, edfdv)

        actual_damping_rate = np.imag(root)
        actual_resonance = np.real(root)
        # run
        mlflow.set_experiment(mod_defaults["mlflow"]["experiment"])
        # modify config
        with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"]) as mlflow_run:
            result, datasets = run(mod_defaults)
            efs = result.ys["fields"]["e"]
            ek1 = 2.0 / mod_defaults["grid"]["nx"] * np.fft.fft(efs, axis=1)[:, 1]
            ek1_mag = np.abs(ek1)
            if real_or_imag == "imag":
                frslc = slice(-100, -50)
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
                frslc = slice(-480, -240)
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
