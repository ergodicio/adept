#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml, pytest

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

from jax import numpy as jnp
import mlflow

from theory import electrostatic
from runner import run


def _modify_defaults_(defaults, rng, gamma):
    rand_k0 = np.round(rng.uniform(0.25, 0.4), 3)
    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["physics"]["electron"]["gamma"] = gamma
    if gamma == "kinetic":
        root = np.real(electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0))
        defaults["mlflow"]["run"] = "kinetic"
    else:
        root = np.sqrt(1.0 + 3.0 * rand_k0**2.0)
        defaults["mlflow"]["run"] = "bohm-gross"

    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax
    defaults["mlflow"]["experiment"] = "test-resonance"

    return defaults, float(root)


@pytest.mark.parametrize("gamma", ["kinetic", 3.0])
def test_single_resonance(gamma):
    with open("./tests/configs/resonance.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    # modify config
    rng = np.random.default_rng()
    mod_defaults, actual_resonance = _modify_defaults_(defaults, rng, gamma)

    # run
    mlflow.set_experiment(mod_defaults["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"]) as mlflow_run:
        result = run(mod_defaults)

    kx = (
        np.fft.fftfreq(
            mod_defaults["save"]["x"]["nx"], d=mod_defaults["save"]["x"]["ax"][2] - mod_defaults["save"]["x"]["ax"][1]
        )
        * 2.0
        * np.pi
    )
    one_over_kx = np.zeros_like(kx)
    one_over_kx[1:] = 1.0 / kx[1:]
    efs = jnp.real(
        jnp.fft.ifft(
            1j
            * one_over_kx[None, :]
            * jnp.fft.fft(result.ys["x"]["ion"]["n"][:, :] - result.ys["x"]["electron"]["n"][:, :])
        )
    )
    ek1 = np.fft.fft(efs, axis=1)[:, 1]
    env, freq = electrostatic.get_nlfs(ek1, result.ts[1] - result.ts[0])
    frslc = slice(-80, -10)
    print(
        f"Frequency check \n"
        f"measured: {np.round(np.mean(freq[frslc]), 5)}, "
        f"desired: {np.round(actual_resonance, 5)}, "
    )
    measured_resonance = np.mean(freq[frslc])
    np.testing.assert_almost_equal(measured_resonance, actual_resonance, decimal=2)


if __name__ == "__main__":
    for gamma in ["kinetic", 3.0]:
        test_single_resonance(gamma)
