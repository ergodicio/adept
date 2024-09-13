#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml

import numpy as np
from jax import config

from adept import ergoExo

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

from jax import numpy as jnp
import mlflow

from adept import electrostatic


def _modify_defaults_(defaults, rng):
    rand_k0 = np.round(rng.uniform(0.25, 0.4), 3)

    wepw = np.sqrt(1.0 + 3.0 * rand_k0**2.0)
    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)
    print(rand_k0, wepw, root)

    defaults["physics"]["landau_damping"] = True
    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(wepw)
    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax
    defaults["save"]["kx"]["kxmax"] = rand_k0
    defaults["mlflow"]["experiment"] = "test-landau-damping"

    return defaults, float(np.imag(root))


def test_single_resonance():
    with open("tests/test_tf1d/configs/resonance.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    # modify config
    rng = np.random.default_rng()
    mod_defaults, actual_damping_rate = _modify_defaults_(defaults, rng)

    exo = ergoExo()
    exo.setup(mod_defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    with mlflow.start_run(run_id=run_id, log_system_metrics=True) as mlflow_run:
        kx = (
            np.fft.fftfreq(
                mod_defaults["save"]["x"]["nx"],
                d=mod_defaults["save"]["x"]["ax"][2] - mod_defaults["save"]["x"]["ax"][1],
            )
            * 2.0
            * np.pi
        )
        one_over_kx = np.zeros_like(kx)
        one_over_kx[1:] = 1.0 / kx[1:]
        efs = jnp.real(jnp.fft.ifft(1j * one_over_kx[None, :] * jnp.fft.fft(1 - result.ys["x"]["electron"]["n"][:, :])))
        ek1 = (2.0 / mod_defaults["grid"]["nx"] * np.abs(np.fft.fft(efs, axis=1)[:, 1])) ** 2.0
        frslc = slice(-100, -50)
        measured_damping_rate = np.mean(np.gradient(ek1[frslc], (result.ts[1] - result.ts[0])) / ek1[frslc])
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
