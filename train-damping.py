#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

from jax import numpy as jnp
import mlflow
import xarray as xr

from theory import electrostatic
from utils.runner import run


def _modify_defaults_(defaults, params):
    k0 = params["k_0"]
    a0 = params["a_0"]
    nuee = params["nu_ee"]

    wepw = np.sqrt(1.0 + 3.0 * k0**2.0)
    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, k0)

    defaults["physics"]["landau_damping"] = True
    defaults["drivers"]["ex"]["0"]["k0"] = float(k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(wepw)
    defaults["drivers"]["ex"]["0"]["a0"] = float(a0)
    xmax = float(2.0 * np.pi / k0)
    # defaults["save"]["field"]["xmax_to_store"] = float(2.0 * np.pi / k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax

    return defaults  # , float(np.imag(root))


def train_loop():
    with open("./damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    # modify config
    params = xr.open_dataset("./params.nc")
    ek1s = xr.open_dataset("./t_vs_e_squared.nc")["average_es"]

    sim_index = np.arange(len(params.coords["sim_index"].data))
    rng = np.random.default_rng()
    rng.shuffle(sim_index)

    for ib in sim_index:
        these_params = {k: params[k][ib].data for k in params.keys()}
        mod_defaults = _modify_defaults_(defaults, these_params)

        # run
        mlflow.set_experiment(mod_defaults["mlflow"]["experiment"])
        # modify config
        with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"]) as mlflow_run:
            result = run(mod_defaults)

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
        ek1 = 2.0 / mod_defaults["grid"]["nx"] * np.abs(np.fft.fft(efs, axis=1)[:, 1])
        interp_ek1 = jnp.interp(ek1s.coords["t"].data, mod_defaults["save"]["t"]["ax"], ek1**2.)
        dt = ek1s.coords["t"].data[2] - ek1s.coords["t"].data[1]
        frslc = slice(-600, -400)
        measured_damping_rate = np.mean(np.gradient(interp_ek1[frslc], dt) / interp_ek1[frslc])
        actual_damping_rate = np.mean(np.gradient(ek1s[ib][frslc], dt) / ek1s[ib][frslc])

        print(
            f"Frequency check \n"
            f"measured: {np.round(measured_damping_rate, 5)}, "
            f"bohm-gross: {np.round(actual_damping_rate, 5)}, "
        )

    # np.testing.assert_almost_equal(measured_damping_rate, actual_damping_rate, decimal=2)


if __name__ == "__main__":
    train_loop()
