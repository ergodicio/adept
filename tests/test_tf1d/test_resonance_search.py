#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from typing import Dict, Tuple
import yaml, pytest, os
from itertools import product


import numpy as np
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import mlflow, optax
import pytest

from jax import numpy as jnp, Array, random as jr, devices
import time

import equinox as eqx
from tqdm import tqdm

from adept import ergoExo
from adept.tf1d import BaseTwoFluid1D
from adept.electrostatic import get_roots_to_electrostatic_dispersion


class Resonance(BaseTwoFluid1D):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, params: Dict, args: Dict) -> Dict:
        args = self.args
        args["drivers"]["ex"]["0"]["w0"] = params["w0"]
        solver_result = super().__call__(params, args)
        soln = solver_result["solver result"]
        nk1 = jnp.abs(jnp.fft.fft(soln.ys["x"]["electron"]["n"], axis=1)[:, 1])
        return -jnp.amax(nk1), solver_result

    def vg(self, params: Dict, args: Dict) -> Tuple[float, Array, Dict]:
        return eqx.filter_jit(eqx.filter_value_and_grad(self.__call__, has_aux=True))(params, args)


def load_cfg(rand_k0, gamma, adjoint):
    with open("tests/test_tf1d/configs/resonance_search.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["physics"]["electron"]["gamma"] = gamma
    defaults["adjoint"] = adjoint

    if gamma == "kinetic":
        wepw = np.real(get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0))
        defaults["mlflow"]["run"] = "kinetic"
    else:
        wepw = np.sqrt(1.0 + 3.0 * rand_k0**2.0)
        defaults["mlflow"]["run"] = "bohm-gross"

    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax

    return defaults, wepw


@pytest.mark.parametrize("adjoint", ["Recursive", "Backsolve"])
@pytest.mark.parametrize("gamma", ["kinetic", 3.0])
def test_resonance_search(gamma, adjoint):
    mlflow.set_experiment("tf1d-resonance-search")
    with mlflow.start_run(run_name="res-search-opt", log_system_metrics=True) as mlflow_run:
        # sim_k0, actual_w0 = init_w0(gamma, adjoint)
        rng = np.random.default_rng(420)
        sim_k0 = rng.uniform(0.26, 0.4)

        mod_defaults, actual_w0 = load_cfg(sim_k0, gamma, adjoint)

        rng_key = jr.PRNGKey(420)

        params = {"w0": 1.1 + 0.05 * jr.normal(rng_key, [1], dtype=jnp.float64)[0]}

        optimizer = optax.adam(0.1)
        opt_state = optimizer.init(params)

        t0 = time.time()
        mlflow.log_metrics({"init_time": round(time.time() - t0, 4)})
        mlflow.log_metrics({"w0": float(params["w0"])}, step=0)
        mlflow.log_metrics({"actual_w0": actual_w0}, step=0)
        for i in tqdm(range(40)):
            exo = ergoExo(mlflow_nested=True)
            mod_defaults, _ = load_cfg(sim_k0, gamma, adjoint)
            exo.setup(mod_defaults, Resonance)
            val, grad, results = exo.val_and_grad(params)
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            mlflow.log_metrics({"w0": float(params["w0"]), "actual_w0": actual_w0, "loss": float(val)}, step=i + 1)

        print(f"{gamma=}, {adjoint=}")
        print(f"{actual_w0=}, {float(params['w0'])=}")

        np.testing.assert_allclose(actual_w0, float(params["w0"]), rtol=0.03)


if __name__ == "__main__":
    for gamma, adjoint in product(["kinetic", 3.0], ["Recursive", "Backsolve"]):

        if not any(["gpu" == device.platform for device in devices()]):
            pytest.skip("Takes too long without a GPU")
        else:
            test_resonance_search(gamma, adjoint)
