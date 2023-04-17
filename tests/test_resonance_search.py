#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml, pytest

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import jax, mlflow, diffrax, optax
from jax import numpy as jnp
import tempfile, time
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from utils import misc
from jaxopt import OptaxSolver
import equinox as eqx
from tqdm import tqdm

from es1d import helpers
from theory.electrostatic import get_roots_to_electrostatic_dispersion


def load_cfg(rand_k0, gamma, adjoint):
    with open("./tests/configs/resonance_search.yaml", "r") as file:
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
    mlflow.set_experiment("test-res-search")
    with mlflow.start_run(run_name="res-search-opt") as mlflow_run:
        vg_func, sim_k0, actual_w0 = get_vg_func(gamma, adjoint)

        mod_defaults, _ = load_cfg(sim_k0, gamma, adjoint)
        mod_defaults["grid"] = helpers.get_derived_quantities(mod_defaults["grid"])
        misc.log_params(mod_defaults)
        mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
        mod_defaults = helpers.get_save_quantities(mod_defaults)

        rng_key = jax.random.PRNGKey(420)

        w0 = 1.1 + 0.05 * jax.random.normal(rng_key, [1], dtype=jnp.float64)

        # optimizer = optax.adam(0.1)
        # opt_state = optimizer.init(w0)

        t0 = time.time()
        optimizer = OptaxSolver(fun=vg_func, value_and_grad=True, has_aux=True, opt=optax.adam(learning_rate=0.1))
        opt_state = optimizer.init_state(w0)
        mlflow.log_metrics({"init_time": round(time.time() - t0, 4)})

        mlflow.log_metrics({"w0": float(w0)}, step=0)
        mlflow.log_metrics({"actual_w0": actual_w0}, step=0)
        for i in tqdm(range(100)):
            w0, opt_state, loss = run_one_step(i, w0, vg_func, mod_defaults, optimizer, opt_state)

            mlflow.log_metrics({"w0": float(w0), "actual_w0": actual_w0, "loss": float(loss)}, step=i + 1)

        np.testing.assert_allclose(actual_w0, float(w0), rtol=0.03)


def run_one_step(i, w0, vg_func, mod_defaults, optimizer, opt_state):
    with mlflow.start_run(run_name=f"res-search-run-{i}", nested=True) as mlflow_run:
        mlflow.log_param("w0", w0)
        t0 = time.time()

        # (loss, results), grad = vg_func(w0)
        # updates, opt_state = optimizer.update(grad, opt_state, w0)
        # w0 = optax.apply_updates(w0, updates)

        w0, opt_state = optimizer.update(params=w0, state=opt_state)
        loss = opt_state.error
        results = opt_state.aux

        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            helpers.post_process(results, mod_defaults, td)
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4), "loss": float(loss)})
            # log artifacts
            mlflow.log_artifacts(td)

    return w0, opt_state, loss


def get_vg_func(gamma, adjoint):
    rng = np.random.default_rng(420)
    sim_k0 = rng.uniform(0.26, 0.4)

    defaults, actual_w0 = load_cfg(sim_k0, gamma, adjoint)
    defaults["grid"] = helpers.get_derived_quantities(defaults["grid"])
    misc.log_params(defaults)

    defaults["grid"] = helpers.get_solver_quantities(defaults["grid"])
    defaults = helpers.get_save_quantities(defaults)

    pulse_dict = {"pulse": defaults["drivers"]}
    state = helpers.init_state(defaults)
    loss_fn = get_loss(state, pulse_dict, defaults)
    vg_func = eqx.filter_jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))

    return vg_func, sim_k0, actual_w0


def get_loss(state, pulse_dict, mod_defaults):
    if mod_defaults["adjoint"] == "Recursive":
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    elif mod_defaults["adjoint"] == "Backsolve":
        adjoint = diffrax.BacksolveAdjoint(solver=Tsit5())
    else:
        raise NotImplementedError

    def loss(w0):
        pulse_dict["pulse"]["ex"]["0"]["w0"] = w0
        vf = helpers.VectorField(mod_defaults)
        results = diffeqsolve(
            terms=ODETerm(vf),
            solver=Tsit5(),
            t0=mod_defaults["grid"]["tmin"],
            t1=mod_defaults["grid"]["tmax"],
            max_steps=mod_defaults["grid"]["max_steps"],
            dt0=mod_defaults["grid"]["dt"],
            adjoint=adjoint,
            y0=state,
            args=pulse_dict,
            saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"]),
        )
        nk1 = jnp.abs(jnp.fft.fft(results.ys["electron"]["n"], axis=1)[:, 1])
        return -jnp.amax(nk1), results

    return loss


if __name__ == "__main__":
    test_resonance_search("kinetic")
