#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import jax
from jax import numpy as jnp
import tempfile, time
import mlflow

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from utils import misc
import haiku as hk
import optax

from es1d import helpers


def _modify_defaults_(defaults, rand_k0):
    wepw = np.sqrt(1.0 + 3.0 * rand_k0**2.0)

    defaults["physics"]["landau_damping"] = True
    defaults["physics"]["electron"]["trapping"]["kld"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(wepw)

    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax
    defaults["save"]["kx"]["kxmax"] = float(rand_k0)

    return defaults, wepw


def test_resonance_search():
    mlflow.set_experiment("test-res-search")
    with mlflow.start_run(run_name="res-search-opt") as mlflow_run:
        with open("./tests/configs/resonance.yaml", "r") as file:
            defaults = yaml.safe_load(file)

        rng = np.random.default_rng()
        rand_k0 = rng.uniform(0.26, 0.4)

        defaults, actual_resonance = _modify_defaults_(defaults, rand_k0)
        defaults["grid"] = helpers.get_derived_quantities(defaults["grid"])
        misc.log_params(defaults)

        defaults["grid"] = helpers.get_solver_quantities(defaults["grid"])
        defaults = helpers.get_save_quantities(defaults)

        pulse_dict = {"pulse": defaults["drivers"]}
        state = helpers.init_state(defaults)
        loss = get_loss(state, pulse_dict, defaults)
        init_w0, _ = hk.without_apply_rng(hk.transform(loss))
        w0 = init_w0(jax.random.PRNGKey(420))
        optimizer = optax.adam(0.1)
        opt_state = optimizer.init(w0)
        mlflow.log_metrics({"w0": float(w0["~"]["w0"])}, step=0)
        mlflow.log_metrics({"actual_w0": actual_resonance}, step=0)
        for i in range(48):
            with mlflow.start_run(run_name=f"res-search-run-{i}", nested=True) as mlflow_run:
                with open("./tests/configs/resonance.yaml", "r") as file:
                    mod_defaults = yaml.safe_load(file)
                mod_defaults, _ = _modify_defaults_(mod_defaults, rand_k0)
                mod_defaults["grid"] = helpers.get_derived_quantities(mod_defaults["grid"])
                misc.log_params(mod_defaults)

                mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
                mod_defaults = helpers.get_save_quantities(mod_defaults)
                t0 = time.time()

                state = helpers.init_state(mod_defaults)
                loss = get_loss(state, pulse_dict, mod_defaults)
                _, loss_fn = hk.without_apply_rng(hk.transform(loss))
                vg_func = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
                (loss, results), grad = vg_func(w0)
                mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})
                with tempfile.TemporaryDirectory() as td:
                    t0 = time.time()
                    helpers.post_process(results, mod_defaults, td)
                    mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
                    # log artifacts
                    mlflow.log_artifacts(td)

            updates, opt_state = optimizer.update(grad, opt_state, w0)
            w0 = optax.apply_updates(w0, updates)
            mlflow.log_metrics({"w0": float(w0["~"]["w0"])}, step=i + 1)
            mlflow.log_metrics({"actual_w0": actual_resonance}, step=i + 1)

        np.testing.assert_allclose(actual_resonance, float(w0["~"]["w0"]), rtol=0.03)


def get_loss(state, pulse_dict, mod_defaults):
    mod_defaults["save"]["kx"]["is_on"] = False

    def loss():
        pulse_dict["pulse"]["ex"]["0"]["w0"] = hk.get_parameter(
            "w0", shape=[1], init=hk.initializers.RandomNormal(mean=1.2, stddev=0.2), dtype=jnp.float64
        )[0]
        vf = helpers.VectorField(mod_defaults)
        results = diffeqsolve(
            terms=ODETerm(vf),
            solver=Tsit5(),
            t0=mod_defaults["grid"]["tmin"],
            t1=mod_defaults["grid"]["tmax"],
            max_steps=mod_defaults["grid"]["max_steps"],
            dt0=mod_defaults["grid"]["dt"],
            y0=state,
            args=pulse_dict,
            saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"], fn=mod_defaults["save"]["func"]),
        )
        nk1 = jnp.abs(jnp.fft.fft(results.ys["x"]["electron"]["n"], axis=1)[:, 1])
        return -jnp.amax(nk1), results

    return loss


if __name__ == "__main__":
    test_resonance_search()
