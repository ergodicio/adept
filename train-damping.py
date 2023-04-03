#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml
from itertools import product

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import jax
from jax import numpy as jnp
import xarray as xr
import tempfile, time
import mlflow, optax
from functools import partial
import haiku as hk
from tqdm import tqdm

from es1d import helpers
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from utils import logs


def _modify_defaults_(defaults, k0, a0, nuee):
    wepw = np.sqrt(1.0 + 3.0 * k0**2.0)

    defaults["physics"]["landau_damping"] = True
    defaults["physics"]["electron"]["trapping"]["nuee"] = nuee
    defaults["physics"]["electron"]["trapping"]["kld"] = float(k0)
    defaults["drivers"]["ex"]["0"]["k0"] = float(k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(wepw)
    defaults["drivers"]["ex"]["0"]["a0"] = float(a0)
    xmax = float(2.0 * np.pi / k0)
    # defaults["save"]["field"]["xmax_to_store"] = float(2.0 * np.pi / k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax
    defaults["save"]["kx"]["kxmax"] = float(k0)

    return defaults  # , float(np.imag(root))


def train_loop():
    w_and_b = get_w_and_b()
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(w_and_b)

    # modify config
    fks = xr.open_dataset("./epws.nc")

    nus = np.copy(fks.coords[r"$\nu_{ee}$"].data)
    k0s = np.copy(fks.coords["$k_0$"].data)
    a0s = np.copy(fks.coords["$a_0$"].data)

    rng = np.random.default_rng(420)


    mlflow.set_experiment("train-damping-rates-epw")
    with mlflow.start_run(run_name="damping-opt", nested=True) as mlflow_run:
        for j in range(100):
            rng.shuffle(nus[::4])
            rng.shuffle(k0s[::4])
            rng.shuffle(a0s[::4])
            epoch_loss = 0.0
            for i, (nuee, k0, a0) in (pbar := tqdm(enumerate(product(nus, k0s, a0s)))):
                with open("./damping.yaml", "r") as file:
                    defaults = yaml.safe_load(file)

                mod_defaults = _modify_defaults_(defaults, k0, a0, nuee)
                locs = {"$k_0$": k0, "$a_0$": a0, r"$\nu_{ee}$": nuee}
                actual_nk1 = jnp.array(fks["n-(k_x)"].loc[locs].data[:, 1])
                with mlflow.start_run(run_name=mod_defaults["mlflow"]["run"], nested=True) as mlflow_run:
                    mod_defaults["grid"] = helpers.get_derived_quantities(mod_defaults["grid"])
                    logs.log_params(mod_defaults)

                    mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
                    mod_defaults = helpers.get_save_quantities(mod_defaults)

                    pulse_dict = {"pulse": mod_defaults["drivers"]}

                    with tempfile.TemporaryDirectory() as td:
                        # run
                        t0 = time.time()

                        def vector_field(t, y, args):
                            dummy_vf = helpers.VectorField(mod_defaults)
                            return dummy_vf(t, y, args)

                        state = helpers.init_state(mod_defaults)
                        _, vf_apply = hk.without_apply_rng(hk.transform(vector_field))

                        def loss(weights_and_biases):
                            vf = partial(vf_apply, weights_and_biases)
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
                            nk1 = (
                                jnp.abs(jnp.fft.fft(results.ys["x"]["electron"]["n"], axis=1)[:, 1])
                                * 2.0
                                / mod_defaults["grid"]["nx"]
                            )
                            return jnp.mean(jnp.square((actual_nk1 - nk1) / np.amax(actual_nk1))), results

                        vg_func = jax.value_and_grad(loss, argnums=0, has_aux=True)
                        (loss_val, results), grad = jax.jit(vg_func)(w_and_b)
                        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

                        t0 = time.time()
                        helpers.post_process(results, mod_defaults, td)
                        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
                        # log artifacts
                        mlflow.log_artifacts(td)

                updates, opt_state = optimizer.update(grad, opt_state, w_and_b)
                w_and_b = optax.apply_updates(w_and_b, updates)
                loss_val = float(loss_val[0])
                mlflow.log_metrics({"run_loss": loss_val}, step=i + j * 100)
                epoch_loss = epoch_loss + loss_val
                pbar.set_description(f"{loss_val=}, {epoch_loss=}, average_loss={epoch_loss/(i+1)}")

            mlflow.log_metrics({"epoch_loss": epoch_loss})


def get_w_and_b():
    with open("./damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)
    defaults["grid"] = helpers.get_derived_quantities(defaults["grid"])
    defaults["grid"] = helpers.get_solver_quantities(defaults["grid"])
    defaults = helpers.get_save_quantities(defaults)

    pulse_dict = {"pulse": defaults["drivers"]}

    def vector_field(t, y, args):
        dummy_vf = helpers.VectorField(defaults)
        return dummy_vf(t, y, args)

    state = helpers.init_state(defaults)
    vf_init, vf_apply = hk.without_apply_rng(hk.transform(vector_field))
    dummy_t = 0.5
    dummy_args = pulse_dict
    key = jax.random.PRNGKey(420)
    return vf_init(key, dummy_t, state, dummy_args)


if __name__ == "__main__":
    train_loop()
