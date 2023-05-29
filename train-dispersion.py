#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import yaml, os

import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import jax
from jax import numpy as jnp
import tempfile, time
import mlflow, optax
import equinox as eqx
from tqdm import tqdm
from matplotlib import pyplot as plt

import helpers
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from utils import misc


def _modify_defaults_(defaults, k0):
    wepw = float(np.sqrt(1.0 + 3.0 * k0**2.0))

    defaults["save"]["func"]["is_on"] = False
    defaults["physics"]["landau_damping"] = True
    defaults["physics"]["electron"]["trapping"]["kld"] = k0
    defaults["drivers"]["ex"]["0"]["k0"] = k0
    defaults["drivers"]["ex"]["0"]["w0"] = wepw
    xmax = float(2.0 * np.pi / k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax
    defaults["save"]["kx"]["kxmax"] = k0

    return defaults


def train_loop():
    weight_key = jax.random.PRNGKey(420)
    dispersion_models = {
        "w_of_k": eqx.nn.MLP(1, 1, 4, 3, activation=jnp.tanh, final_activation=jnp.tanh, key=weight_key)
    }
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(eqx.filter(dispersion_models, eqx.is_array))

    # modify config
    num_ks = 8
    batch_size = 4
    k0s = np.linspace(0.2, 0.4, num_ks)
    all_ks = np.linspace(0.01, 0.5, 1024)
    all_ws = np.sqrt(1 + 3 * all_ks**2.0)
    rng = np.random.default_rng(420)

    mlflow.set_experiment("train-dispersion")
    with mlflow.start_run(run_name="disp-opt", nested=True) as mlflow_run:
        for epoch in range(100):
            rng.shuffle(k0s)
            these_batches = k0s.reshape((-1, batch_size))
            epoch_loss = 0.0

            for batch, this_batch in tqdm(enumerate(these_batches), total=len(these_batches)):
                grads = []
                for sim, k0 in enumerate(this_batch):
                    with open("./tests/configs/resonance.yaml", "r") as file:
                        defaults = yaml.safe_load(file)
                    mod_defaults = _modify_defaults_(defaults, float(k0))
                    with mlflow.start_run(run_name=f"{epoch=}-{batch=}-{sim=}", nested=True) as mlflow_run:
                        mod_defaults["grid"] = helpers.get_derived_quantities(mod_defaults["grid"])
                        misc.log_params(mod_defaults)

                        mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
                        mod_defaults = helpers.get_save_quantities(mod_defaults)

                        with tempfile.TemporaryDirectory() as td:
                            # run
                            t0 = time.time()
                            state = helpers.init_state(mod_defaults)
                            k0 = (k0 - 0.2) / 0.25

                            def loss(models):
                                w0 = jnp.squeeze(0.5 * models["w_of_k"](jnp.array([k0])) + 1.1)
                                mod_defaults["drivers"]["ex"]["0"]["w0"] = w0
                                vf = helpers.VectorField(mod_defaults, models=False)
                                args = {"driver": mod_defaults["drivers"]}

                                results = diffeqsolve(
                                    terms=ODETerm(vf),
                                    solver=Tsit5(),
                                    t0=mod_defaults["grid"]["tmin"],
                                    t1=mod_defaults["grid"]["tmax"],
                                    max_steps=mod_defaults["grid"]["max_steps"],
                                    dt0=mod_defaults["grid"]["dt"],
                                    y0=state,
                                    args=args,
                                    saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"]),
                                )
                                nk1 = (
                                    jnp.abs(jnp.fft.fft(results.ys["electron"]["n"], axis=1)[:, 1])
                                    * 2.0
                                    / mod_defaults["grid"]["nx"]
                                )
                                return -jnp.amax(nk1), results

                            vg_func = eqx.filter_value_and_grad(loss, has_aux=True)
                            (loss_val, results), grad = eqx.filter_jit(vg_func)(dispersion_models)
                            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

                            t0 = time.time()
                            helpers.post_process(results, mod_defaults, td)
                            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
                            # log artifacts
                            mlflow.log_artifacts(td)
                    grads.append(grad)

                grads = misc.all_reduce_gradients(grads, len(grads))
                updates, opt_state = optimizer.update(grads, opt_state, dispersion_models)
                dispersion_models = eqx.apply_updates(dispersion_models, updates)
                loss_val = float(loss_val)
                mlflow.log_metrics({"run_loss": loss_val}, step=sim + epoch * len(k0s))
                epoch_loss = epoch_loss + loss_val

                # pbar.set_description(f"{loss_val=:.2e}, {epoch_loss=:.2e}, average_loss={epoch_loss/(sim+1):.2e}")

            learned_ws = (
                0.5 * eqx.filter_vmap(dispersion_models["w_of_k"])((jnp.array(all_ks[:, None]) - 0.2) / 0.25) + 1.1
            )
            chosen_ws = 0.5 * eqx.filter_vmap(dispersion_models["w_of_k"])((jnp.array(k0s[:, None]) - 0.2) / 0.25) + 1.1

            fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
            ax.plot(all_ks, all_ws, label="actual")
            ax.plot(k0s, chosen_ws, "x", label="training data")
            ax.plot(all_ks, np.squeeze(learned_ws), label="prediction")
            ax.legend(fontsize=14)
            ax.grid()
            ax.set_xlabel(r"$k\lambda_D$", fontsize=14)
            ax.set_ylabel(r"$\omega_0$", fontsize=14)
            with tempfile.TemporaryDirectory() as td:
                fig.savefig(os.path.join(td, f"{epoch=}"), bbox_inches="tight")
                mlflow.log_artifacts(td, "validation-plots")
            plt.close(fig)
            mlflow.log_metrics({"epoch_loss": epoch_loss})
            mlflow.log_metrics({"val_loss": float(np.sqrt(np.mean(np.square(all_ws - learned_ws))))})


if __name__ == "__main__":
    train_loop()
