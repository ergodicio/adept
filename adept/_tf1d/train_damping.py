#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import yaml, os
from itertools import product

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

from jax import numpy as jnp
import xarray as xr
import tempfile, time
import mlflow, optax
import equinox as eqx
from tqdm import tqdm

from adept._tf1d import helpers
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from utils import misc, plotters


def _modify_defaults_(defaults, k0, a0, nuee):
    wepw = float(np.sqrt(1.0 + 3.0 * k0**2.0))

    defaults["physics"]["landau_damping"] = True
    defaults["physics"]["electron"]["trapping"]["nuee"] = nuee
    defaults["physics"]["electron"]["trapping"]["kld"] = k0
    defaults["drivers"]["ex"]["0"]["k0"] = k0
    defaults["drivers"]["ex"]["0"]["w0"] = wepw
    defaults["drivers"]["ex"]["0"]["a0"] = a0
    xmax = float(2.0 * np.pi / k0)
    defaults["grid"]["xmax"] = xmax
    defaults["save"]["x"]["xmax"] = xmax
    defaults["save"]["kx"]["kxmax"] = k0

    return defaults


def train_loop():
    # modify config
    fks = xr.open_dataset("../../epws.nc")

    nus = np.copy(fks.coords[r"$\nu_{ee}$"].data)  # [::4]
    k0s = np.copy(fks.coords["$k_0$"].data)  # [::4]
    a0s = np.copy(fks.coords["$a_0$"].data)  # [::4]

    rng = np.random.default_rng(420)

    mlflow.set_experiment("train-damping-rates-epw")
    with mlflow.start_run(run_name="damping-opt", nested=True) as mlflow_run:
        with open("./damping.yaml", "r") as file:
            defaults = yaml.safe_load(file)
        trapping_models = helpers.get_models(defaults["models"])
        optimizer = optax.adam(0.1)
        opt_state = optimizer.init(eqx.filter(trapping_models, eqx.is_array))

        for epoch in range(100):
            rng.shuffle(nus)
            rng.shuffle(k0s)
            rng.shuffle(a0s)
            epoch_loss = 0.0
            for sim, (nuee, k0, a0) in (pbar := tqdm(enumerate(product(nus, k0s, a0s)))):
                with open("./damping.yaml", "r") as file:
                    defaults = yaml.safe_load(file)

                mod_defaults = _modify_defaults_(defaults, float(k0), float(a0), float(nuee))
                locs = {"$k_0$": k0, "$a_0$": a0, r"$\nu_{ee}$": nuee}
                actual_nk1 = xr.DataArray(fks["n-(k_x)"].loc[locs].data[:, 1], coords=(("t", fks.coords["t"].data),))
                with mlflow.start_run(run_name=f"{epoch=}-{sim=}", nested=True) as mlflow_run:
                    mod_defaults = helpers.get_derived_quantities(mod_defaults)
                    misc.log_params(mod_defaults)

                    mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
                    mod_defaults = helpers.get_save_quantities(mod_defaults)

                    with tempfile.TemporaryDirectory() as td:
                        # run
                        t0 = time.time()
                        state = helpers.init_state(mod_defaults)

                        def loss(models):
                            vf = helpers.VectorField(mod_defaults, models=models)
                            args = {"drivers": mod_defaults["drivers"]}

                            results = diffeqsolve(
                                terms=ODETerm(vf),
                                solver=Tsit5(),
                                t0=mod_defaults["grid"]["tmin"],
                                t1=mod_defaults["grid"]["tmax"],
                                max_steps=mod_defaults["grid"]["max_steps"],
                                dt0=mod_defaults["grid"]["dt"],
                                y0=state,
                                args=args,
                                saveat=SaveAt(
                                    ts=mod_defaults["save"]["t"]["ax"], fn=mod_defaults["save"]["func"]["callable"]
                                ),
                            )
                            nk1 = (
                                jnp.abs(jnp.fft.fft(results.ys["x"]["electron"]["n"], axis=1)[:, 1])
                                * 2.0
                                / mod_defaults["grid"]["nx"]
                            )
                            return jnp.mean(jnp.square((actual_nk1.data - nk1) / np.amax(actual_nk1.data))), results

                        vg_func = eqx.filter_value_and_grad(loss, has_aux=True)
                        (loss_val, results), grad = eqx.filter_jit(vg_func)(trapping_models)
                        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

                        t0 = time.time()
                        helpers.post_process(results, mod_defaults, td)
                        plotters.mva(actual_nk1, mod_defaults, results, td)
                        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
                        # log artifacts
                        mlflow.log_artifacts(td)

                updates, opt_state = optimizer.update(grad, opt_state, trapping_models)
                trapping_models = eqx.apply_updates(trapping_models, updates)
                loss_val = float(loss_val)
                mlflow.log_metrics({"run_loss": loss_val}, step=sim + epoch * 100)
                epoch_loss = epoch_loss + loss_val
                pbar.set_description(f"{loss_val=:.2e}, {epoch_loss=:.2e}, average_loss={epoch_loss / (sim + 1):.2e}")

            mlflow.log_metrics({"epoch_loss": epoch_loss})


def remote_train_loop():
    with open("./damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)
    trapping_models = helpers.get_models(defaults["models"])
    optimizer = optax.adam(0.004)
    opt_state = optimizer.init(eqx.filter(trapping_models, eqx.is_array))

    batch_size = 16

    # modify config
    fks = xr.open_dataset("../../epws.nc")

    nus = np.copy(fks.coords[r"$\nu_{ee}$"].data[::3])
    k0s = np.copy(fks.coords["$k_0$"].data[::2])
    a0s = np.copy(fks.coords["$a_0$"].data[::3])
    all_sims = np.array(list(product(nus, k0s, a0s)))

    rng = np.random.default_rng(420)

    train_sims = rng.choice(
        np.arange(all_sims.shape[0]), int(0.9 * all_sims.shape[0] / batch_size) * batch_size, replace=False
    )
    val_sims = np.array(list(set(np.arange(all_sims.shape[0])) - set(train_sims)))

    mlflow.set_experiment(defaults["mlflow"]["experiment"])
    with mlflow.start_run(run_name="damping-opt", nested=True) as mlflow_run:
        for epoch in range(100):
            epoch_loss = 0.0
            rng.shuffle(train_sims)
            train_batches = all_sims[train_sims].reshape((-1, batch_size, 3))
            for i_batch, batch in (pbar := tqdm(enumerate(train_batches), total=len(train_batches))):
                run_ids, job_done = [], []
                for sim, (nuee, k0, a0) in enumerate(batch):
                    run_ids, job_done = queue_sim(
                        fks, nuee, k0, a0, run_ids, job_done, trapping_models, epoch, i_batch, sim, t_or_v="grad"
                    )
                trapping_models = update_w_and_b(
                    job_done, run_ids, optimizer, opt_state, eqx.filter(trapping_models, eqx.is_array)
                )
                batch_loss = float(
                    np.sum(
                        np.array([misc.get_this_metric_of_this_run("loss", queued_run_id) for queued_run_id in run_ids])
                    )
                )
                mlflow.log_metrics({"batch_loss": batch_loss}, step=i_batch + epoch * len(train_batches))
                epoch_loss = epoch_loss + batch_loss
                pbar.set_description(f"{batch_loss=:.2e}, {epoch_loss=:.2e}, average_loss={epoch_loss / (sim + 1):.2e}")

            mlflow.log_metrics({"epoch_loss": epoch_loss / len(train_sims)}, step=epoch)

            # validation
            run_ids, job_done = [], []
            for sim, (nuee, k0, a0) in enumerate(all_sims[val_sims]):
                run_ids, job_done = queue_sim(
                    fks, nuee, k0, a0, run_ids, job_done, trapping_models, epoch, 0, sim, t_or_v="val"
                )
            wait_for_jobs(job_done, run_ids)
            val_loss = float(
                np.average(
                    np.array([misc.get_this_metric_of_this_run("val_loss", queued_run_id) for queued_run_id in run_ids])
                )
            )
            mlflow.log_metrics({"val_epoch_loss": val_loss}, step=epoch)


def wait_for_jobs(job_done, run_ids):
    while not all(job_done):
        for i, run_id in enumerate(run_ids):
            time.sleep(4.2 / len(run_ids))
            job_done[i] = misc.is_job_done(run_id)


def update_w_and_b(job_done, run_ids, optimizer, opt_state, w_and_b):
    wait_for_jobs(job_done, run_ids)
    gradients = []
    with tempfile.TemporaryDirectory() as td:
        for queued_run_id in run_ids:
            mlflow.artifacts.download_artifacts(run_id=queued_run_id, artifact_path="grads.eqx", dst_path=td)
            gradients.append(eqx.tree_deserialise_leaves(os.path.join(td, "grads.eqx"), w_and_b))

    gradients = misc.all_reduce_gradients(gradients, len(run_ids))
    updates, opt_state = optimizer.update(gradients, opt_state, w_and_b)
    w_and_b = eqx.apply_updates(w_and_b, updates)

    return w_and_b


def queue_sim(fks, nuee, k0, a0, run_ids, job_done, w_and_b, epoch, i_batch, sim, t_or_v="grad"):
    with open("../../configs/tf-1d/damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    mod_defaults = _modify_defaults_(defaults, float(k0), float(a0), float(nuee))
    locs = {"$k_0$": k0, "$a_0$": a0, r"$\nu_{ee}$": nuee}
    actual_nk1 = xr.DataArray(fks["n-(k_x)"].loc[locs].data[:, 1], coords=(("t", fks.coords["t"].data),))
    with mlflow.start_run(run_name=f"{epoch=}-batch={i_batch}-{sim=}", nested=True) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "config.yaml"), "w") as fp:
                yaml.dump(mod_defaults, fp)
            actual_nk1.to_netcdf(os.path.join(td, "ground_truth.nc"))
            # with open(os.path.join(td, "weights.pkl"), "wb") as fi:
            #     pickle.dump(w_and_b, fi)
            eqx.tree_serialise_leaves(os.path.join(td, "weights.eqx"), w_and_b)

            mlflow.log_artifacts(td)
        misc.queue_sim(
            {
                "job_name": f"epw-{t_or_v}-epoch-{epoch}-batch-{i_batch}-sim-{sim}",
                "run_id": mlflow_run.info.run_id,
                "sim_type": "fluid",
                "run_type": t_or_v,
                "machine": "continuum-cpu",
            }
        )
        mlflow.set_tags({"status": "queued"})
        run_ids.append(mlflow_run.info.run_id)
        job_done.append(False)

    return run_ids, job_done


def eval_over_all():
    with open("../../configs/tf-1d/damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)
    trapping_models = helpers.get_models(defaults["models"])

    # batch_size = 16

    # modify config
    fks = xr.open_dataset("../../epws.nc")

    nus = np.copy(fks.coords[r"$\nu_{ee}$"].data[::3])
    k0s = np.copy(fks.coords["$k_0$"].data[::2])
    a0s = np.copy(fks.coords["$a_0$"].data[::3])
    train_val_sims = list(product(list(nus), list(k0s), list(a0s)))

    all_sims = list(
        product(
            list(np.copy(fks.coords[r"$\nu_{ee}$"].data)),
            list(np.copy(fks.coords["$k_0$"].data)),
            list(np.copy(fks.coords["$a_0$"].data)),
        )
    )

    test_sims = np.array(list(set(all_sims) - set(train_val_sims)))
    test_sim_inds = np.arange(test_sims.shape[0])
    np.random.shuffle(test_sim_inds)
    #     print(test_sim_inds)
    #     print(len(test_sim_inds))

    #     raise ValueError

    mlflow.set_experiment(defaults["mlflow"]["experiment"])
    with mlflow.start_run(run_name="test-all-nl-damping-opt", nested=True) as mlflow_run:
        run_ids, job_done = [], []
        for sim in tqdm(test_sim_inds, total=len(test_sim_inds)):
            nuee, k0, a0 = test_sims[sim]
            run_ids, job_done = queue_sim(
                fks, nuee, k0, a0, run_ids, job_done, trapping_models, 0, 0, sim, t_or_v="val"
            )

        wait_for_jobs(job_done, run_ids)
        val_loss = float(
            np.average(
                np.array([misc.get_this_metric_of_this_run("val_loss", queued_run_id) for queued_run_id in run_ids])
            )
        )
        mlflow.log_metrics({"test_loss": val_loss}, step=0)


if __name__ == "__main__":
    eval_over_all()
