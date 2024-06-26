#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import yaml, os, argparse, tempfile
from itertools import product

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

import mlflow, optax
import equinox as eqx
from tqdm import tqdm
import xarray as xr

from utils import misc
from adept.lpse2d import helpers

import parsl
from parsl.config import Config
from parsl.providers import SlurmProvider, PBSProProvider, LocalProvider
from parsl.launchers import SrunLauncher, MpiExecLauncher
from parsl.executors import HighThroughputExecutor
from parsl.app.app import python_app


def _modify_defaults_(defaults, k0, nuee, a0):
    defaults["driver"]["E2"]["k0"] = float(k0)
    defaults["driver"]["E2"]["a0"] = float(1000 * a0)
    defaults["driver"]["E2"]["w0"] = float(1.5 * k0**2.0)
    defaults["terms"]["epw"]["trapping"]["nuee"] = float(nuee)
    defaults["terms"]["epw"]["trapping"]["kld"] = float(k0)

    xmax = float(2.0 * np.pi / k0)
    defaults["grid"]["xmax"] = xmax

    return defaults


@python_app
def remote_run(run_id, t_or_v):
    from jax import config

    config.update("jax_enable_x64", True)

    import mlflow, tempfile, time, os
    import numpy as np
    import xarray as xr
    from diffrax import diffeqsolve, ODETerm, SaveAt
    from jax import numpy as jnp
    import equinox as eqx

    from utils import misc
    from adept.lpse2d import helpers
    from adept.lpse2d.core import integrator

    with mlflow.start_run(run_id=run_id, nested=True) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=os.getenv("BASE_TEMPDIR")) as td:
            mod_defaults = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=td)
            actual_ek1 = xr.open_dataarray(
                misc.download_file("ground_truth.nc", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td)
            )
            mod_defaults = helpers.get_derived_quantities(mod_defaults)
            misc.log_params(mod_defaults)
            mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
            mod_defaults = helpers.get_save_quantities(mod_defaults)
            t0 = time.time()

            state = helpers.init_state(mod_defaults, td)
            mod_defaults["models"]["file"] = misc.download_file(
                "weights.eqx", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td
            )
            models = helpers.get_models(mod_defaults["models"])
            vf = integrator.SpectralPotential(mod_defaults)

            loss_t = np.linspace(200, 400, 64)
            t_factor = np.exp(-2 * (1 - (loss_t / mod_defaults["grid"]["tmax"])))

            interp_actual_ek1 = np.interp(loss_t, actual_ek1.coords["t"].data, actual_ek1.data)
            log_actual = np.log10(interp_actual_ek1.data)

            def loss(these_models):
                args = {"driver": mod_defaults["driver"], "nu_g": these_models["nu_g"]}

                # if not debug:
                results = diffeqsolve(
                    terms=ODETerm(vf),
                    solver=integrator.Stepper(),
                    t0=mod_defaults["grid"]["tmin"],
                    t1=mod_defaults["grid"]["tmax"],
                    max_steps=mod_defaults["grid"]["max_steps"],
                    dt0=mod_defaults["grid"]["dt"],
                    y0=state,
                    args=args,
                    saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"]),
                )
                phi = results.ys["phi"]
                ek1 = 1j * mod_defaults["grid"]["kx"][None, :, None] * phi.view(np.complex128)
                ek1 = jnp.mean(jnp.abs(ek1[:, -1, :]), axis=-1)
                ek1 = ek1 / jnp.amax(ek1) * np.amax(actual_ek1.data)
                interp_ek1 = jnp.interp(loss_t, mod_defaults["save"]["t"]["ax"], ek1)

                return (
                    jnp.mean(jnp.square((log_actual - jnp.log10(interp_ek1)) / np.amax(log_actual)) * t_factor),
                    results,
                )

            if t_or_v == "grad":
                vg_func = eqx.filter_value_and_grad(loss, has_aux=True)
                (loss_val, results), grad = eqx.filter_jit(vg_func)(models)

                # dump gradients
                eqx.tree_serialise_leaves(os.path.join(td, "grads.eqx"), grad)
            else:
                loss_val, results = eqx.filter_jit(loss)(models)
                grad = None

            mlflow.log_metrics({"run_time": round(time.time() - t0, 4), "loss": float(loss_val)})

            t0 = time.time()
            helpers.post_process(results.ts, results.ys, mod_defaults, td)
            helpers.mva(actual_ek1.data, mod_defaults, results, td, actual_ek1.coords)
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            # log artifacts
            mlflow.log_artifacts(td)
            mlflow.set_tags({"status": "completed"})

    misc.export_run(mlflow_run.info.run_id)
    return loss_val, grad


def remote_train_loop(batch_size):
    with open("../adept/configs/envelope-2d/damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)
    trapping_models = helpers.get_models(defaults["models"])
    optimizer = optax.adam(4e-3)
    opt_state = optimizer.init(eqx.filter(trapping_models, eqx.is_array))

    fks = xr.open_dataset("./epws.nc")
    nus = np.copy(fks.coords[r"$\nu_{ee}$"].data[::3])
    k0s = np.copy(fks.coords["$k_0$"].data[::2])
    a0s = np.copy(fks.coords["$a_0$"].data[::3])
    all_sims = np.array(list(product(nus, k0s, a0s)))
    print(f"{len(all_sims)=}")

    rng = np.random.default_rng(420)

    train_sims = rng.choice(
        np.arange(all_sims.shape[0]), int(0.85 * all_sims.shape[0] / batch_size) * batch_size, replace=False
    )
    val_sims = np.array(list(set(np.arange(all_sims.shape[0])) - set(train_sims)))

    mlflow.set_experiment(defaults["mlflow"]["experiment"])
    val_loss = 1.0e10

    with mlflow.start_run(run_name="damping-opt", nested=True) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=os.environ["BASE_TEMPDIR"]) as td:
            with open("../adept/configs/envelope-2d/damping.yaml", "r") as file:
                dump_defaults = yaml.safe_load(file)
            with open(os.path.join(td, "config.yaml"), "w") as file:
                yaml.dump(dump_defaults, file)
            mlflow.log_artifacts(td)
        misc.export_run(mlflow_run.info.run_id, prefix="ingest")

        print(f"optimizer {mlflow_run.info.run_id=}")
        for epoch in range(20):
            epoch_loss = 0.0
            rng.shuffle(train_sims)
            train_batches = all_sims[train_sims].reshape((-1, batch_size, 3))
            for i_batch, batch in (pbar := tqdm(enumerate(train_batches), total=len(train_batches))):
                futures = [
                    queue_sim(fks, nuee, k0, a0, trapping_models, epoch, i_batch, sim)
                    for sim, (nuee, k0, a0) in enumerate(batch)
                ]
                results = [future.result() for future in futures]
                loss_vals = [result[0] for result in results]
                grads = [result[1] for result in results]
                trapping_models = misc.update_w_and_b(
                    grads, optimizer, opt_state, eqx.filter(trapping_models, eqx.is_array)
                )

                batch_loss = float(np.sum(np.array(loss_vals)))
                mlflow.log_metrics({"batch_loss": batch_loss}, step=i_batch + epoch * len(train_batches))
                epoch_loss = epoch_loss + batch_loss
                pbar.set_description(
                    f"{batch_loss=:.2e}, avg_loss={epoch_loss / (len(batch) * (i_batch+ 1)):.2e}, {val_loss=:.2e}"
                )

            mlflow.log_metrics({"epoch_loss": epoch_loss / len(train_sims)}, step=epoch)

            # validation
            futures = [
                queue_sim(fks, nuee, k0, a0, trapping_models, epoch, i_batch, sim, t_or_v="val")
                for sim, (nuee, k0, a0) in enumerate(all_sims[val_sims])
            ]
            results = [future.result() for future in futures]
            loss_vals = [result[0] for result in results]
            val_loss = float(np.mean(np.array(loss_vals)))

            mlflow.log_metrics({"val_epoch_loss": val_loss}, step=epoch)
            pbar.set_description(
                f"{batch_loss=:.2e}, avg_loss={epoch_loss / (len(batch) * (i_batch+ 1)):.2e}, {val_loss=:.2e}"
            )

            misc.export_run(mlflow_run.info.run_id, prefix="parent", step=epoch)


def queue_sim(fks, nuee, k0, a0, w_and_b, epoch, i_batch, sim, t_or_v="grad"):
    with open("../adept/configs/envelope-2d/damping.yaml", "r") as file:
        defaults = yaml.safe_load(file)

    mod_defaults = _modify_defaults_(defaults, k0=float(k0), a0=float(a0), nuee=float(nuee))
    locs = {"$k_0$": k0, "$a_0$": a0, r"$\nu_{ee}$": nuee}
    actual_ek1 = xr.DataArray(fks["force-(k_x)"].loc[locs].data[:, 1], coords=(("t", fks.coords["t"].data),))
    with mlflow.start_run(run_name=f"{t_or_v}-{epoch=}-batch={i_batch}-{sim=}", nested=True) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=os.getenv("BASE_TEMPDIR")) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fp:
                yaml.dump(mod_defaults, fp)
            actual_ek1.to_netcdf(os.path.join(td, "ground_truth.nc"))
            eqx.tree_serialise_leaves(os.path.join(td, "weights.eqx"), w_and_b)

            mlflow.log_artifacts(td)

        mlflow.set_tags({"status": "queued"})

    return remote_run(mlflow_run.info.run_id, t_or_v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train damping")
    parser.add_argument("--mode", help="enter mode")
    parser.add_argument("--run_id", help="enter run id")
    parser.add_argument("--machine", help="enter machine")
    args = parser.parse_args()

    if args.mode == "train":
        shared_args = dict(walltime="8:00:00", cmd_timeout=120, nodes_per_block=1)
        if args.machine == "storm":
            # typhoon
            this_provider = PBSProProvider
            sched_args = [
                # "#PBS -S /bin/bash",
                # "#PBS -l nodes=1:ppn=36",
                "#PBS -N train-nlepw",
                "#PBS -j oe",
            ]
            provider_args = dict(
                queue="slpi",
                # account="m4434",
                scheduler_options="\n".join(sched_args),
                worker_init="export PYTHONPATH='$PYTHONPATH:/home/ajog/laplax/'; \
                            export BASE_TEMPDIR='/b1/ajog/tmp/'; \
                            export MLFLOW_TRACKING_URI='/b1/ajog/mlflow'; \
                            module load anaconda/5.0.1; conda activate laplax-cpu",
                cpus_per_node=12,
                launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=12 --ppn 1"),
            )
            provider_args = {**provider_args, **shared_args}
        elif args.machine == "nersc":
            # nersc
            this_provider = SlurmProvider
            sched_args = ["#SBATCH -C cpu", "#SBATCH --qos=regular"]
            provider_args = dict(
                partition=None,
                account="m4434",
                scheduler_options="\n".join(sched_args),
                worker_init="module load python; source /global/homes/a/archis/laplax/venv/bin/activate; \
                        export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/laplax/'; \
                        export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                        export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';",
                launcher=SrunLauncher(overrides="-c 128"),
                cores_per_node=128,
            )
            provider_args = {**provider_args, **shared_args}
        elif args.machine == "local":
            # nersc
            this_provider = LocalProvider
            provider_args = dict(
                # scheduler_options="\n".join(sched_args),
                worker_init="module load python; source /global/homes/a/archis/laplax/venv/bin/activate; \
                        export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/laplax/'; \
                        export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                        export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';",
                init_blocks=1,
                max_blocks=1,
            )
        else:
            raise NotImplementedError(f"{args.machine} not implemented as a provider")

        batch_size = 34
        htex = HighThroughputExecutor(
            label="train-nlepw",
            provider=this_provider(**provider_args),
            cores_per_worker=int(128 // batch_size),
            max_workers=batch_size,
            cpu_affinity="block",
        )
        print(f"{htex.workers_per_node=}")
        config = Config(executors=[htex], retries=4)

        # load the Parsl config
        parsl.load(config)

        remote_train_loop(batch_size)
    elif args.mode == "run":
        remote_run(run_id=args.run_id, t_or_v="val")
    else:
        raise NotImplementedError
    # remote_train_loop(batch_size)
