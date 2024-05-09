import parsl
from parsl.config import Config
from parsl.providers import SlurmProvider, LocalProvider
from parsl.launchers import SrunLauncher
from parsl.executors import HighThroughputExecutor
from parsl.app.app import python_app
import logging, os

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

from utils import misc


def run_one_val_and_grad(these_params, run_id):
    """
    This function is the main entry point for running a simulation. It takes a configuration dictionary and returns a
    ``diffrax.Solution`` object and a dictionary of datasets.

    Args:
        cfg: A dictionary containing the configuration for the simulation.

    Returns:
        A tuple of a Solution object and a dictionary of ``xarray.dataset``s.

    """

    import os, time, pickle, tempfile, logging

    logger = logging.getLogger(__name__)

    if "BASE_TEMPDIR" in os.environ:
        BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
    else:
        BASE_TEMPDIR = None

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from typing import Dict
    from adept.lpse2d import helpers
    from utils.misc import log_params
    from diffrax import diffeqsolve, SaveAt
    from matplotlib import pyplot as plt
    import jax.numpy as jnp
    from equinox import filter_value_and_grad, filter_jit
    import mlflow, yaml

    # log to logger
    logging.info(f"Running a run")

    with open(f"/global/homes/a/archis/adept/configs/envelope-2d/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    logging.info(f"Config is loaded")

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(log_system_metrics=True, nested=True, run_id=run_id) as mlflow_run:
        t__ = time.time()  # starts the timer

        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            # NB - this is not yet solver specific but should be
            cfg = helpers.write_units(cfg, td)  # writes the units to the temporary directory

            # NB - this is solver specific
            cfg = helpers.get_derived_quantities(cfg)  # gets the derived quantities from the configuration
            log_params(cfg)  # logs the parameters to mlflow

            # NB - this is solver specific
            cfg["grid"] = helpers.get_solver_quantities(cfg)  # gets the solver quantities from the configuration
            cfg = helpers.get_save_quantities(cfg)  # gets the save quantities from the configuration

            logging.info(f"Solver quantities have been determined")

            # create the dictionary of time quantities that is given to the time integrator and save manager
            tqs = {
                "t0": 0.0,
                "t1": cfg["grid"]["tmax"],
                "max_steps": cfg["grid"]["max_steps"],
                "save_t0": 0.0,  # cfg["grid"]["tmin"],
                "save_t1": cfg["grid"]["tmax"],
                "save_nt": cfg["grid"]["tmax"],
            }

            # initialize the state for the solver - NB - this is solver specific
            state, args = helpers.init_state(cfg, td)
            logging.info(f"State is initialized")

            # NB - this is solver specific
            # Remember that we rely on the diffrax library to provide the ODE (time, usually) integrator
            # So we need to create the diffrax terms, solver, and save objects
            diffeqsolve_quants = helpers.get_diffeqsolve_quants(cfg)

            # run
            t0 = time.time()

            # args["drivers"]["E0"]["delta_omega"] = params["drivers"]["E0"]["delta_omega"]
            args["drivers"]["E0"]["amplitudes"] = these_params["drivers"]["E0"]["amplitudes"]
            args["drivers"]["E0"]["initial_phase"] = these_params["drivers"]["E0"]["initial_phase"]

            def _run_(_args_, time_quantities: Dict):

                # _args_["drivers"]["E0"]["delta_omega"] = jnp.tanh(_args_["drivers"]["E0"]["delta_omega"])
                _args_["drivers"]["E0"]["amplitudes"] = jnp.tanh(_args_["drivers"]["E0"]["amplitudes"])
                _args_["drivers"]["E0"]["initial_phase"] = jnp.tanh(_args_["drivers"]["E0"]["initial_phase"])

                _args_["drivers"]["E0"]["delta_omega"] = jnp.linspace(
                    -cfg["drivers"]["E0"]["delta_omega_max"],
                    cfg["drivers"]["E0"]["delta_omega_max"],
                    num=cfg["drivers"]["E0"]["num_colors"],
                )

                _args_["drivers"]["E0"]["amplitudes"] *= 2.0  # from [-1, 1] to [-2, 2]
                _args_["drivers"]["E0"]["amplitudes"] -= 2.0  # from [-2, 2] to [-4, 0]
                _args_["drivers"]["E0"]["amplitudes"] = jnp.power(
                    10.0, _args_["drivers"]["E0"]["amplitudes"]
                )  # from [-4, 0] to [1e-4, 1]
                _args_["drivers"]["E0"]["amplitudes"] /= jnp.sqrt(
                    jnp.sum(jnp.square(_args_["drivers"]["E0"]["amplitudes"]))
                )  # normalize

                _args_["drivers"]["E0"]["initial_phase"] *= jnp.pi  # from [-1, 1] to [-pi, pi]

                if "terms" in cfg.keys():
                    args["terms"] = cfg["terms"]

                result = diffeqsolve(
                    terms=diffeqsolve_quants["terms"],
                    solver=diffeqsolve_quants["solver"],
                    t0=time_quantities["t0"],
                    t1=time_quantities["t1"],
                    max_steps=cfg["grid"]["max_steps"],  # time_quantities["max_steps"],
                    dt0=cfg["grid"]["dt"],
                    y0=state,
                    args=_args_,
                    saveat=SaveAt(**diffeqsolve_quants["saveat"]),
                )

                e_sq = jnp.sum(jnp.abs(result.ys["epw"].view(jnp.complex128)) ** 2)

                return e_sq, (result, _args_)

            # _log_flops_(_run_, args, tqs)
            logging.info(f"Compile and run has begun")
            vg_func = filter_jit(filter_value_and_grad(_run_, has_aux=True))
            (val, (result, used_args)), grad = vg_func(args, tqs)

            logging.info(f"Run completed, post processing has begun")
            with open(os.path.join(td, "used_args.pkl"), "wb") as fi:
                pickle.dump(used_args, fi)

            dw_over_w = used_args["drivers"]["E0"]["delta_omega"]  # / cfg["units"]["derived"]["w0"] - 1
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
            ax[0].plot(dw_over_w, used_args["drivers"]["E0"]["amplitudes"], "o")
            ax[0].grid()
            ax[0].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
            ax[0].set_ylabel("$E$", fontsize=14)
            ax[1].semilogy(dw_over_w, used_args["drivers"]["E0"]["amplitudes"], "o")
            ax[1].grid()
            ax[1].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
            ax[1].set_ylabel("$E$", fontsize=14)
            plt.savefig(os.path.join(td, "learned_bandwidth.png"), bbox_inches="tight")
            plt.close()

            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

            t0 = time.time()
            # NB - this is solver specific
            datasets = helpers.post_process(result, cfg, td)  # post-processes the result
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})  # logs the post-process time to mlflow
            mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

            mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})  # logs the total time to mlflow

        # fin
        logging.info(f"Done!")

    return val, grad


def setup_parsl(parsl_provider="local"):

    if parsl_provider == "local":
        logging.info("Running with local provider")
        this_provider = LocalProvider
        provider_args = dict(
            worker_init="source /pscratch/sd/a/archis/venvs/adept-gpu/bin/activate; \
                    module load cudnn/8.9.3_cuda12.lua; \
                    export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'; \
                    export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                    export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow'; \
                    export MLFLOW_EXPORT=True",
            init_blocks=1,
            max_blocks=1,
        )

        htex = HighThroughputExecutor(
            available_accelerators=1, label="tpd-learn", provider=this_provider(**provider_args), cpu_affinity="block"
        )
        print(f"{htex.workers_per_node=}")

    elif parsl_provider == "gpu":
        logging.info("Running with Slurm provider")

        this_provider = SlurmProvider
        sched_args = ["#SBATCH -C gpu", "#SBATCH --qos=debug"]
        provider_args = dict(
            partition=None,
            account="m4490_g",
            scheduler_options="\n".join(sched_args),
            worker_init="source /pscratch/sd/a/archis/venvs/adept-gpu/bin/activate; \
                    module load cudnn/8.9.3_cuda12.lua; \
                    export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'; \
                    export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                    export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';\
                    export JAX_ENABLE_X64=True;\
                    export MLFLOW_EXPORT=True",
            launcher=SrunLauncher(overrides="--gpus-per-node 4 -c 64"),
            walltime="0:10:00",
            cmd_timeout=120,
            nodes_per_block=1,
        )

        htex = HighThroughputExecutor(
            available_accelerators=4, label="tpd-learn", provider=this_provider(**provider_args), cpu_affinity="block"
        )
        print(f"{htex.workers_per_node=}")

    config = Config(executors=[htex], retries=4)

    # load the Parsl config
    parsl.load(config)


if __name__ == "__main__":
    import uuid

    logging.basicConfig(filename=f"adept-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    run_parsl = True

    if run_parsl:
        # use the logger to note that we're running a parsl job
        logging.info("Running with parsl")

        import jax

        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_enable_x64", True)

        import optax

        setup_parsl("gpu")
        run_one_val_and_grad = python_app(run_one_val_and_grad)
    else:
        import optax

    import yaml, mlflow, tempfile, os
    import numpy as np

    with open(f"/global/homes/a/archis/adept/configs/envelope-2d/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["opt"] = {"learning_rate": 0.1, "clip": 1e5, "optimizer": "adam"}
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name="learn-tpd") as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        misc.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    misc.export_run(parent_run_id)

    # # restart mlflow run
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.chain(optax.clip(cfg["opt"]["clip"]), optax.adam(learning_rate=cfg["opt"]["learning_rate"]))
        params = {
            "drivers": {
                "E0": {
                    k: np.random.normal(size=cfg["drivers"]["E0"]["num_colors"])
                    for k in ["amplitudes", "initial_phase"]
                }
            }
        }
        opt_state = opt.init(params)

        for i in range(1000):
            old_params = params
            with mlflow.start_run(
                nested=True, run_name=f"log_amp-lines={cfg['drivers']['E0']['num_colors']}-sim={i}"
            ) as nested_run:
                pass

            val_and_grad = run_one_val_and_grad(old_params, run_id=nested_run.info.run_id)
            if run_parsl:
                val, grad = val_and_grad.result()
            else:
                val, grad = val_and_grad

            mlflow.log_metrics({"loss": float(val)}, step=i)
            misc.export_run(parent_run_id, prefix="parent", step=i)

            misc.export_run(nested_run.info.run_id)
            updates, opt_state = opt.update(grad, opt_state)
            params = optax.apply_updates(old_params, updates)
