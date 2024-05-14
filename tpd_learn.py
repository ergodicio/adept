import parsl
from parsl.config import Config
from parsl.providers import SlurmProvider, LocalProvider
from parsl.launchers import SrunLauncher
from parsl.executors import HighThroughputExecutor
from parsl.app.app import python_app
import logging, os
import dill as pickle
import equinox as eqx

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

from utils import misc


def run_one_val_and_grad(model_path, Te, L, I0, cfg_path, run_id):
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
    from diffrax import diffeqsolve, SaveAt
    from matplotlib import pyplot as plt
    import jax.numpy as jnp
    from equinox import filter_value_and_grad, filter_jit
    import mlflow, yaml
    from jax.flatten_util import ravel_pytree

    from adept.lpse2d import helpers, nn
    from utils.misc import log_params, export_run

    # log to logger
    logging.info(f"Running a run")

    with open(cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    logging.info(f"Config is loaded")

    cfg["density"]["gradient scale length"] = f"{L}um"
    cfg["units"]["laser intensity"] = f"{I0:.2e}W/cm^2"
    cfg["units"]["reference electron temperature"] = f"{Te}eV"

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

            # args["driver model"] = this_model

            def _run_(this_model, _args_, time_quantities: Dict):

                amps, phases = this_model(jnp.array([Te, L, I0]))
                # _args_ = {"drivers": {"E0": {}}}
                _args_["drivers"]["E0"]["amplitudes"] = jnp.tanh(amps)
                _args_["drivers"]["E0"]["initial_phase"] = jnp.tanh(phases)

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

                return jnp.log10(e_sq), (result, _args_["drivers"])

            # _log_flops_(_run_, args, tqs)
            model, hyperparams = nn.load(model_path)
            nn.save(os.path.join(td, f"model.eqx"), hyperparams, model)  # save the model

            logging.info(f"Compile and run has begun")
            vg_func = filter_jit(filter_value_and_grad(_run_, has_aux=True))
            (val, (result, used_driver)), grad = vg_func(model, args, tqs)

            logging.info(f"Run completed, post processing has begun")
            with open(os.path.join(td, "used_driver.pkl"), "wb") as fi:
                pickle.dump(used_driver, fi)

            dw_over_w = used_driver["E0"]["delta_omega"]  # / cfg["units"]["derived"]["w0"] - 1
            fig, ax = plt.subplots(1, 3, figsize=(13, 5), tight_layout=True)
            ax[0].plot(dw_over_w, used_driver["E0"]["amplitudes"], "o")
            ax[0].grid()
            ax[0].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
            ax[0].set_ylabel("$|E|$", fontsize=14)
            ax[1].semilogy(dw_over_w, used_driver["E0"]["amplitudes"], "o")
            ax[1].grid()
            ax[1].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
            ax[1].set_ylabel("$|E|$", fontsize=14)
            ax[2].plot(dw_over_w, used_driver["E0"]["initial_phase"], "o")
            ax[2].grid()
            ax[2].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
            ax[2].set_ylabel(r"$\angle E$", fontsize=14)
            plt.savefig(os.path.join(td, "learned_bandwidth.png"), bbox_inches="tight")
            plt.close()

            flat_grad, _ = ravel_pytree(grad)
            mlflow.log_metrics({"grad_l2_norm": float(jnp.linalg.norm(flat_grad))})
            mlflow.log_metrics({"grad_l1_norm": float(jnp.linalg.norm(flat_grad, ord=1))})
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

            t0 = time.time()
            # NB - this is solver specific
            datasets = helpers.post_process(result, cfg, td)  # post-processes the result
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})  # logs the post-process time to mlflow
            mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

            mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})  # logs the total time to mlflow

        # fin
        logging.info(f"Done!")

    export_run(run_id)
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
                    export JAX_ENABLE_X64=True;\
                    export MLFLOW_EXPORT=True",
            init_blocks=1,
            max_blocks=1,
        )

        htex = HighThroughputExecutor(
            available_accelerators=4, label="tpd-learn", provider=this_provider(**provider_args), cpu_affinity="block"
        )
        print(f"{htex.workers_per_node=}")

    elif parsl_provider == "gpu":
        logging.info("Running with Slurm provider")

        this_provider = SlurmProvider
        sched_args = ["#SBATCH -C gpu&hbm80g", "#SBATCH --qos=regular"]
        provider_args = dict(
            partition=None,
            account="m4490_g",
            scheduler_options="\n".join(sched_args),
            worker_init="export SLURM_CPU_BIND='cores';\
                    source /pscratch/sd/a/archis/venvs/adept-gpu/bin/activate; \
                    module load cudnn/8.9.3_cuda12.lua; \
                    export PYTHONPATH='$PYTHONPATH:/global/homes/a/archis/adept/'; \
                    export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                    export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';\
                    export JAX_ENABLE_X64=True;\
                    export MLFLOW_EXPORT=True",
            launcher=SrunLauncher(overrides="--gpus-per-node 4 -c 128"),
            walltime="1:00:00",
            cmd_timeout=120,
            nodes_per_block=1,
            # init_blocks=1,
            max_blocks=3,
        )

        htex = HighThroughputExecutor(
            available_accelerators=4, label="tpd-learn", provider=this_provider(**provider_args), cpu_affinity="block"
        )
        print(f"{htex.workers_per_node=}")

    config = Config(executors=[htex], retries=4)

    # load the Parsl config
    parsl.load(config)


if __name__ == "__main__":
    import uuid, operator
    from itertools import product
    from tqdm import tqdm

    logging.basicConfig(filename=f"adept-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    # use the logger to note that we're running a parsl job
    logging.info("Running with parsl")

    import jax
    from jax.flatten_util import ravel_pytree

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    import optax

    setup_parsl("gpu")
    run_one_val_and_grad = python_app(run_one_val_and_grad)

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx
    from adept.lpse2d import nn

    with open(f"/global/homes/a/archis/adept/configs/envelope-2d/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["opt"] = {"learning_rate": 0.002, "optimizer": "adam"}
    cfg["nn-hp"] = hyperparams = {
        "encoder_width": 4,
        "encoder_depth": 2,
        "decoder_width": 4,
        "decoder_depth": 2,
        "input_width": 3,
        "output_width": 64,
        "latent_width": 4,
        "key": 42,
    }

    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name="learn-tpd") as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        misc.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    misc.export_run(parent_run_id)

    # create the dataset with the appropriate independent variables
    input_names = ("Te", "L", "I0")

    # 125 simulations in total
    Tes = np.linspace(2000, 4000, 4)
    Ls = np.linspace(200, 500, 4)
    I0s = np.linspace(3e14, 5e15, 6)

    rng = np.random.default_rng(42)

    # batch logic
    all_inputs = np.array(list(product(Tes, Ls, I0s)))

    batch_size = 16
    num_batches = all_inputs.shape[0] // batch_size
    batch_inds = np.arange(num_batches)

    # restart mlflow run that was initialized in the optimizer
    # we did that over there so that the appropriate nesting of the mlflow run can take place
    # remember this is a separate process than what is happening in __main__
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=cfg["opt"]["learning_rate"])
        # optax.chain(optax.clip(cfg["opt"]["clip"]), o)  # initialize the optimizer
        model = nn.DriverModel(**hyperparams)  # initialize the model
        opt_state = opt.init(eqx.filter(model, eqx.is_array))  # initialize the optimizer state

        with tempfile.TemporaryDirectory(
            dir=BASE_TEMPDIR
        ) as td:  # create a temporary directory for optimizer run artifacts
            os.makedirs(os.path.join(td, "model-history"), exist_ok=True)  # create a directory for model history
            with open(cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            for i in range(1000):  # 1000 epochs
                rng.shuffle(all_inputs)

                for j, batch_ind in tqdm(enumerate(batch_inds), total=num_batches):  # iterate over the batches
                    nn.save(
                        model_path := os.path.join(td, "model-history", f"model-e-{i}-b-{j}.eqx"), hyperparams, model
                    )  # save the model
                    mlflow.log_artifacts(td)
                    misc.export_run(parent_run_id, prefix="artifact", step=i)
                    # this model and model_path are passed to the run_one_val_and_grad function for use in the simulation

                    batch_loss = 0.0  # initialize the batch loss

                    batch = all_inputs[batch_ind * batch_size : (batch_ind + 1) * batch_size]  # get the batch
                    val_and_grads = []  # initialize the list to store the values and gradient Futures
                    run_ids = []
                    for Te, L, I0 in batch:

                        with mlflow.start_run(nested=True, run_name=f"{Te=}-{L=}-{I0=:.2e}") as nested_run:
                            mlflow.log_params({"indep_var.Te": Te, "indep_var.L": L, "indep_var.I0": I0})

                        # get the futures for all the inputs
                        val_and_grads.append(
                            run_one_val_and_grad(model_path, Te, L, I0, cfg_path, run_id=nested_run.info.run_id)
                        )
                        run_ids.append(nested_run.info.run_id)  # store the run_id

                    vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                    val = np.mean([v for v, _ in vgs])  # get the mean of the loss values

                    avg_grad = misc.all_reduce_gradients(
                        [g for _, g in vgs], batch_size
                    )  # get the average of the gradients
                    flat_grad, _ = ravel_pytree(avg_grad)
                    mlflow.log_metrics({"batch grad norm": float(np.linalg.norm(flat_grad))})

                    # with open("./completed_run_ids.txt", "a") as f:
                    #     f.write("\n".join(run_ids) + "\n")

                    mlflow.log_metrics({"batch loss": float(val)}, step=i * batch_size + j)
                    misc.export_run(parent_run_id, prefix="parent", step=i)
                    updates, opt_state = opt.update(avg_grad, opt_state, model)
                    model = eqx.apply_updates(model, updates)

                    batch_loss += val

                mlflow.log_metrics({"epoch loss": float(batch_loss / num_batches)}, step=i)
