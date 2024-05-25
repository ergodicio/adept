from parsl.app.app import python_app
import logging, os
import equinox as eqx

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

from utils import misc


def run_one_val_and_grad(weights, cfg, run_id, mode="bwd"):
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

    from matplotlib import pyplot as plt
    import jax.numpy as jnp
    import mlflow, yaml
    from jax.flatten_util import ravel_pytree

    from adept.lpse2d import helpers, nn, run_fns
    from utils.misc import log_params, export_run

    # log to logger
    logging.info(f"Running a run")

    with open(cfg_path, "r") as fi:
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

            # _log_flops_(_run_, args, tqs)
            _run_ = run_fns.bandwidth(cfg, diffeqsolve_quants, state, indep_quants={}, mode=mode)

            logging.info(f"Compile and run has begun")

            (val, aux_out), grad = _run_(model, args, tqs)

            result = aux_out[0]
            used_driver = aux_out[1]

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
            metrics = {
                "grad_l2_norm": float(jnp.linalg.norm(flat_grad)),
                "grad_l1_norm": float(jnp.linalg.norm(flat_grad, ord=1)),
                "loss": float(val),
                "run_time": round(time.time() - t0, 4),
            }
            mlflow.log_metrics(metrics)
            if cfg["model"]["type"] == "VAE":
                mlflow.log_metrics({"kl_loss": float(jnp.sum(aux_out[2]))})

            t0 = time.time()
            # NB - this is solver specific
            datasets = helpers.post_process(result, cfg, td, {"drivers": used_driver})  # post-processes the result
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})  # logs the post-process time to mlflow
            mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

            mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})  # logs the total time to mlflow

        # fin
        logging.info(f"Done!")

    export_run(run_id)
    return val, grad


if __name__ == "__main__":
    import uuid
    from itertools import product
    from tqdm import tqdm

    logging.basicConfig(filename=f"runlog-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    # use the logger to note that we're running a parsl job
    logging.info("Running with parsl")

    import jax
    from jax.flatten_util import ravel_pytree

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    import optax

    misc.setup_parsl("local")
    run_one_val_and_grad = python_app(run_one_val_and_grad)

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx
    from adept.lpse2d import nn

    with open(f"/global/homes/a/archis/adept/configs/envelope-2d/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name="learn-tpd") as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        misc.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    misc.export_run(parent_run_id)

    rng = np.random.default_rng(487)
    initial_amps = rng.uniform(0, 1, cfg["drivers"]["E0"]["num_colors"])
    initial_phases = rng.uniform(0, 1, cfg["drivers"]["E0"]["num_colors"])
    weights = {"initial_amps": initial_amps, "initial_phases": initial_phases}

    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=cfg["opt"]["learning_rate"])
        opt_state = opt.init(weights)  # initialize the optimizer state

        with tempfile.TemporaryDirectory(
            dir=BASE_TEMPDIR
        ) as td:  # create a temporary directory for optimizer run artifacts
            os.makedirs(os.path.join(td, "model-history"), exist_ok=True)  # create a directory for model history
            with open(cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            for i in range(1000):  # 1000 epochs
                with mlflow.start_run(nested=True, run_name=f"epoch-{i}") as nested_run:
                    pass

                val, grad = run_one_val_and_grad(weights, cfg, run_id=nested_run.info.run_id, mode="bwd")

                flat_grad, _ = ravel_pytree(grad)
                mlflow.log_metrics({"grad norm": float(np.linalg.norm(flat_grad))}, step=i)
                mlflow.log_metrics({"loss": float(val)}, step=i)
                misc.export_run(parent_run_id, prefix="parent", step=i)
                updates, opt_state = opt.update(grad, opt_state, model)
                model = eqx.apply_updates(model, updates)
