import os, time
from typing import Dict

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import config, numpy as jnp
from equinox import filter_value_and_grad, filter_jit

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow, tempfile
import numpy as np
from diffrax import diffeqsolve, SaveAt
from adept.lpse2d import helpers
from utils.runner import run, run_job
from utils.misc import export_run, log_params


def run_one_val_and_grad(cfg):
    """
    This function is the main entry point for running a simulation. It takes a configuration dictionary and returns a
    ``diffrax.Solution`` object and a dictionary of datasets.

    Args:
        cfg: A dictionary containing the configuration for the simulation.

    Returns:
        A tuple of a Solution object and a dictionary of ``xarray.dataset``s.

    """
    t__ = time.time()  # starts the timer

    # helpers = get_helpers(cfg["mode"])  # gets the right helper functions depending on the desired simulation

    with tempfile.TemporaryDirectory() as td:
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

        # create the dictionary of time quantities that is given to the time integrator and save manager
        tqs = {
            "t0": 0.0,
            "t1": cfg["grid"]["tmax"],
            "max_steps": cfg["grid"]["max_steps"],
            "save_t0": 0.0,  # cfg["grid"]["tmin"],
            "save_t1": cfg["grid"]["tmax"],
            "save_nt": cfg["grid"]["tmax"],
        }

        # in case you are using ML models
        models = helpers.get_models(cfg["models"]) if "models" in cfg else None

        # initialize the state for the solver - NB - this is solver specific
        state, args = helpers.init_state(cfg, td)

        # NB - this is solver specific
        # Remember that we rely on the diffrax library to provide the ODE (time, usually) integrator
        # So we need to create the diffrax terms, solver, and save objects
        diffeqsolve_quants = helpers.get_diffeqsolve_quants(cfg)

        # run
        t0 = time.time()

        def _run_(_args_, time_quantities: Dict):

            _args_["drivers"]["E0"]["delta_omega"] = _args_["delta_omega"] * cfg["drivers"]["delta_omega_max"]

            _args_["drivers"]["E0"]["amplitude"] *= 2.0  # from -1 to 1 to -2 to 2
            _args_["drivers"]["E0"]["amplitude"] -= 1.0  # from -2 to 2 to -3 to 1
            _args_["drivers"]["E0"]["amplitude"] = jnp.power(
                10.0, _args_["drivers"]["E0"]["amplitude"]
            )  # from -3 to 1 to 1e-3 to 1
            _args_["drivers"]["E0"]["amplitude"] /= jnp.sum(jnp.square(_args_["drivers"]["E0"]["amplitude"]))

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

            e_sq = jnp.sum(jnp.abs(result.ys["epw"].view(jnp.complex128)[-100:]) ** 2)

            return e_sq, result

        # _log_flops_(_run_, args, tqs)
        vg_func = filter_jit(filter_value_and_grad(_run_, has_aux=True))
        (val, result), grad = vg_func(args, tqs)

        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

        t0 = time.time()
        # NB - this is solver specific
        datasets = helpers.post_process(result, cfg, td)  # post-processes the result
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})  # logs the post-process time to mlflow
        mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

        mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})  # logs the total time to mlflow

    # fin
    return result, datasets


def _run_():

    # for ll in np.linspace(120, 200, 5):

    #     for I0 in np.linspace(14, 15, 5):
    #         # with open("configs/tf-1d/damping.yaml", "r") as fi:
    #         with open(f"configs/envelope-2d/tpd.yaml", "r") as fi:
    #             cfg = yaml.safe_load(fi)

    #         cfg["units"]["laser intensity"] = f"{10**I0:.2E}W/cm^2"
    #         cfg["density"]["gradient scale length"] = f"{ll}um"
    #         mlflow.set_experiment(cfg["mlflow"]["experiment"])
    #         # modify config
    #         with mlflow.start_run(run_name=cfg["units"]["laser intensity"], log_system_metrics=True) as mlflow_run:
    #             result, datasets = run(cfg)

    #     # export_run(mlflow_run.info.run_id)
    pass


if __name__ == "__main__":

    with open(f"configs/envelope-2d/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    run_one_val_and_grad(cfg)
