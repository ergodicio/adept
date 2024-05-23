from typing import Dict, Tuple
import os, time, tempfile, yaml


from diffrax import diffeqsolve, SaveAt, Solution
import numpy as np
import equinox as eqx
import mlflow, pint, jax


from utils import misc

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def get_helpers(mode):
    if mode == "tf-1d":
        from adept.tf1d import helpers
    elif mode == "sh-2d":
        from adept.sh2d import helpers
    elif mode == "vlasov-1d":
        from adept.vlasov1d import helpers
    elif mode == "vlasov-1d2v":
        from adept.vlasov1d2v import helpers
    elif mode == "vlasov-2d":
        from adept.vlasov2d import helpers
    elif mode == "envelope-2d":
        from adept.lpse2d import helpers
    elif mode == "vfp-2d":
        from adept.vfp1d import helpers
    else:
        raise NotImplementedError("This solver approach has not been implemented yet")

    return helpers


def run_job(run_id, nested):
    with mlflow.start_run(run_id=run_id, nested=nested) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
            cfg = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=temp_path)
        run(cfg)


def run(cfg: Dict) -> Tuple[Solution, Dict]:
    """
    This function is the main entry point for running a simulation. It takes a configuration dictionary and returns a
    ``diffrax.Solution`` object and a dictionary of datasets.

    Args:
        cfg: A dictionary containing the configuration for the simulation.

    Returns:
        A tuple of a Solution object and a dictionary of ``xarray.dataset``s.

    """
    t__ = time.time()  # starts the timer

    helpers = get_helpers(cfg["mode"])  # gets the right helper functions depending on the desired simulation

    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
        with open(os.path.join(td, "config.yaml"), "w") as fi:
            yaml.dump(cfg, fi)

        # NB - this is not yet solver specific but should be
        cfg = helpers.write_units(cfg, td)  # writes the units to the temporary directory

        # NB - this is solver specific
        cfg = helpers.get_derived_quantities(cfg)  # gets the derived quantities from the configuration
        misc.log_params(cfg)  # logs the parameters to mlflow

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

        @eqx.filter_jit
        def _run_(_args_, time_quantities: Dict):

            if "terms" in cfg.keys():
                args["terms"] = cfg["terms"]

            return diffeqsolve(
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

        _log_flops_(_run_, args, tqs)
        result = _run_(args, tqs)
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

        t0 = time.time()
        # NB - this is solver specific
        datasets = helpers.post_process(result, cfg, td, args)  # post-processes the result
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})  # logs the post-process time to mlflow
        mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

        mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})  # logs the total time to mlflow

    # fin
    return result, datasets


def _log_flops_(_run_, models, tqs):
    """
    Logs the number of flops to mlflow

    Args:
        _run_: The function that runs the simulation
        models: The models used in the simulation
        tqs: The time quantities used in the simulation

    """
    wrapped = jax.xla_computation(_run_)
    computation = wrapped(models, tqs)
    module = computation.as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
    flops_sum = analysis["flops"]
    mlflow.log_metrics({"total GigaFLOP": flops_sum / 1e9})  # logs the flops to mlflow
