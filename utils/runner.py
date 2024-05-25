from typing import Dict, Tuple
import os, time, tempfile, yaml


from diffrax import Solution
import mlflow, jax


from utils import misc

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def get_helpers(solver):
    if solver == "tf-1d":
        from adept.tf1d import helpers
    elif solver == "sh-2d":
        from adept.sh2d import helpers
    elif solver == "vlasov-1d":
        from adept.vlasov1d import helpers
    elif solver == "vlasov-1d2v":
        from adept.vlasov1d2v import helpers
    elif solver == "vlasov-2d":
        from adept.vlasov2d import helpers
    elif solver == "envelope-2d":
        from adept.lpse2d import helpers
    elif solver == "vfp-2d":
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

    helpers = get_helpers(cfg["solver"])  # gets the right helper functions depending on the desired simulation

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
        models = helpers.get_models(cfg["models"]) if "models" in cfg else {}

        # initialize the state for the solver - NB - this is solver specific
        state, args = helpers.init_state(cfg, td)

        # run
        t0 = time.time()
        _run_ = helpers.get_run_fn(cfg)

        _log_flops_(_run_, models, state, args, tqs)
        run_output = _run_(models, state, args, tqs)
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

        t0 = time.time()
        # NB - this is solver specific
        post_processing_output = helpers.post_process(run_output, cfg, td, args)  # post-processes the result
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})  # logs the post-process time to mlflow
        mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

        mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})  # logs the total time to mlflow

    # fin
    return run_output, post_processing_output


def _log_flops_(_run_, models, state, args, tqs):
    """
    Logs the number of flops to mlflow

    Args:
        _run_: The function that runs the simulation
        models: The models used in the simulation
        tqs: The time quantities used in the simulation

    """
    wrapped = jax.xla_computation(_run_)
    computation = wrapped(models, state, args, tqs)
    module = computation.as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
    flops_sum = analysis["flops"]
    mlflow.log_metrics({"total GigaFLOP": flops_sum / 1e9})  # logs the flops to mlflow
