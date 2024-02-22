from typing import Dict, Tuple
import os, time, tempfile, yaml


from diffrax import diffeqsolve, SaveAt, Solution
import numpy as np
import equinox as eqx
import mlflow, pint


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
    elif mode == "vlasov-2d":
        from adept.vlasov2d import helpers
    elif mode == "envelope-2d":
        from adept.lpse2d import helpers
    else:
        raise NotImplementedError("This solver approach has not been implemented yet")

    return helpers


def write_units(cfg, td):
    ureg = pint.UnitRegistry()
    _Q = ureg.Quantity

    n0 = _Q(cfg["units"]["normalizing density"]).to("1/cc")
    T0 = _Q(cfg["units"]["normalizing temperature"]).to("eV")

    wp0 = np.sqrt(n0 * ureg.e**2.0 / (ureg.m_e * ureg.epsilon_0)).to("rad/s")
    tp0 = (1 / wp0).to("fs")

    v0 = np.sqrt(2.0 * T0 / ureg.m_e).to("m/s")
    x0 = (v0 / wp0).to("nm")
    c_light = _Q(1.0 * ureg.c).to("m/s") / v0
    beta = (v0 / ureg.c).to("dimensionless")

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * x0).to("microns")
    if "ymax" in cfg["grid"].keys():
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * x0).to("microns")
    else:
        box_width = "inf"
    sim_duration = (cfg["grid"]["tmax"] * tp0).to("ps")

    # collisions
    logLambda_ee = 23.5 - np.log(n0.magnitude**0.5 / T0.magnitude**-1.25)
    logLambda_ee -= (1e-5 + (np.log(T0.magnitude) - 2) ** 2.0 / 16) ** 0.5
    nuee = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0.magnitude**1.5, "Hz")
    nuee_norm = nuee / wp0

    all_quantities = {
        "wp0": wp0,
        "tp0": tp0,
        "n0": n0,
        "v0": v0,
        "T0": T0,
        "c_light": c_light,
        "beta": beta,
        "x0": x0,
        "nuee": nuee,
        "logLambda_ee": logLambda_ee,
        "box_length": box_length,
        "box_width": box_width,
        "sim_duration": sim_duration,
    }

    cfg["units"]["derived"] = all_quantities

    cfg["grid"]["beta"] = beta.magnitude

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump({k: str(v) for k, v in all_quantities.items()}, fi)

    return cfg


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

    with tempfile.TemporaryDirectory() as td:  # creates a temporary directory to store the simulation data
        with open(os.path.join(td, "config.yaml"), "w") as fi:  # writes the configuration to the temporary directory
            yaml.dump(cfg, fi)

        # NB - this is not yet solver specific but should be
        cfg = write_units(cfg, td)  # writes the units to the temporary directory

        # NB - this is solver specific
        cfg = helpers.get_derived_quantities(cfg)  # gets the derived quantities from the configuration
        misc.log_params(cfg)  # logs the parameters to mlflow

        # NB - this is solver specific
        cfg["grid"] = helpers.get_solver_quantities(cfg)  # gets the solver quantities from the configuration
        cfg = helpers.get_save_quantities(cfg)  # gets the save quantities from the configuration

        # create the dictionary of time quantities that is given to the time integrator and save manager
        tqs = {
            "t0": cfg["grid"]["tmin"],
            "t1": cfg["grid"]["tmax"],
            "max_steps": cfg["grid"]["max_steps"],
            "save_t0": cfg["grid"]["tmin"],
            "save_t1": cfg["grid"]["tmax"],
            "save_nt": cfg["grid"]["tmax"],
        }

        # in case you are using ML models
        models = helpers.get_models(cfg["models"]) if "models" in cfg else None

        # initialize the state for the solver - NB - this is solver specific
        state = helpers.init_state(cfg)

        # NB - this is solver specific
        # Remember that we rely on the diffrax library to provide the ODE (time, usually) integrator
        # So we need to create the diffrax terms, solver, and save objects
        diffeqsolve_quants = helpers.get_diffeqsolve_quants(cfg)

        # run
        t0 = time.time()

        @eqx.filter_jit
        def _run_(these_models, time_quantities: Dict):
            args = {"drivers": cfg["drivers"]}
            if these_models is not None:
                args["models"] = these_models
            if "terms" in cfg.keys():
                args["terms"] = cfg["terms"]

            return diffeqsolve(
                terms=diffeqsolve_quants["terms"],
                solver=diffeqsolve_quants["solver"],
                t0=time_quantities["t0"],
                t1=time_quantities["t1"],
                max_steps=time_quantities["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=state,
                args=args,
                saveat=SaveAt(**diffeqsolve_quants["saveat"]),
            )

        result = _run_(models, tqs)
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})  # logs the run time to mlflow

        t0 = time.time()
        # NB - this is solver specific
        datasets = helpers.post_process(result, cfg, td)  # post-processes the result
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})  # logs the post-process time to mlflow
        mlflow.log_artifacts(td)  # logs the temporary directory to mlflow

        mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})  # logs the total time to mlflow

    # fin
    return result, datasets
