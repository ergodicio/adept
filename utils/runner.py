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

    lambda0 = _Q(cfg["units"]["laser wavelength"])
    w0 = (2 * np.pi / lambda0 * ureg.c).to("rad/s")
    t0 = (1 / w0).to("fs")
    n0 = (w0**2 * ureg.m_e * ureg.epsilon_0 / ureg.e**2.0).to("1/cc")
    T0 = _Q(cfg["units"]["electron temperature"]).to("eV")
    v0 = np.sqrt(2.0 * T0 / (ureg.m_e)).to("m/s")
    debye_length = (v0 / w0).to("nm")

    logLambda_ee = 23.5 - np.log(n0.magnitude**0.5 / T0.magnitude**-1.25)
    logLambda_ee -= (1e-5 + (np.log(T0.magnitude) - 2) ** 2.0 / 16) ** 0.5
    nuee = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0.magnitude**1.5, "Hz")
    nuee_norm = nuee / w0

    # if (Ti * me / mi) < Te:
    #     if Te > 10 * Z ^ 2:
    #         logLambda_ei = 24 - np.log(ne.magnitude**0.5 / Te.magnitude)
    #     else:
    #         logLambda_ei = 23 - np.log(ne.magnitude**0.5 * Z * Te.magnitude**-1.5)
    # else:
    #     logLambda_ei = 30 - np.log(ni.magnitude**0.5 * Z**2 / mu * Ti.magnitude**-1.5)

    # nuei = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0**1.5, "Hz")
    # nuee_norm = nuee / w0

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * debye_length).to("microns")
    if "ymax" in cfg["grid"].keys():
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * debye_length).to("microns")
    else:
        box_width = "inf"
    sim_duration = (cfg["grid"]["tmax"] * t0).to("ps")

    all_quantities = {
        "w0": str(w0),
        "t0": str(t0),
        "n0": str(n0),
        "v0": str(v0),
        "T0": str(T0),
        "lambda_D": str(debye_length),
        "logLambda_ee": str(logLambda_ee),
        "nuee": str(nuee),
        "nuee_norm": str(nuee_norm),
        "box_length": str(box_length),
        "box_width": str(box_width),
        "sim_duration": str(sim_duration),
    }

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump(all_quantities, fi)


def run_job(run_id, nested):
    with mlflow.start_run(run_id=run_id, nested=nested) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
            cfg = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=temp_path)
        run(cfg)


def run(cfg: Dict) -> Tuple[Solution, Dict]:
    t__ = time.time()

    helpers = get_helpers(cfg["mode"])

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "config.yaml"), "w") as fi:
            yaml.dump(cfg, fi)

        # get derived quantities
        cfg["grid"] = helpers.get_derived_quantities(cfg["grid"])
        misc.log_params(cfg)

        cfg["grid"] = helpers.get_solver_quantities(cfg)
        cfg = helpers.get_save_quantities(cfg)

        write_units(cfg, td)

        models = helpers.get_models(cfg["models"]) if "models" in cfg else None
        state = helpers.init_state(cfg)

        # run
        t0 = time.time()

        diffeqsolve_quants = helpers.get_diffeqsolve_quants(cfg)

        @eqx.filter_jit
        def _run_(these_models):
            args = {"drivers": cfg["drivers"]}
            if these_models is not None:
                args["models"] = these_models

            return diffeqsolve(
                terms=diffeqsolve_quants["terms"],
                solver=diffeqsolve_quants["solver"],
                t0=cfg["grid"]["tmin"],
                t1=cfg["grid"]["tmax"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=state,
                args=args,
                saveat=SaveAt(**diffeqsolve_quants["saveat"]),
            )

        result = _run_(models)
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

        t0 = time.time()
        datasets = helpers.post_process(result, cfg, td)
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
        mlflow.log_artifacts(td)

        mlflow.log_metrics({"total_time": round(time.time() - t__, 4)})

    # fin

    return result, datasets
