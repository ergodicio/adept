from typing import Dict
import os, time, tempfile, yaml

import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Solution
import numpy as np
import equinox as eqx

import mlflow, pint

from adept.sh2d.utils import helpers, save
from utils import misc


def write_units(cfg, td):
    ureg = pint.UnitRegistry()
    _Q = ureg.Quantity

    lambda0 = _Q(cfg["units"]["laser_wavelength"])
    w0 = (2 * np.pi / lambda0 * ureg.c).to("rad/s")
    t0 = (1 / w0).to("fs")
    n0 = (w0**2 * ureg.m_e * ureg.epsilon_0 / ureg.e**2.0).to("1/cc")
    nee = _Q(cfg["units"]["density for collisions"])
    T0 = _Q(cfg["units"]["electron temperature"]).to("eV")
    v0 = np.sqrt(2.0 * T0 / (ureg.m_e)).to("m/s")
    debye_length = (v0 / w0).to("nm")

    logLambda_ee = 23.5 - np.log(nee.magnitude**0.5 * T0.magnitude**-1.25)
    logLambda_ee -= (1e-5 + (np.log(T0.magnitude) - 2) ** 2.0 / 16) ** 0.5
    nuee = _Q(2.91e-6 * nee.magnitude * logLambda_ee / T0.magnitude**1.5, "Hz")
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
    sim_duration = (cfg["grid"]["tmax"] * t0).to("ps")

    all_quantities = {
        "w0": w0,
        "t0": t0,
        "n0": n0,
        "v0": v0,
        "T0": T0,
        "lambda_D": debye_length,
        "logLambda_ee": logLambda_ee,
        "nuee": nuee,
        "nuee_norm": nuee_norm,
        "box_length": box_length,
        "sim_duration": sim_duration,
    }
    all_quantities_str = {k: str(v) for k, v in all_quantities.items()}

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump(all_quantities_str, fi)

    print("units: ")
    print(all_quantities_str)
    print()

    mlflow.log_artifacts(td)

    return all_quantities


def run(cfg: Dict) -> Solution:
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "config.yaml"), "w") as fi:
            yaml.dump(cfg, fi)

        # get derived quantities
        cfg["grid"] = helpers.get_derived_quantities(cfg["grid"])
        misc.log_params(cfg)

        cfg["grid"] = helpers.get_solver_quantities(cfg["grid"])
        cfg = helpers.get_save_quantities(cfg)

        cfg["units"]["derived"] = write_units(cfg, td)

        state = helpers.init_state(cfg)

        # run
        t0 = time.time()

        @eqx.filter_jit
        def _run_():
            vvf = helpers.VlasovVectorField(cfg)
            cvf = helpers.FokkerPlanckVectorField(cfg)
            args = {"driver": cfg["drivers"], "b_ext": 0.0}
            return diffeqsolve(
                terms=diffrax.MultiTerm(ODETerm(vvf), ODETerm(cvf)),
                solver=helpers.ExplicitEStepper(),
                t0=cfg["grid"]["tmin"],
                t1=cfg["grid"]["tmax"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=state,
                args=args,
                # adjoint=diffrax.DirectAdjoint(),
                saveat=SaveAt(ts=cfg["save"]["t"]["ax"]),  # , fn=cfg["save"]["func"]["callable"]),
            )

        print("starting run")
        result = _run_()
        print("finished run")
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

        t0 = time.time()
        post_process(result, cfg, td)
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
        # log artifacts
        mlflow.log_artifacts(td)

    # fin

    return result


def post_process(result, cfg: Dict, td: str) -> None:
    os.makedirs(os.path.join(td, "binary"))
    os.makedirs(os.path.join(td, "plots"))

    xrs = save.save_arrays(result, td, cfg)
