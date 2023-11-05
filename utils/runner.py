from typing import Dict
import os, time, tempfile, yaml

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Solution
from jax import numpy as jnp
import numpy as np

import mlflow, pint
import equinox as eqx
import xarray as xr


from utils import plotters, misc


def get_helpers(mode):
    if mode == "tf-1d":
        from adept.tf1d import helpers
    elif mode == "sh-2d":
        from adept.sh2d import helpers
    elif mode == "vlasov-2d":
        from adept.vlasov2d import helpers
    elif mode == "envelope-2d":
        from adept.envelope2d import helpers
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
        "box_width": box_width,
        "sim_duration": sim_duration,
    }

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump(all_quantities, fi)


def run(cfg: Dict) -> Solution:
    helpers = get_helpers(cfg["mode"])

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "config.yaml"), "w") as fi:
            yaml.dump(cfg, fi)

        # get derived quantities
        cfg["grid"] = helpers.get_derived_quantities(cfg["grid"])
        misc.log_params(cfg)

        cfg = helpers.get_solver_quantities(cfg)
        cfg = helpers.get_save_quantities(cfg)

        write_units(cfg, td)

        models = helpers.get_models(cfg["models"])
        state = helpers.init_state(cfg)

        # run
        t0 = time.time()

        # @eqx.filter_jit
        t_and_s = helpers.get_terms_and_solver(cfg)

        def _run_(these_models):
            args = {"driver": cfg["drivers"]}
            if "models" in cfg:
                args["models"] = these_models

            return diffeqsolve(
                terms=t_and_s["terms"],
                solver=t_and_s["solver"],
                t0=cfg["grid"]["tmin"],
                t1=cfg["grid"]["tmax"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=state,
                args=args,
                # adjoint=diffrax.DirectAdjoint(),
                saveat=SaveAt(ts=cfg["save"]["t"]["ax"], fn=cfg["save"]["func"]["callable"]),
            )

        result = _run_(models)
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

        t0 = time.time()
        helpers.post_process(result, cfg, td)
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
        # log artifacts
        mlflow.log_artifacts(td)

    # fin

    return result


def remote_gradient(run_id):
    with mlflow.start_run(run_id=run_id, nested=True) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            mod_defaults = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=td)
            actual_nk1 = xr.open_dataarray(
                misc.download_file("ground_truth.nc", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td)
            )
            mod_defaults["grid"] = helpers.get_derived_quantities(mod_defaults["grid"])
            misc.log_params(mod_defaults)
            mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
            mod_defaults = helpers.get_save_quantities(mod_defaults)
            t0 = time.time()

            state = helpers.init_state(mod_defaults)
            mod_defaults["models"]["file"] = misc.download_file(
                "weights.eqx", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td
            )
            models = helpers.get_models(mod_defaults["models"])

            def loss(these_models):
                vf = helpers.VectorField(mod_defaults, models=these_models)
                args = {"driver": mod_defaults["drivers"]}

                results = diffeqsolve(
                    terms=ODETerm(vf),
                    solver=Tsit5(),
                    t0=mod_defaults["grid"]["tmin"],
                    t1=mod_defaults["grid"]["tmax"],
                    max_steps=mod_defaults["grid"]["max_steps"],
                    dt0=mod_defaults["grid"]["dt"],
                    y0=state,
                    args=args,
                    saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"], fn=mod_defaults["save"]["func"]["callable"]),
                )
                nk1 = (
                    jnp.abs(jnp.fft.fft(results.ys["x"]["electron"]["n"], axis=1)[:, 1])
                    * 2.0
                    / mod_defaults["grid"]["nx"]
                )
                return (
                    jnp.mean(
                        jnp.square(
                            (np.log10(actual_nk1.data + 1e-20) - jnp.log10(nk1 + 1e-20))
                            / np.log10(np.amax(actual_nk1.data))
                        )
                        * jnp.exp(-2 * (1 - (mod_defaults["save"]["t"]["ax"] / mod_defaults["save"]["t"]["tmax"])))
                    ),
                    results,
                )

            vg_func = eqx.filter_value_and_grad(loss, has_aux=True)
            (loss_val, results), grad = eqx.filter_jit(vg_func)(models)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4), "loss": float(loss_val)})

            # dump gradients
            eqx.tree_serialise_leaves(os.path.join(td, "grads.eqx"), grad)

            t0 = time.time()
            helpers.post_process(results, mod_defaults, td)
            plotters.mva(actual_nk1.data, mod_defaults, results, td, actual_nk1.coords)
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            # log artifacts
            mlflow.log_artifacts(td)
            mlflow.set_tags({"status": "completed"})


def remote_val(run_id):
    with mlflow.start_run(run_id=run_id, nested=True) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            mod_defaults = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=td)
            actual_nk1 = xr.open_dataarray(
                misc.download_file("ground_truth.nc", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td)
            )
            mod_defaults["grid"] = helpers.get_derived_quantities(mod_defaults["grid"])
            misc.log_params(mod_defaults)
            mod_defaults["grid"] = helpers.get_solver_quantities(mod_defaults["grid"])
            mod_defaults = helpers.get_save_quantities(mod_defaults)
            t0 = time.time()

            state = helpers.init_state(mod_defaults)
            mod_defaults["models"]["file"] = misc.download_file(
                "weights.eqx", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td
            )
            models = helpers.get_models(mod_defaults["models"])

            def loss(these_models):
                vf = helpers.VectorField(mod_defaults, models=these_models)
                args = {"driver": mod_defaults["drivers"]}
                results = diffeqsolve(
                    terms=ODETerm(vf),
                    solver=Tsit5(),
                    t0=mod_defaults["grid"]["tmin"],
                    t1=mod_defaults["grid"]["tmax"],
                    max_steps=mod_defaults["grid"]["max_steps"],
                    dt0=mod_defaults["grid"]["dt"],
                    y0=state,
                    args=args,
                    saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"], fn=mod_defaults["save"]["func"]["callable"]),
                )
                nk1 = (
                    jnp.abs(jnp.fft.fft(results.ys["x"]["electron"]["n"], axis=1)[:, 1])
                    * 2.0
                    / mod_defaults["grid"]["nx"]
                )
                return (
                    jnp.mean(
                        jnp.square(
                            (np.log10(actual_nk1.data + 1e-20) - jnp.log10(nk1 + 1e-20))
                            / np.log10(np.amax(actual_nk1.data))
                        )
                        * jnp.exp(-2 * (1 - (mod_defaults["save"]["t"]["ax"] / mod_defaults["save"]["t"]["tmax"])))
                    ),
                    results,
                )

            loss_val, results = eqx.filter_jit(loss)(models)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4), "val_loss": float(loss_val)})

            t0 = time.time()
            helpers.post_process(results, mod_defaults, td)
            plotters.mva(actual_nk1.data, mod_defaults, results, td, actual_nk1.coords)
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            # log artifacts
            mlflow.log_artifacts(td)
            mlflow.set_tags({"status": "completed"})
