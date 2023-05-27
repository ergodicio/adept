from typing import Dict
import os, time, tempfile

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Solution
from jax import numpy as jnp
import numpy as np

import mlflow
import equinox as eqx
import xarray as xr

import es1d
from utils import plotters, misc


def start_run(run_type, run_id):
    if run_type == "forward":
        just_forward(run_id, nested=False)
    elif run_type == "grad":
        remote_gradient(run_id)
    elif run_type == "val":
        remote_val(run_id)
    else:
        raise NotImplementedError


def run(cfg: Dict) -> Solution:
    if cfg["mode"] == "es-1d":
        helpers = es1d.helpers
    else:
        raise NotImplementedError("This mode hasn't been implemented yet")

    # get derived quantities
    cfg["grid"] = helpers.get_derived_quantities(cfg["grid"])
    misc.log_params(cfg)

    cfg["grid"] = helpers.get_solver_quantities(cfg["grid"])
    cfg = helpers.get_save_quantities(cfg)

    models = helpers.get_models(cfg["models"])
    state = helpers.init_state(cfg)

    with tempfile.TemporaryDirectory() as td:
        # run
        t0 = time.time()

        # @eqx.filter_jit
        def _run_():
            vf = helpers.VectorField(cfg, models=models)
            return diffeqsolve(
                terms=ODETerm(vf),
                solver=Tsit5(),
                t0=cfg["grid"]["tmin"],
                t1=cfg["grid"]["tmax"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=state,
                args={"driver": cfg["drivers"]},
                saveat=SaveAt(ts=cfg["save"]["t"]["ax"], fn=cfg["save"]["func"]["callable"]),
            )

        result = _run_()
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
            mod_defaults["grid"] = es1d.helpers.get_derived_quantities(mod_defaults["grid"])
            misc.log_params(mod_defaults)
            mod_defaults["grid"] = es1d.helpers.get_solver_quantities(mod_defaults["grid"])
            mod_defaults = es1d.helpers.get_save_quantities(mod_defaults)
            t0 = time.time()

            state = es1d.helpers.init_state(mod_defaults)
            mod_defaults["models"]["file"] = misc.download_file(
                "weights.eqx", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td
            )
            models = es1d.helpers.get_models(mod_defaults["models"])

            def loss(these_models):
                vf = es1d.helpers.VectorField(mod_defaults, models=these_models)
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
            es1d.helpers.post_process(results, mod_defaults, td)
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
            mod_defaults["grid"] = es1d.helpers.get_derived_quantities(mod_defaults["grid"])
            misc.log_params(mod_defaults)
            mod_defaults["grid"] = es1d.helpers.get_solver_quantities(mod_defaults["grid"])
            mod_defaults = es1d.helpers.get_save_quantities(mod_defaults)
            t0 = time.time()

            state = es1d.helpers.init_state(mod_defaults)
            mod_defaults["models"]["file"] = misc.download_file(
                "weights.eqx", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td
            )
            models = es1d.helpers.get_models(mod_defaults["models"])

            def loss(these_models):
                vf = es1d.helpers.VectorField(mod_defaults, models=these_models)
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
            es1d.helpers.post_process(results, mod_defaults, td)
            plotters.mva(actual_nk1.data, mod_defaults, results, td, actual_nk1.coords)
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            # log artifacts
            mlflow.log_artifacts(td)
            mlflow.set_tags({"status": "completed"})
