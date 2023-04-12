from typing import Dict
from functools import partial
import os, pickle, time, tempfile

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Solution, Kvaerno5, PIDController
from jax import jit, value_and_grad
from jax import numpy as jnp
import numpy as np
import haiku as hk
import mlflow
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

    with tempfile.TemporaryDirectory() as td:
        # run
        t0 = time.time()

        def vector_field(t, y, args):
            dummy_vf = helpers.VectorField(cfg)
            return dummy_vf(t, y, args)

        state = helpers.init_state(cfg)

        vf_init, vf_apply = hk.without_apply_rng(hk.transform(vector_field))

        if "weights" in cfg:
            with open(cfg["weights"], "rb") as fi:
                weights = pickle.load(fi)
        else:
            weights = vf_init()

        @jit
        def _run_():
            vf = partial(vf_apply, weights)
            return diffeqsolve(
                terms=ODETerm(vf),
                solver=Tsit5(),
                t0=cfg["grid"]["tmin"],
                t1=cfg["grid"]["tmax"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=state,
                args={"pulse": cfg["drivers"]},
                # stepsize_controller=PIDController(rtol=1e-8, atol=1e-8),
                saveat=SaveAt(ts=cfg["save"]["t"]["ax"], fn=cfg["save"]["func"]),
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
            w_and_b = misc.get_weights(artifact_uri=mlflow_run.info.artifact_uri, temp_path=td)
            actual_nk1 = xr.open_dataarray(
                misc.download_file("ground_truth.nc", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td)
            )
            mod_defaults["grid"] = es1d.helpers.get_derived_quantities(mod_defaults["grid"])
            misc.log_params(mod_defaults)
            mod_defaults["grid"] = es1d.helpers.get_solver_quantities(mod_defaults["grid"])
            mod_defaults = es1d.helpers.get_save_quantities(mod_defaults)
            pulse_dict = {"pulse": mod_defaults["drivers"]}
            t0 = time.time()

            def vector_field(t, y, args):
                dummy_vf = es1d.helpers.VectorField(mod_defaults)
                return dummy_vf(t, y, args)

            state = es1d.helpers.init_state(mod_defaults)
            _, vf_apply = hk.without_apply_rng(hk.transform(vector_field))

            def loss(weights_and_biases):
                vf = partial(vf_apply, weights_and_biases)
                results = diffeqsolve(
                    terms=ODETerm(vf),
                    solver=Tsit5(),
                    t0=mod_defaults["grid"]["tmin"],
                    t1=mod_defaults["grid"]["tmax"],
                    max_steps=mod_defaults["grid"]["max_steps"],
                    dt0=mod_defaults["grid"]["dt"],
                    y0=state,
                    args=pulse_dict,
                    saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"], fn=mod_defaults["save"]["func"]),
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

            vg_func = value_and_grad(loss, argnums=0, has_aux=True)
            (loss_val, results), grad = jit(vg_func)(w_and_b)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4), "loss": float(loss_val)})

            # dump gradients
            with open(os.path.join(td, "grads.pkl"), "wb") as fi:
                pickle.dump(grad, fi)

            t0 = time.time()
            es1d.helpers.post_process(results, mod_defaults, td)
            plotters.mva(actual_nk1, mod_defaults, results, td)
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            # log artifacts
            mlflow.log_artifacts(td)
            mlflow.set_tags({"status": "completed"})


def remote_val(run_id):
    with mlflow.start_run(run_id=run_id, nested=True) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            mod_defaults = misc.get_cfg(artifact_uri=mlflow_run.info.artifact_uri, temp_path=td)
            w_and_b = misc.get_weights(artifact_uri=mlflow_run.info.artifact_uri, temp_path=td)
            actual_nk1 = xr.open_dataarray(
                misc.download_file("ground_truth.nc", artifact_uri=mlflow_run.info.artifact_uri, destination_path=td)
            )
            mod_defaults["grid"] = es1d.helpers.get_derived_quantities(mod_defaults["grid"])
            misc.log_params(mod_defaults)
            mod_defaults["grid"] = es1d.helpers.get_solver_quantities(mod_defaults["grid"])
            mod_defaults = es1d.helpers.get_save_quantities(mod_defaults)
            pulse_dict = {"pulse": mod_defaults["drivers"]}
            t0 = time.time()

            def vector_field(t, y, args):
                dummy_vf = es1d.helpers.VectorField(mod_defaults)
                return dummy_vf(t, y, args)

            state = es1d.helpers.init_state(mod_defaults)
            _, vf_apply = hk.without_apply_rng(hk.transform(vector_field))

            def loss(weights_and_biases):
                vf = partial(vf_apply, weights_and_biases)
                results = diffeqsolve(
                    terms=ODETerm(vf),
                    solver=Tsit5(),
                    t0=mod_defaults["grid"]["tmin"],
                    t1=mod_defaults["grid"]["tmax"],
                    max_steps=mod_defaults["grid"]["max_steps"],
                    dt0=mod_defaults["grid"]["dt"],
                    y0=state,
                    args=pulse_dict,
                    saveat=SaveAt(ts=mod_defaults["save"]["t"]["ax"], fn=mod_defaults["save"]["func"]),
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

            loss_val, results = jit(loss)(w_and_b)
            mlflow.log_metrics({"run_time": round(time.time() - t0, 4), "val_loss": float(loss_val)})

            t0 = time.time()
            es1d.helpers.post_process(results, mod_defaults, td)
            plotters.mva(actual_nk1, mod_defaults, results, td)
            mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})
            # log artifacts
            mlflow.log_artifacts(td)
            mlflow.set_tags({"status": "completed"})
