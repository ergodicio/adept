import tempfile, time
from typing import Dict
import mlflow
import xarray as xr
import os
from matplotlib import pyplot as plt


from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from es1d import helpers
from utils import logging


def run(cfg: Dict):
    # get derived quantities
    cfg["grid"] = helpers.get_derived_quantities(cfg["grid"])
    logging.log_params(cfg)

    cfg["grid"] = helpers.get_solver_quantities(cfg["grid"])
    cfg = helpers.get_save_quantities(cfg)

    with tempfile.TemporaryDirectory() as td:
        # run
        t0 = time.time()
        result = solve_everything(cfg)
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

        t0 = time.time()
        post_process(result, cfg, td)
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})

        # log artifacts
        mlflow.log_artifacts(td)

    # fin

    return result


def solve_everything(cfg):
    vector_field = helpers.get_vector_field(cfg)

    state = helpers.init_state(cfg)

    result = diffeqsolve(
        terms=ODETerm(vector_field),
        solver=Tsit5(),
        t0=0,
        t1=cfg["grid"]["tmax"],
        max_steps=cfg["grid"]["nt"] + 4,
        dt0=cfg["grid"]["dt"],
        y0=state,
        saveat=SaveAt(ts=cfg["save"]["t_save"]),
    )

    return result


def post_process(result, cfg, td):
    os.makedirs(os.path.join(td, "binary"))
    saved_arrays_xr = xr.Dataset(
        data_vars={
            k: xr.DataArray(v, coords=(("t", cfg["save"]["t_save"]), ("x", cfg["grid"]["x"])))
            for k, v in result.ys.items()
        }
    )
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", "stored_state.nc"))

    os.makedirs(os.path.join(td, "plots"))
    for k, v in saved_arrays_xr.items():
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
        v.plot(ax=ax, cmap="gist_ncar")
        fig.savefig(os.path.join(td, "plots", f"{k}.png"), bbox_inches="tight")
        plt.close(fig)
