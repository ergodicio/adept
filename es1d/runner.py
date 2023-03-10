import tempfile, time
from typing import Dict
import mlflow
import xarray as xr
import os
from matplotlib import pyplot as plt


from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from es1d import helpers


def run(cfg: Dict):
    # get derived quantities
    cfg = helpers.get_derived_quantities(cfg)
    mlflow.log_params(cfg)

    cfg = helpers.get_array_quantities(cfg)

    with tempfile.TemporaryDirectory() as td:
        # run
        t0 = time.time()
        results_and_metrics = solve_everything(cfg)
        mlflow.log_metrics({"run_time": round(time.time() - t0, 4)})

        t0 = time.time()
        post_process(results_and_metrics, cfg, td)
        mlflow.log_metrics({"postprocess_time": round(time.time() - t0, 4)})

        # log artifacts
        mlflow.log_artifacts(td)

    # fin

    pass


def solve_everything(cfg):
    # initialize runners
    vector_field = helpers.get_vector_field(cfg)

    state = helpers.init_state(cfg)

    result = diffeqsolve(
        terms=ODETerm(vector_field),
        solver=Tsit5(),
        t0=0,
        t1=cfg["tmax"],
        max_steps=cfg["nt"] + 4,
        dt0=cfg["dt"],
        y0=state,
        saveat=SaveAt(ts=cfg["t_save"]),
    )

    return result


def post_process(result, cfg, td):
    saved_arrays_xr = xr.Dataset(
        data_vars={k: xr.DataArray(v, coords=(("t", cfg["t_save"]), ("x", cfg["x"]))) for k, v in result.ys.items()}
    )
    saved_arrays_xr.to_netcdf(os.path.join(td, "stored_state.nc"))

    for k, v in saved_arrays_xr.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
        v.plot(ax=ax, cmap="gist_ncar")
        fig.savefig(os.path.join(td, f"{k}.png"), bbox_inches="tight")
        plt.close(fig)
