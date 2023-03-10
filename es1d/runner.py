from typing import Dict
import mlflow

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from es1d import helpers


def run(cfg: Dict):
    # log parameters
    mlflow.log_params(cfg)

    # get derived quantities
    cfg = helpers.get_derived_quantities(cfg)

    # run
    results_and_metrics = solve_everything(cfg)

    # log metrics

    # postprocess

    # log artifacts

    # fin

    pass


def solve_everything(cfg):
    # initialize runners
    vector_field = helpers.get_vector_field(cfg)


    helpers.init_state(cfg)

    result = diffeqsolve(
        terms=ODETerm(vector_field),
        solver=Tsit5(),
        t0=0,
        t1=cfg["tmax"],
        max_steps=cfg["nt"] + 4,
        dt0=cfg["dt"],
        y0=cfg["state"],
        saveat=SaveAt(ts=cfg["t_save"]),
    )

    return result
