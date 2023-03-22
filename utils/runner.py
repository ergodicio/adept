import tempfile, time
from typing import Dict
import mlflow

import es1d
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, RESULTS
from utils import logs
from jax import jit


def run(cfg: Dict) -> RESULTS:
    if cfg["mode"] == "es-1d":
        helpers = es1d.helpers
    else:
        raise NotImplementedError("This mode hasn't been implemented yet")

    # get derived quantities
    cfg["grid"] = helpers.get_derived_quantities(cfg["grid"])
    logs.log_params(cfg)

    cfg["grid"] = helpers.get_solver_quantities(cfg["grid"])
    cfg = helpers.get_save_quantities(cfg)

    with tempfile.TemporaryDirectory() as td:
        # run
        t0 = time.time()
        vector_field = helpers.get_vector_field(cfg)
        state = helpers.init_state(cfg)

        @jit
        def _run_():
            return diffeqsolve(
                terms=ODETerm(vector_field),
                solver=Tsit5(),
                t0=cfg["grid"]["tmin"],
                t1=cfg["grid"]["tmax"],
                max_steps=cfg["grid"]["max_steps"],
                dt0=cfg["grid"]["dt"],
                y0=state,
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
