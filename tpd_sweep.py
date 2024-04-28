import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow
import numpy as np
from utils.runner import run, run_job
from utils.misc import export_run


def _run_():

    for ll in np.linspace(120, 200, 5):

        for I0 in np.linspace(14, 15, 5):
            # with open("configs/tf-1d/damping.yaml", "r") as fi:
            with open(f"configs/envelope-2d/tpd.yaml", "r") as fi:
                cfg = yaml.safe_load(fi)

            cfg["units"]["laser intensity"] = f"{10**I0:.2E}W/cm^2"
            cfg["density"]["gradient scale length"] = f"{ll}um"
            mlflow.set_experiment(cfg["mlflow"]["experiment"])
            # modify config
            with mlflow.start_run(run_name=cfg["units"]["laser intensity"], log_system_metrics=True) as mlflow_run:
                result, datasets = run(cfg)

        # export_run(mlflow_run.info.run_id)


if __name__ == "__main__":

    run_id = _run_()
