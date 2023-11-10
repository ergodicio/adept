import os

from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow
from utils.runner import run
from utils.misc import export_run

if __name__ == "__main__":
    # with open("configs/tf-1d/damping.yaml", "r") as fi:
    with open("configs/vlasov-2d/base.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        result, datasets = run(cfg)

    if "MLFLOW_EXPORT" in os.environ:
        export_run(mlflow_run.info.run_id)
