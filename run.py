import argparse
import os

from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow
from utils.runner import run, run_job
from utils.misc import export_run


def _run_(cfg_path):
    # with open("configs/tf-1d/damping.yaml", "r") as fi:
    with open(f"{cfg_path}.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        result, datasets = run(cfg)

    return mlflow_run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Differentiation Enabled Plasma Transport")
    parser.add_argument("--cfg", help="enter path to cfg")
    parser.add_argument("--run_id", help="enter run_id to continue")
    args = parser.parse_args()

    if args.mode == "local":
        run_id = _run_(args.cfg)

    elif args.mode == "remote":
        run_job(args.run_id, nested=None)
        run_id = args.run_id

    if "MLFLOW_EXPORT" in os.environ:
        export_run(run_id)
