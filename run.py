from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow
from utils.runner import run

if __name__ == "__main__":
    with open("configs/wp.yaml", "r") as fi:
    # with open("tests/configs/resonance.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        run(cfg)
