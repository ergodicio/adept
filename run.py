from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow
from adept.es1d.runner import run
from adept.sh2d.runner import run as run_sh2d

if __name__ == "__main__":
    # with open("configs/es1d/damping.yaml", "r") as fi:
    with open("configs/sh2d/landau_damping.yaml", "r") as fi:
        # with open("tests/configs/resonance.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        # run(cfg)
        run_sh2d(cfg)
