from parsl.app.app import python_app
import logging, os
import equinox as eqx

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

from utils import misc


def run_once(Te, L, I0, _amp_):
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from jax import config

    config.update("jax_enable_x64", True)
    # config.update("jax_disable_jit", True)

    import yaml, mlflow
    from utils.runner import run
    from utils.misc import export_run

    with open("/global/homes/a/archis/adept/configs/envelope-2d/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["density"]["gradient scale length"] = f"{L}um"
    cfg["units"]["laser intensity"] = f"{I0:.2e}W/cm^2"
    cfg["units"]["reference electron temperature"] = f"{Te}eV"
    if _amp_ == "mono":
        cfg["drivers"]["E0"]["num_colors"] = 1

    cfg["drivers"]["E0"]["amplitude_shape"] = _amp_

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=f"Te={Te:.2f}, L={L:.2f}, I0={I0:.2e}, amp={_amp_}") as mlflow_run:
        result, datasets = run(cfg)

    export_run(mlflow_run.info.run_id)


if __name__ == "__main__":
    import uuid
    from itertools import product
    from tqdm import tqdm
    import numpy as np

    logging.basicConfig(filename=f"runlog-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    # use the logger to note that we're running a parsl job
    logging.info("Running with parsl")

    import jax
    from jax.flatten_util import ravel_pytree

    # jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    misc.setup_parsl("gpu", 4, 32)
    # misc.setup_parsl("local", 1)
    run_once = python_app(run_once)

    # create the dataset with the appropriate independent variables

    # 125 simulations in total
    Tes = np.linspace(2000, 4000, 5)
    Ls = np.linspace(200, 800, 7)
    I0s = np.logspace(14, 17, 10)
    amp_spec = ["uniform", "mono"]

    all_inputs = list(product(Tes, Ls, I0s, amp_spec))

    res = []
    for Te, L, I0, amp in all_inputs:
        res.append(run_once(Te, L, I0, amp))

    for r in tqdm(res):
        print(r.result())
