from parsl.app.app import python_app
import logging, os
import equinox as eqx

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

from utils import misc


def run_once(Te, L, I0, dw, nc):
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

    cfg["mlflow"]["experiment"] = "tpd-8nc-scan"
    cfg["density"]["gradient scale length"] = f"{L}um"
    cfg["units"]["laser intensity"] = f"{I0:.2e}W/cm^2"
    cfg["units"]["reference electron temperature"] = f"{Te}eV"
    if dw == 0.0:
        cfg["drivers"]["E0"]["num_colors"] = 1
    else:
        cfg["drivers"]["E0"]["num_colors"] = nc

    # cfg["drivers"]["E0"]["amplitude_shape"] = _amp_
    cfg["drivers"]["E0"]["delta_omega_max"] = float(dw)
    # collisions off
    cfg["terms"]["epw"]["damping"]["collisions"] = False

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(
        run_name=f"nonu-Te={Te:.2f}, L={L:.2f}, I0={I0:.2e}, dw={float(dw)}, nc={float(nc)}"
    ) as mlflow_run:
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

    # from jax.flatten_util import ravel_pytree

    # jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    misc.setup_parsl("gpu", 4, 16)
    # misc.setup_parsl("local", 4)
    run_once = python_app(run_once)

    # create the dataset with the appropriate independent variables

    # 125 simulations in total
    Tes = np.linspace(2000, 4000, 3)
    Ls = np.linspace(200, 400, 3)
    I0s = np.linspace(2, 10, 5)[:, None] * 10 ** np.linspace(13, 16, 4)[None, :]
    I0s = I0s.flatten(order="F")
    # amp_spec = ["uniform", "mono"]
    dws = np.linspace(0.0, 0.03, 3)
    ncs = [8, 16, 32]

    all_inputs = list(product(Tes, Ls, I0s, dws, ncs))

    res = []
    done_runs = []
    for Te, L, I0, dw, nc in all_inputs:
        # for I0 in I0s:
        if dw == 0.0:
            if (Te, L, I0, dw, 1) in done_runs:
                continue
            else:
                res.append(run_once(Te=Te, L=L, I0=I0, dw=dw, nc=nc))
                done_runs.append((Te, L, I0, dw, 1))
        else:
            res.append(run_once(Te=Te, L=L, I0=I0, dw=dw, nc=nc))

    for r in tqdm(res):
        print(r.result())
