from parsl.app.app import python_app
import logging, os
import equinox as eqx
import pickle

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

from utils import misc


def run_one_val_and_grad(cfg, run_id):
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from jax import config

    config.update("jax_enable_x64", True)

    import mlflow
    from utils.runner import run
    from utils.misc import export_run

    with mlflow.start_run(run_id=run_id) as mlflow_run:
        solver_output, postprocessing_output = run(cfg)
        mlflow.log_artifact(cfg["models"]["bandwidth"]["file"])

    export_run(mlflow_run.info.run_id)

    val = solver_output[0][0]
    grad = solver_output[1]

    return val, grad


if __name__ == "__main__":
    import uuid

    logging.basicConfig(filename=f"runlog-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    # use the logger to note that we're running a parsl job
    # logging.info("Running with parsl")

    import jax
    from jax.flatten_util import ravel_pytree

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    import optax

    misc.setup_parsl("gpu", num_gpus=4, max_blocks=8)
    run_one_val_and_grad = python_app(run_one_val_and_grad)

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx
    from adept.lpse2d import nn

    with open(f"/global/homes/a/archis/adept/configs/envelope-2d/tpd-opt.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name="gen-tpd") as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        misc.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    misc.export_run(parent_run_id)

    rng = np.random.default_rng(6367)

    if "hyperparams" in cfg["models"]["bandwidth"]:
        weights = nn.GenerativeDriver(**cfg["models"]["bandwidth"]["hyperparams"])
    else:
        initial_amps = rng.uniform(0, 1, cfg["drivers"]["E0"]["num_colors"])
        initial_phases = rng.uniform(0, 1, cfg["drivers"]["E0"]["num_colors"])
        weights = {"amps": initial_amps, "phases": initial_phases}

    cfg["mode"] = "optimize-bandwidth"
    batch_size = 32
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=cfg["opt"]["learning_rate"])
        opt_state = opt.init(eqx.filter(weights, eqx.is_array))  # initialize the optimizer state

        with tempfile.TemporaryDirectory(
            dir=BASE_TEMPDIR
        ) as td:  # create a temporary directory for optimizer run artifacts

            os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)  # create a directory for model history
            with open(cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            for i in range(1000):  # 1000 epochs

                if "hyperparams" in cfg["models"]["bandwidth"]:
                    nn.save(
                        weights_path := os.path.join(td, "weights-history", f"weights-{i}.eqx"),
                        cfg["models"]["bandwidth"],
                        weights,
                    )
                else:
                    with open(weights_path := os.path.join(td, "weights-history", f"weights-{i}.pkl"), "wb") as fi:
                        pickle.dump(weights, fi)
                cfg["models"]["bandwidth"]["file"] = weights_path

                if batch_size == 1:
                    with mlflow.start_run(nested=True, run_name=f"epoch-{i}") as nested_run:
                        pass
                    val, avg_grad = run_one_val_and_grad(cfg, run_id=nested_run.info.run_id)
                else:
                    val_and_grads = []
                    for j in range(batch_size):
                        with mlflow.start_run(nested=True, run_name=f"epoch-{i}-sim-{j}") as nested_run:
                            # val, grad = run_one_val_and_grad(cfg, run_id=nested_run.info.run_id).result()
                            val_and_grads.append(run_one_val_and_grad(cfg, run_id=nested_run.info.run_id))

                    vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                    val = np.mean([v for v, _ in vgs])  # get the mean of the loss values

                    avg_grad = misc.all_reduce_gradients(
                        [g for _, g in vgs], batch_size
                    )  # get the average of the gradients

                grad_bandwidth = avg_grad["bandwidth"]
                flat_grad, _ = ravel_pytree(grad_bandwidth)
                mlflow.log_metrics({"grad norm": float(np.linalg.norm(flat_grad))}, step=i)
                mlflow.log_metrics({"loss": float(val)}, step=i)
                misc.export_run(parent_run_id, prefix="parent", step=i)
                updates, opt_state = opt.update(grad_bandwidth, opt_state, weights)
                weights = eqx.apply_updates(weights, updates)
