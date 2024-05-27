from parsl.app.app import python_app
import logging, os
import equinox as eqx

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

from utils import misc


def run_one_val_and_grad(model_path, Te, L, I0, cfg_path, run_id):
    """
    This function is the main entry point for running a simulation. It takes a configuration dictionary and returns a
    ``diffrax.Solution`` object and a dictionary of datasets.

    Args:
        cfg: A dictionary containing the configuration for the simulation.

    Returns:
        A tuple of a Solution object and a dictionary of ``xarray.dataset``s.

    """

    import os, logging

    logger = logging.getLogger(__name__)

    if "BASE_TEMPDIR" in os.environ:
        BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
    else:
        BASE_TEMPDIR = None

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import mlflow, yaml
    from utils.misc import export_run
    from utils.runner import run

    # log to logger
    logging.info(f"Running a run")

    with open(cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    logging.info(f"Config is loaded")

    cfg["density"]["gradient scale length"] = f"{L}um"
    cfg["units"]["laser intensity"] = f"{I0:.2e}W/cm^2"
    cfg["units"]["reference electron temperature"] = f"{Te}eV"

    cfg["models"]["bandwidth"]["file"] = model_path
    cfg["mode"] = "train-bandwidth"

    with mlflow.start_run(run_id=run_id) as mlflow_run:
        solver_output, postprocessing_output = run(cfg)
        mlflow.log_artifact(model_path)

    export_run(mlflow_run.info.run_id)

    val = solver_output[0][0]
    grad = solver_output[1]

    return val, grad


if __name__ == "__main__":
    import uuid
    from itertools import product
    from tqdm import tqdm

    logging.basicConfig(filename=f"runlog-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    # use the logger to note that we're running a parsl job
    logging.info("Running with parsl")

    import jax
    from jax.flatten_util import ravel_pytree

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    import optax

    misc.setup_parsl("gpu", 4)
    run_one_val_and_grad = python_app(run_one_val_and_grad)

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx
    from adept.lpse2d import nn

    with open(f"/global/homes/a/archis/adept/configs/envelope-2d/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name="learn-tpd") as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        misc.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    misc.export_run(parent_run_id)

    # create the dataset with the appropriate independent variables
    input_names = ("Te", "L", "I0")

    # 125 simulations in total
    Tes = np.linspace(2000, 4000, 3)
    Ls = np.linspace(150, 450, 4)
    I0s = np.logspace(14, 16, 7)

    rng = np.random.default_rng(487)

    # batch logic
    all_inputs = np.array(list(product(Tes, Ls, I0s)))

    batch_size = 12
    num_batches = all_inputs.shape[0] // batch_size
    batch_inds = np.arange(num_batches)

    # restart mlflow run that was initialized in the optimizer
    # we did that over there so that the appropriate nesting of the mlflow run can take place
    # remember this is a separate process than what is happening in __main__
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=cfg["opt"]["learning_rate"])

        if cfg["models"]["bandwidth"]["type"] == "VAE":
            model = nn.DriverVAE(**cfg["models"]["bandwidth"]["hyperparams"])
        elif cfg["models"]["bandwidth"]["type"] == "MLP":
            model = nn.DriverModel(**cfg["models"]["bandwidth"]["hyperparams"])
        else:
            raise ValueError("Invalid model type")

        opt_state = opt.init(eqx.filter(model, eqx.is_array))  # initialize the optimizer state

        with tempfile.TemporaryDirectory(
            dir=BASE_TEMPDIR
        ) as td:  # create a temporary directory for optimizer run artifacts
            os.makedirs(os.path.join(td, "model-history"), exist_ok=True)  # create a directory for model history
            with open(cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            for i in range(1000):  # 1000 epochs
                rng.shuffle(all_inputs)

                for j, batch_ind in tqdm(enumerate(batch_inds), total=num_batches):  # iterate over the batches
                    nn.save(
                        model_path := os.path.join(td, "model-history", f"model-e-{i}-b-{j}.eqx"),
                        cfg["models"]["bandwidth"],
                        model,
                    )  # save the model
                    mlflow.log_artifacts(td)
                    misc.export_run(parent_run_id, prefix="artifact", step=i)
                    # this model and model_path are passed to the run_one_val_and_grad function for use in the simulation

                    batch_loss = 0.0  # initialize the batch loss

                    batch = all_inputs[batch_ind * batch_size : (batch_ind + 1) * batch_size]  # get the batch
                    val_and_grads = []  # initialize the list to store the values and gradient Futures
                    run_ids = []
                    for Te, L, I0 in batch:

                        with mlflow.start_run(nested=True, run_name=f"epoch={i}-{Te=}-{L=}-{I0=:.2e}") as nested_run:
                            mlflow.log_params({"indep_var.Te": Te, "indep_var.L": L, "indep_var.I0": I0})

                        # get the futures for all the inputs
                        val_and_grads.append(
                            run_one_val_and_grad(model_path, Te, L, I0, cfg_path, run_id=nested_run.info.run_id)
                        )
                        run_ids.append(nested_run.info.run_id)  # store the run_id

                    # for run_id in prev_run_ids:
                    #     artifact_dir = mlflow.get_artifact_uri(run_id)
                    #     shutil.rmtree(artifact_dir)

                    vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                    val = np.mean([v for v, _ in vgs])  # get the mean of the loss values

                    avg_grad = misc.all_reduce_gradients(
                        [g for _, g in vgs], batch_size
                    )  # get the average of the gradients
                    flat_grad, _ = ravel_pytree(avg_grad)
                    mlflow.log_metrics({"batch grad norm": float(np.linalg.norm(flat_grad))})

                    # with open("./completed_run_ids.txt", "a") as f:
                    #     f.write("\n".join(run_ids) + "\n")

                    mlflow.log_metrics({"batch loss": float(val)}, step=i * batch_size + j)
                    misc.export_run(parent_run_id, prefix="parent", step=i)
                    updates, opt_state = opt.update(avg_grad["bandwidth"], opt_state, model)
                    model = eqx.apply_updates(model, updates)

                    batch_loss += val

                mlflow.log_metrics({"epoch loss": float(batch_loss / num_batches)}, step=i)
