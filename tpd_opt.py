from parsl.app.app import python_app
import logging, os
import equinox as eqx

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_one_val_and_grad(run_id, module_path):
    import os, yaml

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from jax import config, numpy as jnp

    config.update("jax_enable_x64", True)

    from adept import ergoExo
    from adept.utils.misc import export_run
    from adept.lpse2d.base import BaseLPSE2D

    with open(f"/global/homes/a/archis/adept/configs/envelope-2d/tpd-opt.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["models"]["bandwidth"]["file"] = module_path

    class TPDModule(BaseLPSE2D):
        def __init__(self, cfg) -> None:
            super().__init__(cfg)

        def __call__(self, trainable_modules, args=None):
            out_dict = super()(trainable_modules, args)
            phi_xy = out_dict["solver result"].ys["fields"]["epw"]
            phi_k = jnp.fft.fft2(phi_xy)
            ex_k = -1j * self.cfg["grid"]["kx"] * phi_k
            ey_k = -1j * self.cfg["grid"]["ky"] * phi_k
            e_sq = jnp.abs(ex_k) ** 2 + jnp.abs(ey_k) ** 2
            return e_sq, out_dict

    exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
    modules = exo.setup(cfg, adept_module=TPDModule)
    val, grad, (sol, ppo, _) = exo.val_and_grad(modules)
    export_run(run_id)

    return val, grad


if __name__ == "__main__":
    import uuid
    from adept.utils import misc
    from adept import ergoExo

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
    from adept.lpse2d.modules.driver import save as save_driver

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

    exo = ergoExo()
    modules = exo.setup(cfg)

    # if "hyperparams" in cfg["models"]["bandwidth"]:
    #     weights = nn.GenerativeDriver(**cfg["models"]["bandwidth"]["hyperparams"])
    # else:
    #     initial_amps = rng.uniform(0, 1, cfg["drivers"]["E0"]["num_colors"])
    #     initial_phases = rng.uniform(0, 1, cfg["drivers"]["E0"]["num_colors"])
    #     weights = {"amps": initial_amps, "phases": initial_phases}

    # cfg["mode"] = "optimize-bandwidth"
    batch_size = 1
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=cfg["opt"]["learning_rate"])
        opt_state = opt.init(eqx.filter(modules, eqx.is_array))  # initialize the optimizer state

        with tempfile.TemporaryDirectory(
            dir=BASE_TEMPDIR
        ) as td:  # create a temporary directory for optimizer run artifacts

            os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)  # create a directory for model history
            # with open(cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
            #     yaml.dump(cfg, fi)

            for i in range(1000):  # 1000 epochs
                save_driver(
                    module_path := os.path.join(td, "weights-history", f"weights-{i}.eqx"),
                    model_cfg=cfg["modules"]["bandwidth"]["params"],
                    model=modules["bandwidth"],
                )

                if batch_size == 1:
                    with mlflow.start_run(nested=True, run_name=f"epoch-{i}") as nested_run:
                        pass
                    val, avg_grad = run_one_val_and_grad(cfg, run_id=nested_run.info.run_id)
                else:
                    val_and_grads = []
                    for j in range(batch_size):
                        with mlflow.start_run(nested=True, run_name=f"epoch-{i}-sim-{j}") as nested_run:
                            # val, grad = run_one_val_and_grad(cfg, run_id=nested_run.info.run_id).result()
                            val_and_grads.append(
                                run_one_val_and_grad(run_id=nested_run.info.run_id, module_path=module_path)
                            )

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
