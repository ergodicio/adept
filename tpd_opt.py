from parsl.app.app import python_app
import logging, os


logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_one_val_and_grad(run_id, cfg_path):
    import os, yaml

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from jax import numpy as jnp
    from equinox import filter_value_and_grad

    from adept import ergoExo
    from adept.utils import export_run
    from adept.lpse2d.base import BaseLPSE2D

    with open(cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    class TPDModule(BaseLPSE2D):
        def __init__(self, cfg) -> None:
            super().__init__(cfg)

        def __call__(self, trainable_modules, args=None):
            out_dict = super().__call__(trainable_modules, args)
            phi_xy = out_dict["solver result"].ys["fields"]["epw"][-4:]
            phi_k = jnp.fft.fft2(phi_xy.view(jnp.complex128), axes=(-2, -1))
            ex_k = -1j * self.cfg["save"]["fields"]["kx"][None, :, None] * phi_k
            ey_k = -1j * self.cfg["save"]["fields"]["ky"][None, None, :] * phi_k
            log10e_sq = jnp.log10(jnp.sum(jnp.abs(ex_k) ** 2 + jnp.abs(ey_k) ** 2))
            return log10e_sq, out_dict

        def vg(self, trainable_modules, args=None):
            return filter_value_and_grad(self.__call__, has_aux=True)(trainable_modules, args)

    exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
    modules = exo.setup(cfg, adept_module=TPDModule)
    val, grad, (sol, ppo, _) = exo.val_and_grad(modules)
    export_run(run_id)

    return val, grad


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD optimization.")
    parser.add_argument("--config", type=str, help="The config file")
    args = parser.parse_args()
    cfg_path = args.config

    import uuid
    from copy import deepcopy
    from adept.utils import misc
    from adept import ergoExo

    logging.basicConfig(filename=f"runlog-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    from jax.flatten_util import ravel_pytree

    import optax

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx
    from adept.lpse2d.modules.driver import save as save_driver

    with open(f"/global/homes/a/archis/adept/{cfg_path}", "r") as fi:
        cfg = yaml.safe_load(fi)

    if cfg["drivers"]["E0"]["shape"] != "file":
        cfg["drivers"]["E0"]["params"]["key"] = np.random.randint(0, 2**10)
        cfg["mlflow"]["run"] = f"{cfg['mlflow']['run']}-{cfg['drivers']['E0']['params']['key']}"
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        misc.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    misc.export_run(parent_run_id)

    orig_cfg = deepcopy(cfg)

    exo = ergoExo()
    modules = exo.setup(cfg)

    if cfg["opt"]["batch_size"] > 1:
        misc.setup_parsl("local", num_gpus=4, max_blocks=5)
        run_one_val_and_grad = python_app(run_one_val_and_grad)

    lr_sched = optax.cosine_decay_schedule(
        init_value=cfg["opt"]["learning_rate"], decay_steps=cfg["opt"]["decay_steps"]
    )
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=lr_sched)
        opt_state = opt.init(eqx.filter(modules["driver"], eqx.is_array))  # initialize the optimizer state

        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)  # create a directory for model history
            for i in range(cfg["opt"]["decay_steps"] + 1):
                save_driver(
                    module_path := os.path.join(td, "weights-history", f"weights-{i}.eqx"),
                    model_cfg=cfg["drivers"]["E0"]["params"],
                    model=modules["driver"],
                )
                orig_cfg["drivers"]["E0"]["file"] = module_path

                with open(cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
                    yaml.dump(orig_cfg, fi)

                if cfg["opt"]["batch_size"] == 1:
                    with mlflow.start_run(nested=True, run_name=f"epoch-{i}") as nested_run:
                        pass
                    val, avg_grad = run_one_val_and_grad(run_id=nested_run.info.run_id, cfg_path=cfg_path)
                else:
                    val_and_grads = []
                    for j in range(cfg["opt"]["batch_size"]):
                        with mlflow.start_run(nested=True, run_name=f"epoch-{i}-sim-{j}") as nested_run:
                            mlflow.log_artifact(module_path)
                            # val, grad = run_one_val_and_grad(cfg, run_id=nested_run.info.run_id).result()
                            val_and_grads.append(run_one_val_and_grad(run_id=nested_run.info.run_id, cfg_path=cfg_path))

                    vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                    val = np.mean([v for v, _ in vgs])  # get the mean of the loss values

                    avg_grad = misc.all_reduce_gradients([g for _, g in vgs], cfg["opt"]["batch_size"])

                grad_bandwidth = avg_grad["driver"]
                flat_grad, _ = ravel_pytree(grad_bandwidth)
                mlflow.log_metrics({"grad norm": float(np.linalg.norm(flat_grad))}, step=i)
                mlflow.log_metrics({"loss": float(val)}, step=i)
                if i % 10 == 0:
                    mlflow.log_artifacts(td)
                misc.export_run(parent_run_id, prefix="parent", step=i)
                updates, opt_state = opt.update(grad_bandwidth, opt_state, modules["driver"])
                modules["driver"] = eqx.apply_updates(modules["driver"], updates)
