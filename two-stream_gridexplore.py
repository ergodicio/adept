import xarray as xr, numpy as np, os, sys
import yaml
import mlflow
from tqdm import tqdm
import time
import itertools
import copy

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVIES"] = "1"
sys.path.append(os.getcwd())  # To load adept

import jax
from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
import equinox as eqx
import optax

# os.chdir('../../adept')

from adept import ergoExo
from adept._vlasov1d.modules import BaseVlasov1D
from adept._vlasov1d.helpers import _initialize_total_distribution_

import multiprocess as mp
from multiprocess import get_context
from mlflow.tracking import MlflowClient  # for run_id-based logging


def set_dict_leaves(src, dst, key=None):
    for k, v in src.items():
        if isinstance(dst[k], dict):
            set_dict_leaves(v, dst[k])
        else:
            dst[k] = v


class OptVlasov1D(BaseVlasov1D):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, params: dict, args: dict) -> dict:
        if args is None:
            args = self.args
        solver_result = super().__call__(self.cfg, self.state, args)
        return solver_result

    # def vg(self, params: dict, args: dict) -> tuple[float, dict, dict]:
    #     return eqx.filter_value_and_grad(self.__call__, has_aux=True)(params, args)


def run_one(args):
    dt, nx, nv, base_cfg = args

    # Local imports (optional) to avoid large globals during fork
    import time, copy

    t0 = time.time()

    # Deep copy the config so each worker can mutate independently
    cfg = copy.deepcopy(base_cfg)
    cfg["grid"]["nx"] = nx
    cfg["grid"]["nv"] = nv
    cfg["grid"]["dt"] = dt
    cfg["mlflow"]["run"] = f"nx-{nx}_nv-{nv}_dt-{dt}"

    # Ensure MLflow experiment is set inside this process
    mlflow.set_experiment(cfg["mlflow"].get("experiment", "twostream-gridsearch"))

    # Run the simulation
    exo = ergoExo(mlflow_nested=True)
    exo.setup(cfg=cfg, adept_module=OptVlasov1D)
    sim_out, post_processed_output, mlflow_run_id = exo(exo.cfg)

    # Compute your value
    val = float(
        jnp.mean(
            jnp.nan_to_num(
                jnp.log10(sim_out["solver result"].ys["default"]["mean_e2"][1:])
            )
        )
    )

    elapsed = time.time() - t0

    # Log metrics against the run_id to avoid relying on implicit active run state
    try:
        client = MlflowClient()
        client.log_metric(mlflow_run_id, key="time_sim", value=elapsed)
        client.log_metric(mlflow_run_id, key="val", value=val)
    except Exception:
        # Fallback to implicit logging if needed (not ideal in multi-proc)
        mlflow.log_metrics({"time_sim": elapsed, "val": val})

    return {
        "dt": dt,
        "nx": nx,
        "nv": nv,
        "time_sim": elapsed,
        "mlflow_run_id": mlflow_run_id,
        "val": val,
    }



if __name__ == "__main__":
    n_procs=20
    # Load base config
    with open("tests/test_vlasov1d/configs/twostream_opt.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    # Set experiment (also done per worker)
    cfg["mlflow"]["experiment"] = "twostream-gridsearch"
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    # Parameter grid
    nx_list = [8, 16, 32, 64, 128]
    nv_list = [16, 64, 256, 1024, 4096]
    dt_list = [0.1, 1, 10]

    tasks = [(dt, nx, nv, cfg) for dt, nx, nv in itertools.product(dt_list, nx_list, nv_list)]
    total = len(tasks)

    # Choose number of processes
    if n_procs is None:
        n_procs = os.cpu_count() or 1

    # Use spawn context for JAX/MLflow friendliness
    ctx = get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        # chunksize=1 ensures fair scheduling; increase for fewer, heavier tasks
        results_iter = pool.imap_unordered(run_one, tasks, chunksize=1)
        for res in tqdm(results_iter, total=total):
            tqdm.write(f"time: {res['time_sim']:.3f} run_id: {res['mlflow_run_id']} val: {res['val']:.6g}")