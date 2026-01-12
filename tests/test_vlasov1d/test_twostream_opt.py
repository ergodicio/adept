import xarray as xr, numpy as np, os, sys
import yaml
import mlflow
from tqdm import tqdm
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.append(os.getcwd())  # To load adept

from jax import config
import jax
import jax.numpy as jnp
config.update("jax_enable_x64", True)
import equinox as eqx
import optax

from diffrax import diffeqsolve, SaveAt

from adept import ergoExo
from adept._vlasov1d.modules import BaseVlasov1D
from adept._vlasov1d.helpers import _initialize_total_distribution_

import matplotlib
matplotlib.use("Agg")


def set_dict_leaves(src, dst, key=None):
    for k, v in src.items():
        if isinstance(dst[k], dict):
            set_dict_leaves(v, dst[k])
        else:
            dst[k] = v


class OptVlasov1D(BaseVlasov1D):
    def __init__(self, cfg):
        super().__init__(cfg)

    def reinitialize_distribution(self, cfg, state):
        # return super().init_state_and_args()
        _, f = _initialize_total_distribution_(cfg, cfg["grid"])
        state["electron"] = f

        return state

    def __call__(self, params: dict, args: dict) -> dict:
        if args is None:
            args = self.args
        # Overwrite cfg with passed args
        cfg = self.cfg
        set_dict_leaves(params, cfg)
        # Reinitialize the distribution based on args
        state = self.reinitialize_distribution(cfg, self.state)
        # Solve the equations
        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=cfg["grid"]["max_steps"],
            dt0=cfg["grid"]["dt"],
            y0=state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )
         # Compute the mean growth rate at the start of the simulation
        opt_quantity = jnp.mean(jnp.log10(solver_result.ys['default']['mean_e2'][10:200]))
        return_val = (opt_quantity, {"solver result": solver_result})
        return return_val

    def vg(self, params: dict, args: dict) -> tuple[float, dict, dict]:
        return eqx.filter_value_and_grad(self.__call__, has_aux=True)(params, args)


if __name__ == "__main__":
    
    with open("tests/test_vlasov1d/configs/twostream_opt.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)
    cfg['mlflow']['experiment'] = "twostream-optimize"
    mlflow.set_experiment("twostream-optimize")

    params = {"density": {
            "species-electron1": {
                "v0": jnp.array(cfg["density"]["species-electron1"]["v0"]),
                "T0": jnp.array(cfg["density"]["species-electron1"]["T0"]),
                },
            "species-electron2": {
                "v0": jnp.array(cfg["density"]["species-electron2"]["v0"]),
                "T0": jnp.array(cfg["density"]["species-electron2"]["T0"]),
                }
            }
        }
    
    mlflow.log_metrics({
        "e1_v0": params["density"]["species-electron1"]["v0"].item(),
        "e1_T0": params["density"]["species-electron1"]["T0"].item(),
        "e2_v0": params["density"]["species-electron2"]["v0"].item(),
        "e2_T0": params["density"]["species-electron2"]["T0"].item(),
        }, step=0)

    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(params)

    loop_t0 = time.time()
    mlflow.log_metrics({"time_loop": time.time() - loop_t0}, step=0)
    for i in tqdm(range(5)):
        iter_t0 = time.time()
        cfg['mlflow']['run'] = f"opt-iter-{i}"

        exo = ergoExo(mlflow_nested=True)
        exo.setup(cfg=cfg, adept_module=OptVlasov1D)
        # Potential optimization to shift post-processing to another thread
        val, grad, (sim_out, post_processed_output, mlflow_run_id) = exo.val_and_grad(params)

        mlflow.log_metrics({
            "gamma-e2": val.item(),
            "grad_l2": jnp.linalg.norm(jnp.array(jax.tree.flatten(grad)[0])).item(),
            "e1_v0": params["density"]["species-electron1"]["v0"].item(),
            "e1_T0": params["density"]["species-electron1"]["T0"].item(),
            "e2_v0": params["density"]["species-electron2"]["v0"].item(),
            "e2_T0": params["density"]["species-electron2"]["T0"].item(),
        }, step=i+1)

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        print(f"Mean-log e2 growth rate : {val}")

        mlflow.log_metrics({
            "time_iter": time.time() - iter_t0,
            "time_loop": time.time() - loop_t0,
        }, step=i+1)

    # The final parameter values are not logged because they do not correspond to
    #   the final optimized quantity (the update step has been applied to them)
    np.testing.assert_almost_equal(val, -8.572186655748087, decimal=2)
    np.testing.assert_almost_equal(np.abs(params["density"]["species-electron1"]["v0"]), 
                                   np.abs(params["density"]["species-electron2"]["v0"]), decimal=2)
    np.testing.assert_almost_equal(params["density"]["species-electron1"]["T0"], 
                                   params["density"]["species-electron2"]["T0"], decimal=2)
