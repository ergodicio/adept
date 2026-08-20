import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import yaml

# Parse args early to determine solver type before importing JAX-heavy modules
parser = argparse.ArgumentParser(description="Automatic Differentiation Enabled Plasma Transport")
parser.add_argument("--cfg", help="enter path to cfg")
parser.add_argument("--run_id", help="enter run_id to continue")
parser.add_argument("--tmax", help="override grid.tmax and save.t.tmax (for smoke/timing runs)")
parser.add_argument("--save-nt", type=int, help="override save.t.nt")
parser.add_argument("--run-name", help="override mlflow.run")
args = parser.parse_args()

# Enable float64 for kinetic solvers (must be done before importing adept)
if args.run_id is None and args.cfg:
    with open(f"{os.path.join(os.getcwd(), args.cfg)}.yaml") as fi:
        cfg = yaml.safe_load(fi)

    if args.tmax is not None:
        cfg["grid"]["tmax"] = args.tmax
        cfg.setdefault("save", {}).setdefault("t", {})["tmax"] = args.tmax
    if args.save_nt is not None:
        cfg.setdefault("save", {}).setdefault("t", {})["nt"] = args.save_nt
    if args.run_name is not None:
        cfg.setdefault("mlflow", {})["run"] = args.run_name

    if cfg.get("solver") != "envelope-2d":
        from jax import config

        config.update("jax_enable_x64", True)

# Now safe to import adept (which imports JAX)
from adept import ergoExo

if __name__ == "__main__":
    exo = ergoExo()

    if args.run_id is None:
        # Config already loaded above
        modules = exo.setup(cfg=cfg)
        sol, post_out, run_id = exo(modules)

    else:
        exo.run_job(args.run_id, nested=None)
        run_id = args.run_id
