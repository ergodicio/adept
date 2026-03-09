import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import yaml

# Parse args early to determine solver type before importing JAX-heavy modules
parser = argparse.ArgumentParser(description="Automatic Differentiation Enabled Plasma Transport")
parser.add_argument("--cfg", help="enter path to cfg")
parser.add_argument("--run_id", help="enter run_id to continue")
args = parser.parse_args()

# Enable float64 for kinetic solvers (must be done before importing adept)
if args.run_id is None and args.cfg:
    with open(f"{os.path.join(os.getcwd(), args.cfg)}.yaml") as fi:
        cfg = yaml.safe_load(fi)

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
