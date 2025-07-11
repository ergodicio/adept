import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml

from adept import ergoExo
from adept._sbsbs1d.base import SBSBS_CBET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Differentiation Enabled Plasma Transport")
    parser.add_argument("--cfg", help="enter path to cfg")
    parser.add_argument("--run_id", help="enter run_id to continue")
    args = parser.parse_args()

    exo = ergoExo()

    if args.run_id is None:
        with open(f"{os.path.join(os.getcwd(), args.cfg)}.yaml") as fi:
            cfg = yaml.safe_load(fi)
        modules = exo.setup(cfg=cfg, adept_module=SBSBS_CBET)
        sol, post_out, run_id = exo(modules)

    else:
        exo.run_job(args.run_id, nested=None)
        run_id = args.run_id
