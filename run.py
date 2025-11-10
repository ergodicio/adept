import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml

from adept import ergoExo
from adept._sbsbs1d.base import SBSBS_CBET, Train_SBSBS_CBET
import numpy as np
import equinox as eqx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Differentiation Enabled Plasma Transport")
    parser.add_argument("--cfg", help="enter path to cfg")
    parser.add_argument("--run_id", help="enter run_id to continue")
    args = parser.parse_args()

    exo = ergoExo()

    if args.run_id is None:
        file_string = f"{os.path.join(os.getcwd(), args.cfg)}"
        file_string = file_string if file_string.endswith('.yaml') else file_string+'.yaml'
        with open(file_string) as fi:
            cfg = yaml.safe_load(fi)
        cfg = cfg|{"nn_inputs":{"laser_powers":np.ones([8,512]),"design":np.random.rand(12),"t":np.zeros(1)}}
        cfg["intensities"] = np.ones(8)
        cfg["reflectivity"] = 0.1*np.ones(8)
        
        all_modules = exo.setup(cfg=cfg,adept_module=Train_SBSBS_CBET)
        diff_modules, static_modules = {}, {}
        # diff_modules['lpi'], static_modules['lpi'] = eqx.partition(all_modules['lpi'],all_modules['lpi'].get_partition_spec())
        diff_modules['hohlnet'], static_modules['hohlnet'] = eqx.partition(all_modules['hohlnet'],all_modules['hohlnet'].get_partition_spec())
        # static_modules = exo.setup(cfg=cfg, adept_module=Train_SBSBS_CBET)
        # diff_modules = {'Jr_target':0.5*np.ones(4)}
        val, grad, (sol, post_out, run_id) = exo.val_and_grad(diff_modules,args={'static_modules':static_modules,'reflectivity':cfg["reflectivity"]})

    else:
        exo.run_job(args.run_id, nested=None)
        run_id = args.run_id
