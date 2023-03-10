from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml

from es1d.runner import run


if __name__ == "__main__":
    with open("es1d.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    run(cfg)
