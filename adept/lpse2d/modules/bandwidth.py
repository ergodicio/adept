from typing import Dict
import equinox as eqx
from jax import numpy as jnp
import jax
import numpy as np
from adept.lpse2d.core import nn


class Bandwidth(eqx.Module):
    cfg: Dict
    bandwidth_model: eqx.Module
    hps: Dict

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.bandwidth_model = self.get_bandwidth_model(cfg)
        self.hps = {}

    def unnorm_weights(self, amps_and_phases: Dict, args: Dict):
        amps = amps_and_phases["amps"]
        phases = amps_and_phases["phases"]

        args["drivers"]["E0"]["amplitudes"] = jnp.tanh(amps)
        args["drivers"]["E0"]["initial_phase"] = jnp.tanh(phases)

        args["drivers"]["E0"]["delta_omega"] = jnp.linspace(
            -self.cfg["drivers"]["E0"]["delta_omega_max"],
            self.cfg["drivers"]["E0"]["delta_omega_max"],
            num=self.cfg["drivers"]["E0"]["num_colors"],
        )

        args["drivers"]["E0"]["amplitudes"] *= 2.0  # from [-1, 1] to [-2, 2]
        args["drivers"]["E0"]["amplitudes"] -= 2.0  # from [-2, 2] to [-4, 0]
        args["drivers"]["E0"]["amplitudes"] = jnp.power(
            10.0, args["drivers"]["E0"]["amplitudes"]
        )  # from [-4, 0] to [1e-4, 1]
        args["drivers"]["E0"]["amplitudes"] /= jnp.sqrt(
            jnp.sum(jnp.square(args["drivers"]["E0"]["amplitudes"]))
        )  # normalize
        args["drivers"]["E0"]["initial_phase"] *= jnp.pi  # from [-1, 1] to [-pi, pi]

        return args

    def get_bandwidth_model(self):
        if self.cfg["models"]["bandwidth"]["type"] == "direct":
            if "file" in self.cfg["models"]["bandwidth"]:
                file_path = self.cfg["models"]["bandwidth"]["file"]
                if file_path.endswith(".pkl"):
                    import pickle

                    with open(file_path, "rb") as fi:
                        self.bandwidth_model = pickle.load(fi)
            else:
                self.bandwidth_model = np.random.uniform(0, 1, self.cfg["drivers"]["E0"]["num_colors"] * 2)

        elif self.cfg["models"]["bandwidth"]["type"] == "generative":

            if "file" in self.cfg["models"]["bandwidth"]:
                file_path = self.cfg["models"]["bandwidth"]["file"]
                self.bandwidth_model, self.hps = nn.load(file_path)
            else:
                self.bandwidth_model = nn.GenerativeDriver(**self.cfg["models"]["bandwidth"]["hyperparams"])
        elif self.cfg["models"]["bandwidth"]["type"] == "function":
            if "file" in self.cfg["models"]["bandwidth"]:
                file_path = self.cfg["models"]["bandwidth"]["file"]
                self.bandwidth_model, self.hps = nn.load(file_path)
            else:
                self.bandwidth_model = nn.DriverVAE(**self.cfg["models"]["bandwidth"]["hyperparams"])

    def __call__(self, time_quantities, y, args):

        if self.cfg["models"]["bandwidth"]["type"] == "direct":
            weights = self.bandwidth_model

        elif self.cfg["models"]["bandwidth"]["type"] == "generative":
            rand_nums = jax.random.normal(
                jax.random.PRNGKey(seed=np.random.randint(2**20)),
                shape=(self.hps["input_width"],),
            )
            weights = self.bandwidth_model(rand_nums)
        elif self.cfg["models"]["bandwidth"]["type"] == "function":
            weights = self.bandwidth_model(jnp.array([self.cfg["Te"], self.cfg["L"], self.cfg["I0"]]))

        args = self.unnorm_weights(weights, args)
        return None, y, args
