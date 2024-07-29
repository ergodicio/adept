from typing import Dict
import json

from jax import numpy as jnp, Array
from jax.lax import stop_gradient
import equinox as eqx
import numpy as np


from adept.lpse2d.modules.nn import driver as driver_nn


def save(filename, model_cfg, model):
    with open(filename, "wb") as f:
        model_cfg_str = json.dumps(model_cfg)
        f.write((model_cfg_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(cfg):
    filename = cfg["modules"]["bandwidth"]["file"]
    with open(filename, "rb") as f:
        model_cfg = json.loads(f.readline().decode())
        hyperparams = model_cfg["hyperparams"]
        cfg["modules"]["bandwidth"] = cfg["modules"]["bandwidth"] | hyperparams
        model = BandwidthModule(cfg)

        return eqx.tree_deserialise_leaves(f, model)


class BandwidthModule(eqx.Module):
    num_colors: float
    shape: str
    delta_omega: np.ndarray
    initial_phase: Array
    intensities: Array
    envelope: Dict
    model: eqx.Module

    def __init__(self, cfg: Dict):
        super().__init__()
        self.num_colors = cfg["drivers"]["E0"]["num_colors"]
        self.shape = cfg["drivers"]["E0"]["shape"]
        self.model = {}
        self.envelope = cfg["drivers"]["E0"]["derived"]

        if self.num_colors == 1:
            self.delta_omega = np.zeros(1)
            self.initial_phase = np.zeros(1)
            self.intensities = np.ones(1)

        else:
            delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
            self.delta_omega = jnp.linspace(-delta_omega_max, delta_omega_max, self.num_colors)
            self.initial_phase = np.random.uniform(0, 2 * np.pi, self.num_colors)

            if self.shape == "uniform":
                self.intensities = np.ones(self.num_colors)

            elif self.shape == "gaussian":
                self.intensities = (
                    2
                    * np.log(2)
                    / delta_omega_max
                    / np.sqrt(np.pi)
                    * np.exp(-4 * np.log(2) * (self.delta_omega / delta_omega_max) ** 2.0)
                )
                # self.intensities = jnp.sqrt(intensities)

            elif self.shape == "lorentzian":
                self.intensities = (
                    1 / np.pi * (delta_omega_max / 2) / (self.delta_omega**2.0 + (delta_omega_max / 2) ** 2.0)
                )
                # self.intensities = np.sqrt(intensities)

            elif self.shape == "learned":
                if "ml" in self.shape:
                    if "gen" in self.shape:
                        self.model = driver_nn.GenerativeDriver(**cfg["modules"]["bandwidth"]["model"]["params"])
                    else:
                        raise NotImplementedError("This ")
                else:
                    self.intensities = jnp.ones_like(self.delta_omega)
            else:
                raise NotImplementedError(
                    f"Amplitude shape - {self.shape} - not implemented. If you want monochromatic light, set num_colors to 1."
                )

        self.intensities /= jnp.sum(self.intensities)

    def stop_gradients(self):
        self.num_colors = stop_gradient(self.num_colors)
        self.delta_omega = stop_gradient(self.delta_omega)

    # elif self.cfg["drivers"]["E0"]["shape"] == "file":
    #     import tempfile

    #     with tempfile.TemporaryDirectory() as td:

    #         import pickle

    #         if self.cfg["drivers"]["E0"]["file"].startswith("s3"):
    #             import boto3

    #             fname = self.cfg["drivers"]["E0"]["file"]

    #             bucket = fname.split("/")[2]
    #             key = "/".join(fname.split("/")[3:])
    #             s3 = boto3.client("s3")
    #             s3.download_file(bucket, key, local_fname := os.path.join(td, "drivers.pkl"))
    #         else:
    #             local_fname = self.cfg["drivers"]["E0"]["file"]

    #         with open(local_fname, "rb") as fi:
    #             drivers = pickle.load(fi)
    # else:
    #     raise NotImplemented

    # drivers["E0"]["intensities"] /= jnp.sum(drivers["E0"]["intensities"])

    # return drivers

    def __call__(self, state: Dict, args: Dict) -> tuple:
        if "learned" in self.shape.casefold():
            if "ml" in self.shape.casefold():
                if "gen" in self.shape.casefold():
                    inputs = None
                else:
                    raise NotImplementedError(f"The -- {self.shape} -- model type has not been implemented")
                amp = self.model(inputs)
        else:
            amp = self.intensities

        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "initial_phase": self.initial_phase,
            "intensities": amp,
            "xr": self.envelope["xr"],
            "yr": self.envelope["yr"],
            "tr": self.envelope["tr"],
            "tw": self.envelope["tw"],
            "tc": self.envelope["tc"],
            "xw": self.envelope["xw"],
            "yw": self.envelope["yw"],
            "xc": self.envelope["xc"],
            "yc": self.envelope["yc"],
        }

        return state, args
