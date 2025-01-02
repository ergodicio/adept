from typing import Dict
import json, tempfile

from jax import numpy as jnp, Array, tree_util as jtu
from jax.random import PRNGKey

import equinox as eqx
import numpy as np

from adept.utils import download_from_s3


class PRNGKeyArray:
    def __init__(self, key: Array):
        self.key = key


def load(cfg: Dict, DriverModule: eqx.Module) -> eqx.Module:
    filename = cfg["drivers"]["E0"]["file"]
    with tempfile.TemporaryDirectory() as td:
        # download the file if it is on s3
        if "s3" in filename:
            from os.path import join

            cfg["drivers"]["E0"]["file"] = join(td, filename.split("/")[-1])
            cfg["drivers"]["E0"]["file"] = download_from_s3(filename, cfg["drivers"]["E0"]["file"])
        else:
            cfg["drivers"]["E0"]["file"] = filename

        # load the model
        if "pkl" in cfg["drivers"]["E0"]["file"]:
            loaded_model = DriverModule(cfg)
        elif "eqx" in cfg["drivers"]["E0"]["file"]:
            with open(cfg["drivers"]["E0"]["file"], "rb") as f:
                # read the model config
                model_cfg = json.loads(f.readline().decode())
                cfg["drivers"]["E0"]["params"] = model_cfg
                model = DriverModule(cfg)

                loaded_model = eqx.tree_deserialise_leaves(f, model)
        else:
            raise NotImplementedError(f"File type not recognized: {filename}. Must be .pkl or .eqx")

    return loaded_model


def choose_driver(shape: str) -> eqx.Module:
    if shape == "uniform":
        return UniformDriver

    elif shape == "gaussian":
        return GaussianDriver

    elif shape == "lorentzian":
        return LorentzianDriver

    elif shape == "arbitrary":
        return ArbitraryDriver

    else:
        raise NotImplementedError(f"Amplitude shape -- {shape} -- not implemented")


class UniformDriver(eqx.Module):
    intensities: Array
    delta_omega: Array
    phases: Array
    envelope: Dict

    def __init__(self, cfg: Dict):
        super().__init__()
        driver_cfg = cfg["drivers"]["E0"]
        self.intensities = jnp.array(np.ones(cfg["drivers"]["E0"]["num_colors"]))
        self.delta_omega = jnp.linspace(
            -driver_cfg["delta_omega_max"], driver_cfg["delta_omega_max"], driver_cfg["num_colors"]
        )
        phase_rng = np.random.default_rng(seed=cfg["drivers"]["E0"]["params"]["phases"]["seed"])
        self.phases = jnp.array(phase_rng.uniform(-1, 1, driver_cfg["num_colors"]))
        self.envelope = driver_cfg["derived"]

    def save(self, filename: str) -> None:
        """
        Save the model to a file

        Parameters
        ----------
        filename : str
            The name of the file to save the model to

        """

        with open(filename, "wb") as f:
            model_cfg_str = json.dumps(self.model_cfg)
            f.write((model_cfg_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    def __call__(self, state: Dict, args: Dict) -> tuple:
        intensities = self.intensities / jnp.sum(self.intensities)
        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope
        return state, args


class ArbitraryDriver(UniformDriver):
    model_cfg: Dict

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        driver_cfg = cfg["drivers"]["E0"]
        self.model_cfg = cfg["drivers"]["E0"]["params"]
        if self.model_cfg["amplitudes"]["init"] == "random":
            self.intensities = jnp.array(np.random.uniform(-1, 1, driver_cfg["num_colors"]))
        elif self.model_cfg["amplitudes"]["init"] == "uniform":
            self.intensities = jnp.ones(driver_cfg["num_colors"])
        else:
            raise NotImplementedError(
                f"Initialization type -- {self.model_cfg['amplitudes']['init']} -- not implemented"
            )

    def scale_intensities(self, intensities):
        if self.model_cfg["amplitudes"]["activation"] == "linear":
            ints = 0.5 * (jnp.tanh(intensities) + 1.0)
        elif self.model_cfg["amplitudes"]["activation"] == "log":
            ints = 3 * (jnp.tanh(intensities) + 1.0) - 3
            ints = 10**ints
        else:
            raise NotImplementedError(
                f"Amplitude Output type -- {self.model_cfg['amplitudes']['activation']} -- not implemented"
            )

        return ints

    def get_partition_spec(self):
        """
        Get the partition spec for the model

        Only amplitudes and phases can be learned

        Returns
        -------
        filter_spec : pytree with the same structure as the model

        """
        filter_spec = jtu.tree_map(lambda _: False, self)

        if self.model_cfg["amplitudes"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.intensities, filter_spec, replace=True)

        if self.model_cfg["phases"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.phases, filter_spec, replace=True)

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:
        intensities = self.scale_intensities(self.intensities)
        intensities = intensities / jnp.sum(intensities)

        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope
        return state, args


class GaussianDriver(UniformDriver):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        self.intensities = jnp.array(
            2
            * np.log(2)
            / delta_omega_max
            / np.sqrt(np.pi)
            * np.exp(-4 * np.log(2) * (self.delta_omega / delta_omega_max) ** 2.0)
        )


class LorentzianDriver(UniformDriver):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        self.intensities = jnp.array(
            1 / np.pi * (delta_omega_max / 2) / (self.delta_omega**2.0 + (delta_omega_max / 2) ** 2.0)
        )
