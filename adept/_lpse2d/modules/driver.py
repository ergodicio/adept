from typing import Dict
import json, tempfile

from jax import numpy as jnp, Array
from jax.lax import stop_gradient
from jax.random import normal, PRNGKey, uniform

import equinox as eqx
import numpy as np

from adept.utils import download_from_s3


class PRNGKeyArray:
    def __init__(self, key: Array):
        self.key = key


def load(cfg: Dict, DriverModule: eqx.Module) -> eqx.Module:
    filename = cfg["drivers"]["E0"]["file"]
    with tempfile.TemporaryDirectory() as td:
        if "s3" in filename:
            from os.path import join

            cfg["drivers"]["E0"]["file"] = join(td, filename.split("/")[-1])
            cfg["drivers"]["E0"]["file"] = download_from_s3(filename, cfg["drivers"]["E0"]["file"])
        else:
            cfg["drivers"]["E0"]["file"] = filename

        if "pkl" in cfg["drivers"]["E0"]["file"]:
            loaded_model = DriverModule(cfg)
        elif "eqx" in cfg["drivers"]["E0"]["file"]:
            with open(cfg["drivers"]["E0"]["file"], "rb") as f:
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


class ArbitraryDriver(eqx.Module):
    intensities: Array
    delta_omega: Array
    initial_phase: Array
    envelope: Dict
    amp_output: str
    phase_output: str
    model_cfg: Dict

    def __init__(self, cfg: Dict):
        super().__init__()
        self.model_cfg = cfg["drivers"]["E0"]["params"]
        self.intensities = np.ones(cfg["drivers"]["E0"]["num_colors"])
        self.delta_omega = np.linspace(
            -cfg["drivers"]["E0"]["delta_omega_max"],
            cfg["drivers"]["E0"]["delta_omega_max"],
            cfg["drivers"]["E0"]["num_colors"],
        )
        self.initial_phase = jnp.array(np.random.uniform(-np.pi, np.pi, cfg["drivers"]["E0"]["num_colors"]))
        self.envelope = cfg["drivers"]["E0"]["derived"]
        self.amp_output = "none"
        self.phase_output = "none"

    def scale_ints_and_phases(self, intensities, phases) -> tuple:
        if self.amp_output == "linear":
            ints = 0.5 * (jnp.tanh(ints) + 1.0)
        elif self.amp_output == "log":
            ints = 3 * (jnp.tanh(intensities) + 1.0) - 6
            ints = 10**ints
        else:
            raise NotImplementedError(f"Amplitude Output type -- {self.amp_output} -- not implemented")

        if self.phase_output == "learned":
            phases = jnp.tanh(phases) * jnp.pi * 4
        elif self.phase_output == "random":
            phases = stop_gradient(
                uniform(self.phase_key.key, (self.initial_phase.size,), minval=-jnp.pi, maxval=jnp.pi)
            )
        else:
            raise NotImplementedError(f"Phase Output type -- {self.phase_output} -- not implemented")

        ints /= jnp.sum(ints)

        return ints, phases

    def save(self, filename: str) -> None:
        """
        Save the model to a file

        Parameters
        ----------
        filename : str
            The name of the file to save the model to

        """
        new_filter_spec = lambda f, x: (
            None if isinstance(x, PRNGKeyArray) else eqx.default_serialise_filter_spec(f, x)
        )
        with open(filename, "wb") as f:
            model_cfg_str = json.dumps(self.model_cfg)
            f.write((model_cfg_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self, filter_spec=new_filter_spec)

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints = self.intensities / jnp.sum(self.intensities)
        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": self.initial_phase,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
        return state, args


class UniformDriver(ArbitraryDriver):

    phase_key: PRNGKeyArray

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        num_colors = cfg["drivers"]["E0"]["num_colors"]
        self.intensities = jnp.array(np.ones(num_colors))
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        self.delta_omega = jnp.linspace(-delta_omega_max, delta_omega_max, num_colors)
        self.initial_phase = jnp.array(np.random.uniform(-np.pi, np.pi, num_colors))
        self.phase_key = PRNGKeyArray(PRNGKey(seed=np.random.randint(2**20)))


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
