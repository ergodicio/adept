from typing import Dict
import json, tempfile

from jax import numpy as jnp, Array
from jax.lax import stop_gradient
from jax.random import normal, PRNGKey, uniform

import equinox as eqx
import numpy as np


from adept._lpse2d.modules.nn import driver as driver_nn
from adept.utils import download_from_s3


def save(filename: str, model_cfg: Dict, model: eqx.Module) -> None:
    new_filter_spec = lambda f, x: (
        None if isinstance(x, driver_nn.PRNGKeyArray) else eqx.default_serialise_filter_spec(f, x)
    )
    with open(filename, "wb") as f:
        model_cfg_str = json.dumps(model_cfg)
        f.write((model_cfg_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model, filter_spec=new_filter_spec)


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

    elif shape == "file":
        return FileDriver

    elif shape == "vae":
        return ITLnVAE

    elif shape == "generative":
        return GenerativeDriver

    else:
        raise NotImplementedError(f"Amplitude shape -- {shape} -- not implemented")


class UniformDriver(eqx.Module):
    intensities: Array
    delta_omega: Array
    initial_phase: Array
    envelope: Dict
    phase_key: driver_nn.PRNGKeyArray

    def __init__(self, cfg: Dict):
        super().__init__()
        num_colors = cfg["drivers"]["E0"]["num_colors"]
        self.intensities = jnp.array(np.ones(num_colors))
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        self.delta_omega = jnp.linspace(-delta_omega_max, delta_omega_max, num_colors)
        self.initial_phase = jnp.array(np.random.uniform(-np.pi, np.pi, num_colors))
        self.envelope = cfg["drivers"]["E0"]["derived"]
        self.phase_key = driver_nn.PRNGKeyArray(PRNGKey(seed=np.random.randint(2**20)))

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

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints = self.intensities / jnp.sum(self.intensities)
        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": self.initial_phase,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
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

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints = self.intensities / jnp.sum(self.intensities)
        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": self.initial_phase,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
        return state, args


class LorentzianDriver(UniformDriver):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        self.intensities = jnp.array(
            1 / np.pi * (delta_omega_max / 2) / (self.delta_omega**2.0 + (delta_omega_max / 2) ** 2.0)
        )

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints = self.intensities / jnp.sum(self.intensities)
        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": self.initial_phase,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
        return state, args


class FileDriver(UniformDriver):
    amp_output: str
    phase_output: str

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self.amp_output = cfg["drivers"]["E0"]["output"]["amp"]
        self.phase_output = cfg["drivers"]["E0"]["output"]["phase"]

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints, phases = self.scale_ints_and_phases(self.intensities, self.initial_phase)

        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": phases,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
        return state, args


class GenerativeDriver(UniformDriver):
    input_width: int
    model: eqx.Module
    amp_output: str
    phase_output: str

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self.input_width = cfg["drivers"]["E0"]["params"]["input_width"]
        cfg["drivers"]["E0"]["params"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        self.model = driver_nn.GenerativeModel(**cfg["drivers"]["E0"]["params"])
        self.amp_output = cfg["drivers"]["E0"]["output"]["amp"]
        self.phase_output = cfg["drivers"]["E0"]["output"]["phase"]

    def __call__(self, state: Dict, args: Dict) -> tuple:
        inputs = normal(PRNGKey(seed=np.random.randint(2**20)), shape=(self.input_width,))
        ints_and_phases = self.model(inputs)
        ints, phases = self.scale_ints_and_phases(ints_and_phases["amps"], ints_and_phases["phases"])
        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": phases,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
        return state, args


class ITLnVAE(UniformDriver):
    model: eqx.Module
    amp_output: str
    phase_output: str
    inputs: tuple

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        cfg["drivers"]["E0"]["params"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        cfg["drivers"]["E0"]["params"]["input_width"] = 3
        self.model = driver_nn.VAE2(**cfg["drivers"]["E0"]["params"])
        self.amp_output = cfg["drivers"]["E0"]["output"]["amp"]
        self.phase_output = cfg["drivers"]["E0"]["output"]["phase"]

        from astropy.units import Quantity as _Q

        I0 = _Q(cfg["units"]["laser intensity"]).to("W/cm^2").value
        Te = _Q(cfg["units"]["reference electron temperature"]).to("eV").value
        Ln = _Q(cfg["density"]["gradient scale length"]).to("um").value

        I00 = 1e14
        Te0 = 1000
        Ln0 = 500

        rescaled_I0 = np.log10(I0 / I00)
        rescaled_Te = Te / Te0
        rescaled_Ln = Ln / Ln0

        self.inputs = jnp.array((rescaled_I0, rescaled_Te, rescaled_Ln))

    def __call__(self, state: Dict, args: Dict) -> tuple:
        ints_and_phases = self.model(self.inputs)
        ints, phases = self.scale_ints_and_phases(ints_and_phases["amps"], ints_and_phases["phases"])
        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": phases,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
        return state, args