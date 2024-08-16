from typing import Dict

from jax import numpy as jnp, Array
import equinox as eqx
import numpy as np

from adept.lpse2d.modules.nn import driver as driver_nn


class BandwidthModule(eqx.Module):
    num_colors: float
    amplitude_shape: str
    delta_omega: np.ndarray
    initial_phase: Array
    amplitudes: Array
    envelope: Dict
    model: eqx.Module

    def __init__(self, cfg: Dict):
        super().__init__()
        self.num_colors = cfg["drivers"]["E0"]["num_colors"]
        self.amplitude_shape = cfg["drivers"]["E0"]["amplitude_shape"]
        self.model = {}
        self.envelope = cfg["drivers"]["E0"]["derived"]

        if self.num_colors == 1:
            self.delta_omega = np.zeros(1)
            self.initial_phase = np.zeros(1)
            self.amplitudes = np.ones(1)

        else:
            delta_omega_max = self.cfg["drivers"]["E0"]["delta_omega_max"]
            self.delta_omega = jnp.linspace(-delta_omega_max, delta_omega_max, self.num_colors)
            self.initial_phase = np.random.uniform(0, 2 * np.pi, self.num_colors)
            self.amplitudes = np.ones(self.num_colors)
            if self.amplitude_shape == "gaussian":
                amplitudes = (
                    2
                    * np.log(2)
                    / delta_omega_max
                    / np.sqrt(np.pi)
                    * np.exp(-4 * np.log(2) * (self.delta_omega / delta_omega_max) ** 2.0)
                )
                self.amplitudes = jnp.sqrt(amplitudes)

            elif self.amplitude_shape == "lorentzian":
                amplitudes = 1 / np.pi * (delta_omega_max / 2) / (self.delta_omega**2.0 + (delta_omega_max / 2) ** 2.0)
                self.amplitudes = np.sqrt(amplitudes)

            elif self.amplitude_shape == "learned":
                if "ml" in self.amplitude_shape:
                    if "gen" in self.amplitude_shape:
                        self.model = driver_nn.GenerativeDriver(**cfg["modules"]["bandwidth"]["model"]["params"])
                    else:
                        raise NotImplementedError("This ")
                else:
                    self.amplitudes = jnp.ones_like(self.delta_omega)
            else:
                raise NotImplementedError(
                    f"Amplitude shape - {self.amplitude_shape} - not implemented. If you want monochromatic light, set num_colors to 1."
                )

        self.amplitudes /= jnp.sum(self.amplitudes)

    def __call__(self, state: Dict, args: Dict) -> tuple:
        if "learned" in self.amplitude_shape.casefold():
            if "ml" in self.amplitude_shape.casefold():
                if "gen" in self.amplitude_shape.casefold():
                    inputs = None
                else:
                    raise NotImplementedError(f"The -- {self.amplitude_shape} -- model type has not been implemented")
                amp = self.model(inputs)
        else:
            amp = self.amplitudes

        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "initial_phase": self.initial_phase,
            "amplitudes": amp,
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
