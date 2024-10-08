from typing import Dict, Tuple
from jax import numpy as jnp, Array


class Light:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.E0_source = cfg["units"]["derived"]["E0_source"]
        self.c = cfg["units"]["derived"]["c"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.dE0x = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]))
        self.x = cfg["grid"]["x"]

    def laser_update(self, t: float, y: jnp.ndarray, light_wave: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function updates the laser field at time t

        :param t: time
        :param y: state variables
        :return: updated laser field
        """
        # if self.cfg["laser"]["time"] == "static":
        wpe = self.w0 * jnp.sqrt(y["background_density"])[None]
        k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + light_wave["delta_omega"][:, None, None]) ** 2 - wpe**2 / self.w0**2)
        E0_static = (
            (1 + 0j - wpe**2.0 / (self.w0 * (1 + light_wave["delta_omega"][:, None, None])) ** 2) ** -0.25
            * self.E0_source
            * jnp.sqrt(light_wave["intensities"][:, None, None])
            * jnp.exp(1j * k0 * self.x[None, :, None] + 1j * light_wave["initial_phase"][:, None, None])
        )
        dE0y = E0_static * jnp.exp(-1j * light_wave["delta_omega"][:, None, None] * self.w0 * t)
        E0 = jnp.stack([self.dE0x, jnp.sum(dE0y, axis=0)], axis=-1)
        # else:
        #     raise NotImplementedError

        return E0

    def calc_ey_at_one_point(self, t: float, density: Array, light_wave: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function is used to calculate the coherence time of the laser

        :param t: time
        :param y: state variables
        :return: updated laser field
        """

        wpe = self.w0 * jnp.sqrt(density)[None, 0, 0]
        k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + light_wave["delta_omega"]) ** 2 - wpe**2 / self.w0**2)
        E0_static = (
            (1 + 0j - wpe**2.0 / (self.w0 * (1 + light_wave["delta_omega"])) ** 2) ** -0.25
            * self.E0_source
            * jnp.sqrt(light_wave["intensities"])
            * jnp.exp(1j * k0 * self.x[0] + 1j * light_wave["initial_phase"])
        )
        dE0y = E0_static * jnp.exp(-1j * light_wave["delta_omega"] * self.w0 * t)
        return jnp.sum(dE0y, axis=0)
