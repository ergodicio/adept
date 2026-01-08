from jax import Array
from jax import numpy as jnp


class Light:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.E0_source = cfg["units"]["derived"]["E0_source"]
        self.c = cfg["units"]["derived"]["c"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.dE0x = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]))
        self.x = cfg["grid"]["x"]
        self.background_density = cfg["grid"]["background_density"]

    def laser_update(self, t: float, y: jnp.ndarray, light_wave: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function updates the laser field at time t

        :param t: time
        :param y: state variables
        :return: updated laser field
        """

        dE0y = jnp.zeros((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"]), dtype=jnp.complex128)
        for i in range(len(light_wave["delta_omega"])):
            delta_omega = light_wave["delta_omega"][i]
            intensity = light_wave["intensities"][i]
            phase = light_wave["phases"][i]

            wpe = self.w0 * jnp.sqrt(self.background_density)
            k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + delta_omega) ** 2 - wpe**2 / self.w0**2)
            E0_static = (
                (1 + 0j - wpe**2.0 / (self.w0 * (1 + delta_omega)) ** 2) ** -0.25
                * self.E0_source
                * jnp.sqrt(intensity)
                * jnp.exp(1j * k0 * self.x[:, None] + 1j * phase)
            )
            dE0y += E0_static * jnp.exp(-1j * delta_omega * self.w0 * t)

        return jnp.stack([self.dE0x, dE0y], axis=-1)

