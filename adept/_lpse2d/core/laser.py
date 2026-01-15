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

        # Speckle envelope (if configured) - computed once since RPP/CPP are static
        self.speckle_envelope = None
        speckle_profile = cfg["drivers"]["E0"].get("speckle_profile")
        speckle_key = cfg["drivers"]["E0"].get("speckle_key")

        if speckle_profile is not None:
            # Convert y-coordinates to SI units (meters)
            y_si = cfg["grid"]["y"] * 1e-6  # um -> m

            # Evaluate speckle at x=0 (center of beam), y_grid, t=0
            # Shape for evaluate: (nx, ny, nt)
            ny = len(y_si)
            t_eval = jnp.zeros((1, ny, 1))

            fnum = speckle_profile.focal_length / jnp.linalg.norm(speckle_profile.beam_aperture)

            # Calculate the average magnitude of the entire speckle envelope so we can normalize by that.
            # Michel Fig 9.2 -- the entire speckle profile has a size
            # on the order of f lambda_0 / delta_x_RPP
            delta_x_RPP = speckle_profile.beam_aperture[0] / speckle_profile.n_beamlets[0]
            delta_x = fnum * speckle_profile.lambda0 / delta_x_RPP
            delta_y_RPP = speckle_profile.beam_aperture[0] / speckle_profile.n_beamlets[0]
            delta_y = fnum * speckle_profile.lambda0 / delta_y_RPP

            xs = jnp.linspace(-delta_x, delta_x, 1000)
            ys = jnp.linspace(-delta_y, delta_y, 1000)

            whole_focal_plane = jnp.meshgrid(xs, ys, jnp.array([0.0]), indexing="ij")
            whole_envelope = speckle_profile.evaluate(*whole_focal_plane, speckle_key)
            average = jnp.mean(jnp.abs(whole_envelope))

            x_eval = jnp.zeros((1, ny, 1))
            y_eval = jnp.broadcast_to(y_si[None, :, None], (1, ny, 1))

            # Get speckle envelope (complex) - static for RPP/CPP
            envelope = speckle_profile.evaluate(x_eval, y_eval, t_eval, speckle_key)
            # Shape: (1, ny, 1) -> (ny,)
            self.speckle_envelope = envelope[0, :, 0] / average

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
            intensity = light_wave["intensities"][i, :]
            phase = light_wave["phases"][i, :]

            wpe = self.w0 * jnp.sqrt(self.background_density)
            k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + delta_omega) ** 2 - wpe**2 / self.w0**2)
            E0_static = (
                (1 + 0j - wpe**2.0 / (self.w0 * (1 + delta_omega)) ** 2) ** -0.25
                * self.E0_source
                * jnp.sqrt(intensity[None, :])
                * jnp.exp(1j * k0 * self.x[:, None] + 1j * phase[None, :])
            )
            dE0y += E0_static * jnp.exp(-1j * delta_omega * self.w0 * t)

        # Apply speckle envelope if configured (same for all colors)
        if self.speckle_envelope is not None:
            dE0y = dE0y * self.speckle_envelope[None, :]

        return jnp.stack([self.dE0x, dE0y], axis=-1)
