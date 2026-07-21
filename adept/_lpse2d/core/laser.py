from jax import Array
from jax import numpy as jnp


class Light:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.E0_source = cfg["units"]["derived"]["E0_source"]
        self.c = cfg["units"]["derived"]["c"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.wp0 = cfg["units"]["derived"]["wp0"]  # reference (envelope-density) plasma frequency
        self.nx = cfg["grid"]["nx"]
        self.ny = cfg["grid"]["ny"]
        self.dx = cfg["grid"]["dx"]
        self.dk = 2.0 * jnp.pi / (self.nx * self.dx)  # matches MATLAB makeKspaceAxes: dk = 2*pi/(N*dx)
        self.Lx = self.nx * self.dx  # box length in x (um)
        self.dE0x = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]))
        self.x = cfg["grid"]["x"]
        self.background_density = cfg["grid"]["background_density"]

        # Speckle state
        self.speckle_profile = None
        self.speckle_normalization = 1.0
        self.y_si = None  # y-coordinates in meters

        speckle_profile = cfg["drivers"]["E0"].get("speckle_profile")

        if speckle_profile is not None:
            # Convert y-coordinates to SI units (meters)
            y_si_m = cfg["grid"]["y"] * 1e-6  # um -> m
            self.y_si = y_si_m
            self.speckle_profile = speckle_profile

            # Calculate the normalization factor (average magnitude over focal plane)
            # Michel Fig 9.2 -- the entire speckle profile has a size
            # on the order of f lambda_0 / delta_x_RPP

            # All lengths are in units of meters
            f_m = speckle_profile.focal_length
            delta_x_RPP_m = speckle_profile.beam_aperture[0] / speckle_profile.n_beamlets[0]
            delta_x_m = f_m * speckle_profile.lambda0 / delta_x_RPP_m
            delta_y_RPP_m = speckle_profile.beam_aperture[1] / speckle_profile.n_beamlets[1]
            delta_y_m = f_m * speckle_profile.lambda0 / delta_y_RPP_m

            xs_m = jnp.linspace(-delta_x_m, delta_x_m, 1000)
            ys_m = jnp.linspace(-delta_y_m, delta_y_m, 1000)
            whole_x, whole_y = jnp.meshgrid(xs_m, ys_m, indexing="ij")
            whole_envelope = speckle_profile.evaluate(whole_x, whole_y, 0.0)
            self.speckle_normalization = jnp.mean(jnp.abs(whole_envelope))

    def laser_update(self, t_ps: float, y: jnp.ndarray, light_wave: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function updates the laser field.

        :param t_ps: time in picoseconds
        :param y: state variables
        :return: updated laser field
        """

        # Build the pump in k-space, matching the MATLAB default path
        # (flag.buildStaticFieldsInRealSpace = false; m201805_matlabLpse_v11.m:1551-1575).
        # Each color is a single plane wave whose wavenumber is computed at the *reference*
        # (envelope-density) plasma frequency wp0 and snapped to the nearest FFT grid mode, so the
        # pump is exactly periodic on the grid. The local density swelling is applied as an
        # amplitude factor *after* the transform (not as a spatially-varying phase).
        E0y_k = jnp.zeros((self.nx, self.ny), dtype=jnp.complex128)
        for i in range(len(light_wave["delta_omega"])):
            delta_omega = light_wave["delta_omega"][i]
            intensity = light_wave["intensities"][i, :]  # (ny,)
            phase = light_wave["phases"][i, :]  # (ny,)

            # reference-density wavenumber, snapped to the FFT grid (MATLAB lines 1558-1559)
            k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega) ** 2 - self.wp0**2 / self.w0**2)
            k_index = (jnp.round(k0 / self.dk).astype(int) + self.nx // 2) % self.nx
            phase_shift = self.Lx / 2.0 * k0  # MATLAB line 1561: matches the x-space construction

            # complex amplitude placed at the snapped k-mode (MATLAB lines 1564-1565)
            amp = (
                self.E0_source
                * jnp.sqrt(intensity)
                * jnp.exp(-1j * (delta_omega * self.w0 * t_ps - (phase - phase_shift)))
            )
            E0y_k = E0y_k.at[k_index, :].add(amp)

        # k-space -> x-space; the nx factor undoes the 1/nx in ifft (MATLAB line 1569: N*ifft)
        dE0y = self.nx * jnp.fft.ifft(jnp.fft.ifftshift(E0y_k, axes=0), axis=0)

        # local field swelling, applied once to the summed field (MATLAB line 1572: uses local wpe)
        wpe = self.w0 * jnp.sqrt(self.background_density)
        dE0y = dE0y * (1.0 - wpe**2 / self.w0**2) ** -0.25

        # Apply speckle envelope if configured (same for all colors)
        if self.speckle_profile is not None:
            t_s = t_ps * 1e-12  # ps -> s
            x_eval, y_eval = jnp.meshgrid(jnp.array([0.0]), self.y_si, indexing="ij")
            envelope = self.speckle_profile.evaluate(x_eval, y_eval, t_s)
            # Shape: (1, ny) -> (ny,)
            dE0y = dE0y * (envelope[0, :] / self.speckle_normalization)[None, :]

        return jnp.stack([self.dE0x, dE0y], axis=-1)

    def calc_ey_at_one_point(self, t: float, density: Array, light_wave: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
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
            * jnp.exp(1j * k0 * self.x[0] + 1j * light_wave["phases"])
        )
        dE0y = E0_static * jnp.exp(-1j * light_wave["delta_omega"] * self.w0 * t)
        return jnp.sum(dE0y, axis=0)
