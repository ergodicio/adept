import numpy as np
from jax import Array
from jax import numpy as jnp

from adept._lpse2d.core.kap import KapPhases
from adept._lpse2d.core.pulse import PulseShape


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
        self.y = cfg["grid"]["y"]
        self.dy = cfg["grid"]["dy"]
        self.dky = 2.0 * jnp.pi / (self.ny * self.dy)
        self.background_density = cfg["grid"]["background_density"]
        # static-field swelling (drivers.E0.swelling): LPSE's default is the constant (1 - n_env)^(-1/4)
        # (LightSolver::computeStaticE0_fft / _xSpace); "local" is its useSpatiallyVaryingFieldSwelling
        # form (1 - n)^(-1/4), zero from 0.9999 n_c and at most 10 (the MATLAB prototype's convention)
        swelling = str(cfg["drivers"].get("E0", {}).get("swelling", "constant"))
        if swelling == "constant":
            self.swelling = float((1.0 - float(cfg["units"]["envelope density"])) ** -0.25)
        elif swelling == "local":
            n = np.asarray(self.background_density, dtype=np.float64)
            below = n < 0.9999
            factor = np.where(below, np.minimum((1.0 - np.where(below, n, 0.0)) ** -0.25, 10.0), 0.0)
            self.swelling = jnp.asarray(factor)
        else:
            raise ValueError(f"drivers.E0.swelling must be 'constant' or 'local', got {swelling!r}")
        # in-plane angle of incidence from +x (drivers.E0.angle, degrees; LPSE laser.N.direction).
        # The pump is one k-mode snapped to the grid in kx *and* ky (LPSE makeStaticField),
        # polarised in the plane perpendicular to the snapped k (LPSE polarization 0)
        self.angle = float(np.deg2rad(float(cfg["drivers"].get("E0", {}).get("angle", 0.0))))
        pump = cfg["drivers"].get("E0", {}).get("derived", {})
        self.beam_angle = np.atleast_1d(np.asarray(pump.get("beam_angle", [self.angle]), dtype=np.float64))
        self.beam_fraction = np.atleast_1d(np.asarray(pump.get("beam_fraction", [1.0]), dtype=np.float64))
        self.beam_phase = np.atleast_1d(np.asarray(pump.get("beam_phase", [0.0]), dtype=np.float64))
        self.beam_delta_omega = np.atleast_1d(np.asarray(pump.get("beam_delta_omega", [0.0]), dtype=np.float64))
        # polarization about the beam axis (rad): cos(psi) in the plane, sin(psi) along z (LPSE rotateBeam)
        self.polarization = float(pump.get("polarization", 0.0))
        self.beam_polarization = np.atleast_1d(
            np.asarray(pump.get("beam_polarization", [self.polarization]), dtype=np.float64)
        )
        self.kap_bandwidth = float(pump.get("kap_bandwidth", 0.0) or 0.0)
        self.kap_seed = int(pump.get("kap_seed", 0) or 0)
        # LPSE KAP bandwidth: random dwell times per beam (core/kap.py)
        self.kap = KapPhases(self.kap_bandwidth, self.w0, len(self.beam_angle), cfg["grid"]["tmax"], self.kap_seed)
        # LPSE laser.pulseShape: the static field scales as sqrt(max(shape, 1e-12))
        # (LightSolver::applyPulseShapeStatic)
        self.pulse = PulseShape(pump)
        self.multi_beam = len(self.beam_angle) > 1 or float(self.beam_angle[0]) != 0.0

        # Speckle state
        self.speckle_profile = None
        self.speckle_normalization = 1.0
        self.y_si = None  # y-coordinates in meters

        speckle_profile = cfg["drivers"].get("E0", {}).get("speckle_profile")

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
        # the prescribed pump replaces the state's E0 wholesale, so it carries the state's
        # component count (3 = x, y, z since plan 2 F.1; 2 for the older tests); the static
        # p-polarised pump has no z component
        nc = int(y["E0"].shape[-1]) if isinstance(y, dict) and "E0" in y else 3
        if self.multi_beam or self.kap_bandwidth > 0.0 or self.pulse.active:
            return self._oblique_update(t_ps, light_wave, nc)
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

        # field swelling, applied once to the summed field (drivers.E0.swelling)
        dE0y = dE0y * self.swelling

        # Apply speckle envelope if configured (same for all colors)
        if self.speckle_profile is not None:
            t_s = t_ps * 1e-12  # ps -> s
            x_eval, y_eval = jnp.meshgrid(jnp.array([0.0]), self.y_si, indexing="ij")
            envelope = self.speckle_profile.evaluate(x_eval, y_eval, t_s)
            # Shape: (1, ny) -> (ny,)
            dE0y = dE0y * (envelope[0, :] / self.speckle_normalization)[None, :]

        cos_psi, sin_psi = float(np.cos(self.polarization)), float(np.sin(self.polarization))
        if nc == 2:
            if sin_psi != 0.0:
                raise ValueError("an out-of-plane (s-polarised) pump needs three-component light fields")
            return jnp.stack([self.dE0x, dE0y], axis=-1)
        # p-polarised (psi = 0): the y component is dE0y exactly (x 1.0) and E0z = 0
        return jnp.stack([self.dE0x, cos_psi * dE0y if cos_psi != 1.0 else dE0y, sin_psi * dE0y], axis=-1)

    def _oblique_update(self, t_ps: float, light_wave: dict, nc: int = 3) -> jnp.ndarray:
        """Oblique static pump: each color is the single grid mode nearest to
        ``k0(delta_omega) (cos a, sin a)`` (LPSE ``makeStaticField`` rounds both components),
        built in x-space, with the field along ``(-sin a', cos a')`` of the *snapped* direction
        so that E0 is exactly transverse; the swelling factor is applied as in the normal case."""
        E0 = jnp.zeros((self.nx, self.ny, nc), dtype=jnp.complex128)
        xx = self.x[:, None]
        yy = self.y[None, :]
        pulse = self.pulse.static_field_factor(t_ps)
        for b in range(len(self.beam_angle)):
            angle = float(self.beam_angle[b])
            dw_b = float(self.beam_delta_omega[b])
            beam_phase = float(self.beam_phase[b]) + self._kap_phase(t_ps, b)
            for i in range(len(light_wave["delta_omega"])):
                delta_omega = light_wave["delta_omega"][i] + dw_b
                intensity = light_wave["intensities"][i, :] * float(self.beam_fraction[b])  # (ny,)
                phase = light_wave["phases"][i, :] + beam_phase  # (ny,)
                k0 = self.w0 / self.c * jnp.sqrt((1.0 + delta_omega) ** 2 - self.wp0**2 / self.w0**2)
                kx = jnp.round(k0 * jnp.cos(angle) / self.dk) * self.dk
                ky = jnp.round(k0 * jnp.sin(angle) / self.dky) * self.dky
                k_snapped = jnp.sqrt(kx**2 + ky**2)
                # LPSE rotateBeam: cos(psi) along the in-plane unit vector perpendicular to the
                # snapped k, sin(psi) along z
                cos_psi, sin_psi = float(np.cos(self.beam_polarization[b])), float(np.sin(self.beam_polarization[b]))
                if nc == 2 and sin_psi != 0.0:
                    raise ValueError("an out-of-plane (s-polarised) pump needs three-component light fields")
                in_plane = jnp.stack([-ky / k_snapped, kx / k_snapped])
                pol = jnp.concatenate([in_plane * cos_psi if cos_psi != 1.0 else in_plane, jnp.full(nc - 2, sin_psi)])
                amp = (
                    pulse * self.E0_source * jnp.sqrt(intensity) * jnp.exp(-1j * (delta_omega * self.w0 * t_ps - phase))
                )
                carrier = jnp.exp(1j * (kx * xx + ky * yy))
                E0 = E0 + (amp[None, :] * carrier)[..., None] * pol[None, None, :]
        E0 = E0 * (self.swelling if isinstance(self.swelling, float) else self.swelling[..., None])
        if self.speckle_profile is not None:
            raise NotImplementedError("drivers.E0.speckle with a non-zero drivers.E0.angle is not supported")
        return E0

    def _kap_phase(self, t_ps, beam: int):
        """Kubo-Anderson phase of ``beam`` at ``t_ps`` (LPSE's process, core/kap.py)."""
        return self.kap.phase(t_ps, beam)

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
