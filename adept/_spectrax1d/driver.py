"""External field driver for Spectrax-1D (Hermite-Fourier solver)."""

from jax import Array
from jax import numpy as jnp

from adept._base_ import get_envelope


class Driver:
    """External field driver for Spectrax-1D (Hermite-Fourier solver).

    Computes external driving fields and returns them in 3D Fourier space format
    ready for use in the VectorField. Handles all FFT transformations internally.
    """

    def __init__(self, xax: Array, Nx: int, Ny: int, Nz: int, driver_key: str = "ex"):
        """
        Initialize driver.

        Args:
            xax: Real-space x grid
            Nx: Number of Fourier modes in x (for output shape)
            Ny: Number of Fourier modes in y (for output shape)
            Nz: Number of Fourier modes in z (for output shape)
            driver_key: Which field component to drive ("ex", "ey", or "ez")
        """
        self.xax = xax
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.driver_key = driver_key

        # Determine which component index this driver targets
        self.component_map = {"ex": 0, "ey": 1, "ez": 2}
        self.component_idx = self.component_map[driver_key]

    def __call__(self, driver_config: dict, t: float) -> Array:
        """
        Compute external field in Fourier space at time t.

        Args:
            driver_config: Driver configuration dict from YAML
            t: Current time

        Returns:
            Complex array of shape (3, Ny, Nx, Nz) with Fourier-space field values.
            Only the component specified by driver_key is non-zero.
        """
        pulse_configs = driver_config.get(self.driver_key, {})

        # Compute total real-space field
        total_field_real = jnp.zeros_like(self.xax)
        for pulse_name, pulse_params in pulse_configs.items():
            total_field_real += self._get_single_pulse(pulse_params, t)

        # Convert to Fourier space with 3D shape
        # For 1D case (Ny=1, Nz=1), expand to 3D shape: (1, Nx, 1)
        field_real_3d = total_field_real[None, :, None]
        field_fourier_3d = jnp.fft.fftn(field_real_3d, axes=(-3, -2, -1), norm="forward")

        # Create output array with shape (3, Ny, Nx, Nz)
        # Only set the component for this driver (Ex, Ey, or Ez)
        driver_array = jnp.zeros((3, self.Ny, self.Nx, self.Nz), dtype=jnp.complex128)
        driver_array = driver_array.at[self.component_idx].set(field_fourier_3d)

        return driver_array

    def _get_single_pulse(self, pulse: dict, t: float) -> Array:
        """
        Compute spatiotemporal envelope for a single pulse in real space.

        Args:
            pulse: Pulse parameters (k0, w0, a0, envelopes, etc.)
            t: Current time

        Returns:
            Array of shape (Nx,) with real-space field
        """
        kk = pulse["k0"]
        ww = pulse["w0"]
        a0 = pulse["a0"]
        dw = pulse.get("dw0", 0.0)

        left_bound = pulse.get("t_center", 0.0) - 0.5 * pulse.get("t_width", 1e10)
        right_bound = pulse.get("t_center", 0.0) + 0.5 * pulse.get("t_width", 1e10)

        # Temporal envelope
        envelope_t = get_envelope(pulse.get("t_rise", 10.0), pulse.get("t_rise", 10.0), left_bound, right_bound, t)

        left_bound = pulse.get("x_center", 0.0) - 0.5 * pulse.get("x_width", 1e10)
        right_bound = pulse.get("x_center", 0.0) + 0.5 * pulse.get("x_width", 1e10)

        # Spatial envelope
        envelope_x = get_envelope(
            pulse.get("x_rise", 10.0), pulse.get("x_rise", 10.0), left_bound, right_bound, self.xax
        )

        # Real-space field: a0 * sin(k*x - w*t)
        # Note: |k0| factor matches Vlasov1D convention
        return envelope_t * envelope_x * jnp.abs(kk) * a0 * jnp.sin(kk * self.xax - (ww + dw) * t)
