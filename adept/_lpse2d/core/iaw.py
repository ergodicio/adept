"""Ion-acoustic-wave evolution for the Envelope-2D solver.

This ports the ``isEvolveIaw`` split-step path from
``m201805_matlabLpse_v11.m``. The real state variables are the fractional ion
density perturbation ``n`` (MATLAB ``Nelf``) and the ion-velocity divergence
``w`` (MATLAB ``W``):

    d w / dt = -laplacian(PP)
    d n / dt = -w

where the ponderomotive potential is

    PP = cs^2 n + Z e^2/(4 me mi) [|E_epw|^2/wp0^2
                                    + |E0|^2/w0^2
                                    + |E1|^2/w1^2].

The update order matches MATLAB's ``iawSplitStep``: kick ``w`` from the
potential, apply twice the requested amplitude Landau rate to ``w``, then
drift ``n``. Boundary damping acts on ``w`` and collisions act on ``n`` after
the split step, as in the original time loop.
"""

from jax import Array
from jax import numpy as jnp


class IonAcousticWave:
    """Advance the ion-acoustic density and velocity-divergence fields."""

    def __init__(self, cfg: dict):
        grid = cfg["grid"]
        derived = cfg["units"]["derived"]
        iaw = cfg["terms"]["iaw"]

        self.dt = grid["dt"]
        self.dx = grid["dx"]
        self.dy = grid["dy"]
        self.ny = grid["ny"]
        self.kx = grid["kx"]
        self.ky = grid["ky"]
        self.k_sq = self.kx[:, None] ** 2 + self.ky[None, :] ** 2
        self.filter = grid["low_pass_filter_grid"] * grid["zero_mask"]
        self.boundary = grid["iaw_absorbing_boundaries"]

        self.cs = derived["cs"]
        self.wp0 = derived["wp0"]
        self.w0 = derived["w0"]
        self.w1 = derived["w1"]
        self.ponderomotive_prefactor = (
            cfg["units"]["ionization state"] * derived["e"] ** 2 / (4.0 * derived["me"] * derived["mi"])
        )

        damping = iaw["damping"]
        self.landau_rate = damping["landau"] * self.cs * jnp.sqrt(self.k_sq)
        self.collisional_factor = 1.0 - damping["collisions"] * self.dt
        self.max_density_perturbation = iaw["max_density_perturbation"]

    def laplacian(self, field: Array) -> Array:
        """Second-order periodic finite-difference Laplacian used by MATLAB."""
        lap = (jnp.roll(field, -1, axis=0) - 2.0 * field + jnp.roll(field, 1, axis=0)) / self.dx**2
        if self.ny > 1:
            lap = lap + (jnp.roll(field, -1, axis=1) - 2.0 * field + jnp.roll(field, 1, axis=1)) / self.dy**2
        return lap

    def epw_fields(self, phi_k: Array) -> tuple[Array, Array]:
        """Return the real-space EPW electric-field envelopes from ``phi_k``."""
        ex = jnp.fft.ifft2(-1j * self.kx[:, None] * phi_k)
        ey = jnp.fft.ifft2(-1j * self.ky[None, :] * phi_k)
        return ex, ey

    def ponderomotive_potential(self, phi_k: Array, E0: Array, E1: Array, density: Array) -> Array:
        """Build the acoustic plus EPW/pump/Raman ponderomotive potential."""
        ex, ey = self.epw_fields(phi_k)
        epw_intensity = jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2
        pump_intensity = jnp.sum(jnp.abs(E0) ** 2, axis=-1)
        raman_intensity = jnp.sum(jnp.abs(E1) ** 2, axis=-1)
        return self.cs**2 * density + self.ponderomotive_prefactor * (
            epw_intensity / self.wp0**2 + pump_intensity / self.w0**2 + raman_intensity / self.w1**2
        )

    def __call__(self, y: dict[str, Array]) -> dict[str, Array]:
        """Advance one MATLAB-order IAW split step and return the full state."""
        potential = self.ponderomotive_potential(y["epw"], y["E0"], y["E1"], y["iaw_density"])

        velocity_divergence = y["iaw_velocity_divergence"] - self.dt * self.laplacian(potential)
        velocity_k = jnp.fft.fft2(velocity_divergence)
        velocity_k = velocity_k * jnp.exp(-2.0 * self.landau_rate * self.dt) * self.filter
        velocity_divergence = jnp.real(jnp.fft.ifft2(velocity_k)) * self.boundary

        density = (y["iaw_density"] - self.dt * velocity_divergence) * self.collisional_factor
        if self.max_density_perturbation is not None:
            density = jnp.clip(
                density,
                -self.max_density_perturbation,
                self.max_density_perturbation,
            )

        return {
            **y,
            "iaw_density": density,
            "iaw_velocity_divergence": velocity_divergence,
        }
