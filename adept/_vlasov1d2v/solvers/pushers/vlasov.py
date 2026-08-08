"""Vlasov advection pushers for f(x, v_par, v_perp).

The electric field only accelerates v_par and streaming is v_par * df/dx, so
both pushers act on axis 1 of the (nx, nv, nvperp) distribution; the
perpendicular axis is a pure spectator (axisymmetric cylindrical geometry).
"""

from jax import numpy as jnp


class VelocityExponential2V:
    """Spectral v_par advection under electric and ponderomotive forces."""

    def __init__(self, species_grids, species_params):
        """Store per-species velocity grids and charge-to-mass parameters."""
        self.species_grids = species_grids
        self.species_params = species_params

    def __call__(self, f_dict, e, pond, dt):
        """Apply the spectral velocity push to each species distribution."""
        result = {}
        for species_name, f in f_dict.items():
            kv_real = self.species_grids[species_name]["kvr"]
            q = self.species_params[species_name]["charge"]
            m = self.species_params[species_name]["mass"]
            force = q * e + (q**2 / m) * pond
            accel = force / m
            result[species_name] = jnp.real(
                jnp.fft.irfft(
                    jnp.exp(-1j * kv_real[None, :, None] * dt * accel[:, None, None]) * jnp.fft.rfft(f, axis=1),
                    axis=1,
                )
            )
        return result


class SpaceExponential2V:
    """Spectral configuration-space advection for each species."""

    def __init__(self, x, species_grids):
        """Precompute x-space wavenumbers."""
        self.kx_real = jnp.fft.rfftfreq(len(x), d=x[1] - x[0]) * 2 * jnp.pi
        self.species_grids = species_grids

    def __call__(self, f_dict, dt):
        """Advect every species in configuration space for one timestep."""
        result = {}
        for species_name, f in f_dict.items():
            v = self.species_grids[species_name]["v"] * dt
            result[species_name] = jnp.real(
                jnp.fft.irfft(
                    jnp.exp(-1j * self.kx_real[:, None, None] * v[None, :, None]) * jnp.fft.rfft(f, axis=0),
                    axis=0,
                )
            )
        return result
