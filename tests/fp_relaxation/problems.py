#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Initial condition factories for Fokker-Planck relaxation tests.

Each factory returns a normalized distribution function f(v) on the given grid.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .registry import VelocityGrid


@eqx.filter_jit
def _normalize(f: Array, grid: VelocityGrid) -> Array:
    """Normalize distribution function to unit density."""
    if grid.spherical:
        norm = 4.0 * jnp.pi * jnp.sum(grid.v**2 * f * grid.dv)
    else:
        norm = jnp.sum(f * grid.dv)
    return f / norm


@eqx.filter_jit
def maxwellian(grid: VelocityGrid, T: float = 1.0) -> Array:
    """
    Create a normalised Maxwellian distribution: f(v) = exp(-v^2 / (2T)) / Z

    Args:
        grid: VelocityGrid instance
        T: Temperature (default 1.0)

    Returns:
        Normalized distribution function, shape (nv,)
    """
    f = jnp.exp(-(grid.v**2) / (2.0 * T))
    return _normalize(f, grid)


@eqx.filter_jit
def supergaussian(grid: VelocityGrid, m: int = 5, T: float = 1.0) -> Array:
    """
    Create a normlaised super-Gaussian distribution: f(v) = exp(-(v/vth)^m) / Z  where vth = sqrt(2T)

    Args:
        grid: VelocityGrid instance
        m: Super-Gaussian order (default 5)
        T: Temperature scale (default 1.0)

    Returns:
        Normalized distribution function, shape (nv,)
    """
    vth = jnp.sqrt(2.0 * T)
    f = jnp.exp(-(jnp.abs(grid.v / vth) ** m))
    return _normalize(f, grid)


@eqx.filter_jit
def bump_on_tail(grid: VelocityGrid, v_bump: float = 3.0, narrow: bool = True) -> Array:
    """
    Create a bump-on-tail distribution: f(v) = f_bulk(v) + A * exp(-((v - v_bump) / width)^2)

    Args:
        grid: VelocityGrid instance
        v_bump: Center of the bump in units of vth (default 3.0)
        narrow: If True, use narrow bump (2*dv width), else use wide bump (width=1.0)

    Returns:
        Normalized distribution function, shape (nv,)
    """
    width = 2.0 * grid.dv if narrow else 1.0

    # Bulk Maxwellian
    f_bulk = jnp.exp(-(grid.v**2) / 2.0)

    # Bump
    if grid.spherical:
        # For spherical, bump is at positive v
        f_bump = 0.1 * jnp.exp(-(((grid.v - v_bump) / width) ** 2))
    else:
        # For cartesian, add bumps at +/- v_bump for symmetry
        f_bump = 0.1 * (jnp.exp(-(((grid.v - v_bump) / width) ** 2)) + jnp.exp(-(((grid.v + v_bump) / width) ** 2)))

    return _normalize(f_bulk + f_bump, grid)


@eqx.filter_jit
def two_temperature(
    grid: VelocityGrid,
    T_cold: float = 0.5,
    T_hot: float = 2.0,
    frac_cold: float = 0.7,
) -> Array:
    """
    Create a two-temperature distribution.

    f(v) = frac_cold * Maxwellian(T_cold) + (1 - frac_cold) * Maxwellian(T_hot)

    Args:
        grid: VelocityGrid instance
        T_cold: Cold temperature (default 0.5)
        T_hot: Hot temperature (default 2.0)
        frac_cold: Fraction of cold population (default 0.7)

    Returns:
        Normalized distribution function, shape (nv,)
    """
    f_cold = maxwellian(grid, T_cold)
    f_hot = maxwellian(grid, T_hot)
    f = frac_cold * f_cold + (1.0 - frac_cold) * f_hot
    return _normalize(f, grid)


@eqx.filter_jit
def shifted_maxwellian(grid: VelocityGrid, v_shift: float = 1.0, T: float = 1.0) -> Array:
    """
    Create a shifted Maxwellian distribution: f(v) = exp(-(v - v_shift)^2 / (2T)) / Z

    For cartesian geometry, the peak is at v = v_shift.
    For spherical geometry (v >= 0), this gives a beam peaked at v_shift.

    Args:
        grid: VelocityGrid instance
        v_shift: Mean velocity shift (default 1.0)
        T: Temperature (default 1.0)

    Returns:
        Normalized distribution function, shape (nv,)
    """
    f = jnp.exp(-((grid.v - v_shift) ** 2) / (2.0 * T))
    return _normalize(f, grid)


@eqx.filter_jit
def monoenergetic_beam(grid: VelocityGrid, v_beam: float = 1.8) -> Array:
    """
    Create a monoenergetic beam: delta function at v_beam on the discrete grid.

    Zero everywhere except at the grid point closest to v_beam.

    Args:
        grid: VelocityGrid instance
        v_beam: Beam velocity (default 1.8, matching Buet rescaled to vmax=6)

    Returns:
        Normalized distribution function, shape (nv,)
    """
    idx = jnp.argmin(jnp.abs(grid.v - v_beam))
    f = jnp.zeros_like(grid.v).at[idx].set(1.0)
    return _normalize(f, grid)
