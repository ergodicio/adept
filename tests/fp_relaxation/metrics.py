#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Metric computation functions for Fokker-Planck relaxation tests.

All metrics handle both cartesian and spherical geometries appropriately.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from adept.driftdiffusion import _find_self_consistent_beta_single, discrete_temperature

from .problems import shifted_maxwellian
from .registry import VelocityGrid


class RelaxationMetrics(eqx.Module):
    """Metrics computed at each snapshot during relaxation."""

    time: Array
    density: Array  # integral(f dv) or integral(4*pi*v^2*f dv)
    energy_discrete: Array  # T = <v^2> or <v^4>/(3<v^2>)
    energy_self_consistent: Array  # T from self-consistent beta
    positivity_violation: Array  # integral(|f| where f<0)
    rmse_final_maxwellian: Array  # sqrt(sum((f-f_eq)^2 * dv)) to expected equilibrium
    rmse_instant_maxwellian: Array  # sqrt(sum((f-f_eq)^2 * dv)) to Maxwellian with current T
    entropy: Array  # -integral(f*log(f) dv), NaN if f<=0 anywhere
    momentum: Array  # <v> for cartesian, NaN for spherical


def compute_density(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the density (zeroth moment).

    For cartesian: n = integral(f dv)
    For spherical: n = 4*pi * integral(v^2 * f dv)
    """
    if grid.spherical:
        return 4.0 * jnp.pi * jnp.sum(grid.v**2 * f * grid.dv)
    else:
        return jnp.sum(f * grid.dv)


def compute_momentum(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the mean velocity (first moment).

    For cartesian: <v> = integral(v * f dv) / n
    For spherical: Not applicable (returns NaN)
    """
    if grid.spherical:
        return jnp.nan
    n = jnp.sum(f * grid.dv)
    return jnp.sum(grid.v * f * grid.dv) / n


def compute_self_consistent_temperature(f: Array, grid: VelocityGrid, max_steps: int = 3) -> Array:
    """
    Compute temperature from self-consistent beta.

    Finds beta* such that Maxwellian(beta*) has the same discrete temperature as f,
    then returns T = 1/(2*beta*).
    """
    beta = _find_self_consistent_beta_single(
        f,
        grid.v,
        grid.dv,
        rtol=1e-8,
        atol=1e-12,
        max_steps=max_steps,
        spherical=grid.spherical,
    )
    return 1.0 / (2.0 * beta)


def compute_positivity_violation(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the integral of |f| where f < 0.

    This measures how badly positivity is violated.
    """
    negative_f = jnp.where(f < 0, -f, 0.0)
    if grid.spherical:
        return 4.0 * jnp.pi * jnp.sum(grid.v**2 * negative_f * grid.dv)
    else:
        return jnp.sum(negative_f * grid.dv)


def compute_entropy(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the entropy: -integral(f * log(f) dv).

    Returns NaN if f <= 0 anywhere (entropy undefined).
    """
    if grid.spherical:
        entropy = -4.0 * jnp.pi * jnp.sum(grid.v**2 * f * jnp.log(jnp.where(f > 0, f, 1.0)) * grid.dv)
    else:
        entropy = -jnp.sum(f * jnp.log(jnp.where(f > 0, f, 1.0)) * grid.dv)
    return jnp.where(jnp.any(f <= 0), jnp.nan, entropy)


def compute_rmse(f: Array, f_ref: Array, grid: VelocityGrid) -> Array:
    """
    Compute the RMSE between f and a reference distribution.

    RMSE = sqrt(4π * sum(v^2 * (f - f_ref)^2 * dv))  for spherical
    RMSE = sqrt(sum((f - f_ref)^2 * dv))              for cartesian
    """
    diff_sq = (f - f_ref) ** 2
    if grid.spherical:
        return jnp.sqrt(4.0 * jnp.pi * jnp.sum(grid.v**2 * diff_sq * grid.dv))
    else:
        return jnp.sqrt(jnp.sum(diff_sq * grid.dv))


@eqx.filter_jit
def compute_metrics(
    f: Array,
    grid: VelocityGrid,
    time: float,
    T_final: float,
    n_initial: float,
    vbar_initial: float,
) -> RelaxationMetrics:
    """
    Compute all relaxation metrics for a given distribution.

    Args:
        f: Distribution function, shape (nv,)
        grid: VelocityGrid instance
        time: Current simulation time
        T_final: Expected final equilibrium temperature
        n_initial: Initial density (for normalization)
        vbar_initial: Initial mean velocity (0.0 for spherical/symmetric)

    Returns:
        RelaxationMetrics with all computed metrics
    """
    # Density
    density = compute_density(f, grid)

    # Temperatures
    T_discrete = discrete_temperature(f, grid.v, grid.dv, spherical=grid.spherical)
    # Always use 2 SC iterations for the diagnostic (independent of simulation sc)
    T_sc = compute_self_consistent_temperature(f, grid, max_steps=2)

    # Positivity
    positivity_violation = compute_positivity_violation(f, grid)

    # Current momentum (0.0 for spherical)
    momentum = compute_momentum(f, grid)
    vbar_current = jnp.where(jnp.isnan(momentum), 0.0, momentum)

    # Reference Maxwellians (shifted to match momentum)
    # Final: Maxwellian at initial (vbar, T, n) — the expected equilibrium
    f_final_ref = shifted_maxwellian(grid, v_shift=vbar_initial, T=T_final)
    f_final_ref = f_final_ref * n_initial

    # Instantaneous: Maxwellian at current (vbar, T, n)
    current_density = compute_density(f, grid)
    f_instant_ref = shifted_maxwellian(grid, v_shift=vbar_current, T=T_discrete)
    f_instant_ref = f_instant_ref * current_density

    # RMSE errors
    rmse_final = compute_rmse(f, f_final_ref, grid)
    rmse_instant = compute_rmse(f, f_instant_ref, grid)

    # Entropy
    entropy = compute_entropy(f, grid)

    return RelaxationMetrics(
        time=jnp.asarray(time),
        density=density,
        energy_discrete=T_discrete,
        energy_self_consistent=T_sc,
        positivity_violation=positivity_violation,
        rmse_final_maxwellian=rmse_final,
        rmse_instant_maxwellian=rmse_instant,
        entropy=entropy,
        momentum=momentum,
    )
