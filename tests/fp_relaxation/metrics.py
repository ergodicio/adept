#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Metric computation functions for Fokker-Planck relaxation tests.

All metrics handle both cartesian and spherical geometries appropriately.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from adept.driftdiffusion import _find_self_consistent_beta_single, discrete_temperature

from .problems import shifted_maxwellian
from .registry import VelocityGrid


class RelaxationMetrics(eqx.Module):
    """Metrics computed at each snapshot during relaxation.

    Conservation quantities (density, temperature, entropy) are stored as raw
    values.  Relative discrepancies are computed on the fly for MLflow logging
    and assertions (using index-0 as reference), matching the entropy pattern.
    """

    time: Array
    rel_density: Array  # (n - n0) / n0
    temperature_discrete: Array  # total-energy discrete T
    temperature_sc: Array  # total-energy self-consistent T
    positivity_violation: Array  # integral(|f| where f<0)
    momentum_drift: Array  # p - p0 (absolute, not relative; p0 can be 0)
    rmse_expected: Array  # RMSE to expected equilibrium Maxwellian
    rmse_instant: Array  # RMSE to instantaneous Maxwellian(n, p, T)
    entropy: Array  # -integral(f*log(f) dv), NaN if f<=0 anywhere


def compute_density(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the density (zeroth moment).

    For cartesian: n = integral(f dv)
    For spherical: n = 4*pi * integral(v^2 * f dv)

    Supports arbitrary leading batch dimensions.
    """
    if grid.spherical:
        return 4.0 * jnp.pi * jnp.sum(grid.v**2 * f * grid.dv, axis=-1)
    else:
        return jnp.sum(f * grid.dv, axis=-1)


def compute_momentum(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the mean velocity (first moment).

    For cartesian: <v> = integral(v * f dv) / n
    For spherical: Returns 0 (symmetric about origin)

    Supports arbitrary leading batch dimensions.
    """
    if grid.spherical:
        return jnp.zeros(f.shape[:-1])
    n = jnp.sum(f * grid.dv, axis=-1)
    return jnp.sum(grid.v * f * grid.dv, axis=-1) / n


def compute_positivity_violation(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the integral of |f| where f < 0.

    This measures how badly positivity is violated.

    Supports arbitrary leading batch dimensions.
    """
    negative_f = jnp.where(f < 0, -f, 0.0)
    if grid.spherical:
        return 4.0 * jnp.pi * jnp.sum(grid.v**2 * negative_f * grid.dv, axis=-1)
    else:
        return jnp.sum(negative_f * grid.dv, axis=-1)


def compute_entropy(f: Array, grid: VelocityGrid) -> Array:
    """
    Compute the entropy: -integral(f * log(f) dv).

    Returns NaN if f < -1e-20 anywhere (entropy undefined).

    Supports arbitrary leading batch dimensions.
    """
    safe_f = jnp.where(f > 0, f, 1.0)  # log(1)=0, so 0*0=0 for f=0 points
    if grid.spherical:
        entropy = -4.0 * jnp.pi * jnp.sum(grid.v**2 * f * jnp.log(safe_f) * grid.dv, axis=-1)
    else:
        entropy = -jnp.sum(f * jnp.log(safe_f) * grid.dv, axis=-1)
    return jnp.where(jnp.any(f < -1e-20, axis=-1), jnp.nan, entropy)


def compute_rmse(f: Array, f_ref: Array, grid: VelocityGrid) -> Array:
    """
    Compute the RMSE between f and a reference distribution.

    RMSE = sqrt(4π * sum(v^2 * (f - f_ref)^2 * dv))  for spherical
    RMSE = sqrt(sum((f - f_ref)^2 * dv))              for cartesian

    Supports arbitrary leading batch dimensions (f and f_ref must broadcast).
    """
    diff_sq = (f - f_ref) ** 2
    if grid.spherical:
        return jnp.sqrt(4.0 * jnp.pi * jnp.sum(grid.v**2 * diff_sq * grid.dv, axis=-1))
    else:
        return jnp.sqrt(jnp.sum(diff_sq * grid.dv, axis=-1))


@eqx.filter_jit
def compute_metrics(
    f: Array,
    grid: VelocityGrid,
    times: Array,
) -> RelaxationMetrics:
    """
    Compute all relaxation metrics for a batch of distributions.

    Initial values for relative/drift quantities are taken from f[0].

    Args:
        f: Distribution functions, shape (n_snapshots, nv)
        grid: VelocityGrid instance
        times: Simulation times, shape (n_snapshots,)

    Returns:
        RelaxationMetrics with all arrays shape (n_snapshots,)
    """
    # Density and relative density
    density = compute_density(f, grid)
    n_initial = density[0]
    rel_density = (density - n_initial) / n_initial

    # Momentum and drift
    vbar = compute_momentum(f, grid)
    vbar_initial = vbar[0]
    momentum_drift = vbar - vbar_initial

    # Temperatures (vmap the external functions)
    T_discrete = jax.vmap(discrete_temperature, in_axes=(0, None, None, None))(f, grid.v, grid.dv, grid.spherical)
    beta_sc = jax.vmap(_find_self_consistent_beta_single, in_axes=(0, None, None, None, None, None))(
        f, grid.v, grid.dv, grid.spherical, None, 2
    )
    T_sc = 1.0 / (2.0 * beta_sc)

    # Positivity violation
    positivity_violation = compute_positivity_violation(f, grid)

    # Thermal self-consistent temperature (with vbar subtraction) for RMSE references
    beta_sc_thermal = jax.vmap(_find_self_consistent_beta_single, in_axes=(0, None, None, None, 0, None))(
        f, grid.v, grid.dv, grid.spherical, vbar, 2
    )
    T_sc_thermal = 1.0 / (2.0 * beta_sc_thermal)
    T_sc_initial = T_sc_thermal[0]

    # Expected equilibrium: Maxwellian matching initial thermal temperature (same for all)
    f_expected = shifted_maxwellian(grid, v_shift=vbar_initial, T=T_sc_initial)
    f_expected = f_expected * n_initial
    rmse_expected = compute_rmse(f, f_expected, grid)  # f_expected broadcasts

    # Instantaneous: Maxwellian matching current thermal temperature (different for each)
    def make_instant_maxwellian(vbar_i, T_i, n_i):
        f_m = shifted_maxwellian(grid, v_shift=vbar_i, T=T_i)
        return f_m * n_i

    f_instant = jax.vmap(make_instant_maxwellian)(vbar, T_sc_thermal, density)
    rmse_instant = compute_rmse(f, f_instant, grid)

    # Entropy
    entropy = compute_entropy(f, grid)

    return RelaxationMetrics(
        time=times,
        rel_density=rel_density,
        temperature_discrete=T_discrete,
        temperature_sc=T_sc,
        positivity_violation=positivity_violation,
        momentum_drift=momentum_drift,
        rmse_expected=rmse_expected,
        rmse_instant=rmse_instant,
        entropy=entropy,
    )
