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

    Returns NaN if f < -1e-20 anywhere (entropy undefined).
    """
    safe_f = jnp.where(f > 0, f, 1.0)  # log(1)=0, so 0*0=0 for f=0 points
    if grid.spherical:
        entropy = -4.0 * jnp.pi * jnp.sum(grid.v**2 * f * jnp.log(safe_f) * grid.dv)
    else:
        entropy = -jnp.sum(f * jnp.log(safe_f) * grid.dv)
    return jnp.where(jnp.any(f < -1e-20), jnp.nan, entropy)


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
    T_sc_initial: float,
    n_initial: float,
    vbar_initial: float,
) -> RelaxationMetrics:
    """
    Compute all relaxation metrics for a given distribution.

    Conservation quantities (temperature, entropy) are stored as raw values;
    relative discrepancies are computed on the fly using index-0 as reference.

    Args:
        f: Distribution function, shape (nv,)
        grid: VelocityGrid instance
        time: Current simulation time
        T_sc_initial: Initial *thermal* self-consistent temperature (Maxwellian
            parameter that reproduces the initial thermal discrete T on the
            finite grid; used for the expected-equilibrium RMSE reference)
        n_initial: Initial density
        vbar_initial: Initial mean velocity (0.0 for spherical/symmetric)

    Returns:
        RelaxationMetrics with all computed metrics
    """
    # Density discrepancy
    density = compute_density(f, grid)
    rel_density = (density - n_initial) / n_initial

    # Momentum (compute before temperature — needed for vbar subtraction)
    momentum = compute_momentum(f, grid)
    momentum_drift = jnp.where(jnp.isnan(momentum), 0.0, momentum - vbar_initial)
    vbar_current = jnp.where(jnp.isnan(momentum), 0.0, momentum)

    # Raw temperatures — total energy (no vbar subtraction).
    # Relative discrepancies computed on the fly from index-0 (like entropy).
    T_discrete = discrete_temperature(f, grid.v, grid.dv, spherical=grid.spherical)
    beta_sc = _find_self_consistent_beta_single(f, grid.v, grid.dv, spherical=grid.spherical, vbar=None, max_steps=2)
    T_sc = 1.0 / (2.0 * beta_sc)

    # Positivity
    positivity_violation = compute_positivity_violation(f, grid)

    # RMSE references — use *thermal* self-consistent temperature (with vbar)
    # so the reference Maxwellian has the correct discrete thermal temperature.
    beta_sc_thermal = _find_self_consistent_beta_single(
        f, grid.v, grid.dv, spherical=grid.spherical, vbar=vbar_current, max_steps=2
    )
    T_sc_thermal = 1.0 / (2.0 * beta_sc_thermal)

    # Expected equilibrium: Maxwellian matching initial thermal temperature
    f_expected = shifted_maxwellian(grid, v_shift=vbar_initial, T=T_sc_initial)
    f_expected = f_expected * n_initial

    # Instantaneous: Maxwellian matching current thermal temperature
    f_instant = shifted_maxwellian(grid, v_shift=vbar_current, T=T_sc_thermal)
    f_instant = f_instant * density

    rmse_expected = compute_rmse(f, f_expected, grid)
    rmse_instant = compute_rmse(f, f_instant, grid)

    # Entropy (raw; relative change computed on the fly for MLflow logging)
    entropy = compute_entropy(f, grid)

    return RelaxationMetrics(
        time=jnp.asarray(time),
        rel_density=rel_density,
        temperature_discrete=T_discrete,
        temperature_sc=T_sc,
        positivity_violation=positivity_violation,
        momentum_drift=momentum_drift,
        rmse_expected=rmse_expected,
        rmse_instant=rmse_instant,
        entropy=entropy,
    )
