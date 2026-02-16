#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Time-stepping loop for Fokker-Planck relaxation tests.

Provides run_relaxation() which evolves the distribution and collects snapshots.
Uses diffrax with ADEPT's Stepper for efficient JIT-compiled time stepping.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, diffeqsolve
from jax import Array

from adept._base_ import Stepper
from adept.driftdiffusion import _find_self_consistent_beta_single, discrete_temperature

from .metrics import (
    RelaxationMetrics,
    compute_density,
    compute_metrics,
    compute_momentum,
)
from .registry import VelocityGrid


@dataclass
class TimeStepperConfig:
    """Configuration for the time-stepping loop."""

    n_collision_times: float = 10.0  # Run for this many tau = 1/nu
    n_snapshots: int = 10  # Number of snapshots to collect
    nu: float = 1.0  # Collision frequency (tau = 1/nu)
    dt_over_tau: float = 1.0  # Time step in units of collision time


@dataclass
class RelaxationResult:
    """Results from a relaxation run."""

    snapshots: RelaxationMetrics = None  # Batched pytree, each leaf shape (n_snapshots,)
    f_history: Array = None  # Shape (n_snapshots, nv) or None
    times: Array = None  # Shape (n_snapshots,)
    f_initial: Array = None
    f_final: Array = None
    T_initial: float = None
    config: TimeStepperConfig = None


def run_relaxation(
    f0: Array,
    grid: VelocityGrid,
    vector_field: eqx.Module,
    config: TimeStepperConfig,
    store_f_history: bool = False,
) -> RelaxationResult:
    """
    Run a Fokker-Planck relaxation simulation using diffrax + Stepper.

    This implementation uses ADEPT's Stepper pattern for efficient JIT compilation:
    - diffeqsolve handles the time stepping loop
    - SaveAt collects snapshots at specified times

    Args:
        f0: Initial distribution function, shape (nv,)
        grid: VelocityGrid instance
        vector_field: Vector field module (from factory) compatible with diffrax
        config: TimeStepperConfig instance
        store_f_history: If True, store full distribution at each snapshot

    Returns:
        RelaxationResult with snapshots, f_history, etc.
    """
    # Time stepping parameters
    tau = 1.0 / config.nu  # Collision time

    n_initial = compute_density(f0, grid)
    vbar_initial = compute_momentum(f0, grid)
    # For spherical (NaN), use 0.0 so reference Maxwellians are zero-centered
    vbar_initial = jnp.where(jnp.isnan(vbar_initial), 0.0, vbar_initial)
    # T_initial: total energy (no vbar) — for conservation metrics
    T_initial = discrete_temperature(f0, grid.v, grid.dv, spherical=grid.spherical)
    # T_sc_initial: thermal SC temperature (with vbar) — for RMSE reference Maxwellians
    beta_sc_initial = _find_self_consistent_beta_single(
        f0, grid.v, grid.dv, spherical=grid.spherical, vbar=vbar_initial
    )
    T_sc_initial = 1.0 / (2.0 * beta_sc_initial)

    result = RelaxationResult(f_initial=f0, T_initial=float(T_initial), config=config)

    # Compute snapshot times:
    if config.dt_over_tau >= 10.0:
        # Large time steps: only initial and final
        snapshot_times = jnp.array([0.0, config.n_collision_times * tau])
    else:
        # Dump every tau (collision time)
        n_tau_dumps = int(config.n_collision_times) + 1  # +1 for t=0
        snapshot_times = jnp.array([i * tau for i in range(n_tau_dumps)])

    # Set up diffrax solver with ADEPT's Stepper
    term = ODETerm(vector_field)
    stepper = Stepper()

    # Run the simulation
    # max_steps needs some headroom for diffrax internals
    solution = diffeqsolve(
        term,
        stepper,
        t0=0.0,
        t1=config.n_collision_times * tau,
        dt0=config.dt_over_tau * tau,
        y0=f0,
        saveat=SaveAt(ts=snapshot_times),
        max_steps=int(config.n_collision_times / config.dt_over_tau) + 4,
    )

    # Compute all metrics in one vectorised call
    # solution.ys: (n_snapshots, nv), solution.ts: (n_snapshots,)
    result.snapshots = jax.vmap(compute_metrics, in_axes=(0, None, 0, None, None, None))(
        solution.ys, grid, solution.ts, T_sc_initial, n_initial, vbar_initial
    )
    result.times = solution.ts
    result.f_history = solution.ys if store_f_history else None
    result.f_final = solution.ys[-1]
    return result
