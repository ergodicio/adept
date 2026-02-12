#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Time-stepping loop for Fokker-Planck relaxation tests.

Provides run_relaxation() which evolves the distribution and collects snapshots.
Uses diffrax with ADEPT's Stepper for efficient JIT-compiled time stepping.
"""

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax import ODETerm, SaveAt, diffeqsolve
from jax import Array

from adept._base_ import Stepper
from adept.driftdiffusion import _find_self_consistent_beta_single, discrete_temperature

from .metrics import (
    RelaxationMetrics,
    compute_density,
    compute_entropy,
    compute_metrics,
    compute_momentum,
)
from .registry import VelocityGrid


def collision_time(f, grid):
    """Buet collision time (includes 2π factor from rho=n/(2π))."""
    n = compute_density(f, grid)
    T = discrete_temperature(f, grid.v, grid.dv, spherical=grid.spherical)
    return 3.0 * jnp.sqrt(jnp.pi) * discrete_temperature**1.5 / n / 4.0


@dataclass
class TimeStepperConfig:
    """Configuration for the time-stepping loop."""

    n_collision_times: float = 10.0  # Run for this many tau = 1/nu
    n_snapshots: int = 100  # Number of snapshots to collect
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


class FPVectorField(eqx.Module):
    """
    Fokker-Planck vector field for use with diffrax + Stepper.

    This equinox Module wraps the implicit FP step so that:
    1. All parameters are pytree leaves (arrays) or static fields
    2. The __call__ method returns the NEW state directly (not dy/dt)
       because Stepper just passes through: y1 = vf(t0, y0, args)

    Key design: floats like dv, nu, dt are stored as 0-d JAX arrays
    so they're pytree leaves. Changing them doesn't trigger recompilation.
    Only truly shape-affecting values (nv, spherical, sc_iterations) are static.
    """

    # All arrays (including 0-d for scalars) - pytree leaves, no recompilation
    v: Array  # Velocity grid centers, shape (nv,)
    v_edge: Array  # Velocity grid edges, shape (nv-1,)
    dv: Array = eqx.field(converter=jnp.asarray)  # Grid spacing, 0-d array
    nu: Array = eqx.field(converter=jnp.asarray)  # Collision frequency, 0-d array
    dt: Array = eqx.field(converter=jnp.asarray)  # Time step, 0-d array
    model: eqx.Module  # FP model (LenardBernstein, etc.)
    scheme: eqx.Module  # Differencing scheme (ChangCooper, etc.)

    # Only truly static (shape-affecting) values
    nv: int = eqx.field(static=True)
    spherical: bool = eqx.field(static=True)
    sc_iterations: int = eqx.field(static=True)

    @jax.jit
    def __call__(self, t: float, f: Array, args) -> Array:
        """
        Take one implicit FP step and return the NEW state.

        Note: Returns f_new directly, NOT (f_new - f) / dt, because
        ADEPT's Stepper just passes through: y1 = vf(t0, y0, args).

        Args:
            t: Current time (unused, but required by diffrax)
            f: Distribution function, shape (nv,)
            args: Additional arguments (unused)

        Returns:
            f_new: Updated distribution function, shape (nv,)
        """
        del t, args  # Unused

        vbar = self.model.compute_vbar(f)
        beta = _find_self_consistent_beta_single(
            f,
            self.v,
            self.dv,
            spherical=self.spherical,
            vbar=vbar,
            max_steps=self.sc_iterations,
        )

        # Get model coefficients (D from beta, vbar already computed)
        D = self.model.compute_D(f, beta)

        # Compute C_edge using the general formula: C = 2*beta*D*v
        # This ensures correct Maxwellian equilibrium for all models
        v_eff = self.v_edge if vbar is None else (self.v_edge - vbar)
        C_edge = 2.0 * beta * D * v_eff

        # Build operator and solve
        # For spherical geometry, nu ~ 1/v^2 to account for the Jacobian
        if self.spherical:
            nu_arr = self.nu / self.v**2
        else:
            nu_arr = self.nu * jnp.ones(self.nv)

        op = self.scheme.get_operator(C_edge=C_edge, D=D, nu=nu_arr, dt=self.dt)
        f_new = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value

        return f_new


def run_relaxation(
    f0: Array,
    grid: VelocityGrid,
    model,
    scheme,
    config: TimeStepperConfig,
    sc_iterations: int = 0,
    store_f_history: bool = False,
) -> RelaxationResult:
    """
    Run a Fokker-Planck relaxation simulation using diffrax + Stepper.

    This implementation uses ADEPT's Stepper pattern for efficient JIT compilation:
    - Vector field is created ONCE with all parameters as pytree leaves
    - diffeqsolve handles the time stepping loop
    - SaveAt collects snapshots at specified times

    Args:
        f0: Initial distribution function, shape (nv,)
        grid: VelocityGrid instance
        model: Fokker-Planck model instance (LenardBernstein, Dougherty, etc.)
        scheme: Differencing scheme instance (ChangCooper, CentralDifferencing)
        config: TimeStepperConfig instance
        sc_iterations: Number of self-consistency iterations for beta
        store_f_history: If True, store full distribution at each snapshot

    Returns:
        RelaxationResult with snapshots, f_history, etc.
    """
    # Time stepping parameters
    tau = 1.0 / config.nu  # Collision time
    dt = config.dt_over_tau * tau
    t_final = config.n_collision_times * tau
    n_steps = int(t_final / dt)

    # Initial state — compute vbar first (needed for thermal SC temperature)
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
    entropy_initial = compute_entropy(f0, grid)

    result = RelaxationResult(
        f_initial=f0,
        T_initial=float(T_initial),
        config=config,
    )

    # Compute snapshot times:
    # - If dt >= 10*tau, only dump initial and final
    # - Otherwise, dump every tau
    if config.dt_over_tau >= 10.0:
        # Large time steps: only initial and final
        snapshot_times = jnp.array([0.0, t_final])
    else:
        # Dump every tau (collision time)
        n_tau_dumps = int(config.n_collision_times) + 1  # +1 for t=0
        snapshot_times = jnp.array([i * tau for i in range(n_tau_dumps)])

    # Create the vector field ONCE with 0-d arrays for floats
    # This avoids JIT recompilation when parameters change
    vf = FPVectorField(
        v=grid.v,
        v_edge=grid.v_edge,
        dv=grid.dv,
        nu=config.nu,
        dt=dt,
        nv=grid.nv,
        spherical=grid.spherical,
        sc_iterations=sc_iterations,
        model=model,
        scheme=scheme,
    )

    # Set up diffrax solver with ADEPT's Stepper
    term = ODETerm(vf)
    stepper = Stepper()

    # Run the simulation
    # max_steps needs some headroom for diffrax internals
    solution = diffeqsolve(
        term,
        stepper,
        t0=0.0,
        t1=t_final,
        dt0=dt,
        y0=f0,
        saveat=SaveAt(ts=snapshot_times),
        max_steps=n_steps + 4,
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
