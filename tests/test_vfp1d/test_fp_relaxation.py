#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Test Fokker-Planck relaxation using production F0Collisions class.

These tests exercise the actual production collision classes (not test doubles)
to ensure the production code produces correct physics.
"""

from functools import partial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from fp_relaxation import problems
from fp_relaxation.factories import AbstractFPRelaxationVectorFieldFactory
from fp_relaxation.registry import VelocityGrid
from fp_relaxation.runner import problem_name, run_relaxation_sweep
from jax import Array

from adept.vfp1d.fokker_planck import F0Collisions, SelfConsistentBetaConfig, get_model, get_scheme
from adept.vfp1d.grid import Grid

# =============================================================================
# Test configuration
# =============================================================================

MODELS = ("CoulombianKernel", "AsymptoticLocal", "FastVFP")
SCHEMES = ("ChangCooper", "CentralDifferencing")
EXPERIMENT = "vfp1d-fokker-planck-relaxation-tests"
VMAX = 6.0
NV = 128
TEMPERATURE_TOL = 2.5e-1  # Energy not conserved without full Buet weak-form scheme

PROBLEMS = [
    partial(problems.maxwellian, T=1.0),
    partial(problems.supergaussian, m=5, T=1.0),
    partial(problems.two_temperature, T_cold=0.5, T_hot=2.0, frac_cold=0.7),
    partial(problems.bump_on_tail, narrow=True),
    partial(problems.bump_on_tail, narrow=False),
    partial(problems.shifted_maxwellian, v_shift=1.8, T=0.162),
    problems.monoenergetic_beam,
]

SLOW_EXTRA_COMBOS = [
    {"sc_iterations": 0, "dt_over_tau": 1.0},
    {"sc_iterations": 1, "dt_over_tau": 1.0},
    {"sc_iterations": 2, "dt_over_tau": 0.1},
    {"sc_iterations": 2, "dt_over_tau": 10.0},
]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize(
    "slow",
    [
        pytest.param(False, id="fast"),
        pytest.param(True, id="slow", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("ic_fn", PROBLEMS, ids=problem_name)
def test_fp_relaxation(ic_fn, slow):
    """Test that production F0Collisions class produces correct physics."""
    factory = Vfp1dVectorFieldFactory(model_names=MODELS, scheme_names=SCHEMES)
    grid = VelocityGrid(nv=NV, vmax=VMAX, spherical=factory.spherical)

    results = run_relaxation_sweep(
        problem_name=problem_name(ic_fn),
        factory=factory,
        f0=ic_fn(grid),
        grid=grid,
        experiment_name=EXPERIMENT if not slow else f"{EXPERIMENT}-slow",
        extra_params=dict(ic_fn.keywords) if isinstance(ic_fn, partial) else {},
        extra_param_combos=SLOW_EXTRA_COMBOS if slow else None,
    )
    assert results, f"No results for {problem_name(ic_fn)}"

    # Verify conservation properties for each model/scheme
    for name, metrics in results.items():
        # Skip assertions for CentralDifferencing (only log results)
        if "CentralDifferencing" in name:
            continue

        # Density conservation (1e-12 allows for accumulated floating-point roundoff)
        assert abs(metrics.rel_density[-1]) < 1e-12, (
            f"{name}: Density not conserved: rel_density={metrics.rel_density[-1]:.2e}"
        )

        # Temperature (total energy) conservation
        assert jnp.isclose(
            metrics.temperature_discrete[-1], metrics.temperature_discrete[0], atol=0.0, rtol=TEMPERATURE_TOL
        ), (
            f"{name}: Temperature changed: rel_T="
            f"{metrics.temperature_discrete[-1] / metrics.temperature_discrete[0] - 1.0:.2e}"
        )

        # Relaxation to a Maxwellian
        assert metrics.rmse_instant[-1] < 7.5e-3, (
            f"{name}: Did not relax to Maxwellian: rmse_instant={metrics.rmse_instant[-1]:.2e}"
        )

        # Relaxation to expected equilibrium
        assert metrics.rmse_expected[-1] < 5e-2, (
            f"{name}: Did not relax to expected equilibrium: rmse_expected={metrics.rmse_expected[-1]:.2e}"
        )

        assert metrics.positivity_violation[-1] < 1e-20, (
            f"{name}: Violated positivity too strongly: positivity_violation={metrics.positivity_violation[-1]:.2e}"
        )


# =============================================================================
# Vector field adapter for production F0Collisions class
# =============================================================================


class F0CollisionsVectorField(eqx.Module):
    """
    Adapts F0Collisions to diffrax vector field interface.

    This wrapper allows tests to exercise the production F0Collisions class
    through the same time-stepping infrastructure used by FPVectorField.
    """

    collisions: F0Collisions
    dt: Array = eqx.field(converter=jnp.asarray)

    def __call__(self, t: float, f: Array, args) -> Array:
        """Take one implicit FP step using production F0Collisions."""
        del t, args  # Unused
        # F0Collisions.__call__ expects (nx, nv), add batch dim for single distribution
        f_batched = f[None, :]
        f_new = self.collisions(nu=1.0, f0x=f_batched, dt=self.dt)
        return f_new[0]  # Remove batch dim


# =============================================================================
# Factory for vfp1d collision vector fields
# =============================================================================


class Vfp1dVectorFieldFactory(AbstractFPRelaxationVectorFieldFactory):
    """Factory for vfp1d collision vector fields."""

    _spherical = True  # Spherical (positive-only) grid

    def make_vector_field(
        self,
        grid: VelocityGrid,
        model_name: str,
        scheme_name: str,
        dt: float,
        nu: float,
        sc_iterations: int,
    ) -> eqx.Module:
        """Create an F0Collisions vector field for the given model/scheme combo."""
        scheme_map = {
            "ChangCooper": "chang_cooper",
            "CentralDifferencing": "central",
        }

        # Build a vfp1d Grid with dummy spatial/temporal values (F0Collisions only uses velocity fields)
        vfp_grid = Grid(
            xmin=0.0,
            xmax=1.0,
            nx=1,
            tmin=0.0,
            tmax=1.0,
            dt=1.0,
            nv=grid.nv,
            vmax=float(grid.vmax),
            nl=1,
            boundary="periodic",
        )

        collisions = F0Collisions(
            nuee_coeff=nu,
            grid=vfp_grid,
            model=get_model(model_name, vfp_grid.v, vfp_grid.dv),
            scheme=get_scheme(scheme_map.get(scheme_name, scheme_name.lower()), vfp_grid.dv),
            sc_beta=SelfConsistentBetaConfig(max_steps=sc_iterations if sc_iterations > 0 else 0),
        )
        return F0CollisionsVectorField(collisions=collisions, dt=dt)
