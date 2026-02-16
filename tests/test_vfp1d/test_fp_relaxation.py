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
from fp_relaxation.runner import problem_name, run_relaxation_sweep_and_assert
from jax import Array

from adept.vfp1d.fokker_planck import F0Collisions

# =============================================================================
# Test configuration
# =============================================================================

MODELS = ("FastVFP",)
SCHEMES = ("ChangCooper", "CentralDifferencing")
EXPERIMENT = "vfp1d-fokker-planck-relaxation-tests"
VMAX = 6.0
NV = 128
TEMPERATURE_TOL = 5e-2

PROBLEMS = [
    {"ic_fn": partial(problems.maxwellian, T=1.0), "extra_checks": "rmse", "equilibrium": True},
    {"ic_fn": partial(problems.supergaussian, m=5, T=1.0)},
    {"ic_fn": partial(problems.two_temperature, T_cold=0.5, T_hot=2.0, frac_cold=0.7)},
    {"ic_fn": partial(problems.bump_on_tail, narrow=True)},
    {"ic_fn": partial(problems.bump_on_tail, narrow=False)},
    {"ic_fn": partial(problems.shifted_maxwellian, v_shift=1.8, T=0.162)},
    {"ic_fn": problems.monoenergetic_beam},
]

# =============================================================================
# Tests
# =============================================================================

SLOW_EXTRA_COMBOS = [
    {"sc_iterations": 0, "dt_over_tau": 1.0},
    {"sc_iterations": 1, "dt_over_tau": 1.0},
    {"sc_iterations": 2, "dt_over_tau": 0.1},
    {"sc_iterations": 2, "dt_over_tau": 10.0},
]


@pytest.mark.parametrize(
    "slow",
    [
        pytest.param(False, id="fast"),
        pytest.param(True, id="slow", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: problem_name(p["ic_fn"]))
def test_fp_relaxation(problem, slow):
    """Test that production F0Collisions class produces correct physics."""
    factory = Vfp1dVectorFieldFactory(model_names=MODELS, scheme_names=SCHEMES)
    run_relaxation_sweep_and_assert(
        factory=factory,
        experiment_name=EXPERIMENT if not slow else f"{EXPERIMENT}-slow",
        problem=problem,
        slow=slow,
        slow_extra_combos=SLOW_EXTRA_COMBOS if slow else [],
        nv=NV,
        vmax=VMAX,
        temperature_tol=TEMPERATURE_TOL,
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
    """Factory for vfp1d collision vector fields (FastVFP only)."""

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
        # Build config
        cfg = self._make_config(grid, scheme_name, nu, sc_iterations)

        # Create production class and adapter
        collisions = F0Collisions(cfg)
        return F0CollisionsVectorField(collisions=collisions, dt=dt)

    def _make_config(
        self,
        grid: VelocityGrid,
        scheme_name: str,
        nu: float,
        sc_iterations: int,
    ) -> dict:
        """Build F0Collisions config dict."""
        scheme_map = {
            "ChangCooper": "chang_cooper",
            "CentralDifferencing": "central",
        }
        return {
            "grid": {
                "v": np.asarray(grid.v),
                "dv": float(grid.dv),
                "nv": grid.nv,
            },
            "terms": {
                "fokker_planck": {
                    # Direct nuee_coeff override for dimensionless testing
                    "nuee_coeff": nu,
                    "f00": {
                        "model": "FastVFP",
                        "scheme": scheme_map.get(scheme_name, scheme_name.lower()),
                    },
                    "self_consistent_beta": {
                        "enabled": sc_iterations > 0,
                        "max_steps": sc_iterations,
                    },
                },
            },
        }
