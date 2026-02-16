#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Test Fokker-Planck relaxation using production Collisions class.

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

from adept._vlasov1d.solvers.pushers.fokker_planck import Collisions

# =============================================================================
# Test configuration
# =============================================================================

MODELS = ("LenardBernstein", "Dougherty")
SCHEMES = ("ChangCooper", "CentralDifferencing")
EXPERIMENT = "vlasov1d-fokker-planck-relaxation-tests"
VMAX = 6.0
NV = 128
TEMPERATURE_TOL = 5e-2

PROBLEMS = [
    {"ic_fn": partial(problems.maxwellian, T=1.0), "extra_checks": "rmse", "equilibrium": True},
    {"ic_fn": partial(problems.supergaussian, m=5, T=1.0)},
    {"ic_fn": partial(problems.two_temperature, T_cold=0.5, T_hot=2.0, frac_cold=0.7)},
    {"ic_fn": partial(problems.bump_on_tail, narrow=True)},
    {"ic_fn": partial(problems.bump_on_tail, narrow=False)},
    {
        "ic_fn": partial(problems.shifted_maxwellian, v_shift=1.8, T=0.162),
        "extra_checks": "momentum",
        "equilibrium": True,
    },
    {
        "ic_fn": partial(problems.shifted_maxwellian, v_shift=1.0, T=1.0),
        "extra_checks": "momentum",
        "equilibrium": True,
    },
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
    """Test that production Collisions class produces correct physics."""
    factory = Vlasov1dVectorFieldFactory(model_names=MODELS, scheme_names=SCHEMES)
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
# Vector field adapter for production Collisions class
# =============================================================================


class CollisionsVectorField(eqx.Module):
    """
    Adapts Collisions to diffrax vector field interface.

    This wrapper allows tests to exercise the production Collisions class
    (vlasov1d) through the same time-stepping infrastructure.
    """

    collisions: Collisions
    dt: Array = eqx.field(converter=jnp.asarray)
    nu: Array = eqx.field(converter=jnp.asarray)

    def __call__(self, t: float, f: Array, args) -> Array:
        """Take one implicit FP step using production Collisions."""
        del t, args  # Unused
        # Collisions.__call__ expects (nx, nv), add batch dim for single distribution
        f_batched = f[None, :]
        nu_fp = jnp.array([self.nu])
        nu_K = jnp.zeros(1)  # No Krook collisions
        f_new = self.collisions(nu_fp, nu_K, f_batched, self.dt)
        return f_new[0]  # Remove batch dim


# =============================================================================
# Factory for vlasov1d collision vector fields
# =============================================================================


class Vlasov1dVectorFieldFactory(AbstractFPRelaxationVectorFieldFactory):
    """Factory for vlasov1d collision vector fields (LenardBernstein, Dougherty)."""

    _spherical = False  # Cartesian grid

    def make_vector_field(
        self,
        grid: VelocityGrid,
        model_name: str,
        scheme_name: str,
        dt: float,
        nu: float,
        sc_iterations: int,
    ) -> eqx.Module:
        """Create a Collisions vector field for the given model/scheme combo."""
        # Build config
        cfg = self._make_config(grid, model_name, scheme_name, sc_iterations)

        # Create production class and adapter
        collisions = Collisions(cfg)
        return CollisionsVectorField(collisions=collisions, dt=dt, nu=nu)

    def _make_config(
        self,
        grid: VelocityGrid,
        model_name: str,
        scheme_name: str,
        sc_iterations: int,
    ) -> dict:
        """Build Collisions config dict."""
        # Map (model, scheme) to fokker_planck type string
        fp_type_map = {
            ("LenardBernstein", "ChangCooper"): "chang_cooper",
            ("LenardBernstein", "CentralDifferencing"): "lenard_bernstein",
            ("Dougherty", "ChangCooper"): "chang_cooper_dougherty",
            ("Dougherty", "CentralDifferencing"): "dougherty",
        }
        fp_type = fp_type_map.get((model_name, scheme_name))
        if fp_type is None:
            raise ValueError(f"Unknown model/scheme combo: {model_name}/{scheme_name}")

        return {
            "grid": {
                "species_grids": {
                    "electron": {
                        "v": np.asarray(grid.v),
                        "dv": float(grid.dv),
                    },
                },
            },
            "terms": {
                "fokker_planck": {
                    "type": fp_type,
                    "is_on": True,
                    "self_consistent_beta": {
                        "enabled": sc_iterations > 0,
                        "max_steps": sc_iterations,
                    },
                },
                "krook": {"is_on": False},
            },
        }
