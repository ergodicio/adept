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
from fp_relaxation.runner import problem_name, run_relaxation_sweep
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
TEMPERATURE_TOL = 5e-3

PROBLEMS = [
    partial(problems.maxwellian, T=1.0),
    partial(problems.supergaussian, m=5, T=1.0),
    partial(problems.two_temperature, T_cold=0.5, T_hot=2.0, frac_cold=0.7),
    partial(problems.bump_on_tail, narrow=True),
    partial(problems.bump_on_tail, narrow=False),
    partial(problems.shifted_maxwellian, v_shift=1.8, T=0.162),
    partial(problems.shifted_maxwellian, v_shift=1.0, T=1.0),
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
    """Test that production Collisions class produces correct physics."""
    factory = Vlasov1dVectorFieldFactory(model_names=MODELS, scheme_names=SCHEMES)
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

        # Density conservation
        assert abs(metrics.rel_density[-1]) < 2e-13, (
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
        assert metrics.rmse_instant[-1] < 1e-4, (
            f"{name}: Did not relax to Maxwellian: rmse_instant={metrics.rmse_instant[-1]:.2e}"
        )

        assert metrics.positivity_violation[-1] < 1e-20(
            f"{name}: Violated positivity too strongly: positivity_violation={metrics.positivity_violation[-1]:.2e}"
        )

        # skip LB - it doesn't conserve momentum
        if "LenardBernstein" not in name:
            assert metrics.rmse_expected[-1] < 1e-2, (
                f"{name}: Did not relax to expected equilibrium: rmse_expected={metrics.rmse_expected[-1]:.2e}"
            )

            assert abs(metrics.momentum_drift[-1]) < 5e-5, (
                f"{name}: Momentum not conserved: drift={metrics.momentum_drift[-1]:.2e}"
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
