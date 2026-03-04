"""Test quasineutral initialization for single-species Vlasov-1D."""

import copy
from pathlib import Path

import numpy as np
import yaml
from jax import numpy as jnp

from adept._vlasov1d.modules import BaseVlasov1D
from adept._vlasov1d.solvers.pushers.field import SpectralPoissonSolver


def test_nonuniform_density_quasineutral():
    """Test quasineutrality with a non-uniform density profile.

    A uniform density only has a k=0 Fourier mode, which gets zeroed in the
    spectral Poisson solve. This test uses a tanh profile to ensure
    there are non-zero k modes that would reveal if quasineutrality is broken.

    For single-species simulations with quasineutrality enabled, the static ion
    background (ion_charge) should exactly cancel the electron charge, giving
    zero total charge density and thus zero E-field from the Poisson solve.
    """
    config_path = Path(__file__).parent / "configs" / "resonance.yaml"

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Modify to have a non-uniform density using tanh basis
    config_dict = copy.deepcopy(config_dict)
    config_dict["density"]["species-background"]["basis"] = "tanh"
    config_dict["density"]["species-background"]["baseline"] = 1.0
    config_dict["density"]["species-background"]["bump_height"] = 0.1
    config_dict["density"]["species-background"]["bump_or_trough"] = "bump"
    config_dict["density"]["species-background"]["width"] = 10.0
    config_dict["density"]["species-background"]["center"] = 10.0
    config_dict["density"]["species-background"]["rise"] = 2.0

    # Disable the driver
    config_dict["drivers"]["ex"] = {}

    # Create and initialize module
    module = BaseVlasov1D(config_dict)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()

    # Get grid quantities
    grid = module.simulation.grid
    species_grids = module.cfg["grid"]["species_grids"]
    species_params = module.cfg["grid"]["species_params"]
    ion_charge = module.cfg["grid"]["ion_charge"]

    # Verify the ion_charge (static background) is non-uniform
    ion_charge_variation = float(np.max(ion_charge) - np.min(ion_charge))
    assert ion_charge_variation > 0.01, "Ion charge background should be non-uniform for this test"

    # Create the Poisson solver with static ion background
    poisson_solver = SpectralPoissonSolver(
        one_over_kx=grid.one_over_kx,
        species_grids=species_grids,
        species_params=species_params,
        static_charge_density=ion_charge,
    )

    # Build f_dict from state
    f_dict = {k: v for k, v in module.state.items() if k in species_grids}

    # Verify total charge density is approximately zero (quasineutral)
    total_charge_density = poisson_solver.compute_charge_density(f_dict)
    np.testing.assert_allclose(
        total_charge_density,
        0.0,
        atol=1e-6,
        err_msg="Total charge density should be zero for quasineutral initialization",
    )

    # Solve Poisson - E-field should be zero for quasineutral plasma
    e_field = poisson_solver(f_dict, prev_ex=jnp.zeros(grid.nx), dt=grid.dt)

    np.testing.assert_allclose(
        e_field,
        0.0,
        atol=1e-6,
        err_msg="Quasineutral initialization with non-uniform density should give zero E-field",
    )
