"""Test multi-species initialization and state structure."""

from pathlib import Path

import numpy as np
import yaml

from adept._vlasov1d.modules import BaseVlasov1D


def test_multispecies_state_initialization():
    """Test that multi-species state initialization creates correct structures."""
    config_path = Path(__file__).parent / "configs" / "multispecies_ion_acoustic.yaml"

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Create module
    module = BaseVlasov1D(config_dict)

    # Initialize (following the pattern from the module)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()

    # Check state structure
    assert "electron" in module.state
    assert "ion" in module.state
    assert "e" in module.state
    assert "de" in module.state

    # Check electron distribution shape (nx=32, nv=512 from config)
    assert module.state["electron"].shape == (32, 512)

    # Check ion distribution shape (nx=32, nv=256 from config)
    assert module.state["ion"].shape == (32, 256)

    # Check species_grids
    assert "species_grids" in module.cfg["grid"]
    assert "electron" in module.cfg["grid"]["species_grids"]
    assert "ion" in module.cfg["grid"]["species_grids"]

    # Check electron velocity grid
    electron_grid = module.cfg["grid"]["species_grids"]["electron"]
    assert electron_grid["nv"] == 512
    assert electron_grid["vmax"] == 6.4
    assert len(electron_grid["v"]) == 512

    # Check ion velocity grid
    ion_grid = module.cfg["grid"]["species_grids"]["ion"]
    assert ion_grid["nv"] == 256
    assert ion_grid["vmax"] == 0.15
    assert len(ion_grid["v"]) == 256

    # Check species_params
    assert "species_params" in module.cfg["grid"]
    assert "electron" in module.cfg["grid"]["species_params"]
    assert "ion" in module.cfg["grid"]["species_params"]

    # Check electron params
    electron_params = module.cfg["grid"]["species_params"]["electron"]
    assert electron_params["charge"] == -1.0
    assert electron_params["mass"] == 1.0
    assert electron_params["charge_to_mass"] == -1.0

    # Check ion params
    ion_params = module.cfg["grid"]["species_params"]["ion"]
    assert ion_params["charge"] == 10.0
    assert ion_params["mass"] == 18360.0
    assert np.isclose(ion_params["charge_to_mass"], 10.0 / 18360.0)

    # Check quasineutrality handling for multi-species
    # For multi-species, ion_charge should be zeros
    assert np.allclose(module.cfg["grid"]["ion_charge"], 0.0)


def test_backward_compatible_state_initialization():
    """Test that single-species configs still work correctly."""
    config_path = Path(__file__).parent / "configs" / "resonance.yaml"

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Create module
    module = BaseVlasov1D(config_dict)

    # Initialize
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()

    # Check state structure (should have electron only)
    assert "electron" in module.state
    assert "e" in module.state
    assert "de" in module.state

    # For backward compatibility, electron should have shape (nx, nv) with grid-level nv
    nx = module.cfg["grid"]["nx"]
    nv = module.cfg["grid"]["nv"]
    assert module.state["electron"].shape == (nx, nv)

    # Check that single-species still has grid-level velocity grid
    assert "v" in module.cfg["grid"]
    assert len(module.cfg["grid"]["v"]) == nv

    # Check quasineutrality handling for single-species
    # For single-species, ion_charge should equal n_prof_total
    assert np.allclose(module.cfg["grid"]["ion_charge"], module.cfg["grid"]["n_prof_total"])


if __name__ == "__main__":
    test_multispecies_state_initialization()
    test_backward_compatible_state_initialization()
    print("All initialization tests passed!")
