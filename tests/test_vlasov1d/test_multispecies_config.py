"""Test multi-species configuration parsing and validation."""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from adept._vlasov1d.datamodel import ConfigModel, SpeciesConfig


def test_multispecies_config_parsing():
    """Test that multi-species config files parse correctly."""
    config_path = Path(__file__).parent / "configs" / "multispecies_ion_acoustic.yaml"

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Validate with Pydantic model
    config = ConfigModel(**config_dict)

    # Check that species are defined
    assert config.terms.species is not None
    assert len(config.terms.species) == 2

    # Check electron species
    electron = config.terms.species[0]
    assert electron.name == "electron"
    assert electron.charge == -1.0
    assert electron.mass == 1.0
    assert electron.vmax == 6.4
    assert electron.nv == 512
    assert "species-electron-background" in electron.density_components

    # Check ion species
    ion = config.terms.species[1]
    assert ion.name == "ion"
    assert ion.charge == 10.0
    assert ion.mass == 18360.0
    assert ion.vmax == 0.15
    assert ion.nv == 256
    assert "species-ion-background" in ion.density_components


def test_backward_compatible_config():
    """Test that old configs without terms.species still work."""
    config_path = Path(__file__).parent / "configs" / "resonance.yaml"

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Validate with Pydantic model
    config = ConfigModel(**config_dict)

    # Check that species is None (not provided in old configs)
    assert config.terms.species is None


def test_species_config_validation():
    """Test SpeciesConfig validation."""
    # Valid species config
    species = SpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0,
        vmax=6.4,
        nv=512,
        density_components=["species-background"],
    )
    assert species.name == "electron"

    # Test that all required fields must be present
    with pytest.raises(ValidationError):
        SpeciesConfig(
            name="electron",
            charge=-1.0,
            # Missing mass, vmax, nv, density_components
        )


def test_multiple_density_components():
    """Test species with multiple density components."""
    species = SpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0,
        vmax=6.4,
        nv=512,
        density_components=["species-background", "species-beam"],
    )
    assert len(species.density_components) == 2
    assert "species-background" in species.density_components
    assert "species-beam" in species.density_components


if __name__ == "__main__":
    # Run tests
    test_multispecies_config_parsing()
    test_backward_compatible_config()
    test_species_config_validation()
    test_multiple_density_components()
    print("All tests passed!")
