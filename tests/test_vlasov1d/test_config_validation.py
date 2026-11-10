"""Tests that all vlasov-1d config files pass schema validation.

This test ensures that configuration files in configs/vlasov-1d/ and
tests/test_vlasov1d/configs/ can be successfully loaded and validated
by the BaseVlasov1D constructor.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from adept._vlasov1d.datamodel import HouLiFilterConfig
from adept.vlasov1d import BaseVlasov1D

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "vlasov-1d"
TEST_CONFIGS_DIR = Path(__file__).parent / "configs"


def get_config_files():
    """Get all YAML config files from configs/vlasov-1d/ and tests/test_vlasov1d/configs/."""
    return list(CONFIGS_DIR.glob("*.yaml")) + list(TEST_CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
def test_config_validates_and_constructs(config_path: Path):
    """Test that config file can be loaded and passed to BaseVlasov1D constructor.

    This validates the config against the EMDriverSetConfig schema and other
    pydantic models used during simulation construction.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # This should not raise - if it does, the config doesn't match the schema
    BaseVlasov1D(cfg)


# --- Hou-Li filter: configuration space only -------------------------------------------
#
# The Hou-Li filter is FFT-based, hence periodic in whichever axis it filters. x is
# periodic so filtering there is correct; v is a bounded domain (f -> 0 at +/-vmax), so
# an FFT filter in v wraps the forward tail onto the -v edge and corrupts f(v). Velocity
# -space filtering was therefore removed outright, and the config validator is what keeps
# it from being reintroduced from a deck. These tests guard that validator.


@pytest.mark.parametrize("dimensions", [["v"], ["x", "v"], ["v", "x"]])
def test_hou_li_filter_rejects_velocity_dimensions(dimensions: list[str]):
    """Velocity-space Hou-Li filtering must be rejected at config validation."""
    with pytest.raises(ValidationError, match="velocity-space Hou-Li filtering has been removed"):
        HouLiFilterConfig(is_on=True, dimensions=dimensions)


def test_hou_li_filter_accepts_configuration_space():
    """Filtering in x is the supported case and must still validate."""
    assert HouLiFilterConfig(is_on=True, dimensions=["x"]).dimensions == ["x"]


def test_hou_li_filter_defaults_to_configuration_space():
    """The default must be x-only so an unspecified deck cannot filter in v."""
    assert HouLiFilterConfig(is_on=False).dimensions == ["x"]
