"""Tests that all vlasov-1d config files pass schema validation.

This test ensures that configuration files in configs/vlasov-1d/ can be
successfully loaded and validated by the BaseVlasov1D constructor.
"""

from pathlib import Path

import pytest
import yaml

from adept.vlasov1d import BaseVlasov1D

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "vlasov-1d"


def get_config_files():
    """Get all YAML config files from configs/vlasov-1d/."""
    return list(CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
def test_config_validates_and_constructs(config_path: Path):
    """Test that config file can be loaded and passed to BaseVlasov1D constructor.

    This validates the config against the EMDriverSetModel schema and other
    pydantic models used during simulation construction.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # This should not raise - if it does, the config doesn't match the schema
    BaseVlasov1D(cfg)
