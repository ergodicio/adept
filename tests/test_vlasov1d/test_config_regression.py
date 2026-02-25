"""Regression tests for Vlasov1D configuration logging.

These tests verify that the configuration files dumped during ergoExo._setup_()
match expected baselines. This ensures the refactoring from cfg-based data
threading to domain objects doesn't change the logged configuration output.
"""

import pickle
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from adept import ergoExo

CONFIGS_DIR = Path(__file__).parent / "configs"

# List of config files to test
CONFIG_FILES = [
    "resonance.yaml",
    "fokker_planck_conservation.yaml",
    "multispecies_ion_acoustic.yaml",
]


def round_float(value, sig_figs=14):
    """Round a float to a given number of significant figures.

    This avoids platform-specific floating-point representation differences
    (e.g., 1.9625000000000001 vs 1.9625) that cause test failures when
    comparing across macOS and Linux.
    """
    if value == 0:
        return 0.0
    from math import floor, log10

    magnitude = floor(log10(abs(value)))
    return round(value, sig_figs - 1 - magnitude)


def normalize_for_regression(obj):
    """Recursively convert JAX/numpy arrays and Pint quantities for regression testing."""
    # Check for Pint Quantity first (has magnitude and units attributes)
    if hasattr(obj, "magnitude") and hasattr(obj, "units"):
        return str(obj)
    # Handle numpy scalar types (np.float64, np.int64, etc.)
    elif isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float):
            return round_float(val)
        return val
    elif isinstance(obj, float):
        return round_float(obj)
    elif hasattr(obj, "tolist"):
        # Convert array to list first, then normalize each element
        return normalize_for_regression(np.array(obj).tolist())
    elif isinstance(obj, dict):
        return {k: normalize_for_regression(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_for_regression(item) for item in obj]
    else:
        return obj


# Cache for setup results to avoid running _setup_() multiple times per config
_setup_cache = {}


def get_setup_files(config_name: str) -> dict:
    """Run _setup_() for a config and return the dumped files. Results are cached."""
    if config_name in _setup_cache:
        return _setup_cache[config_name]

    config_path = CONFIGS_DIR / config_name

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg = deepcopy(cfg)

    exo = ergoExo()
    with tempfile.TemporaryDirectory() as td:
        # Patch mlflow.log_params to avoid MLflow dependency
        with patch("adept.utils.mlflow.log_params"):
            exo._setup_(cfg, td, log=True)

        files = {}

        # Use unsafe_load for all YAML files since they contain Python objects
        # (Pint Quantity, numpy scalars, etc.) from our setup process
        with open(f"{td}/config.yaml") as f:
            files["config"] = yaml.unsafe_load(f)

        with open(f"{td}/units.yaml") as f:
            files["units"] = yaml.unsafe_load(f)

        with open(f"{td}/derived_config.yaml") as f:
            files["derived_config"] = yaml.unsafe_load(f)

        with open(f"{td}/array_config.pkl", "rb") as f:
            files["array_config"] = pickle.load(f)

    _setup_cache[config_name] = files
    return files


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_config_yaml(config_name, data_regression):
    """Test that config.yaml matches baseline."""
    files = get_setup_files(config_name)
    basename = config_name.replace(".yaml", "_config")
    normalized = normalize_for_regression(files["config"])
    data_regression.check(normalized, basename=basename)


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_units_yaml(config_name, data_regression):
    """Test that units.yaml matches baseline."""
    files = get_setup_files(config_name)
    basename = config_name.replace(".yaml", "_units")
    # Normalize Pint quantities to strings
    normalized = normalize_for_regression(files["units"])
    data_regression.check(normalized, basename=basename)


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_derived_config_yaml(config_name, data_regression):
    """Test that derived_config.yaml matches baseline."""
    files = get_setup_files(config_name)
    basename = config_name.replace(".yaml", "_derived_config")
    normalized = normalize_for_regression(files["derived_config"])
    data_regression.check(normalized, basename=basename)


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_array_config_pkl(config_name, data_regression):
    """Test that array_config.pkl matches baseline.

    Arrays are converted to lists for comparison via data_regression.
    """
    files = get_setup_files(config_name)
    basename = config_name.replace(".yaml", "_array_config")

    # Normalize arrays to lists for comparison
    normalized = normalize_for_regression(files["array_config"])
    data_regression.check(normalized, basename=basename)
