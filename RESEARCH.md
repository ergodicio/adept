# Research: Configuration Logging Regression Tests for Vlasov1D

## Overview

This document captures research findings for implementing regression tests for the Vlasov1D configuration logging. The goal is to create tests that verify the exact contents of configuration files dumped during `ergoExo._setup_()`.

## Context: The Refactoring Goal

The broader refactoring effort aims to:
1. Replace `cfg`-based data threading with domain objects (`Vlasov1DSimulation`, `PlasmaNormalization`, `XGrid`, `Species`, `VGrid`)
2. Move simulation setup into the constructor of a new `Vlasov1DSimulation` class
3. Maintain backwards compatibility by constructing domain objects upfront, then "pretending" to construct them bit-by-bit in lifecycle methods

**Why regression tests first?** These tests will ensure the configuration logging output doesn't change during refactoring, catching any accidental behavioral changes.

---

## Configuration Files to Test

### Vlasov1D Test Configs

Located in `tests/test_vlasov1d/configs/`:

| Config File | Description | Key Features |
|-------------|-------------|--------------|
| `resonance.yaml` | Landau damping resonance tests | Single species (electron), Poisson field, leapfrog time |
| `fokker_planck_conservation.yaml` | Fokker-Planck conservation tests | Single species, Lenard-Bernstein FP operator |
| `multispecies_ion_acoustic.yaml` | Multi-species ion acoustic wave | Electron + ion species, sixth-order time integration |

---

## Configuration Logging Flow

### ergoExo._setup_() (adept/_base_.py:315-354)

The `_setup_()` method writes **4 files** to a temporary directory:

```
1. config.yaml          - Raw config (deepcopy of input)
2. units.yaml           - Output of write_units() - derived physical units
3. derived_config.yaml  - Config after get_derived_quantities() - scalars computed
4. array_config.pkl     - Config after get_solver_quantities() - contains JAX arrays
```

### Lifecycle Method Sequence

```
write_units()            -> cfg["units"]["derived"], cfg["grid"]["beta"]
get_derived_quantities() -> cfg["grid"]["dx"], cfg["grid"]["dt"], cfg["grid"]["nt"],
                            cfg["grid"]["tmax"], cfg["terms"]["species"] normalized
get_solver_quantities()  -> cfg["grid"] arrays (x, t, kx, species_grids, etc.)
init_state_and_args()    -> self.state, self.args (not logged)
init_diffeqsolve()       -> cfg["save"] updated with axes/functions (not logged)
```

### Key Config Modifications Per Stage

**After write_units():**
```python
cfg["units"]["derived"] = {
    "wp0": plasma_frequency,
    "tp0": time_scale,
    "n0": density,
    "v0": velocity,
    "T0": temperature,
    "c_light": speed_of_light,
    "beta": v0/c,
    "x0": spatial_scale,
    "nuee": collision_freq,
    "logLambda_ee": coulomb_log,
    "box_length", "box_width", "sim_duration"
}
cfg["grid"]["beta"] = float_value
```

**After get_derived_quantities():**
```python
cfg["grid"]["dx"] = xmax / nx
cfg["grid"]["dt"] = possibly_adjusted
cfg["grid"]["nt"] = int
cfg["grid"]["max_steps"] = int
cfg["grid"]["tmax"] = adjusted
cfg["save"][type]["t"]["tmin"/"tmax"] = defaults
cfg["terms"]["species"] = normalized_species_list  # with density_components
```

**After get_solver_quantities() (in array_config.pkl):**
```python
cfg["grid"]["x"] = jax_array
cfg["grid"]["t"] = jax_array
cfg["grid"]["kx"], cfg["grid"]["kxr"] = jax_arrays
cfg["grid"]["species_grids"] = {name: {v, dv, nv, vmax, kv, ...}}
cfg["grid"]["species_params"] = {name: {charge, mass, charge_to_mass}}
cfg["grid"]["species_distributions"] = {name: (n_prof, f_s, v_ax)}
```

---

## Testing Approach Analysis

### Current Test Setup

- Tests use `ergoExo().setup(cfg)` which requires MLflow
- No pytest-regression currently in use (not in pyproject.toml)
- Tests use `np.testing.assert_almost_equal()` for numerical comparisons

### _setup_() Method Signature

```python
def _setup_(self, cfg: dict, td: str, adept_module: ADEPTModule = None, log: bool = True) -> dict[str, Module]:
```

The `log` parameter controls whether files are written. We can call `_setup_()` directly with a temp directory, bypassing MLflow.

### Approach

**MLflow bypass:** Call `_setup_()` directly with a temp directory. This avoids MLflow entirely and is the simplest approach.

**Pickle file with JAX arrays:** Convert JAX arrays to numpy, then use `num_regression` with numerical tolerance for comparison.

**YAML dict ordering:** Load YAML back to dict and use `data_regression` for comparison (handles ordering automatically).

**Pint quantities in units.yaml:** String comparison is deterministic and sufficient.

---

## pytest-regression

### What It Does

`pytest-regression` provides fixtures for regression testing:
- `data_regression` - compares dicts/lists to YAML files
- `file_regression` - compares text file contents
- `num_regression` - compares numpy arrays with tolerance

### Installation

Add to `pyproject.toml` dev dependencies:
```toml
dev = [
    ...
    "pytest-regressions",  # note: the package name has an 's'
]
```

### Usage Pattern

```python
def test_config_regression(data_regression):
    result = compute_config()
    data_regression.check(result)  # Compares to tests/test_foo/test_config_regression.yml
```

On first run, creates the baseline file. On subsequent runs, compares against it.

---

## Key Files to Modify/Create

| File | Purpose |
|------|---------|
| `pyproject.toml` | Add pytest-regressions to dev dependencies |
| `tests/test_vlasov1d/test_config_regression.py` | New test file |
| `tests/test_vlasov1d/test_config_regression/` | Directory for regression baselines (auto-created) |

---

## Implementation

### Test Pattern

```python
from adept import ergoExo
import tempfile
import yaml

def test_config_logging(data_regression):
    with open("tests/test_vlasov1d/configs/resonance.yaml") as f:
        cfg = yaml.safe_load(f)

    exo = ergoExo()
    with tempfile.TemporaryDirectory() as td:
        exo._setup_(cfg, td, log=True)

        # Load and compare each file
        with open(f"{td}/config.yaml") as f:
            config = yaml.safe_load(f)
        data_regression.check(config, basename="resonance_config")
```

### Handling array_config.pkl

```python
import pickle
import numpy as np

def normalize_config_for_regression(cfg):
    """Convert JAX arrays to lists for regression testing"""
    result = {}
    for key, value in cfg.items():
        if hasattr(value, 'tolist'):  # JAX/numpy array
            result[key] = np.array(value).tolist()
        elif isinstance(value, dict):
            result[key] = normalize_config_for_regression(value)
        else:
            result[key] = value
    return result
```

### Handling Pint Quantities in units.yaml

Pint quantities serialize as deterministic strings (e.g., `"1.5e+21 1 / centimeter ** 3"`). String comparison via `data_regression` is sufficient.

---

## Test Structure

```
tests/test_vlasov1d/
├── test_config_regression.py          # New test file
├── test_config_regression/             # Baseline files (auto-created)
│   ├── test_resonance_config_config.yml
│   ├── test_resonance_config_units.yml
│   ├── test_resonance_config_derived.yml
│   ├── test_fokker_planck_config_config.yml
│   └── ...
└── configs/
    ├── resonance.yaml
    ├── fokker_planck_conservation.yaml
    └── multispecies_ion_acoustic.yaml
```

---

## Resolved Decisions

1. **Array tolerance**: Use `num_regression` with numerical tolerance for array comparisons.

2. **Baseline regeneration**: Use pytest-regression's `--force-regen` flag when intentional changes are made.

3. **CI integration**: Tests will be fast and run on every PR, piggybacking on the existing vlasov1d workflow (no CI config changes needed).

4. **units.yaml determinism**: Pint string representations are deterministic across runs/platforms.
