# Research: Introducing Vlasov1DSimulation with PlasmaNormalization

## Task Summary

Introduce a new `Vlasov1DSimulation` class that will gradually "eat up" the initialization logic from `BaseVlasov1D`. In this first step, the simulation holds only a `PlasmaNormalization` object.

---

## Current Architecture

### Lifecycle Flow

The `ergoExo` class manages simulation lifecycle by calling methods on `ADEPTModule` subclasses:

```
ergoExo._setup_()
  ├── ADEPTModule.__init__(cfg)     # Store cfg
  ├── write_units()                  # Compute derived units, log to units.yaml
  ├── get_derived_quantities()       # Compute grid scalars, log to derived_config.yaml
  ├── get_solver_quantities()        # Compute arrays, log to array_config.pkl
  ├── init_state_and_args()          # Initialize simulation state
  └── init_diffeqsolve()             # Setup ODE solver
```

Each lifecycle method modifies `self.cfg` and the `ergoExo` logs the config state after certain methods.

### Current BaseVlasov1D Structure (`adept/_vlasov1d/modules.py`)

```python
class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.ureg = pint.UnitRegistry()  # Note: unused after refactor

    def write_units(self) -> dict:
        norm = electron_debye_normalization(
            self.cfg["units"]["normalizing_density"],
            self.cfg["units"]["normalizing_temperature"]
        )
        # Uses norm.L0, norm.tau, norm.n0, norm.v0, norm.T0, etc.
        # Sets self.cfg["units"]["derived"] and self.cfg["grid"]["beta"]
        ...
```

Key observation: `write_units()` already creates a `PlasmaNormalization` via `electron_debye_normalization()`, uses it, but doesn't store it. The refactoring will store this object in `self.simulation.plasma_norm`.

---

## PlasmaNormalization Class (`adept/_vlasov1d/normalization.py`)

Already implemented:

```python
@dataclass
class PlasmaNormalization:
    m0: UREG.Quantity      # Unit mass
    q0: UREG.Quantity      # Unit charge
    n0: UREG.Quantity      # Reference density [particles/m^3]
    T0: UREG.Quantity      # Reference temperature [eV]
    L0: UREG.Quantity      # Reference length [m]
    v0: UREG.Quantity      # Reference velocity [m/s]
    tau: UREG.Quantity     # Reference time [s]

    def logLambda_ee(self) -> float: ...
    def approximate_ee_collision_frequency(self) -> UREG.Quantity: ...
    def speed_of_light_norm(self) -> float: ...
```

Constructor function:

```python
def electron_debye_normalization(n0_str, T0_str) -> PlasmaNormalization:
    """
    Creates electron thermal normalization:
    - L0 = Debye length
    - v0 = electron thermal velocity
    - tau = 1/wp0 (Langmuir oscillation period)
    """
```

This uses `jpu.UnitRegistry()` (JAX-compatible Pint) instead of vanilla `pint.UnitRegistry()`.

---

## Existing Regression Tests

Located at `tests/test_vlasov1d/test_config_regression.py`.

Tests all 4 logged artifacts against baseline files:
- `config.yaml` - raw config
- `units.yaml` - derived units (Pint quantities as strings)
- `derived_config.yaml` - after `get_derived_quantities()`
- `array_config.pkl` - after `get_solver_quantities()` (contains JAX arrays)

Test configs:
- `resonance.yaml` - single species electron
- `fokker_planck_conservation.yaml` - single species with FP collisions
- `multispecies_ion_acoustic.yaml` - electron + ion species

These tests will catch any unintended changes to the logged output.

---

## Implementation Plan

### Step 1: Create Vlasov1DSimulation Class

Create new file `adept/_vlasov1d/simulation.py`:

```python
class Vlasov1DSimulation:
    """
    Domain object representing a Vlasov-1D simulation setup.
    Holds the physical parameters computed from config.
    """
    def __init__(self, plasma_norm: PlasmaNormalization):
        self.plasma_norm = plasma_norm
```

### Step 2: Create sim_from_cfg() Factory Function

In `adept/_vlasov1d/modules.py`:

```python
def sim_from_cfg(cfg: dict) -> Vlasov1DSimulation:
    """Construct a Vlasov1DSimulation from a config dict."""
    plasma_norm = electron_debye_normalization(
        cfg["units"]["normalizing_density"],
        cfg["units"]["normalizing_temperature"]
    )
    return Vlasov1DSimulation(plasma_norm)
```

### Step 3: Modify BaseVlasov1D Constructor

In `adept/_vlasov1d/modules.py`, add import and modify constructor:

```python
from adept._vlasov1d.simulation import Vlasov1DSimulation

class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.simulation = sim_from_cfg(cfg)
        # Remove self.ureg - no longer used
```

### Step 4: Modify write_units() to Use self.simulation

In `adept/_vlasov1d/modules.py`:

```python
def write_units(self) -> dict:
    norm = self.simulation.plasma_norm  # Use stored object instead of creating new

    box_length = ((self.cfg["grid"]["xmax"] - self.cfg["grid"]["xmin"]) * norm.L0).to("microns")
    # ... rest unchanged
```

---

## What Changes

| Component | Current | After Refactor |
|-----------|---------|----------------|
| `simulation.py` | Does not exist | New file with `Vlasov1DSimulation` class |
| `modules.py` | Has `BaseVlasov1D` | Adds `sim_from_cfg()` factory function |
| `BaseVlasov1D.__init__` | Creates `self.ureg` (unused) | Creates `self.simulation` via `sim_from_cfg()` |
| `write_units()` | Calls `electron_debye_normalization()` | Uses `self.simulation.plasma_norm` |
| `self.ureg` | Exists but unused | Removed |
| Config logging | No change | No change (same data flows) |

---

## Backwards Compatibility

The refactoring maintains backwards compatibility by:

1. **Same config structure**: `sim_from_cfg()` reads from existing config keys (`units.normalizing_density`, `units.normalizing_temperature`)

2. **Same logged output**: `write_units()` writes the same values to `cfg["units"]["derived"]` and `cfg["grid"]["beta"]`

3. **Same lifecycle**: `ergoExo` calls methods in the same order; we're just moving *when* normalization is computed (init vs write_units)

---

## Files to Modify

| File | Changes |
|------|---------|
| `adept/_vlasov1d/simulation.py` | **New file**: `Vlasov1DSimulation` class |
| `adept/_vlasov1d/modules.py` | Add `sim_from_cfg()`, import `Vlasov1DSimulation`, modify `BaseVlasov1D.__init__`, modify `write_units()`, remove unused `self.ureg` |

---

## Testing Strategy

1. **Run existing regression tests**: These verify logged config doesn't change
   ```bash
   uv run pytest tests/test_vlasov1d/test_config_regression.py -v
   ```

2. **Run full vlasov1d test suite**: Verify simulation behavior unchanged
   ```bash
   uv run pytest tests/test_vlasov1d/ -v
   ```

---

## Future Steps (Not This PR)

The `Vlasov1DSimulation` class will eventually hold:
- `x_grid: XGrid` - spatial grid parameters
- `species: dict[str, Species]` - species definitions
- `v_grids: dict[str, VGrid]` - velocity grids per species

Each future PR will:
1. Add a new domain object to `Vlasov1DSimulation`
2. Construct it in `sim_from_cfg()`
3. Extract static methods on `BaseVlasov1D` to convert domain objects back to config dicts for logging

---

## Notes

- The `jpu` package provides a JAX-compatible Pint unit registry. The normalization code already uses `jpu.UnitRegistry()` instead of vanilla `pint`.

- The `self.ureg = pint.UnitRegistry()` line in the current constructor appears to be dead code (leftover from before the normalization refactor). It can be safely removed.

- The `beta` value in `cfg["grid"]["beta"]` is computed as `1.0 / norm.speed_of_light_norm()` which equals `v0 / c`. This is the relativistic beta factor.
