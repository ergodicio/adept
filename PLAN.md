# Vlasov1D Domain Object Refactoring

## Overview

We are gradually refactoring the `_vlasov1d` module to replace "cfg"-based data threading with actual domain objects.

### Current Architecture

The `ergoExo` object calls a series of "lifecycle" methods on the `ADEPTModule` (in this case, `BaseVlasov1D`). These modify the `self.cfg` field iteratively, adding normalization constants, save/diagnostic callback functions, numpy arrays for the grid, etc., all onto that one dict.

### Target Architecture

We want to move all simulation setup into the constructor of a new `Vlasov1DSimulation` class:

```python
class Vlasov1DSimulation():
    def __init__(self, plasma_norm: PlasmaNormalization, grid: Grid, species: dict[str, Species], v_grids: dict[str, VGrid]):
        ...
```

### Compatibility Constraint

We can't simply replace `BaseVlasov1D` because `ergoExo` expects to log the config dict after each lifecycle method. The solution is to construct all domain objects upfront in `__init__`, then "pretend" to construct them bit-by-bit in the lifecycle methods by extracting data back into the cfg dict:

```python
def sim_from_cfg(cfg: dict) -> Vlasov1DSimulation:
    ...

class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        self.simulation = sim_from_cfg(cfg)
        self.cfg = cfg

    def write_units(self) -> dict:
        self.cfg["units"]["derived"] = BaseVlasov1D.derived_units_dict(self.simulation.plasma_norm)
        self.cfg["grid"]["beta"] = self.simulation.plasma_norm.beta
        return self.cfg["units"]["derived"]

    def get_derived_quantities(self):
        self.cfg["terms"]["species"] = BaseVlasov1D.species_config_dict(self.simulation.species)
        self.cfg["grid"] = BaseVlasov1D.grid_scalars_dict(self.simulation.x_grid)
        ...
```

## Completed Steps

### Step 1: Regression tests for config logging
Added `tests/test_vlasov1d/test_config_regression.py` which verifies the exact contents of all 4 artifacts logged during `ergoExo._setup_()`:
- `config.yaml` - raw config
- `units.yaml` - derived physical units
- `derived_config.yaml` - config after `get_derived_quantities()`
- `array_config.pkl` - config after `get_solver_quantities()` (contains JAX arrays)

### Step 2: PlasmaNormalization class
Added `adept/_vlasov1d/normalization.py` with:
- `PlasmaNormalization` dataclass holding reference quantities (m0, q0, n0, T0, L0, v0, tau)
- `electron_debye_normalization()` factory function

### Step 3: Vlasov1DSimulation with PlasmaNormalization
Added `adept/_vlasov1d/simulation.py` with `Vlasov1DSimulation` class that holds `plasma_norm`.

Modified `adept/_vlasov1d/modules.py`:
- Added `sim_from_cfg()` factory that creates `Vlasov1DSimulation` from config
- `BaseVlasov1D.__init__` now creates `self.simulation = sim_from_cfg(cfg)`
- `write_units()` now uses `self.simulation.plasma_norm` instead of recomputing normalization

Current state after Step 3:
```python
# simulation.py
class Vlasov1DSimulation:
    def __init__(self, plasma_norm: PlasmaNormalization):
        self.plasma_norm = plasma_norm

# modules.py
def sim_from_cfg(cfg: dict) -> Vlasov1DSimulation:
    plasma_norm = electron_debye_normalization(
        cfg["units"]["normalizing_density"],
        cfg["units"]["normalizing_temperature"],
    )
    return Vlasov1DSimulation(plasma_norm)

class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.simulation = sim_from_cfg(cfg)

    def write_units(self) -> dict:
        norm = self.simulation.plasma_norm
        # ... rest unchanged, uses norm.L0, norm.tau, etc.
```

## Next Steps

The next step is to define the grid classes. There is a "configuration space" grid which defines
the grids in x and t, and then each species has its own velocity space grid due to different resolution required for ions and electrons.
For now we'll just define the Grid dataclass. It should be constructed _completely_ in sim_from_cfg, probably by
calling into judiciously defined helper functions. Anywhere that it's used further down the call tree, we should pass it
as a dependency alongside `cfg`.

For the logging purposes, in get_derived_quantites, we should define a helper to convert Grid into a dict,
and then filter out the values to just the scalars. This can be merged with the cfg_grid dict inside get_derived_quantites,
and the same without filtering out arrays inside get_solver_quantities.

### Step 4: Grid Class (Configuration Space Grid)

#### 4.1 Grid Dataclass Design

The `Grid` class is an `eqx.Module` representing the spatial-temporal grid for configuration space. It computes all derived quantities in a custom `__init__`:

```python
# adept/_vlasov1d/grid.py
import equinox as eqx
import numpy as np
import jax.numpy as jnp

class Grid(eqx.Module):
    """Configuration space grid (x, t, and their Fourier duals).

    Only the minimal set of input parameters are passed to the constructor.
    All derived quantities (dx, nt, arrays, etc.) are computed in __init__.
    """

    # Stored fields (all final values, no "requested" intermediates)
    xmin: float
    xmax: float
    nx: int
    tmin: float
    tmax: float       # Actual tmax (aligned to dt)
    dt: float         # Actual dt (possibly overridden for EM stability)
    dx: float
    nt: int
    max_steps: int

    x: jnp.ndarray
    t: jnp.ndarray
    kx: jnp.ndarray
    kxr: jnp.ndarray
    one_over_kx: jnp.ndarray
    one_over_kxr: jnp.ndarray
    x_a: jnp.ndarray

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        tmin: float,
        tmax_requested: float,
        dt_requested: float,
        has_ey_driver: bool,
        beta: float,
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.tmin = tmin

        # Compute dx
        self.dx = xmax / nx

        # Override dt for EM wave stability if needed
        if has_ey_driver:
            c_light = 1.0 / beta
            self.dt = float(0.95 * self.dx / c_light)
        else:
            self.dt = dt_requested

        # Compute nt and adjust tmax
        self.nt = int(tmax_requested / self.dt + 1)
        self.tmax = self.dt * self.nt
        self.max_steps = min(self.nt + 4, int(1e6))

        # Build arrays
        self.x = jnp.linspace(xmin + self.dx / 2, xmax - self.dx / 2, nx)
        self.t = jnp.linspace(0, self.tmax, self.nt)

        self.kx = jnp.fft.fftfreq(nx, d=self.dx) * 2.0 * np.pi
        self.kxr = jnp.fft.rfftfreq(nx, d=self.dx) * 2.0 * np.pi

        one_over_kx = np.zeros(nx)
        one_over_kx[1:] = 1.0 / np.array(self.kx)[1:]
        self.one_over_kx = jnp.array(one_over_kx)

        one_over_kxr = np.zeros(len(self.kxr))
        one_over_kxr[1:] = 1.0 / np.array(self.kxr)[1:]
        self.one_over_kxr = jnp.array(one_over_kxr)

        self.x_a = jnp.concatenate([[self.x[0] - self.dx], self.x, [self.x[-1] + self.dx]])
```

**Factory function:**
```python
def grid_from_cfg(cfg: dict, beta: float) -> Grid:
    """Construct Grid from config dict.

    Args:
        cfg: Full config dict
        beta: Speed of light normalization (1/c_norm), needed for EM dt override
    """
    cfg_grid = cfg["grid"]
    has_ey_driver = len(cfg.get("drivers", {}).get("ey", {}).keys()) > 0

    return Grid(
        xmin=cfg_grid["xmin"],
        xmax=cfg_grid["xmax"],
        nx=cfg_grid["nx"],
        tmin=cfg_grid.get("tmin", 0.0),
        tmax_requested=cfg_grid["tmax"],
        dt_requested=cfg_grid["dt"],
        has_ey_driver=has_ey_driver,
        beta=beta,
    )
```

#### 4.2 Serialization Helpers

Generic scalar filtering utilities go in `adept/utils.py`:

```python
# adept/utils.py
import jax.numpy as jnp
import numpy as np

def is_scalar(value) -> bool:
    """Check if a value is a scalar (not an array)."""
    if isinstance(value, (int, float, bool, str, type(None))):
        return True
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return False
    # Handle numpy scalar types
    if hasattr(value, 'ndim') and value.ndim == 0:
        return True
    return False

def filter_scalars(d: dict) -> dict:
    """Filter a dict to only include scalar values (recursively).

    Useful for logging domain object contents before array quantities are needed.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            filtered = filter_scalars(v)
            if filtered:  # Only include non-empty dicts
                result[k] = filtered
        elif is_scalar(v):
            result[k] = v
    return result
```

Since `eqx.Module` is a dataclass, we can use `dataclasses.asdict()` directly:

```python
from dataclasses import asdict

# No custom grid_to_dict function needed!
```

**Usage at call sites:**
- `get_derived_quantities()`: `cfg["grid"].update(filter_scalars(asdict(self.simulation.grid)))`
- `get_solver_quantities()`: `cfg["grid"].update(asdict(self.simulation.grid))`

**Note:** The `filter_scalars()` helper is generic and can be reused for future domain object refactorings (e.g., VGrid, Species).

#### 4.3 Call Sites That Read Grid Data

The following call sites currently read `cfg["grid"]` for Grid-related data. Each needs to be updated to accept a `Grid` parameter alongside or instead of `cfg`.

**Deconstruction principle:** Pass the `grid` object whole through intermediate layers, then deconstruct it at the leaf classes where the values are actually used. This mirrors how `cfg["grid"]` is currently passed: intermediate classes receive `cfg` and pass it to children; only leaf classes extract specific values like `cfg["grid"]["dx"]`. We do NOT want to pass individual scalars like `dt, x_a, x` as a parameter tuple through multiple levels.

**Production code path (ODE solver construction):**

| File | Line(s) | Current Access | Change Required |
|------|---------|----------------|-----------------|
| `modules.py` | 265-268 | `cfg["grid"]["nx"]` | Pass `grid.nx` |
| `modules.py` | 280, 295-300 | `cfg["grid"]["dt"]`, `cfg["grid"]["tmax"]`, `cfg["grid"]["max_steps"]` | Pass `grid.dt`, etc. |
| `vector_field.py:VlasovMaxwell` | 231 | `cfg["grid"]["beta"]`, `cfg["grid"]["dx"]`, `cfg["grid"]["dt"]` | Pass `grid` (deconstruct here) |
| `vector_field.py:VlasovMaxwell` | 233-235 | `cfg["grid"]["dt"]`, `cfg["grid"]["x_a"]`, `cfg["grid"]["x"]` | Pass `grid` (deconstruct here) |
| `vector_field.py:VlasovMaxwell` | 242, 265 | `cfg["grid"]["x"]` | Pass `grid` (deconstruct here) |
| `vector_field.py:TimeIntegrator` | 25 | `cfg["grid"]["x"]` via `SpaceExponential` | Pass `grid` (deconstruct here) |
| `vector_field.py:LeapfrogIntegrator` | 51 | `cfg["grid"]["dt"]` | Pass `grid` (deconstruct here) |
| `vector_field.py:SixthOrderHamIntegrator` | 93 | `cfg["grid"]["dt"]` | Pass `grid` (deconstruct here) |
| `vector_field.py:VlasovPoissonFokkerPlanck` | 188 | `cfg["grid"]["dt"]` | Pass `grid` (deconstruct here) |
| `pushers/field.py:WaveSolver` | 43 | `dx`, `dt` (already passed as args) | No change needed |
| `pushers/field.py:ElectricFieldSolver` | 300, 317, 318, 327 | `cfg["grid"]["one_over_kx"]`, `cfg["grid"]["kx"]`, `cfg["grid"]["dx"]` | Pass `grid` (deconstruct here) |
| `pushers/field.py:Driver` | 10-11 | `xax` (already passed as arg) | No change needed |
| `pushers/vlasov.py:VlasovExternalE` | 20-23 | `cfg["grid"]["x"]`, `cfg["grid"]["xmax"]`, `cfg["grid"]["dt"]` | Pass `grid` (deconstruct here) |
| `pushers/vlasov.py:SpaceExponential` | 86-87 | `x` (already passed as arg) | No change needed |
| `pushers/fokker_planck.py` | 128, 196, 264 | `cfg["grid"]["nx"]` | Pass `grid` (deconstruct here) |

**Storage/I/O path:**

| File | Line(s) | Current Access | Change Required |
|------|---------|----------------|-----------------|
| `storage.py:get_save_quantities` | various | `cfg["grid"]["x"]`, `cfg["grid"]["kx"]`, etc. | Pass `grid` (deconstruct here) |
| `storage.py:store_fields` | 31, 43 | `cfg["grid"]["x"]` | Pass `grid` (deconstruct here) |
| `storage.py:store_fields` | 47, 48 | `cfg["grid"]["dt"]`, `cfg["grid"]["dx"]` | Pass `grid` (deconstruct here) |
| `storage.py:store_f` | 91 | `cfg["grid"]["x"]` | Pass `grid` (deconstruct here) |
| `storage.py:store_diags` | 113-121 | `cfg["grid"]["x"]`, `cfg["grid"]["kx"]` | Pass `grid` (deconstruct here) |
| `storage.py:get_field_save_func` | 133+ | Accesses `cfg["grid"]["species_grids"]` | Separate concern (VGrid) |

**Note:** Storage functions are called during `post_process()` and save callbacks. To eliminate ordering dependencies on cfg population, all storage functions receive `grid` directly.

#### 4.4 Recommended Implementation Order

**Phase A: Create Grid without changing consumers**

1. Create `adept/utils.py` with `is_scalar()` and `filter_scalars()` helpers
2. Create `adept/_vlasov1d/grid.py` with `Grid` class (eqx.Module) and `grid_from_cfg` factory
3. Update `sim_from_cfg()` to construct `Grid` and store in `Vlasov1DSimulation`
4. Update `Vlasov1DSimulation` to hold `grid: Grid`
5. In `get_derived_quantities()`, use `filter_scalars(asdict(...))` to populate `cfg["grid"]`
6. In `get_solver_quantities()`, use `asdict()` to populate `cfg["grid"]`
7. Run regression tests to verify identical logged configs

**Phase B: Thread Grid to vector field (largest refactor)**

The `VlasovMaxwell` class is instantiated once in `init_diffeqsolve()` at line 282:
```python
# modules.py:282
self.diffeqsolve_quants = dict(
    terms=ODETerm(VlasovMaxwell(self.cfg)),
    ...
)
```

Change to:
```python
self.diffeqsolve_quants = dict(
    terms=ODETerm(VlasovMaxwell(self.cfg, grid=self.simulation.grid)),
    ...
)
```

Then update `VlasovMaxwell.__init__` and its nested classes. Each class receives `grid` and deconstructs it locally:

1. `VlasovMaxwell.__init__(cfg, grid)` → stores what it needs from `grid`, passes `grid` to children
2. `VlasovPoissonFokkerPlanck.__init__(cfg, grid)` → stores `self.dt = grid.dt`
3. `TimeIntegrator.__init__(cfg, grid)` → passes `grid.x` to `SpaceExponential`
4. `LeapfrogIntegrator.__init__(cfg, grid)` → stores `self.dt = grid.dt`
5. `SixthOrderHamIntegrator.__init__(cfg, grid)` → stores `self.dt = grid.dt`
6. `ElectricFieldSolver.__init__(cfg, grid)` → stores `self.one_over_kx = grid.one_over_kx`, etc.
7. `Collisions.__init__(cfg, grid)` → stores `grid.nx` for FP operators

**Phase C: Update init_state_and_args**

Update `init_state_and_args()` to use `self.simulation.grid.nx` instead of `cfg["grid"]["nx"]`.

**Phase D: Update __call__ and init_diffeqsolve**

Use `grid.dt`, `grid.tmax`, `grid.max_steps` directly from `self.simulation.grid`.

#### 4.5 Call Stack Diagram

```
BaseVlasov1D.__init__(cfg)
└── sim_from_cfg(cfg)
    ├── electron_debye_normalization(...)
    │   └── PlasmaNormalization(...)
    └── grid_from_cfg(cfg, beta)           # NEW
        └── Grid(...)                      # computes arrays in __post_init__

BaseVlasov1D.write_units()
└── uses self.simulation.plasma_norm

BaseVlasov1D.get_derived_quantities()
└── cfg["grid"].update(filter_scalars(asdict(self.simulation.grid)))        # NEW

BaseVlasov1D.get_solver_quantities()
└── cfg["grid"].update(asdict(self.simulation.grid))                        # NEW

BaseVlasov1D.init_state_and_args()
└── uses cfg["grid"]["nx"]  →  change to self.simulation.grid.nx       # CHANGE

BaseVlasov1D.init_diffeqsolve()
├── get_save_quantities(cfg, grid)                                     # CHANGE
│   └── Reads grid.x, grid.kx, etc. directly from grid object
│       (removes ordering dependency on get_solver_quantities populating cfg first)
└── VlasovMaxwell(cfg, grid)                                           # CHANGE
    ├── VlasovPoissonFokkerPlanck(cfg, grid)
    │   ├── LeapfrogIntegrator(cfg, grid)  OR  SixthOrderHamIntegrator(cfg, grid)
    │   │   ├── self.dt = grid.dt          # deconstruct here
    │   │   ├── ElectricFieldSolver(cfg, grid)
    │   │   │   └── self.one_over_kx = grid.one_over_kx, etc.  # deconstruct here
    │   │   ├── SpaceExponential(grid.x, species_grids)        # deconstruct here
    │   │   └── VelocityExponential(species_grids, species_params)  # no grid needed
    │   └── Collisions(cfg, grid)
    │       └── FokkerPlanck operators: self.nx = grid.nx      # deconstruct here
    └── WaveSolver(c=1/beta, dx=grid.dx, dt=grid.dt)           # deconstruct here

BaseVlasov1D.__call__(...)
├── uses cfg["grid"]["max_steps"]  →  self.simulation.grid.max_steps   # CHANGE
└── uses cfg["grid"]["dt"]         →  self.simulation.grid.dt          # CHANGE
```

#### 4.6 Dependency: beta from PlasmaNormalization

The `Grid` constructor needs `beta` (speed of light normalization) to determine whether to override `dt` for EM wave stability. This creates a dependency:

```python
# In sim_from_cfg:
plasma_norm = electron_debye_normalization(...)
beta = 1.0 / plasma_norm.speed_of_light_norm()
grid = grid_from_cfg(cfg, beta)
```

The `beta` value is also stored in `cfg["grid"]["beta"]` during `write_units()`. We keep that for backward compatibility, but `Grid` construction receives `beta` directly rather than reading from cfg.

#### 4.7 Files to Modify

1. **New file:** `adept/utils.py` - `is_scalar()` and `filter_scalars()` helpers
2. **New file:** `adept/_vlasov1d/grid.py` - `Grid` class (eqx.Module), `grid_from_cfg` factory
3. **Modify:** `adept/_vlasov1d/simulation.py` - Add `grid: Grid` field
4. **Modify:** `adept/_vlasov1d/modules.py`:
   - `sim_from_cfg()`: construct `Grid`
   - `get_derived_quantities()`: use `filter_scalars(asdict(...))`
   - `get_solver_quantities()`: use `asdict()`
   - `init_state_and_args()`: use `grid.nx`
   - `init_diffeqsolve()`: pass `grid` to `get_save_quantities` and `VlasovMaxwell`
   - `__call__()`: use `grid.dt`, `grid.max_steps`
5. **Modify:** `adept/_vlasov1d/storage.py`:
   - `get_save_quantities(cfg, grid)`: receive grid, deconstruct locally
   - All store_* functions: add `grid` parameter, deconstruct locally
6. **Modify:** `adept/_vlasov1d/solvers/vector_field.py`:
   - All integrator classes: add `grid` parameter, deconstruct locally
   - `VlasovMaxwell`: add `grid` parameter
7. **Modify:** `adept/_vlasov1d/solvers/pushers/field.py`:
   - `ElectricFieldSolver`: add `grid` parameter, deconstruct locally
8. **Modify:** `adept/_vlasov1d/solvers/pushers/fokker_planck.py`:
   - All FP operator classes: add `grid` parameter, deconstruct `grid.nx` locally
