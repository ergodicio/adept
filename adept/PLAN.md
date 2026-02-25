# Summary

We want to refactor the collision frequency (nu) profile handling in the _vlasov1d module.
Currently, the nu profile is calculated in a function nu_prof which directly interprets
a config dictionary (from yaml) and constructs a prescribed envelope. 

We should introduce an EnvelopeFunction `eqx.Module` which has a center, width, and rise parameter, and a bump/trough boolean.
It has a __call__ method that accepts a single argument and performs the tanh transformation.

Additionally, introduce a SpaceTimeEnvelopeFunction that wraps two EnvelopeFunctions and has a __call__
that accepts an `x, t` pair.

In the entrypoint code, the Vlasov1DSimulation class should gain `nu_fp_prof` and `nu_K_prof`
fields, each of which are an Optional[SpaceTimeEnvelopeFunction]. We will then pass these as parameters
into the VlasovMaxwell constructor.

# Implementation plan

## Overview

The refactor introduces two new `eqx.Module` classes (`EnvelopeFunction` and `SpaceTimeEnvelopeFunction`) to encapsulate the tanh-based envelope logic currently embedded in `VlasovMaxwell.nu_prof()`. These modules become first-class objects that can be constructed from config, stored on the `Vlasov1DSimulation` domain object, and passed into the solver.

## Step 1: Create `EnvelopeFunction` module

**File**: `adept/utils.py` (add to existing file)

Create an `eqx.Module` that encapsulates a 1D envelope function. The tanh envelope logic is inlined directly (no dependency on `get_envelope`):

```python
import equinox as eqx
from jax import Array
from jax import numpy as jnp


class EnvelopeFunction(eqx.Module):
    """A 1D tanh-based envelope function.

    Evaluates: baseline + bump_height * envelope(x)
    where envelope is a smooth tanh step from 0 to 1 (or 1 to 0 for trough).

    Parameters:
        center: The midpoint of the envelope region.
        width: The full width of the "on" region. The envelope transitions from
               0 to 1 at (center - width/2) and back to 0 at (center + width/2).
        rise: Controls the smoothness of the tanh transitions. Smaller values
              give sharper edges; larger values give more gradual transitions.
        baseline: The minimum value of the envelope (when envelope=0).
        bump_height: The amplitude added to baseline when the envelope is active.
                     Final value ranges from baseline to (baseline + bump_height).
        is_trough: If True, inverts the envelope (1 - envelope), creating a
                   trough (dip) instead of a bump (peak).
    """
    center: float
    width: float
    rise: float
    baseline: float
    bump_height: float
    is_trough: bool  # True if bump_or_trough == "trough"

    def __call__(self, x: Array) -> Array:
        """Evaluate the envelope at position(s) x."""
        left = self.center - self.width * 0.5
        right = self.center + self.width * 0.5
        # Inline tanh envelope: 0.5 * (tanh((x - left) / rise) - tanh((x - right) / rise))
        env = 0.5 * (jnp.tanh((x - left) / self.rise) - jnp.tanh((x - right) / self.rise))
        if self.is_trough:
            env = 1 - env
        return self.baseline + self.bump_height * env

    @staticmethod
    def from_config(cfg: dict) -> "EnvelopeFunction":
        """Construct an EnvelopeFunction from a config dict.

        Args:
            cfg: Dict containing center, width, rise, baseline, bump_height, bump_or_trough

        Returns:
            EnvelopeFunction instance
        """
        return EnvelopeFunction(
            center=cfg["center"],
            width=cfg["width"],
            rise=cfg["rise"],
            baseline=cfg["baseline"],
            bump_height=cfg["bump_height"],
            is_trough=(cfg["bump_or_trough"] == "trough"),
        )
```

**Notes**:
- Parameters directly mirror the config structure (`center`, `width`, `rise`, `baseline`, `bump_height`)
- `is_trough` is a bool derived from `bump_or_trough == "trough"`
- The unused `slope` parameter from the data model is intentionally omitted (it was never used)
- The tanh logic is inlined rather than calling `get_envelope`

## Step 1b: Deprecate `get_envelope` in `adept/_base_.py`

**File**: `adept/_base_.py`

Add a deprecation warning to `get_envelope`:

```python
import warnings

def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    warnings.warn(
        "get_envelope is deprecated. Use adept.utils.EnvelopeFunction instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))
```

## Step 2: Create `SpaceTimeEnvelopeFunction` module

**File**: `adept/utils.py` (same file as Step 1)

```python
class SpaceTimeEnvelopeFunction(eqx.Module):
    """A space-time envelope composed of separate time and space envelopes.

    Evaluates: time_envelope(t) * space_envelope(x)
    """
    time_envelope: EnvelopeFunction
    space_envelope: EnvelopeFunction

    def __call__(self, x: Array, t: float) -> Array:
        """Evaluate the envelope at positions x and time t.

        Returns an array of shape (nx,) representing the spatial profile
        at the given time.
        """
        return self.time_envelope(t) * self.space_envelope(x)

    @staticmethod
    # FIXME: why is the return type quoted? The EnvelopeFunction one is too.
    def from_config(term_config: dict) -> "SpaceTimeEnvelopeFunction":
        """Construct a SpaceTimeEnvelopeFunction from a fokker_planck or krook config dict.

        Args:
            term_config: Dict with 'time' and 'space' sub-dicts, each containing
                         center, width, rise, baseline, bump_height, bump_or_trough

        Returns:
            SpaceTimeEnvelopeFunction ready to be called with (x, t)
        """
        time_env = EnvelopeFunction.from_config(term_config["time"])
        space_env = EnvelopeFunction.from_config(term_config["space"])
        return SpaceTimeEnvelopeFunction(time_envelope=time_env, space_envelope=space_env)
```

## Step 3: Extend `Vlasov1DSimulation` with collision profile fields

**File**: `adept/_vlasov1d/simulation.py`

Add optional collision profile fields:

```python
from typing import Optional
from adept._vlasov1d.grid import Grid
from adept._vlasov1d.normalization import PlasmaNormalization
from adept.utils import SpaceTimeEnvelopeFunction


class Vlasov1DSimulation:
    """
    Domain object representing a Vlasov-1D simulation setup.
    Holds the physical parameters computed from config.
    """

    def __init__(
        self,
        plasma_norm: PlasmaNormalization,
        grid: Grid,
        nu_fp_prof: Optional[SpaceTimeEnvelopeFunction] = None,
        nu_K_prof: Optional[SpaceTimeEnvelopeFunction] = None,
    ):
        self.plasma_norm = plasma_norm
        self.grid = grid
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof
```

## Step 4: Construct envelope functions in `sim_from_cfg`

**File**: `adept/_vlasov1d/modules.py`

Update `sim_from_cfg` to construct the collision profiles:

```python
from adept.utils import SpaceTimeEnvelopeFunction

def sim_from_cfg(cfg: dict) -> Vlasov1DSimulation:
    """Construct a Vlasov1DSimulation from a config dict."""
    plasma_norm = electron_debye_normalization(
        cfg["units"]["normalizing_density"],
        cfg["units"]["normalizing_temperature"],
    )
    beta = 1.0 / plasma_norm.speed_of_light_norm()
    grid = grid_from_cfg(cfg, beta)

    # Construct collision frequency profiles if enabled
    nu_fp_prof = None
    if cfg["terms"]["fokker_planck"]["is_on"]:
        nu_fp_prof = SpaceTimeEnvelopeFunction.from_config(cfg["terms"]["fokker_planck"])

    nu_K_prof = None
    if cfg["terms"]["krook"]["is_on"]:
        nu_K_prof = SpaceTimeEnvelopeFunction.from_config(cfg["terms"]["krook"])

    return Vlasov1DSimulation(plasma_norm, grid, nu_fp_prof, nu_K_prof)
```

## Step 5: Update `VlasovMaxwell` to accept profile modules

**File**: `adept/_vlasov1d/solvers/vector_field.py`

Modify the constructor to accept the envelope function modules:

```python
from typing import Optional
from adept.utils import SpaceTimeEnvelopeFunction

class VlasovMaxwell:
    def __init__(
        self,
        cfg: dict,
        grid: Grid,
        nu_fp_prof: Optional[SpaceTimeEnvelopeFunction] = None,
        nu_K_prof: Optional[SpaceTimeEnvelopeFunction] = None,
    ):
        self.cfg = cfg
        self.grid = grid
        self.nu_fp_prof = nu_fp_prof
        self.nu_K_prof = nu_K_prof
        self.vpfp = VlasovPoissonFokkerPlanck(cfg, grid)
        beta = cfg["grid"]["beta"]
        self.wave_solver = field.WaveSolver(c=1.0 / beta, dx=grid.dx, dt=grid.dt)
        self.dt = grid.dt
        self.ey_driver = field.Driver(grid.x_a, driver_key="ey")
        self.ex_driver = field.Driver(grid.x, driver_key="ex")
```

## Step 6: Update `VlasovMaxwell.__call__` to use stored profiles

**File**: `adept/_vlasov1d/solvers/vector_field.py`

Replace the dynamic `nu_prof` computation with calls to stored modules. The if-statement conditions remain based on `cfg["terms"]` to preserve existing behavior:

```python
def __call__(self, t, y, args):
    # ... driver computation unchanged ...

    # Evaluate collision frequency profiles at current time
    if self.cfg["terms"]["fokker_planck"]["is_on"]:
        nu_fp_val = self.nu_fp_prof(self.grid.x, t)
    else:
        nu_fp_val = None

    if self.cfg["terms"]["krook"]["is_on"]:
        nu_K_val = self.nu_K_prof(self.grid.x, t)
    else:
        nu_K_val = None

    # ... rest unchanged, using nu_fp_val and nu_K_val ...
```

**Also**: Remove the `nu_prof` method from `VlasovMaxwell` since it's no longer needed.

## Step 7: Update `init_diffeqsolve` to pass profiles to VlasovMaxwell

**File**: `adept/_vlasov1d/modules.py`

```python
def init_diffeqsolve(self):
    self.cfg = get_save_quantities(self.cfg)
    grid = self.simulation.grid
    self.time_quantities = {"t0": 0.0, "t1": grid.tmax, "max_steps": grid.max_steps}
    self.diffeqsolve_quants = dict(
        terms=ODETerm(VlasovMaxwell(
            self.cfg,
            grid,
            nu_fp_prof=self.simulation.nu_fp_prof,
            nu_K_prof=self.simulation.nu_K_prof,
        )),
        solver=Stepper(),
        saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in self.cfg["save"].items()}),
    )
```

**Note**: The `args` dict in `init_state_and_args` remains unchanged—it still includes both `"drivers"` and `"terms"` since `args["terms"]` is used elsewhere (e.g., for field solvers and pushers). Only the collision profile computation moves out of args into the stored modules.

## Files Changed Summary

| File | Change |
|------|--------|
| `adept/utils.py` | Add `EnvelopeFunction`, `SpaceTimeEnvelopeFunction` modules |
| `adept/_base_.py` | Add deprecation warning to `get_envelope` |
| `adept/_vlasov1d/simulation.py` | Add `nu_fp_prof`, `nu_K_prof` fields |
| `adept/_vlasov1d/modules.py` | Update `sim_from_cfg` and `init_diffeqsolve` to construct/pass profiles |
| `adept/_vlasov1d/solvers/vector_field.py` | Update `VlasovMaxwell.__init__` and `__call__`, remove `nu_prof` method |

## Testing Considerations

1. Run existing Vlasov1D tests to verify behavior is unchanged
2. Add unit tests for `EnvelopeFunction` and `SpaceTimeEnvelopeFunction`:
   - Test bump vs trough behavior
   - Test factorization: `envelope(x, t) == time(t) * space(x)`
3. Verify that disabling fokker_planck/krook (`is_on: False`) results in `None` profiles

## Future Extensions (Out of Scope)

- The `slope` parameter in config data models is unused; could add linear ramp support to `EnvelopeFunction`
- Species-dependent collision profiles (currently only electrons use FP collisions)
- 2D/3D spatial envelopes (similar pattern to `_lpse2d`)
