# Vlasov-1D Configuration Reference

This document describes how to construct a configuration file for the `vlasov-1d` solver.

## Top-Level Structure

```yaml
solver: vlasov-1d

units:
  # Physical unit normalizations

density:
  # Species definitions

grid:
  # Simulation grid parameters

save:
  # Output configuration

mlflow:
  # Experiment tracking

drivers:
  # External drivers

diagnostics:
  # Diagnostic outputs

terms:
  # Solver configuration
```

## units

Physical unit normalizations for the simulation.

| Field | Type | Description |
|-------|------|-------------|
| `laser_wavelength` | string | Laser wavelength with unit, e.g., `"351nm"` |
| `normalizing_temperature` | string | Reference temperature with unit, e.g., `"2000eV"` |
| `normalizing_density` | string | Reference density with unit, e.g., `"1.5e21/cc"` |
| `Z` | int | Ionization state |
| `Zp` | int | Plasma Z |

Example:
```yaml
units:
  laser_wavelength: 351nm
  normalizing_temperature: 2000eV
  normalizing_density: 1.5e21/cc
  Z: 10
  Zp: 10
```

## density

Species and density configuration. You can define multiple "species" by using keys prefixed with `species-`.

| Field | Type | Description |
|-------|------|-------------|
| `quasineutrality` | bool | Whether to enforce quasineutrality |

### Species Definition

Each species is defined with a key starting with `species-` (e.g., `species-background`, `species-beam`, `species-electron1`).

| Field | Type | Description |
|-------|------|-------------|
| `noise_seed` | int | Random seed for noise initialization |
| `noise_type` | string | `"gaussian"` or `"uniform"` |
| `noise_val` | float | Amplitude of noise |
| `v0` | float | Drift velocity (normalized) |
| `T0` | float | Temperature (normalized to `normalizing_temperature`) |
| `m` | float | Exponent for super-Gaussian distribution. `2.0` is Maxwellian |
| `basis` | string | Spatial profile type (see below) |

#### Basis Types

**`uniform`**: Constant density profile
```yaml
species-background:
  basis: uniform
  # No additional parameters required
```

**`sine`**: Sinusoidal density perturbation
```yaml
species-background:
  basis: sine
  baseline: 1.0        # Base density
  amplitude: 1.0e-4    # Perturbation amplitude
  wavenumber: 0.3      # Wavenumber of perturbation
```

**`tanh`**: Hyperbolic tangent profile (for density gradients)
```yaml
species-background:
  basis: tanh
  baseline: 0.001      # Minimum density
  bump_or_trough: bump # "bump" or "trough"
  center: 2000.0       # Profile center location
  rise: 25.0           # Steepness of transition
  bump_height: 0.999   # Height of bump/trough
  width: 3900.0        # Width of profile
```

**`linear`**: Linear density gradient
```yaml
species-background:
  basis: linear
  center: 1000.0
  width: 500.0
  rise: 10.0
  gradient scale length: "100um"  # Gradient scale length with units
  val at center: 1.0              # Density value at center
```

**`exponential`**: Exponential density gradient
```yaml
species-background:
  basis: exponential
  center: 1000.0
  width: 500.0
  rise: 10.0
  gradient scale length: "100um"
  val at center: 1.0
```

### Multi-Species Example

```yaml
density:
  quasineutrality: true
  species-electron1:
    noise_seed: 420
    noise_type: gaussian
    noise_val: 0.0
    v0: -1.5
    T0: 0.2
    m: 2.0
    basis: sine
    baseline: 0.5
    amplitude: 1.0e-4
    wavenumber: 0.3
  species-electron2:
    noise_seed: 420
    noise_type: gaussian
    noise_val: 0.0
    v0: 1.5
    T0: 0.2
    m: 2.0
    basis: sine
    baseline: 0.5
    amplitude: -1.0e-4
    wavenumber: 0.3
```

## grid

Simulation grid parameters.

| Field | Type | Description |
|-------|------|-------------|
| `dt` | float | Timestep (normalized) |
| `nv` | int | Number of velocity grid points |
| `nx` | int | Number of spatial grid points |
| `tmin` | float | Start time |
| `tmax` | float | End time |
| `vmax` | float | Maximum velocity extent |
| `xmin` | float | Domain minimum x |
| `xmax` | float | Domain maximum x |

Example:
```yaml
grid:
  dt: 0.1
  nv: 256
  nx: 32
  tmin: 0.0
  tmax: 100.0
  vmax: 6.4
  xmin: 0.0
  xmax: 20.94
```

## save

Configures what data to save and at what times.

### Structure

```yaml
save:
  fields:
    t:
      tmin: 0.0
      tmax: 100.0
      nt: 601
  electron:
    t:
      tmin: 0.0
      tmax: 100.0
      nt: 11
```

| Save Key | Description |
|----------|-------------|
| `fields` | Electric and magnetic field data |
| `electron` | Electron distribution function |
| `diag-vlasov-dfdt` | Time derivative from Vlasov operator (optional) |
| `diag-fp-dfdt` | Time derivative from Fokker-Planck operator (optional) |

Each save key contains a `t` sub-key with:

| Field | Type | Description |
|-------|------|-------------|
| `tmin` | float | Start time for saving |
| `tmax` | float | End time for saving |
| `nt` | int | Number of save points |

## mlflow

Experiment tracking configuration.

| Field | Type | Description |
|-------|------|-------------|
| `experiment` | string | MLflow experiment name |
| `run` | string | MLflow run name |

Example:
```yaml
mlflow:
  experiment: basic-epw-for-plots
  run: nonlinear
```

## drivers

External electromagnetic drivers. Both `ex` and `ey` are dictionaries of named drivers.

| Field | Type | Description |
|-------|------|-------------|
| `ex` | dict | Longitudinal electric field drivers |
| `ey` | dict | Transverse electric field drivers (for electromagnetic simulations) |

Each driver is identified by a string key (e.g., `"0"`, `"1"`) and has these parameters:

| Field | Type | Description |
|-------|------|-------------|
| `a0` | float | Driver amplitude |
| `k0` | float | Wavenumber |
| `w0` | float | Frequency |
| `dw0` | float | Frequency offset/chirp |
| `t_center` | float | Temporal envelope center |
| `t_rise` | float | Temporal envelope rise time |
| `t_width` | float | Temporal envelope width |
| `x_center` | float | Spatial envelope center |
| `x_rise` | float | Spatial envelope rise distance |
| `x_width` | float | Spatial envelope width |

### Example: Single ex driver

```yaml
drivers:
  ex:
    '0':
      a0: 1.e-2
      k0: 0.3
      w0: 1.1598
      dw0: 0.0
      t_center: 40.0
      t_rise: 5.0
      t_width: 30.0
      x_center: 0.0
      x_rise: 10.0
      x_width: 4000000.0
  ey: {}
```

### Example: Multiple ey drivers (SRS simulation)

```yaml
drivers:
  ex: {}
  ey:
    '0':
      a0: 1.0
      k0: 1.0
      w0: 2.79
      dw0: 0.0
      t_center: 4000.0
      t_rise: 20.0
      t_width: 7900.0
      x_center: 50.0
      x_rise: 0.1
      x_width: 1.0
    '1':
      a0: 1.e-6
      k0: 1.0
      w0: 1.63
      dw0: 0.0
      t_center: 4000.0
      t_rise: 20.0
      t_width: 7900.0
      x_center: 3950.0
      x_rise: 0.1
      x_width: 1.0
```

## diagnostics

Enable/disable diagnostic outputs.

| Field | Type | Description |
|-------|------|-------------|
| `diag-vlasov-dfdt` | bool | Save df/dt from Vlasov operator |
| `diag-fp-dfdt` | bool | Save df/dt from Fokker-Planck operator |

Example:
```yaml
diagnostics:
  diag-vlasov-dfdt: False
  diag-fp-dfdt: False
```

## terms

Solver algorithm configuration.

| Field | Type | Description |
|-------|------|-------------|
| `field` | string | Electric field solver: `"poisson"`, `"ampere"`, or `"hampere"` |
| `edfdv` | string | Velocity advection scheme: `"exponential"` or `"cubic-spline"` |
| `time` | string | Time integrator: `"sixth"` (6th order Hamiltonian) or `"leapfrog"` |
| `fokker_planck` | object | Fokker-Planck collision operator configuration |
| `krook` | object | Krook collision operator configuration |

### Field Solvers

- `"poisson"`: Spectral Poisson solver (works with any time integrator)
- `"ampere"`: Ampere solver (requires `"leapfrog"` time integrator)
- `"hampere"`: Hamiltonian Ampere solver (requires `"leapfrog"` time integrator)

### fokker_planck

Fokker-Planck collision operator.

| Field | Type | Description |
|-------|------|-------------|
| `is_on` | bool | Enable/disable |
| `type` | string | Collision type: `"Dougherty"` |
| `time` | object | Temporal profile |
| `space` | object | Spatial profile |

### krook

Krook collision operator (relaxation towards Maxwellian).

| Field | Type | Description |
|-------|------|-------------|
| `is_on` | bool | Enable/disable |
| `time` | object | Temporal profile |
| `space` | object | Spatial profile |

### Profile Objects (time/space)

Both `fokker_planck` and `krook` use profile objects to define spatiotemporal variation of collision frequency:

| Field | Type | Description |
|-------|------|-------------|
| `baseline` | float | Base collision frequency |
| `bump_or_trough` | string | `"bump"` or `"trough"` |
| `center` | float | Profile center |
| `rise` | float | Transition steepness |
| `slope` | float | Linear slope (typically 0.0) |
| `bump_height` | float | Height of bump/depth of trough |
| `width` | float | Profile width |

### Example: With Fokker-Planck collisions

```yaml
terms:
  field: poisson
  edfdv: cubic-spline
  time: sixth
  fokker_planck:
    is_on: True
    type: Dougherty
    time:
      baseline: 1.0e-5
      bump_or_trough: bump
      center: 0.0
      rise: 25.0
      slope: 0.0
      bump_height: 0.0
      width: 100000.0
    space:
      baseline: 1.0
      bump_or_trough: bump
      center: 0.0
      rise: 25.0
      slope: 0.0
      bump_height: 0.0
      width: 100000.0
  krook:
    is_on: False
    time:
      baseline: 1.0
      bump_or_trough: bump
      center: 0.0
      rise: 25.0
      slope: 0.0
      bump_height: 0.0
      width: 100000.0
    space:
      baseline: 1.0
      bump_or_trough: bump
      center: 0.0
      rise: 25.0
      slope: 0.0
      bump_height: 0.0
      width: 100000.0
```

## Complete Examples

### Electron Plasma Wave (EPW)

See `configs/vlasov-1d/epw.yaml` - driven electron plasma wave with Fokker-Planck collisions.

### Bump-on-Tail Instability

See `configs/vlasov-1d/bump-on-tail.yaml` - two-species configuration with beam distribution.

### Two-Stream Instability

See `configs/vlasov-1d/twostream.yaml` - counter-streaming electron beams.

### Stimulated Raman Scattering (SRS)

See `configs/vlasov-1d/srs.yaml` - electromagnetic simulation with `ey` drivers.

### Nonlinear EPW with Initial Condition

See `configs/vlasov-1d/nlepw-ic.yaml` - large-amplitude initial perturbation without external driver.
