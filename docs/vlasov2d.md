# Vlasov-2D Configuration Reference

This document describes how to construct a configuration file for the `vlasov-2d` solver. This is a 2D2V (2 spatial dimensions, 2 velocity dimensions) Vlasov-Maxwell solver.

## Top-Level Structure

```yaml
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

krook:
  # Krook collision operator

solver:
  # Solver algorithm configuration
```

## units

Physical unit normalizations for the simulation. Same structure as vlasov-1d.

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

Species and density configuration.

| Field | Type | Description |
|-------|------|-------------|
| `quasineutrality` | bool | Whether to enforce quasineutrality |

### Species Definition

Each species is defined with a key starting with `species-` (e.g., `species-background`).

| Field | Type | Description |
|-------|------|-------------|
| `noise_seed` | int | Random seed for noise initialization |
| `noise_type` | string | `"gaussian"` or `"uniform"` |
| `noise_val` | float | Amplitude of noise |
| `v0` | float | Drift velocity (normalized) |
| `T0` | float | Temperature (normalized to `normalizing_temperature`) |
| `m` | float | Exponent for super-Gaussian distribution. `2.0` is Maxwellian |
| `basis` | string | Spatial profile type. Currently only `"uniform"` is supported |
| `space-profile` | object | Spatial profile configuration |

#### space-profile

| Field | Type | Description |
|-------|------|-------------|
| `baseline` | float | Base density |
| `bump_or_trough` | string | `"bump"` or `"trough"` |
| `center` | float | Profile center |
| `rise` | float | Transition steepness |
| `slope` | float | Linear slope (typically 0.0) |
| `wall_height` | float | Height of profile feature |
| `width` | float | Profile width |

Example:
```yaml
density:
  quasineutrality: true
  species-background:
    noise_seed: 420
    noise_type: gaussian
    noise_val: 0.0
    v0: 0.0
    T0: 1.0
    m: 2.0
    basis: uniform
    space-profile:
      baseline: 1.0
      bump_or_trough: bump
      center: 0.0
      rise: 25.0
      slope: 0.0
      wall_height: 0.0
      width: 100000.0
```

## grid

Simulation grid parameters for 2D2V phase space.

| Field | Type | Description |
|-------|------|-------------|
| `dt` | float | Timestep (normalized) |
| `nx` | int | Number of spatial grid points in x |
| `ny` | int | Number of spatial grid points in y |
| `nvx` | int | Number of velocity grid points in vx |
| `nvy` | int | Number of velocity grid points in vy |
| `tmin` | float | Start time |
| `tmax` | float | End time |
| `vmax` | float | Maximum velocity extent (same for vx and vy) |
| `xmin` | float | Domain minimum x |
| `xmax` | float | Domain maximum x |
| `ymin` | float | Domain minimum y |
| `ymax` | float | Domain maximum y |

Example:
```yaml
grid:
  dt: 0.1
  nx: 24
  ny: 8
  nvx: 128
  nvy: 128
  tmin: 0.0
  tmax: 150.0
  vmax: 6.4
  xmin: 0.0
  xmax: 20.94
  ymin: -4.0
  ymax: 4.0
```

## save

Configures what data to save and at what times.

### Structure

```yaml
save:
  fields:
    t:
      tmin: 0.0
      tmax: 150.0
      nt: 301
  electron:
    t:
      tmin: 0.0
      tmax: 150.0
      nt: 3
```

| Save Key | Description |
|----------|-------------|
| `fields` | Electric field (ex, ey), magnetic field (bz), and driver fields (dex, dey) |
| `electron` | Electron distribution function |

Each save key contains a `t` sub-key with:

| Field | Type | Description |
|-------|------|-------------|
| `tmin` | float | Start time for saving |
| `tmax` | float | End time for saving |
| `nt` | int | Number of save points |

### Optional Spatial Subsampling

Fields can optionally include spatial or spectral subsampling:

```yaml
save:
  fields:
    t:
      tmin: 0.5
      tmax: 99.0
      nt: 64
    x:
      xmin: 0.5
      xmax: 19.0
      nx: 16
    y:
      ymin: -2.0
      ymax: 2.0
      ny: 2
```

## mlflow

Experiment tracking configuration.

| Field | Type | Description |
|-------|------|-------------|
| `experiment` | string | MLflow experiment name |
| `run` | string | MLflow run name |

Example:
```yaml
mlflow:
  experiment: ld-2d2v
  run: envelope-driver-small-amp
```

## drivers

External electromagnetic drivers. The `ex` driver includes y-dimension envelope parameters.

| Field | Type | Description |
|-------|------|-------------|
| `ex` | dict | Longitudinal electric field drivers |
| `ey` | dict | Transverse electric field drivers (typically empty `{}`) |

Each driver is identified by a string key (e.g., `"0"`) and has these parameters:

| Field | Type | Description |
|-------|------|-------------|
| `a0` | float | Driver amplitude |
| `k0` | float | Wavenumber |
| `w0` | float | Frequency |
| `dw0` | float | Frequency offset/chirp |
| `t_center` | float | Temporal envelope center |
| `t_rise` | float | Temporal envelope rise time |
| `t_width` | float | Temporal envelope width |
| `x_center` | float | Spatial envelope center (x) |
| `x_rise` | float | Spatial envelope rise distance (x) |
| `x_width` | float | Spatial envelope width (x) |
| `y_center` | float | Spatial envelope center (y) |
| `y_rise` | float | Spatial envelope rise distance (y) |
| `y_width` | float | Spatial envelope width (y) |

Example:
```yaml
drivers:
  ex:
    '0':
      a0: 1.e-6
      k0: 0.3
      w0: 1.1598
      dw0: 0.0
      t_center: 30.0
      t_rise: 5.0
      t_width: 30.0
      x_center: 0.0
      x_rise: 10.0
      x_width: 4000000.0
      y_center: 0.0
      y_rise: 10.0
      y_width: 200.0
  ey: {}
```

## krook

Krook collision operator configuration (at top level, not under `terms`).

| Field | Type | Description |
|-------|------|-------------|
| `space-profile` | object | Spatial profile for collision frequency |
| `time-profile` | object | Temporal profile for collision frequency |

### Profile Objects

| Field | Type | Description |
|-------|------|-------------|
| `baseline` | float | Base collision frequency |
| `bump_or_trough` | string | `"bump"` or `"trough"` |
| `center` | float | Profile center |
| `rise` | float | Transition steepness |
| `slope` | float | Linear slope (typically 0.0) |
| `wall_height` | float | Height of profile feature |
| `width` | float | Profile width |

Example:
```yaml
krook:
  space-profile:
    baseline: 0.0
    bump_or_trough: bump
    center: 2500
    rise: 10.0
    slope: 0.0
    wall_height: 0.0
    width: 48000000.0
  time-profile:
    baseline: 0.0
    bump_or_trough: bump
    center: 0.0
    rise: 10.0
    slope: 0.0
    wall_height: 0.0
    width: 100000.0
```

## solver

Solver algorithm configuration. Note: This is structured differently from vlasov-1d.

| Field | Type | Description |
|-------|------|-------------|
| `dfdt` | string | Time integrator: `"leapfrog"` |
| `edfdv` | string | Velocity advection scheme: `"exponential"` or `"center_difference"` |
| `vdfdx` | string | Spatial advection scheme: `"exponential"` |
| `field` | string | Field solver: `"ampere"` |
| `fp_operator` | string | Fokker-Planck operator type: `"dougherty"` |
| `num_unroll` | int | Number of steps to unroll for JIT compilation |
| `push_f` | bool | Whether to push the distribution function |

Example:
```yaml
solver:
  dfdt: leapfrog
  edfdv: exponential
  vdfdx: exponential
  field: ampere
  fp_operator: dougherty
  num_unroll: 32
  push_f: True
```

## Complete Example

```yaml
units:
  laser_wavelength: 351nm
  normalizing_temperature: 2000eV
  normalizing_density: 1.5e21/cc
  Z: 10
  Zp: 10

density:
  quasineutrality: true
  species-background:
    noise_seed: 420
    noise_type: gaussian
    noise_val: 0.0
    v0: 0.0
    T0: 1.0
    m: 2.0
    basis: uniform
    space-profile:
      baseline: 1.0
      bump_or_trough: bump
      center: 0.0
      rise: 25.0
      slope: 0.0
      wall_height: 0.0
      width: 100000.0

grid:
  dt: 0.1
  nvx: 128
  nvy: 128
  nx: 24
  ny: 8
  tmin: 0.0
  tmax: 150.0
  vmax: 6.4
  xmax: 20.94
  xmin: 0.0
  ymin: -4.0
  ymax: 4.0

save:
  fields:
    t:
      tmin: 0.0
      tmax: 150.0
      nt: 301
  electron:
    t:
      tmin: 0.0
      tmax: 150.0
      nt: 3

krook:
  space-profile:
    baseline: 0.0
    bump_or_trough: bump
    center: 2500
    rise: 10.0
    slope: 0.0
    wall_height: 0.0
    width: 48000000.0
  time-profile:
    baseline: 0.0
    bump_or_trough: bump
    center: 0.0
    rise: 10.0
    slope: 0.0
    wall_height: 0.0
    width: 100000.0

mlflow:
  experiment: ld-2d2v
  run: my-simulation

drivers:
  ex:
    '0':
      a0: 1.e-6
      k0: 0.3
      t_center: 30.0
      t_rise: 5.0
      t_width: 30.0
      w0: 1.1598
      dw0: 0.0
      x_center: 0.0
      x_rise: 10.0
      x_width: 4000000.0
      y_center: 0.0
      y_rise: 10.0
      y_width: 200.0
  ey: {}

solver:
  dfdt: leapfrog
  edfdv: exponential
  fp_operator: dougherty
  num_unroll: 32
  field: ampere
  vdfdx: exponential
  push_f: True
```

## Example Configurations

### Landau Damping (2D)

See `configs/vlasov-2d/base.yaml` - 2D Landau damping simulation with small-amplitude driver.

See `tests/test_vlasov2d/configs/damping.yaml` - Test configuration for Landau damping verification.
