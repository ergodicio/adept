# LPSE-2D (Envelope-2D) Configuration Reference

This document describes how to construct a configuration file for the `envelope-2d` solver. This is a 2D laser-plasma simulation using envelope equations for electron plasma waves (EPW). It supports two-plasmon decay (TPD), stimulated Raman scattering (SRS), and other laser-plasma instabilities.

## Top-Level Structure

```yaml
solver: envelope-2d

units:
  # Physical unit normalizations

density:
  # Density profile configuration

grid:
  # Simulation grid parameters

save:
  # Output configuration

mlflow:
  # Experiment tracking

drivers:
  # Laser and EPW drivers

terms:
  # Physics terms configuration
```

## units

Physical unit normalizations. Note: This module uses different unit keys than the Vlasov modules.

| Field | Type | Description |
|-------|------|-------------|
| `atomic number` | int | Atomic number of the ion species |
| `envelope density` | float | Reference density as fraction of critical density |
| `ionization state` | int | Ionization state Z |
| `laser intensity` | string | Laser intensity with unit, e.g., `"3.5e+14W/cm^2"` |
| `laser_wavelength` | string | Laser wavelength with unit, e.g., `"351nm"` |
| `reference electron temperature` | string | Electron temperature with unit, e.g., `"2000.0eV"` |
| `reference ion temperature` | string | Ion temperature with unit, e.g., `"1000eV"` |

Example:
```yaml
units:
  atomic number: 40
  envelope density: 0.25
  ionization state: 6
  laser intensity: 1.5e+14W/cm^2
  laser_wavelength: 351nm
  reference electron temperature: 2000.0eV
  reference ion temperature: 1000eV
```

## density

Density profile configuration.

| Field | Type | Description |
|-------|------|-------------|
| `basis` | string | Profile type: `"uniform"` or `"linear"` |
| `gradient scale length` | string | Scale length with unit (for `linear` basis) |
| `max` | float | Maximum density fraction (for `linear` basis) |
| `min` | float | Minimum density fraction (for `linear` basis) |
| `noise` | object | Initial noise configuration |

### noise

| Field | Type | Description |
|-------|------|-------------|
| `max` | float | Maximum noise amplitude |
| `min` | float | Minimum noise amplitude |
| `type` | string | `"uniform"` or `"normal"` |

### Example: Uniform Density

```yaml
density:
  basis: uniform
  noise:
    max: 1.0e-09
    min: 1.0e-10
    type: uniform
```

### Example: Linear Density Gradient

```yaml
density:
  basis: linear
  gradient scale length: 50um
  max: 0.28
  min: 0.18
  noise:
    max: 1.0e-09
    min: 1.0e-10
    type: uniform
```

Note: When using `linear` basis, the grid size is automatically computed from the gradient scale length and density range.

## grid

Simulation grid parameters. Note: Grid values use physical units as strings.

| Field | Type | Description |
|-------|------|-------------|
| `boundary_abs_coeff` | float | Absorbing boundary coefficient |
| `boundary_width` | string | Width of absorbing boundary layer with unit |
| `low_pass_filter` | float | Low-pass filter cutoff as fraction of kmax (0-1) |
| `dt` | string | Timestep with unit |
| `dx` | string | Spatial resolution with unit |
| `tmax` | string | End time with unit |
| `tmin` | string | Start time with unit |
| `ymax` | string | Domain maximum y with unit |
| `ymin` | string | Domain minimum y with unit |

Note: `nx` and `ny` are computed automatically from the grid parameters. The grid is optimized for FFT performance (sizes with small prime factors).

Example:
```yaml
grid:
  boundary_abs_coeff: 1.0e4
  boundary_width: 1.5um
  low_pass_filter: 0.66
  dt: 0.010fs
  dx: 40nm
  tmax: 2ps
  tmin: 0.0ns
  ymax: 0.08um
  ymin: -0.08um
```

## save

Configures what data to save and at what times.

### Structure

```yaml
save:
  fields:
    t:
      dt: 100fs
      tmax: 4ps
      tmin: 0ps
    x:
      dx: 50nm
    y:
      dy: 50nm
```

### fields

| Field | Type | Description |
|-------|------|-------------|
| `t` | object | Temporal save configuration |
| `x` | object | Optional spatial subsampling in x |
| `y` | object | Optional spatial subsampling in y |

#### t (temporal)

| Field | Type | Description |
|-------|------|-------------|
| `dt` | string | Time interval between saves, with unit |
| `tmax` | string | End time for saving, with unit |
| `tmin` | string | Start time for saving, with unit |

#### x (optional)

| Field | Type | Description |
|-------|------|-------------|
| `dx` | string | Spatial resolution for saved data, with unit |

#### y (optional)

| Field | Type | Description |
|-------|------|-------------|
| `dy` | string | Spatial resolution for saved data, with unit |

## mlflow

Experiment tracking configuration.

| Field | Type | Description |
|-------|------|-------------|
| `experiment` | string | MLflow experiment name |
| `run` | string | MLflow run name |

Example:
```yaml
mlflow:
  experiment: tpd
  run: srs-test
```

## drivers

Laser and EPW drivers.

### E0 - Pump Laser Driver

The main laser pump for TPD/SRS simulations.

| Field | Type | Description |
|-------|------|-------------|
| `envelope` | object | Spatiotemporal envelope |
| `delta_omega_max` | float | Maximum frequency spread (optional) |
| `num_colors` | int | Number of laser colors (optional) |
| `shape` | string | Amplitude shape: `"uniform"` (optional) |

#### envelope

All values are strings with physical units.

| Field | Type | Description |
|-------|------|-------------|
| `tc` | string | Temporal center |
| `tr` | string | Temporal rise time |
| `tw` | string | Temporal width |
| `xc` | string | Spatial center (x) |
| `xr` | string | Spatial rise (x) |
| `xw` | string | Spatial width (x) |
| `yc` | string | Spatial center (y) |
| `yr` | string | Spatial rise (y) |
| `yw` | string | Spatial width (y) |

Example:
```yaml
drivers:
  E0:
    delta_omega_max: 0.015
    envelope:
      tc: 200.25ps
      tr: 0.1ps
      tw: 400ps
      xc: 50um
      xr: 0.2um
      xw: 1000um
      yc: 50um
      yr: 0.2um
      yw: 1000um
    num_colors: 1
    shape: uniform
```

### E2 - EPW Driver (Optional)

Direct EPW driver for seeding or testing.

| Field | Type | Description |
|-------|------|-------------|
| `envelope` | object | Same structure as E0 envelope |
| `a0` | float | Amplitude |
| `k0` | float | Wavenumber |
| `w0` | float | Frequency |

Example:
```yaml
drivers:
  E2:
    envelope:
      tw: 200fs
      tr: 25fs
      tc: 150fs
      xw: 500um
      xc: 10um
      xr: 0.2um
      yr: 0.2um
      yc: 0um
      yw: 50um
    a0: 1000
    k0: -10.0
    w0: 20.0
```

## terms

Physics terms configuration.

| Field | Type | Description |
|-------|------|-------------|
| `epw` | object | Electron plasma wave configuration |
| `zero_mask` | bool | Whether to zero out k=0 mode |

### epw

| Field | Type | Description |
|-------|------|-------------|
| `boundary` | object | Boundary conditions |
| `damping` | object | Damping mechanisms |
| `density_gradient` | bool | Include density gradient effects |
| `linear` | bool | Linear mode (disables nonlinear coupling) |
| `source` | object | Source terms |
| `hyperviscosity` | object | Optional hyperviscosity for numerical stability |
| `trapping` | object | Optional trapping model |
| `kinetic real part` | bool | Include kinetic correction to real frequency |

#### boundary

| Field | Type | Description |
|-------|------|-------------|
| `x` | string | `"periodic"` or `"absorbing"` |
| `y` | string | `"periodic"` or `"absorbing"` |

#### damping

| Field | Type | Description |
|-------|------|-------------|
| `collisions` | bool or float | Collisional damping. `true` computes from plasma parameters, or specify rate directly |
| `landau` | bool | Include Landau damping |

#### source

| Field | Type | Description |
|-------|------|-------------|
| `noise` | bool | Add random noise source |
| `tpd` | bool | Include two-plasmon decay source |
| `srs` | bool | Include stimulated Raman scattering source (optional) |

#### hyperviscosity (optional)

| Field | Type | Description |
|-------|------|-------------|
| `coeff` | float | Hyperviscosity coefficient |
| `order` | int | Order of hyperviscosity (must be even) |

#### trapping (optional)

| Field | Type | Description |
|-------|------|-------------|
| `active` | bool | Enable trapping model |
| `kld` | float | k * lambda_D parameter |
| `nuee` | float | Electron-electron collision frequency |

### Example: TPD Simulation

```yaml
terms:
  epw:
    boundary:
      x: absorbing
      y: periodic
    damping:
      collisions: 1.0
      landau: true
    density_gradient: true
    linear: true
    source:
      noise: true
      tpd: false
      srs: true
  zero_mask: true
```

### Example: Simple EPW Test

```yaml
terms:
  epw:
    boundary:
      x: periodic
      y: periodic
    damping:
      collisions: false
      landau: false
    density_gradient: false
    linear: True
    source:
      noise: false
      tpd: false
  zero_mask: false
```

## Complete Example

```yaml
solver: envelope-2d

units:
  atomic number: 40
  envelope density: 0.25
  ionization state: 6
  laser intensity: 1.5e+14W/cm^2
  laser_wavelength: 351nm
  reference electron temperature: 2000.0eV
  reference ion temperature: 1000eV

density:
  basis: linear
  gradient scale length: 50um
  max: 0.28
  min: 0.18
  noise:
    max: 1.0e-09
    min: 1.0e-10
    type: uniform

grid:
  boundary_abs_coeff: 1.0e4
  boundary_width: 1.5um
  low_pass_filter: 0.66
  dt: 0.010fs
  dx: 40nm
  tmax: 2ps
  tmin: 0.0ns
  ymax: 0.08um
  ymin: -0.08um

mlflow:
  experiment: tpd
  run: my-simulation

save:
  fields:
    t:
      dt: 0.2ps
      tmax: 2ps
      tmin: 0ps
    x:
      dx: 50nm
    y:
      dy: 50nm

drivers:
  E0:
    delta_omega_max: 0.015
    envelope:
      tc: 200.25ps
      tr: 0.1ps
      tw: 400ps
      xc: 50um
      xr: 0.2um
      xw: 1000um
      yc: 50um
      yr: 0.2um
      yw: 1000um
    num_colors: 1
    shape: uniform

terms:
  epw:
    boundary:
      x: absorbing
      y: periodic
    damping:
      collisions: 1.0
      landau: true
    density_gradient: true
    linear: true
    source:
      noise: true
      tpd: false
      srs: true
  zero_mask: true
```

## Example Configurations

### EPW Linear Propagation

See `configs/envelope-2d/epw.yaml` - Simple EPW test without instabilities.

### Landau Damping

See `configs/envelope-2d/damping.yaml` - EPW with Landau damping and trapping model.

### Two-Plasmon Decay

See `configs/envelope-2d/tpd.yaml` - TPD simulation with linear density gradient.

### SRS / Reflection

See `configs/envelope-2d/reflection.yaml` - SRS simulation with kinetic corrections.
