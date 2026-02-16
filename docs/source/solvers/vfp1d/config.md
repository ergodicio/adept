# VFP-1D Configuration Reference

This document describes how to construct a configuration file for the `vfp-1d` solver.

## Top-Level Structure

```yaml
solver: vfp-1d

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

terms:
  # Solver configuration
```

## units

Physical unit normalizations for the simulation.

| Field | Type | Description |
|-------|------|-------------|
| `laser_wavelength` | string | Laser wavelength with unit, e.g., `"351nm"` |
| `reference electron temperature` | string | Reference electron temperature with unit, e.g., `"3000eV"` |
| `reference ion temperature` | string | Reference ion temperature with unit, e.g., `"300eV"` |
| `reference electron density` | string | Reference electron density with unit, e.g., `"2.275e21/cm^3"` |
| `Z` | int | Ionization state |
| `Ion` | string | Ion species label, e.g., `"Au+"` |
| `logLambda` | string or float | Coulomb logarithm: `"nrl"` for NRL formula, or a numeric value |

Example:
```yaml
units:
  laser_wavelength: 351nm
  reference electron temperature: 3000eV
  reference ion temperature: 300eV
  reference electron density: 2.275e21/cm^3
  Z: 6
  Ion: Au+
  logLambda: nrl
```

## density

Species and density configuration. Multiple species are supported via keys prefixed with `species-`.

| Field | Type | Description |
|-------|------|-------------|
| `quasineutrality` | bool | Whether to enforce quasineutrality |

### Species Definition

Each species is defined with a key starting with `species-` (e.g., `species-background`, `species-hote`).

| Field | Type | Description |
|-------|------|-------------|
| `noise_seed` | int | Random seed for noise initialization |
| `noise_type` | string | `"gaussian"` or `"uniform"` |
| `noise_val` | float | Amplitude of noise |
| `v0` | float | Drift velocity (normalized) |
| `T0` | float | Temperature (normalized to reference temperature) |
| `m` | float | Super-Gaussian exponent. `2.0` is Maxwellian |
| `n` | object | Density spatial profile (see Basis Types) |
| `T` | object | Temperature spatial profile (see Basis Types) |

### Basis Types

Each of `n` and `T` has a `basis` field that determines the spatial profile.

**`uniform`**: Constant profile
```yaml
n:
  basis: uniform
  baseline: 1.0
  bump_or_trough: bump
  center: 0m
  rise: 1m
  bump_height: 0.0
  width: 1m
```

**`sine`**: Sinusoidal perturbation
```yaml
T:
  basis: sine
  baseline: 1.0
  amplitude: 1.0e-3
  wavelength: 250um
```

**`cosine`**: Cosine perturbation (useful with reflective BCs where dT/dx=0 at boundaries)
```yaml
T:
  basis: cosine
  baseline: 1.0
  amplitude: 1.0e-3
  wavelength: 250um
```

**`tanh`**: Hyperbolic tangent profile
```yaml
T:
  basis: tanh
  baseline: 1.0
  bump_or_trough: bump
  center: 50um
  rise: 5um
  bump_height: 0.1
  width: 20um
```

## grid

Simulation grid parameters. All dimensioned quantities use astropy-style unit strings.

| Field | Type | Description |
|-------|------|-------------|
| `dt` | string | Timestep with unit, e.g., `"1fs"` |
| `nv` | int | Number of velocity grid points |
| `nx` | int | Number of spatial grid cells |
| `tmin` | string | Start time with unit |
| `tmax` | string | End time with unit |
| `vmax` | float | Maximum velocity (normalized to $c$) |
| `xmin` | string | Domain minimum x with unit |
| `xmax` | string | Domain maximum x with unit |
| `nl` | int | Maximum spherical harmonic order (typically `1`) |
| `boundary` | string | Spatial boundary condition: `"periodic"` (default) or `"reflective"` |

Example:
```yaml
grid:
  dt: 1fs
  nv: 256
  nx: 32
  tmin: 0.ps
  tmax: 2ps
  vmax: 8.0
  xmin: 0.0um
  xmax: 500um
  nl: 1
  boundary: periodic
```

### Boundary conditions

- **`periodic`**: Standard periodic wrapping. Suitable for full-wavelength perturbations.
- **`reflective`**: Zero-flux boundaries. Vector quantities ($f_{10}$, $E$) are zero at domain edges. Scalar quantities ($f_0$) use mirror ghost cells. Suitable for half-wavelength (cosine) perturbations where symmetry allows simulating half the domain.

## save

Configures what data to save and at what times.

```yaml
save:
  fields:
    t:
      tmin: 0.0ps
      tmax: 2.0ps
      nt: 11
  electron:
    t:
      tmin: 0.0ps
      tmax: 2.0ps
      nt: 6
```

| Save Key | Description |
|----------|-------------|
| `fields` | Fluid moments and electromagnetic fields ($n$, $T$, $U$, $P$, $v$, $q$, $E$, $B$) |
| `electron` | Electron distribution functions ($f_0$, $f_{10}$) |

Each save key contains a `t` sub-key with:

| Field | Type | Description |
|-------|------|-------------|
| `tmin` | string | Start time for saving |
| `tmax` | string | End time for saving |
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
  experiment: vfp1d
  run: epperlein-short
```

## drivers

External electromagnetic drivers. Currently unused for typical VFP-1D transport simulations but required in the config.

```yaml
drivers:
  ex: {}
  ey: {}
```

## terms

Solver algorithm configuration.

| Field | Type | Description |
|-------|------|-------------|
| `e_solver` | string | Electric field solver: `"oshun"` or `"ampere"` |
| `fokker_planck` | object | Fokker-Planck collision operator configuration |

### e_solver

- **`oshun`**: Implicit E-field via Taylor expansion of $J(E)$ (Tzoufras 2013). Recommended for most simulations.
- **`ampere`**: Explicit Ampere update.

### fokker_planck

Configuration for the two collision operators.

#### fokker_planck.f00

Isotropic ($f_0$) collision operator.

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Collision model: `"LenardBernstein"` or `"Dougherty"` |
| `scheme` | string | Differencing scheme: `"central"` or `"chang_cooper"` |

#### fokker_planck.flm

Anisotropic ($f_{lm}$) collision operator (FLM).

| Field | Type | Description |
|-------|------|-------------|
| `ee` | bool | Include electron-electron collisions. When `false`, uses $Z^*$ Epperlein-Haines scaling instead |

### Example

```yaml
terms:
  e_solver: oshun
  fokker_planck:
    f00:
      model: LenardBernstein
      scheme: central
    flm:
      ee: true
```

## Complete Examples

### Epperlein-Short Heat Transport

See `configs/vfp-1d/epp-short.yaml` - Epperlein-Short heat transport benchmark with sinusoidal temperature perturbation and multi-species (background + hot electrons).

### Hotspot Relaxation

See `configs/vfp-1d/hotspot.yaml` - Temperature hotspot relaxation with `tanh` temperature profile.
