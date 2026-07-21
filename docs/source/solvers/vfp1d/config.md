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
| `logLambda` | string or float | Coulomb logarithm: `"nrl"` (NRL formulary), `"lee-more"` (Lee & More 1984: $\ln\Lambda = \max[2, \tfrac{1}{2}\ln(1 + b_\max^2/b_\min^2)]$ with Debye-Hückel screening floored at the ion-sphere radius and quantum/classical $b_\min$), or a numeric value |

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

**`file`**: Tabulated profile from a two-column (coordinate, value) CSV or whitespace-delimited text file. Header rows are skipped automatically; the profile is interpolated onto the grid and clamped to the endpoint values outside the tabulated range. Density values are normalized by `units.reference electron density`, temperature values by `units.reference electron temperature`.
```yaml
n:
  basis: file
  path: /path/to/electrondensity.csv
  units: 1/cm^3          # units of the value column
  coordinate_units: um   # units of the coordinate column (default um)
```

### Ion block (optional)

An optional `density.ion` block sets a spatially varying ion density and ionization state from a tabulated mass-density profile:

```yaml
density:
  ion:
    A: 9.0122            # ion mass number (amu)
    mass_density:
      basis: file        # only "file" is supported
      path: /path/to/massdensity.csv
      units: g/cm^3
      coordinate_units: um
```

This sets $n_i(x) = \rho(x) / (A \, m_u)$ and $Z(x) = n_e(x)/n_i(x)$ pointwise. When the block is omitted, $Z$ is uniform (`units.Z`) and $n_i = n_e / Z$.

For atomic **mixtures** (e.g. fully ionized CD), add a `mixture` list. The plasma is then represented by a single effective ion species that preserves both quasineutrality ($Z n_i = n_e$) and the electron-ion collision rate ($Z^2 n_i = \sum_i n_i Z_i^2 = Z_\text{eff}\, n_e$), with $Z_\text{eff} = \langle Z^2\rangle / \langle Z\rangle$. **Set `units.Z` to $Z_\text{eff}$** — it is the collisional/transport Z that enters the Epperlein-Haines and SNB coefficients. The mass-density profile is used to cross-check full-ionization quasineutrality ($\bar{Z}\rho/(A m_u)$ vs $n_e$); a warning is printed if they disagree by more than 5%.

```yaml
density:
  ion:
    A: 6.304                                # mean atomic weight (optional; default sum of fractions * A)
    mixture:
      - { Z: 6, A: 12.011, fraction: 0.4 }  # C
      - { Z: 1, A: 2.0141, fraction: 0.6 }  # D
    mass_density:
      basis: file
      path: /path/to/massdensity.csv
      units: g/cm^3
      coordinate_units: um
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
| `geometry` | string | Spatial geometry: `"cartesian"` (default) or `"spherical"` |

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

### Geometry

- **`cartesian`**: Planar 1D geometry (default). The spatial coordinate is $x$.
- **`spherical`**: Radial 1D geometry with spherical symmetry. The spatial coordinate is the radius $r$. Requires `xmin: 0` and `boundary: reflective` (the flux through $r=0$ vanishes by symmetry; the outer edge is a reflecting wall). Velocity space is unchanged. For $n_l = 1$ the only modification to the equations is the $f_0$ advection term becoming a radial divergence,

$$\partial_t f_0 + \frac{v}{3}\frac{1}{r^2}\partial_r \left(r^2 f_{10}\right) = C[f_0], \qquad \partial_t f_{10} + v\, \partial_r f_0 - E\, \partial_v f_0 = C[f_{10}]$$

  (the angular coupling term $\propto (1-\mu^2)/r$ only feeds $f_2$ into the $f_{10}$ equation, which is dropped at $n_l = 1$). The divergence is discretized in conservative finite-volume form with exact cell volumes $(r_+^3 - r_-^3)/3$.

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

External drivers for heating, cooling, and electromagnetic fields.

```yaml
drivers:
  ib: {}                    # Inverse Bremsstrahlung heating (optional)
  maxwellian_heating: {}    # Maxwellian heating/cooling (optional)
```

### drivers.ib

Inverse Bremsstrahlung (IB) laser heating. Augments the Fokker-Planck diffusion coefficient $D$ by $v_\text{osc}^2 g(v) / (6v)$ where $g(v) = [1 + (Z^2 n_i / (\omega_0 v^3))^2]^{-1}$ (Ridgers eq 4.39). Drives the distribution toward a Langdon/super-Gaussian shape.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `intensity_1e15_Wcm2` | float | `0.0` | Laser intensity in units of $10^{15}$ W/cm². |
| `polarisation` | string or float | `"linear"` | `"linear"` ($\alpha=1$), `"circular"` ($\alpha=1/2$), or a float in $(0,1]$. |

The quiver velocity $v_\text{osc}^2$ and normalised laser frequency $\omega_0$ are derived automatically from `units.laser_wavelength` and `units.reference electron temperature`:

$$v_\text{osc}^2 = \frac{0.093373}{\alpha} \cdot \frac{I \lambda_0^2}{T_{e0}} \quad \text{(normalised units, } I \text{ in } 10^{15}\text{ W/cm}^2\text{, } \lambda_0 \text{ in } \mu\text{m, } T_{e0} \text{ in keV)}$$

$$\omega_0 = \omega_L / \omega_p = 2\pi c / (\lambda_0 \omega_p)$$

### drivers.maxwellian_heating

Maxwellian heating or cooling. Augments $D$ by $D_0 v^2$ (Ridgers eq E.2), which changes the temperature while preserving the Maxwellian shape of the distribution.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `D0` | float | `None` | Maxwellian heating/cooling rate (normalised). Positive heats, negative cools. |

The heating rate is $\partial T / \partial t = D_0 / \hat{Z}$ (constant, independent of temperature), where both $T$ and $t$ are in normalised units ($T_0$ and $1/\omega_p$ respectively). In physical units this corresponds to a volumetric power density $P/V = n T_0 \omega_p D_0 / \hat{Z}$.

Example:
```yaml
drivers:
  ib:
    intensity_1e15_Wcm2: 1.0
    polarisation: linear
  # OR for Maxwellian heating:
  # maxwellian_heating:
  #   D0: 0.1
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
| `model` | string | Collision model (see below) |
| `scheme` | string | Differencing scheme: `"central"` or `"chang_cooper"` |

**Available models:**

| Model | Description |
|-------|-------------|
| `CoulombianKernel` | Standard Landau/Rosenbluth operator with kernel $g(\varepsilon,\varepsilon') = \min(\varepsilon^{3/2}, \varepsilon'^{3/2})$. Default. |
| `AsymptoticLocal` | Rank-1 kernel $g(\varepsilon,\varepsilon') = \sqrt{\varepsilon \cdot \varepsilon'}$ from high-velocity limit of linearized collision operator. O(N) computation. |
| `FastVFP` | Beta-based model with $D = 1/(2\beta v)$, giving constant drift $C = 1$. From Bell & Sherlock (2024). |

**Note:** Energy is not conserved for non-Maxwellian distributions with the current Chang-Cooper scheme. A proper Buet weak-form scheme is planned for improved energy conservation.

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
      model: CoulombianKernel
      scheme: chang_cooper
    flm:
      ee: true
```

## diagnostics

Optional post-processing diagnostics configuration. The Spitzer-Härm / SNB / kinetic heat-flux comparison always runs during post-processing (it writes `plots/fields/heat-flux-comparison.png` and `binary/heat-flux-comparison.nc`, and logs the `q_ratio_kinetic_over_sh` and `q_ratio_snb_over_sh` metrics). The `diagnostics.snb` block tunes the SNB model (Schurtz et al. 2000; the "separated" variant recommended by Brodrick et al. 2017):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ngroups` | int | `32` | Number of energy groups in $\beta = v^2/(2T)$ |
| `beta_max` | float | `30.0` | Upper edge of the group grid |
| `r` | float | `2.0` | e-e absorption scaling of the group mfp, $\lambda_g^{ee}/r$ |

```yaml
diagnostics:
  snb:
    ngroups: 32
    beta_max: 30.0
    r: 2.0
```

## Complete Examples

### Epperlein-Short Heat Transport

See `configs/vfp-1d/epp-short.yaml` - Epperlein-Short heat transport benchmark with sinusoidal temperature perturbation and multi-species (background + hot electrons).

### Hotspot Relaxation

See `configs/vfp-1d/hotspot.yaml` - Temperature hotspot relaxation with `tanh` temperature profile.

### Spherical Heat Flow from Tabulated Profiles

See `configs/vfp-1d/spherical-heatflow.yaml` - Radial (spherical) geometry with $n_e(r)$, $T_e(r)$, and mass-density profiles loaded from CSV files, set up for a kinetic vs SNB vs Spitzer-Härm heat-flow comparison.
