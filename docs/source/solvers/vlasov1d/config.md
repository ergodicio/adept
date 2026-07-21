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
| `normalizing_temperature` | string | Reference temperature with unit, e.g., `"2000eV"` |
| `normalizing_density` | string | Reference density with unit, e.g., `"1.5e21/cc"` |

Example:
```yaml
units:
  normalizing_temperature: 2000eV
  normalizing_density: 1.5e21/cc
```

### Normalization convention

All Vlasov-1D quantities are normalized using the following unit system, built from
`normalizing_density` ($n_0$) and `normalizing_temperature` ($T_0$):

| Unit | Definition | Meaning |
|------|-----------|---------|
| time | $\tau = 1/\omega_{p0}$, $\omega_{p0} = \sqrt{n_0 e^2/(\epsilon_0 m_e)}$ | inverse plasma frequency |
| velocity | $v_0 = \sqrt{T_0/m_e}$ | electron thermal speed (RMS / standard-deviation convention) |
| length | $L_0 = v_0/\omega_{p0} = \lambda_{De}$ | electron Debye length |
| wavenumber | $1/L_0$ | a code wavenumber is $k \lambda_{De}$ |

Consequences of the $v_0 = \sqrt{T_0/m_e}$ (σ) convention:

- A Maxwellian at code temperature $T = 1$ is $f \propto e^{-v^2/2}$ (unit variance).
- The Bohm–Gross dispersion in code units is $\omega^2 = 1 + 3 k^2$.
- The normalized speed of light is $\hat c = c/\sqrt{T_0/m_e}$ (e.g. $\hat c = 15.98$ at 2000 eV).

Dimensional inputs (strings with units, e.g. `xmax: 100um`) are converted with these
units; plain numeric inputs are taken to already be in code units and pass through
unchanged.

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
| `v0` | float | Drift velocity in code units of $\sqrt{T_0/m_e}$ (thermal-σ units). Numeric only — dimensional strings are not supported here |
| `T0` | float | Temperature in units of `normalizing_temperature`. The initialized distribution has velocity variance `T0/mass`. Numeric only |
| `m` | float | Exponent for super-Gaussian distribution $f \propto \exp[-\|v/(\alpha v_{th})\|^m]$. `2.0` is Maxwellian. Note: $\alpha$ is chosen to fix the moment ratio $\langle v^4\rangle/\langle v^2\rangle = 3\,T_0/\mathrm{mass}$ for all $m$; the *variance* equals `T0/mass` only at `m: 2`. For flat-top distributions (`m > 2`) the second-moment temperature diagnostic will read higher than `T0` (e.g. ×1.24 at `m: 3`, ×1.37 at `m: 4`) |
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
| `nv` | int | Number of velocity grid points (not needed for multispecies) |
| `nx` | int | Number of spatial grid points |
| `tmin` | float | Start time |
| `tmax` | float | End time |
| `vmax` | float | Upper bound of the velocity grid (not needed for multispecies) |
| `vmin` | float | Lower bound of the velocity grid. Optional; defaults to `-vmax` (symmetric grid). Use to specify an asymmetric velocity extent (not needed for multispecies). |
| `xmin` | float | Domain minimum x |
| `xmax` | float | Domain maximum x |
| `parallel` | `false` or list of `"x"`, `"v"` | Axes to parallelize across devices using `jax.shard_map`. `"x"` shards the `edfdv` push and collision operator along the spatial axis; `"v"` shards the `vdfdx` push along the velocity axis. Requires the corresponding dimension (`nx` or `nv`) to be divisible by the number of JAX devices. Defaults to `false`. |

The velocity grid is uniform and cell-centered: `dv = (vmax - vmin) / nv` with cell centers spanning `vmin + dv/2` to `vmax - dv/2`. An asymmetric grid (`vmin != -vmax`) is useful, for example, for a bump-on-tail distribution where less resolution is needed on one side. As with `vmax`, choose bounds wide enough that `f ~ 0` at *both* edges so that the spectral velocity push and the zero-flux collision boundary conditions remain accurate.

**Note:** For multispecies simulations, `nv`, `vmax`, and `vmin` are defined per-species in `terms.species` and the global values are not used.

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
  parallel: ["x", "v"]   # shard edfdv+collisions over x, vdfdx over v
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
    main:
      t:
        tmin: 0.0
        tmax: 100.0
        nt: 11
```

| Save Key | Description |
|----------|-------------|
| `fields` | Electric and magnetic field data |
| `<species>` | Dict of named distribution function saves for that species |
| `diag-vlasov-dfdt` | Time derivative from Vlasov operator (optional) |
| `diag-fp-dfdt` | Time derivative from Fokker-Planck operator (optional) |

### Multiple distribution saves per species

Each species section is a dict of **named saves** (`main`, `full`, `monitor`, etc.).
Multiple saves with different resolutions and time cadences can be configured under
the same species key. Each produces a `dist-<species>/<label>.nc` file in `binary/`
and is accessible as `result.ys["<species>/<label>"]`.

```yaml
save:
  fields:
    t:
      tmin: 0.0
      tmax: 100.0
      nt: 601
  electron:
    # Full-resolution dump, saved infrequently
    full:
      t:
        tmin: 0.0
        tmax: 100.0
        nt: 5
      x:
        xmin: 0.0
        xmax: 20.94
        nx: 32
      v:
        vmin: -6.4
        vmax: 6.4
        nv: 512
    # Coarser dump at higher frequency for monitoring
    monitor:
      t:
        tmin: 0.0
        tmax: 100.0
        nt: 101
      x:
        xmin: 0.0
        xmax: 20.94
        nx: 16
      v:
        vmin: -3.0
        vmax: 3.0
        nv: 128
```

Each named save contains a `t` sub-key with:

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
| `a0` | float | Normalized vector potential amplitude of the wave (see below) |
| `k0` | float | Wavenumber |
| `w0` | float | Frequency |
| `dw0` | float | Frequency offset/chirp |
| `t_center` | float | Temporal envelope center |
| `t_rise` | float | Temporal envelope rise time |
| `t_width` | float | Temporal envelope width |
| `x_center` | float | Spatial envelope center |
| `x_rise` | float | Spatial envelope rise distance |
| `x_width` | float | Spatial envelope width |
| `source_type` | string | `"extended"` (default) or `"point"`. Only affects `ey` drivers. See below. |

### Driver normalization and `a0`

The parameter `a0` has the same meaning for both `ex` and `ey` drivers: **the normalized vector potential amplitude** of the wave. The difference is how `a0` determines the forcing magnitude in each context:

- **`ex` drivers** (longitudinal electric field): The driver produces an electric field with amplitude `ω · a0`, derived from `E = -∂A/∂t`. This field enters the Vlasov equation directly.
- **`ey` drivers** (transverse wave equation source): The driver produces a source term for the wave equation. The source type controls the spatial profile:
  - `"extended"` (default): Source is `S = -ω² · a0 · envelope(x,t) · sin(kx - ωt)`, spread over the spatial envelope width. The resulting wave amplitude depends on the source geometry.
  - `"point"`: Source is concentrated at a single grid cell (nearest to `x_center`). The amplitude is calibrated using the vacuum Green's function so that the outgoing wave has amplitude `a0`. Point sources radiate equally in both directions; place the source near a boundary with absorbing BCs to get a unidirectional wave.

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
| `hou_li_filter` | object | Hou-Li spectral filter (optional, default off) |
| `species` | list | (Optional) List of species configurations for multispecies simulations |

### species (Multispecies Configuration)

For multispecies simulations, you can define multiple particle species (e.g., electrons and ions) with independent velocity grids and charge-to-mass ratios. Each species is defined as an entry in the `terms.species` list.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Species identifier (e.g., `"electron"`, `"ion"`) |
| `charge` | float | Charge in units of fundamental charge (e.g., `-1.0` for electrons, `10.0` for Z=10 ions) |
| `mass` | float | Mass in units of electron mass (e.g., `1.0` for electrons, `1836.0` for protons) |
| `vmax` | float | Upper bound of the velocity grid for this species |
| `vmin` | float | Lower bound of the velocity grid for this species. Optional; defaults to `-vmax` (symmetric grid) |
| `nv` | int | Number of velocity grid points for this species |
| `density_components` | list | List of density component names (from `density` section) that belong to this species |

**Important notes:**

- Each species can have its own velocity grid parameters (`vmax`, `vmin`, `nv`), allowing heavier species to use smaller velocity extents or asymmetric velocity bounds
- The `density_components` field maps species to their density profiles defined in the `density` section
- For single-species simulations, `terms.species` can be omitted; a default electron species will be auto-generated from the `grid.vmax`, `grid.nv`, and all `species-*` density components
- When using multispecies, `grid.vmax` and `grid.nv` are not used (each species defines its own)

#### Multispecies Example: Ion Acoustic Wave

```yaml
density:
  quasineutrality: true
  species-electron-background:
    noise_seed: 420
    noise_type: gaussian
    noise_val: 0.0
    v0: 0.0
    T0: 1.0              # Electron temperature
    m: 2.0
    basis: sine
    baseline: 1.0
    amplitude: 1.0e-3
    wavenumber: 0.1
  species-ion-background:
    noise_seed: 421
    noise_type: gaussian
    noise_val: 0.0
    v0: 0.0
    T0: 0.01             # Ion temperature (cooler than electrons)
    m: 2.0
    basis: sine
    baseline: 1.0
    amplitude: 1.0e-3
    wavenumber: 0.1

terms:
  field: poisson
  edfdv: exponential
  time: sixth
  species:
    - name: electron
      charge: -1.0
      mass: 1.0
      vmax: 6.4           # Larger velocity range for light electrons
      nv: 512
      density_components:
        - species-electron-background
    - name: ion
      charge: 10.0        # Z=10 ions
      mass: 18360.0       # Ion mass relative to electron mass
      vmax: 0.15          # Smaller velocity range for heavy ions
      nv: 256
      density_components:
        - species-ion-background
  fokker_planck:
    is_on: False
    # ...
```

See `tests/test_vlasov1d/configs/multispecies_ion_acoustic.yaml` for a complete working example.

### hou_li_filter

Hou-Li exponential spectral filter applied after each timestep. Damps high-wavenumber modes to suppress numerical oscillations without significantly affecting well-resolved physics. Can be applied in position space (x), velocity space (v), or both.

The filter kernel in Fourier space is:

```
sigma(j) = exp(-alpha * (j / N)^(2*order))
```

where `j` is the mode index, `N` is the Nyquist mode, and modes near `j = N` are damped strongly while low modes are left nearly unchanged.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `is_on` | bool | — | Enable/disable the filter |
| `alpha` | float | `36.0` | Filter strength. The default `36.0 ≈ -log(float64 machine epsilon)` ensures the Nyquist mode is zeroed to machine precision |
| `order` | int | `36` | Filter order; higher values give a sharper roll-off that preserves more low-wavenumber content |
| `dimensions` | list | `["x", "v"]` | Dimensions to filter. Can be `[]`, `["x"]`, `["v"]`, or `["x", "v"]` |

Example:
```yaml
terms:
  hou_li_filter:
    is_on: True
    alpha: 36.0
    order: 36
    dimensions: ["x", "v"]
```

If `hou_li_filter` is omitted entirely, it defaults to `is_on: False`.

### Field Solvers

- `"poisson"`: Spectral Poisson solver (works with any time integrator)
- `"ampere"`: Ampere solver (requires `"leapfrog"` time integrator)
- `"hampere"`: Hamiltonian Ampere solver (requires `"leapfrog"` time integrator)

### fokker_planck

Fokker-Planck collision operator.

| Field | Type | Description |
|-------|------|-------------|
| `is_on` | bool | Enable/disable |
| `type` | string | Collision type (see table below) |
| `m` | float | Super-Gaussian exponent of the equilibrium (only used by `type: super_gaussian`; default `2.0` = Maxwellian) |
| `self_consistent_beta` | object | Optional Newton refinement of the equilibrium shape parameter (see below) |
| `time` | object | Temporal profile |
| `space` | object | Spatial profile |

Available `type` values (case-insensitive):

| `type` | Model | Scheme | Equilibrium |
|--------|-------|--------|-------------|
| `lenard_bernstein` | Lenard-Bernstein | central differencing | Maxwellian at v=0 |
| `chang_cooper` | Lenard-Bernstein | Chang-Cooper | Maxwellian at v=0 |
| `dougherty` | Dougherty | central differencing | Maxwellian at $\bar v$ |
| `chang_cooper_dougherty` | Dougherty | Chang-Cooper | Maxwellian at $\bar v$ |
| `super_gaussian` | Super-Gaussian Dougherty | Chang-Cooper | Super-Gaussian of order `m` at $\bar v$ |

The Chang-Cooper scheme is positivity-preserving and conserves density exactly; central differencing is provided for comparison only.

#### super_gaussian

Drift-diffusion operator whose equilibrium is a super-Gaussian
$f_0 \propto \exp(-\beta\,|v-\bar v|^m)$ instead of a Maxwellian
($m=2$ recovers `chang_cooper_dougherty`). Use this to *maintain* a prescribed
non-Maxwellian order — e.g. a Langdon/DLM inverse-bremsstrahlung-heated
distribution — against collisional relaxation during Landau-damping or LPI
runs. Note it does not *generate* super-Gaussians dynamically (there is no
competition between heating and e-e collisions as in the VFP-1D
inverse-bremsstrahlung operator); it relaxes toward the prescribed order `m`.

The drift coefficient is the exact finite difference of the equilibrium
potential $\varphi = \beta|v-\bar v|^m$, so the sampled super-Gaussian is the
exact fixed point of the Chang-Cooper discretization. The shape parameter
$\beta$ is set by the energy-conservation closure
$\beta = n/(m\,\langle|v-\bar v|^m\rangle)$ (the generalization of the
Lenard-Bernstein/Dougherty $\beta = 1/(2T)$), refined by a Newton solve of the
discrete energy-flux condition when `self_consistent_beta` is enabled.

Conservation properties:

- **Density**: exact (zero-flux boundary conditions).
- **Energy**: no secular drift at equilibrium when `self_consistent_beta` is
  enabled (recommended for long collisional runs; without it the continuum
  closure leaves an O(dv²) quadrature residual that drifts T by ~5e-7 per
  collision time at nv=128 for m=3 — already negligible unless many collision
  times elapse). A single Newton step (`max_steps: 1`) reduces the drift to
  machine level. Off-equilibrium transients carry a one-time O(nu·dt) offset
  from operator splitting, negligible at production nu·dt.
- **Momentum**: exact for distributions symmetric about $\bar v$; skewed
  transients exchange momentum at O(skewness), unlike $m=2$ where
  $C \propto (v-\bar v)$ makes it exact.

Note on initialization: the super-Gaussian initializer (`m` in the species
config) uses a width convention that fixes $\langle v^4\rangle/\langle v^2\rangle$,
while this operator preserves the super-Gaussian carrying the distribution's
own variance. Any super-Gaussian of order `m` is (approximately) a fixed
point regardless of width convention, so the two compose without drift.

```yaml
terms:
  fokker_planck:
    is_on: True
    type: super_gaussian
    m: 3.0
    self_consistent_beta:
      enabled: True
      max_steps: 1   # one Newton step suffices (the closure lands within O(dv²) of the root)
    time:
      baseline: 1.0e-5
      # ...
    space:
      baseline: 1.0
      # ...
```

#### self_consistent_beta

Optional sub-object controlling the Newton refinement of the equilibrium shape
parameter β. For the Maxwellian operators it matches the *discrete* temperature
of the equilibrium to that of f (eliminating equilibrium drift in Chang-Cooper
schemes); for `super_gaussian` it solves the discrete energy-flux condition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `False` | Enable the Newton solve |
| `max_steps` | int | `3` | Maximum Newton iterations |
| `rtol` | float | `1e-8` | Relative tolerance |
| `atol` | float | `1e-12` | Absolute tolerance |

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
| `baseline` | float | Base collision frequency in code units of $\omega_{p0}$ (i.e. $\hat\nu = \nu[\mathrm{s}^{-1}]/\omega_{p0}$, a true rate — no $2\pi$). For reference, the NRL e–e rate in these units is logged as `nuee_norm` in `units.yaml`. Note the Dougherty/Lenard–Bernstein $\nu$ agrees with the NRL $\nu_{ee}$ only up to an O(1) factor |
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

### Ion Acoustic Wave (Multispecies)

See `tests/test_vlasov1d/configs/multispecies_ion_acoustic.yaml` - two-species (electron + ion) simulation demonstrating multispecies support.
