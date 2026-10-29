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

## Execution backend

`backend: jax` is the default. The optional `backend: native` runs the
electrostatic solver in C++/CUDA/MPI while retaining Python initialization,
save callbacks, xarray/NetCDF postprocessing, and `ergoExo` tracking. It has no
automatic differentiation. Custom `BaseVlasov1D` subclasses passed to
`ergoExo.setup(..., adept_module=MyModule)` are wrapped automatically.

Build the shared library on the target system (CUDA toolkit, MPI and CMake
are required). Install the Python extra with `pip install 'adept[native]'`
or your environment manager's equivalent:

```sh
cmake -S adept/_vlasov1d/native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j4
```

When ADEPT is installed in another project (such as turbulence), locate the
packaged sources in that project's Python environment:

```sh
native_source=$(python -c 'from importlib.util import find_spec; print(find_spec("adept._vlasov1d.native").submodule_search_locations[0])')
cmake -S "$native_source" -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j4
```

On Perlmutter add `-DCMAKE_CXX_COMPILER=CC`, and use the existing allocated
job's `srun` launcher. A two-node example is:

```yaml
backend: native
native:
  library: /absolute/path/build/native/libadept_vlasov.so
  ranks: 8
  workdir: /absolute/shared/scratch/path
  scalar_stride: 1
  launcher: [srun, --overlap, --nodes=2, '--ntasks={ranks}',
             --ntasks-per-node=4, --gpus-per-node=4, --gpu-bind=none,
             --cpu-bind=none]
```

Run one outer Python driver inside the allocation; the launcher starts its
workers. Set `MPICH_GPU_SUPPORT_ENABLED=1` for CUDA-aware Cray MPI. The work
directory and library must be visible on every worker node. Native execution
uses its own velocity-slab decomposition; `native.ranks` controls it.
CUDA device ordinals follow each worker's visible device list, independently
of its MPI local rank. The example exposes all four GPUs per node for
CUDA-aware Cray MPI. A launcher that exposes one GPU per worker uses ordinal
zero within each worker. On Cray systems requiring all GPUs visible for GPU
transport, use `host_mpi: true` and `MPICH_GPU_SUPPORT_ENABLED=0` to test that
visibility mode with host-staged communication.
On the tested shared A100 node, masked-worker NCCL shared-memory imports also
reported an invalid device ordinal. The validated host-transport regression
uses `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_NET=Socket` and
`NCCL_NET_GDR_LEVEL=0`. Keep the all-visible CUDA-aware example above for
the production throughput path.
If an outer Slurm step already restricted `CUDA_VISIBLE_DEVICES`, prefix the
launcher with `[env, -u, CUDA_VISIBLE_DEVICES, srun, ...]` so Slurm can apply
the new worker step's GPU binding. Separate GPU cgroups may each report
ordinal zero and `CUDA_VISIBLE_DEVICES=0`; physical GPUs still differ.

| Native option | Default | Meaning |
|---|---|---|
| `library` | unset | Shared library path; alternatively set `ADEPT_VLASOV_LIBRARY` |
| `device` | `cuda` | `cpu` selects the FFTW validation build (`-DVLASOV_CUDA=OFF`) |
| `ranks` | `1` | One MPI worker per GPU; must divide `nx` and every species' `nv` |
| `launcher` | empty | Argument list, with `{ranks}` substitution; defaults to `mpiexec -n` for multiple ranks |
| `workdir` | `BASE_TEMPDIR` or system temporary directory | Shared staging/output directory, required for multiple ranks |
| `scalar_stride` | `1` | Save scalar diagnostics every this many actual timesteps, plus both endpoints |
| `custom_save_times` | `preserve` | Preserve subclass callback times; `steps` rounds those too |
| `halo_capacity` | `64` | Maximum velocity halo width; exceeding it fails explicitly |
| `host_mpi` | `false` | Stage MPI messages through host memory for MPI without CUDA support |
| `timeout` | unset | Worker timeout in seconds |
| `velocity_grids` | `{}` | Per-species explicit edges or coarse/fine refinement; requires `conservative-cubic-spline` (below) |
| `velocity_boundary` | `zero-flux` | Conservative spline boundaries: `zero-flux` or `periodic` |
| `boundary_tolerance` | `1e-8` | Positive finite tolerance for tail and attempted-escape fractions with zero-flux conservative splines |
| `boundary_action` | `warn` | `warn` reports a small velocity domain; `error` aborts the run |

Standard field/distribution save times round to the nearest timestep and
duplicate times are merged. Outputs record actual times in `result.ts` and
`cfg['save'][key]['t']['ax']`; the number of frames can therefore change.
The native backend streams saved outputs into host memory-mapped files instead
of retaining the full time history in GPU memory. Sparse custom callbacks
preserve their requested times by default for existing project postprocessors;
off-step callbacks require a previous-state GPU copy and interpolation. Use
`custom_save_times: steps` when the postprocessor reads actual result times.
Custom postprocessors that rebuild a uniform scalar axis must also use actual
times when the final scalar interval is shorter than `scalar_stride`.

Kinetic electrons and ions use the existing `terms.species` and `density`
configuration below. Each species has its own cell-centred velocity
grid, charge and mass. All species contribute charge to the shared Poisson
solve at every splitting stage and advance with their own charge-to-mass ratio.
With no explicit species list, the usual electron species and static background
are retained. No additional ion parameters or preprocessing are required.

Supported numerics are `time: sixth`, `edfdv: cubic-spline` or
`conservative-cubic-spline`, spectral spatial
streaming and `field: poisson`. The cubic operator reproduces ADEPT's local C1
Hermite interpolation. Optional collisions are constant-rate
`chang_cooper_dougherty` with analytic beta, applied to the reference species
(electron if present, otherwise the first species), matching the JAX path.
Inter-species collisional exchange is not implemented. Drivers, Krook relaxation,
Hou–Li filtering, evolved df/dt diagnostics and trainable modules are rejected.

This backend uses the `ergoExo`/`BaseVlasov1D` lifecycle. The explicit
`Vlasov1DBuilder` prepares JAX programs and rejects `backend: native`; it does
not silently fall back to JAX. Rebuild the shared library after updating ADEPT;
incompatible native library ABIs fail with an actionable error.

### Native conservative splines and nonuniform velocity grids

Select `terms.edfdv: conservative-cubic-spline` for a primitive-spline remap based on
[Afeyan et al. (2014), sections 2.1–2.2](https://arxiv.org/abs/1409.2086).
This treats the distribution as velocity cell averages. By default it constructs
a natural C2 cubic spline of cumulative mass on the velocity cell edges and
evaluates swept integrals at interior faces. The two outer numerical fluxes
are set to zero, so their divergence telescopes and conserves `sum(f * dv)` to
roundoff without mass renormalization. The primitive continues as a constant
outside the domain. The natural spline endpoint conditions and zero numerical
boundary flux are separate conditions. The existing sixth-order time splitting
and spectral spatial streaming remain in use.

Set `native.velocity_boundary: periodic` to use the paper's periodic remap:
spline the mean-subtracted primitive, then difference translated edge values.
The mean is `sum(f * dv) / (vmax - vmin)`, ensuring a periodic primitive on domains
of any length. Particles crossing an edge then re-enter at the opposite edge.

For example, add these settings to a collisionless native configuration:

```yaml
backend: native
grid:
  nv: 512
  vmin: -8.0
  vmax: 8.0
  # ... spatial grid and time settings ...
terms:
  time: sixth
  edfdv: conservative-cubic-spline
  field: poisson
  # ... fokker_planck and krook disabled ...
native:
  library: /absolute/path/build/native/libadept_vlasov.so
  velocity_boundary: zero-flux
  boundary_tolerance: 1.0e-8
  boundary_action: error
  velocity_grids:
    electron:
      refined_min: 1.2
      refined_max: 1.6
      ratio: 32
```

The refinement preserves the configured `nv` total cells; `ratio` is an integer
at least two. Requested endpoints snap to coarse-grid edges. Inspect the
resolved `cfg['grid']['species_grids'][name]['v_edges']` for the actual interval.
An interval that cannot be resolved at the requested `nv`/ratio is rejected.
Alternatively use `native.velocity_grids.<species>.edges: [...]`, containing
exactly `nv + 1` finite, increasing edges spanning that species' `vmin`/`vmax`.
Do not combine explicit edges and refinement parameters. Omitted species use
uniform cells. Multispecies bounds and cell counts still come from `terms.species`.
The conservative method supports asymmetric domains; `cubic-spline` continues
to require symmetric uniform grids.

Initialization integrates each super-Gaussian component over its velocity cells
and normalizes it using the cell widths. Field solves and standard moments use
those same widths. Full-resolution distribution NetCDF files include velocity
cell bounds and a `velocity_representation: cell_average` attribute. Interpolated
saves remain samples and should not be used to measure conserved mass. Custom
callbacks must reduce `sum(f * species_grid['dv'], axis=1)`; `dv` is now a vector
for this method. Moments use cell-center velocities, so energy and momentum are
diagnostics rather than exactly conserved quantities of the velocity remap.

Zero-flux boundaries retain all mass, but they impose an artificial velocity
cap if appreciable population reaches either boundary. Keep both edges well
into the tails throughout the run. For Maxwellians, a useful starting point is
at least eight thermal standard deviations `sqrt(T0 / mass)` beyond the drift
of every component. Allow extra room for acceleration and heating. Use coarse
outer cells to widen the domain affordably, retaining the required spacing in
the refined region; increasing the extent at fixed `nv` can coarsen that region.
Confirm results by widening the domain while holding interior resolution fixed.
Bounds are never silently changed.

The default checks compare against `native.boundary_tolerance: 1e-8`:

- Initial analytic mass omitted beyond either edge, before normalization, for
  each super-Gaussian component (the maximum across components is reported).
- Population in the outer 5% of velocity extent on each side, using cell-overlap
  widths and `abs(f)`, divided by each spatial row's absolute mass. The initial
  value and maximum before/after every velocity kick are recorded.
- Attempted transport blocked at the outer faces: the maximum fraction per
  kick and cumulative absolute attempted mass on each side. Cumulative mass is
  also divided by the initial total absolute mass. This counts every kick,
  including negative sixth-order substeps, and is a domain diagnostic, not
  mass actually lost.

`boundary_action: warn` emits warnings; `error` rejects an initially narrow
domain or aborts when a runtime check fails. Runtime checks are acted on at
save intervals and the end of the solve, using maxima/cumulative values from
all intervening kicks. Results are in `native stats.velocity_boundaries` by
species and in the saved `native/stats.json`. Periodic and local cubic runs
do not use these checks. The unlimited cubic spline can undershoot; it does
not preserve positivity, even with zero flux.

The global spline solve uses the MPI transpose layout and caches its matrix
factors; `halo_capacity` has no effect on this path. Collisions are rejected
because the current Chang–Cooper implementation assumes uniform spacing.
Drivers and the other native restrictions above still apply. This option is
native-only; rebuild the library (ABI 5) before using it.

## units

Physical unit normalizations for the simulation.

| Field | Type | Description |
|-------|------|-------------|
| `normalizing_temperature` | string | Reference temperature with unit, e.g., `"2000eV"` |
| `normalizing_density` | string | Reference density with unit, e.g., `"1.5e21/cc"` |
| `reference` | string | Reference species for the normalization: `"electron"` (default) or `"ion"` |
| `A` | float | Ion mass number, $m_i = A\,m_p$ (only used with `reference: ion`; default `1.0`) |
| `Z` | float | Ion charge state, $q_i = Z e$ (only used with `reference: ion`; default `1.0`) |

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

### Ion reference (`reference: ion`)

For runs where only ions are evolved kinetically (e.g. `terms.field:
poisson-boltzmann`), set `reference: ion` to build the normalization from the ion
species instead:

| Unit | Definition | Meaning |
|------|-----------|---------|
| time | $\tau = 1/\omega_{pi}$, $\omega_{pi} = \sqrt{n_i Z^2 e^2/(\epsilon_0 m_i)}$ | inverse ion plasma frequency |
| velocity | $v_0 = \sqrt{T_0/m_i}$ | ion thermal speed (same σ convention) |
| length | $L_0 = v_0/\omega_{pi}$ | ion Debye length |

with $m_i = A\,m_p$ and charge $Z e$; `normalizing_density` and
`normalizing_temperature` are then the ion density $n_i$ and ion temperature
$T_i$. Dimensional string inputs (e.g. `xmax: 300um`, `tmax: 50ps`) convert with
these ion units, and the physical quantities logged by `write_units()` (`wp0`,
`tp0`, `x0`, `v0`, ...) refer to the ion reference. The electron-specific
collision entries (`nuee`, `nuee_norm`, `logLambda_ee`) are replaced by their
ion-ion counterparts (`nuii`, `nuii_norm`, `logLambda_ii`, from the NRL
formulary).

Example (deuterium):
```yaml
units:
  normalizing_temperature: 100eV   # T_i
  normalizing_density: 1.0e20/cc   # n_i
  reference: ion
  A: 2.0
  Z: 1.0
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
| `parallel` | `false` or list of `"x"`, `"v"` | Axes to parallelize across devices using `jax.shard_map`. `"x"` shards the `edfdv` push and collision operator along the spatial axis; `"v"` shards the `vdfdx` push along the velocity axis. Requires the corresponding dimension (`nx` or `nv`) to be divisible by the number of JAX devices. Defaults to `false`. See [Running on Multiple GPUs](overview.md#running-on-multiple-gpus) for the memory model, the launch command, and the reverse-mode AD caveat on `"v"`. |

By default the velocity grid is uniform and cell-centered: `dv = (vmax - vmin) / nv` with cell centers spanning `vmin + dv/2` to `vmax - dv/2`. The native conservative spline method additionally supports [nonuniform grids](#native-conservative-splines-and-nonuniform-velocity-grids). An asymmetric grid (`vmin != -vmax`) is useful, for example, for a bump-on-tail distribution where less resolution is needed on one side. As with `vmax`, choose bounds wide enough that `f ~ 0` at *both* edges so that the spectral velocity push and the zero-flux collision boundary conditions remain accurate.

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
| `ex_stochastic` | object | (Optional) Band-limited, time-correlated stochastic forcing of the longitudinal field (see below) |

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

### Stochastic forcing: `ex_stochastic`

Drives the plasma with an external longitudinal electric field built from a set
of periodic-box Fourier modes whose complex amplitudes evolve as independent
Ornstein-Uhlenbeck (OU) processes. This is the standard "compressive,
time-correlated" forcing used for driven-turbulence simulations (e.g. ion-acoustic
turbulence with Boltzmann electrons):

```
dE(x, t) = sum_m Re[ a_m(t) exp(i k_m x) ],   k_m = 2 pi m / L
```

Each `a_m(t)` has correlation time `tau` and stationary RMS `amplitude`. The
realization is drawn once from `seed` on a uniform time grid of spacing
`dt_update` and linearly interpolated in time, so a run is exactly reproducible
and independent of the solver timestep.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `modes` | list[int] | `[1]` | Box mode numbers m to force (k_m = 2πm/L) |
| `amplitude` | float | — | Stationary RMS of each mode's complex amplitude (code-unit E field) |
| `tau` | float | — | OU correlation time (code units) |
| `seed` | int | `42` | RNG seed for the forcing realization |
| `dt_update` | float | `tau/10` | Time resolution of the precomputed OU series (clamped to at most `tau/2`) |

Example (critically balanced box-scale driving — the correlation time is the
ion thermal box-crossing time `tau = L/v_ti` and the amplitude is `v_ti^2/L`,
so the velocity kick per correlation time is ~`v_ti`; here `v_ti = 1`,
`L = 458`):
```yaml
drivers:
  ex: {}
  ey: {}
  ex_stochastic:
    modes: [1]
    amplitude: 2.1834e-3   # v_ti^2 / L
    tau: 458.0             # L / v_ti
    seed: 42
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
| `field` | string | Electric field solver: `"poisson"`, `"poisson-boltzmann"`, `"ampere"`, or `"hampere"` |
| `edfdv` | string | Velocity advection: `"exponential"`, `"cubic-spline"`, `"lagrange7"`, `"pfc3"`, `"sl-weno5"`, or native-only `"conservative-cubic-spline"` |
| `vdfdx` | string | Spatial advection: `"exponential"` (default), `"pfc3"`, or `"sl-weno5"` |
| `time` | string | Time integrator: `"sixth"` (6th order Hamiltonian), `"leapfrog"`, or `"strang"` (explicit second-order electrostatic splitting) |
| `fokker_planck` | object | Fokker-Planck collision operator configuration |
| `krook` | object | Krook collision operator configuration |
| `hou_li_filter` | object | Hou-Li spectral filter (optional, default off) |
| `species` | list | (Optional) List of species configurations for multispecies simulations |
| `boltzmann_electrons` | object | (Optional) Linearized Boltzmann electron closure parameters, used with `field: poisson-boltzmann` |

### Strang splitting with degree-7 velocity interpolation

Select these options in an existing Vlasov-1D configuration:

```yaml
terms:
  field: poisson  # poisson-boltzmann is also supported
  time: strang
  edfdv: lagrange7
  # Retain the configuration's species, collision, and other term settings.
```

`strang` streams in space for half a timestep, evaluates the self-consistent field,
pushes velocity for a full timestep using midpoint external forcing, then streams
for the remaining half timestep. There is one velocity remap per full timestep.
Saved self-consistent and external electric fields are evaluated at the end of
the step. This option supports `poisson` and `poisson-boltzmann`; `ampere` and
`hampere` retain their existing `leapfrog` requirement.

The second-order description applies to the collisionless electrostatic step.
The existing collision, filter, and transverse-wave coupling is unchanged; choosing
`strang` does not raise the order of those separate updates.

`lagrange7` uses eight velocity samples to interpolate a degree-7 polynomial. It
supports species-specific uniform grids (including asymmetric bounds), requires
at least eight velocity cells per species, and works with the existing `grid.parallel`
options. It can also be selected independently with the other time integrators.
The stencil uses `1e-30` for samples outside the velocity grid and returns `1e-30`
for departure points outside the first/last cell centers, without periodic wraparound.
Keep populated structures clear of the velocity edges. The interpolation is unlimited:
it can produce negative values near unresolved structure and does not renormalize
mass lost through the boundaries. Gradients with respect to displacement are
piecewise smooth as the stencil changes between cells.

### Positive conservative semi-Lagrangian advection

Select either `pfc3` or `sl-weno5` independently in each advection direction:

```yaml
terms:
  field: poisson
  time: sixth
  vdfdx: sl-weno5  # or pfc3; omission retains exponential
  edfdv: sl-weno5  # or pfc3
  hou_li_filter:
    is_on: false
  # Retain the configuration's species, collisions and other term settings.
```

These are advection pushers, not new time splitters. Both work with `sixth`,
`strang`, and `leapfrog`, including negative splitting stages. A displacement
of 2.46 cells is an exact two-cell offset followed by a 0.46-cell conservative
remap; it does not trigger subcycling. Positivity adds no advection CFL
restriction. Field evolution, splitting accuracy, collisions and any transverse
wave solver still constrain the usable timestep. Selecting `sixth` does not
establish sixth-order accuracy of the complete discretized and limited scheme.

- **PFC3** uses a quadratic reconstruction whose cell mean is preserved while its
  deviations are scaled to make the polynomial nonnegative throughout the cell.
  It requires at least three cells along the remapped direction and has third-order
  spatial accuracy on smooth, resolved data when the limiter is inactive.
- **SL-WENO5** integrates three quadratic reconstructions with displacement-dependent
  optimal weights and WENO-Z nonlinear weights. Its smooth spatial order is five;
  it requires at least five cells. An outgoing-mass limiter bounds the fractional
  mass leaving each donor between zero and that donor's mass. This local limiter
  guarantees positivity, but does not enforce a local maximum principle. It is not
  an implementation of the full parametrized MPP limiter in Xiong et al.

Both evolve **cell averages** in an `(nx, nv)` array. The existing initializer's
center samples are interpreted as midpoint approximations to these averages;
initialization and existing moment/field quadrature can therefore limit overall
accuracy even when the remap is high order. For a formal spatial convergence
study, supply accurately integrated initial cell averages. There is no automatic
conversion of a point-sampled restart. Ordinary sums times `dx`/`dv` conserve
the represented mass; exact momentum, energy or Casimir conservation is not implied.

Spatial boundaries are periodic. Velocity boundaries use **zero inflow**, including
partial cells at the domain edges, and discard escaped mass without renormalization
or periodic wraparound. This differs from the `1e-30` exterior point values used by
the interpolation pushers. Keep populated structures clear of velocity boundaries
when measuring closed-system conservation or using negative splitting stages.

For finite, nonnegative input cell averages, each remap returns nonnegative cell
averages and preserves mass up to roundoff, except for velocity boundary loss.
It does not repair an already negative checkpoint. To retain this property through
the collisionless split update, select positive pushers in **both** directions and
disable Hou-Li filtering. Mixing in spectral/interpolation advection, filtering, or
other operators without a positivity guarantee removes the full-step guarantee.
The `hampere` field solver requires `vdfdx: exponential`, since its integrated
current is tied to spectral streaming; incompatible selections are rejected.

The implementation uses compact per-line shifts and local gathers, without global
prefix sums or data-dependent loops. Float32 and float64 are supported; half
precision is rejected because the nonlinear weights can overflow. Existing species-specific velocity grids,
ponderomotive forces and `grid.parallel: ["x", "v"]` are supported. Autodiff is
piecewise smooth away from stencil and limiter switches. A100 performance and
nonlinear turbulence accuracy at production timesteps require separate measurement.

Background: [Filbet, Sonnendrücker and Bertrand (2001), conservative/PFC methods](https://doi.org/10.1006/jcph.2001.6818),
[Borges et al. (2008), WENO-Z](https://doi.org/10.1016/j.jcp.2007.11.038), and
[Xiong et al. (2014), positive conservative SL-WENO flux limiting](https://arxiv.org/abs/1401.0076).

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

### boltzmann_electrons (kinetic ions with adiabatic electrons)

With `field: poisson-boltzmann`, only ions are evolved kinetically; the electrons
respond adiabatically through the linearized Boltzmann (screened Poisson) closure

```
e phi / T_e = delta n_i / n_0 + lambda_De^2 d^2/dx^2 (e phi / T_e)
```

solved spectrally as `e phi_k / T_e = (delta n_k / n_0) / (1 + k^2 lambda_De^2)`.
The k=0 force vanishes identically, so no static neutralizing background is
needed. This is the standard setup for ion-acoustic wave and IAW-turbulence
simulations: it removes all electron timescales from the problem, so the
timestep is limited only by ion dynamics.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Te` | float | `1.0` | Electron temperature in code units (units of `normalizing_temperature`) |
| `lambda_De` | float | `sqrt(Te / n_0)` | Screening length in code units. Omit for the physically consistent value; set explicitly to move the screening scale (e.g. to the grid spacing, mimicking PIC references where the cell size takes the place of the Debye length); `0.0` gives the quasineutral closure `e phi / T_e = delta n / n_0` |

**Requirements:** `terms.species` must be given explicitly and all kinetic
species must have positive charge (electrons are the closure, not a species).

**Ion-unit convention:** with a single ion species at `charge: 1.0, mass: 1.0`
and `T0: 1.0`, code time is `1/omega_pi`, velocity is the ion thermal speed
`v_ti`, and length is `v_ti/omega_pi`. Then `Te` is the electron-to-ion
temperature ratio `Te/Ti`, the sound speed is `c_s = sqrt(Te)`, and the electron
Debye length is `lambda_De = sqrt(Te)` in code units. Set `units.reference: ion`
(see [units](#units)) so that dimensional string inputs and the physical
quantities reported by `write_units()` follow this same ion convention — with
the default electron reference they are electron-referenced and therefore
nominal for such runs.

Example (ion-acoustic turbulence, `Te/Ti = 0.05`):
```yaml
terms:
  field: poisson-boltzmann
  boltzmann_electrons:
    Te: 0.05
  edfdv: cubic-spline
  time: sixth
  species:
    - name: ion
      charge: 1.0
      mass: 1.0
      vmax: 6.4
      nv: 512
      density_components:
        - species-ion-background
```

See `configs/vlasov-1d/iaw-turbulence.yaml` for a complete working example with
box-scale stochastic forcing, and `tests/test_vlasov1d/configs/boltzmann_iaw.yaml`
for an ion-acoustic dispersion test setup.

### solver: vlasov-1d-iaw (IAW turbulence module)

`solver: vlasov-1d-iaw` selects `IAWTurbulence1D`, a problem-specific module for
driven ion-acoustic turbulence that extends the base Vlasov-1D module with

- an **nk save stream**: the low-|k| complex spectrum of the total kinetic
  charge density, sampled densely in time (each sample is only
  `2 * (nk_modes + 1)` floats, so it can be saved far more often than full
  moment profiles). Written to `binary/nk.nc` with real/imag parts and
  `P = |n_k|^2`.
- **IAW post-processing**: `plots/iaw/density_spectrum.png` (late-window-averaged
  P(m) with a k*lambda_De axis and m^-2 / m^-3/2 guides),
  `plots/iaw/nk_spectrogram.png` (P(m, t)), and `phase_space_dfx.png`
  (f - <f>_x panels) for every configured distribution save.

Options live in an optional top-level `iaw_diagnostics` block:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nk_modes` | int | `1024` | Number of box modes in the nk stream (clipped to `nx // 2`) |
| `nk_nt` | int | `2001` | Number of nk time samples |
| `spectrum_window` | [float, float] | `[0.5, 1.0]` | Averaging window for the spectrum plot, as fractions of `tmax` |

```yaml
solver: vlasov-1d-iaw

iaw_diagnostics:
  nk_modes: 2048
  nk_nt: 4001
  spectrum_window: [0.4, 1.0]
```

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
- `"poisson-boltzmann"`: Spectral screened-Poisson solver for kinetic ions with linearized Boltzmann electrons (works with any time integrator; see `boltzmann_electrons` above)
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
| `dougherty_nodrag` | Dougherty (diffusion of the deviation) | central differencing | Maxwellian at $\bar v$ |
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

#### dougherty_nodrag

Diffusion-of-the-deviation operator
$C[f] = \nu\, T\, \partial_v^2 \left(f - f_M[n, \bar v, T]\right)$.
It uses the same moments machinery as `dougherty`, but the implicit solve runs
with the drag coefficient zeroed (pure diffusion) and the diffusion of the
self-consistent Maxwellian is subtracted explicitly using the same discrete
zero-flux stencil, so $f_M$ is a discrete fixed point and $n$, $P$, and $E$
are conserved. Because the operator carries no drag acting on the deviation,
it is parity-preserving in $v - v_\phi$ — useful as a control for isolating
the role of collisional drag (e.g. in ratchet/pawl-style transport
experiments) while retaining Maxwellian-preserving velocity diffusion.

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

### Ion-Acoustic Turbulence (Boltzmann Electrons)

See `configs/vlasov-1d/iaw-turbulence.yaml` - kinetic ions with the linearized Boltzmann electron closure (`field: poisson-boltzmann`), driven by box-scale time-correlated stochastic forcing (`drivers.ex_stochastic`).
