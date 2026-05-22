# Vlasov-2D Configuration Reference

The `vlasov-2d` solver evolves the 2D2V Vlasov–Maxwell system

```
∂f/∂t + v·∇_x f + (q/m)(E + v × B)·∇_v f = C[f]
∂Ex/∂t =  c² ∂Bz/∂y − Jx
∂Ey/∂t = −c² ∂Bz/∂x − Jy
∂Bz/∂t = ∂Ex/∂y − ∂Ey/∂x
```

with `f(x, y, vx, vy)` on a periodic 2D box. Time stepping is Strang-split:

1. ½ x-streaming (spectral)  →  ½ y-streaming (spectral)
2. Velocity push: ½ E_x → ½ E_y → full Bz-rotation (2D SL) → ½ E_y → ½ E_x
3. ½ y-streaming  →  ½ x-streaming
4. Spectral Maxwell update with currents `(Jx_self + Jx_driver, Jy_self + Jy_driver)`
5. Collisions (Dougherty FP, Krook) and optional Hou–Li filter

## Top-level keys

| Key | Required | Description |
| --- | --- | --- |
| `solver` | yes | Must be `vlasov-2d`. |
| `units` | yes | Plasma normalization. |
| `density` | yes | Density-component definitions. |
| `grid` | yes | Spatial and temporal grid. |
| `terms` | no | Time-integration & physics toggles. |
| `drivers` | no | External EM current drivers. |
| `save` | yes | Output / diagnostic save points. |
| `mlflow` | yes | Experiment name and run name. |

## `units`

```yaml
units:
  laser_wavelength: 351nm    # optional
  normalizing_temperature: 2000eV
  normalizing_density: 1.5e21/cc
```

Sets `n0`, `T0`, `v0 = √(T0/m_e)`, `λ_D`, `ωp0`. Lengths in the config are in
Debye lengths, times in `1/ωp0`, velocities in `v0`, fields normalized to
`m_e v0 ωp0 / e`.

## `grid`

```yaml
grid:
  dt: 0.1
  nx: 32
  ny: 16
  nvx: 64
  nvy: 64
  tmin: 0.0
  tmax: 60.0
  vmax: 6.0
  xmin: 0.0
  xmax: 20.94
  ymin: -10.0
  ymax: 10.0
  parallel: false
```

`dt` will be capped to `0.5 * min(dx, dy) / c_norm` whenever any EM driver is
enabled (Maxwell CFL).

## `density`

Like the 1D solver, each `species-<name>` key under `density` describes one
additive component that contributes a density profile and a bi-supergaussian
in (vx, vy):

```yaml
density:
  quasineutrality: true
  species-background:
    noise_type: gaussian       # uniform | gaussian | none
    noise_val: 0.0
    noise_seed: 420
    v0x: 0.0                   # drift in vx
    v0y: 0.0                   # drift in vy
    T0: 1.0                    # temperature
    m: 2.0                     # supergaussian order; 2 = Maxwellian
    basis: sine                # uniform | sine
    baseline: 1.0
    amplitude: 1.0e-3          # for `basis: sine`
    wavenumber-x: 0.3
    wavenumber-y: 0.0
    # Optional spatial masks (multiplied in):
    space-x: {center: 10.0, rise: 1.0, width: 1e6, baseline: 0.0, bump_height: 1.0}
    space-y: {center: 0.0,  rise: 1.0, width: 1e6, baseline: 0.0, bump_height: 1.0}
```

For multi-species runs add a `terms.species` list of explicit `SpeciesConfig`
entries (name, charge, mass, vmax, nvx, nvy, density_components).

## `terms`

```yaml
terms:
  edfdv: exponential    # exponential | sl   (sl = 2D cubic interpolation)
  vdfdx: exponential    # exponential (only option)
  fokker_planck:
    is_on: false
    type: dougherty
    time:    {center: 0.0, rise: 1.0, width: 1e6, baseline: 0.0, bump_height: 1.0}
    space-x: {center: 0.0, rise: 1.0, width: 1e6, baseline: 0.0, bump_height: 1.0}
    space-y: {center: 0.0, rise: 1.0, width: 1e6, baseline: 0.0, bump_height: 1.0}
  krook:
    is_on: false
    time:    {...}      # same shape as fokker_planck
    space-x: {...}
    space-y: {...}
  hou_li_filter:
    is_on: false
    alpha: 36.0
    order: 36
    dimensions: ["x", "y", "vx", "vy"]
```

- The Dougherty FP relaxes `f` toward a local Maxwellian centred at the local
  mean velocity, separable in vx then vy (Lie split). Conserves density,
  momentum, and energy along each axis exactly.
- The Krook operator drags `f` toward `n(x,y) · M(vx) · M(vy)` with unit
  thermal speed.
- The Hou–Li filter is separable; pick any subset of `{x, y, vx, vy}`.

## `drivers`

External EM current sources. Each driver adds
`J = −w² · a0 · env(x, y, t) · sin(kx·x + ky·y − w·t)` to either Jx or Jy:

```yaml
drivers:
  ex: {}
  ey:
    "0":
      params: {a0: 1.0e-5, k0x: 0.0, k0y: 0.5, w0: 1.5, dw0: 0.0}
      envelope:
        time:    {center: 5.0, rise: 1.0, width: 30.0, baseline: 0.0, bump_height: 1.0}
        space-x: {center: 10.0, rise: 1.0, width: 1e6, baseline: 0.0, bump_height: 1.0}
        space-y: {center: 0.0,  rise: 1.0, width: 1e6, baseline: 0.0, bump_height: 1.0}
      polarization: y      # "x" or "y"
```

The driver source enters Maxwell as an additional current at time `t + dt/2`.

## `save`

```yaml
save:
  fields:
    t: {nt: 32}              # only {t} is supported for fields right now
  electron:
    snapshots:
      t:  {nt: 5}
      x:  {xmin: 0.0,  xmax: 20.94, nx: 16}
      y:  {ymin: -10.0, ymax: 10.0, ny: 8}
      vx: {vxmin: -6.0, vxmax: 6.0, nvx: 32}
      vy: {vymin: -6.0, vymax: 6.0, nvy: 32}
```

- `fields` writes species moments (n, ux, uy, Txx, Tyy, T) plus shared fields
  (Ex, Ey, Bz, Jx_driver, Jy_driver).
- Distribution saves under a species name accept either `{t}` (full
  resolution) or `{t, x, y, vx, vy}` (linear interpolation onto a coarser
  output grid). 4D output files are large; the coarse grid keeps things sane.

## `mlflow`

```yaml
mlflow:
  experiment: vlasov-2d-tests
  run: my-run-name
```

## Example: 2D Landau damping

```yaml
solver: vlasov-2d
units: {normalizing_temperature: 2000eV, normalizing_density: 1.5e21/cc}
density:
  quasineutrality: true
  species-background:
    v0x: 0.0
    v0y: 0.0
    T0: 1.0
    m: 2.0
    basis: sine
    baseline: 1.0
    amplitude: 1.0e-3
    wavenumber-x: 0.3
    wavenumber-y: 0.0
grid:
  dt: 0.1
  nx: 32
  ny: 16
  nvx: 64
  nvy: 64
  tmax: 60.0
  vmax: 6.0
  xmin: 0.0
  xmax: 20.94
  ymin: -10.0
  ymax: 10.0
terms: {edfdv: exponential, vdfdx: exponential}
drivers: {ex: {}, ey: {}}
save:
  fields: {t: {nt: 32}}
  electron:
    snapshots:
      t: {nt: 5}
mlflow: {experiment: vlasov-2d-tests, run: landau-damping}
```

A full template ships in `configs/vlasov-2d/base.yaml`.
