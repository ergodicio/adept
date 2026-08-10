# Two-Fluid-1D Configuration Reference

The `tf-1d` solver evolves a 1D electrostatic two-fluid (electron + ion) system
closed with a kinetic dispersion table and an optional particle-trapping model:

```
∂n/∂t + ∂x(n u) = 0
∂u/∂t + u ∂x u = −(∂x p)/n − (q/m) E + ν_L[u, E, δ]
∂p/∂t + u ∂x p + γ p ∂x u = −2 n u (q/m) E
∂δ/∂t = −v_φ ∂x δ + Γ(E) |E| / (1 + δ²)
```

with `p ≡ P/m` the mass-normalized pressure. The field is obtained spectrally
from the charge density, `E_k = i (q_i n_{i,k} + q_e n_{e,k}) / k`.

The Landau damping term `ν_L` is interpolated from a complex-frequency table
(see `adept.electrostatic.get_complex_frequency_table`) and may be reduced
locally by the trapping variable `δ`, whose growth rate `Γ` comes from the
optional learned closure (see [`models`](#models)).

Example configs live in `configs/tf-1d/`.

## Top-level keys

| Key | Required | Description |
| --- | --- | --- |
| `solver` | yes | Must be `tf-1d`. |
| `mlflow` | yes | Experiment name and run name. |
| `units` | yes | Plasma normalization. |
| `grid` | yes | Spatial and temporal grid. |
| `save` | yes | Output / diagnostic save points. |
| `physics` | yes | Per-species fluid and closure parameters. |
| `drivers` | yes | External electrostatic drivers. |
| `models` | no | Learned-closure (neural network) specification. |
| `adjoint` | no | Adjoint method hint. Currently unused by the solver. |

## `mlflow`

```yaml
mlflow:
  experiment: tf1d-epw-test
  run: test
```

## `units`

```yaml
units:
  normalizing_temperature: 2000eV
  normalizing_density: 1.5e21/cc
```

Sets `n0`, `T0`, `v0 = √(T0/m_e)`, `ωp0`, and `x0 = v0/ωp0`. Lengths in the
config are in Debye lengths and times in `1/ωp0`.

## `grid`

```yaml
grid:
  nx: 16
  xmin: 0.0
  xmax: 20.94
  tmin: 0.0
  tmax: 500.0
```

| Field | Type | Description |
| --- | --- | --- |
| `nx` | int | Number of spatial cells (periodic, spectral derivatives). |
| `xmin`, `xmax` | float | Box extent in Debye lengths. |
| `tmin`, `tmax` | float | Simulation time window in `1/ωp0`. |

`dx`, `dt = 0.05 dx`, and `nt` are derived at runtime; `tmax` is snapped to a
whole number of steps and the step count is capped at `1e6`.

## `save`

```yaml
save:
  t:
    tmin: 0.5
    tmax: 500.0
    nt: 1000
  x:
    xmin: 0.0
    xmax: 20.94
    nx: 16
  kx:            # optional
    kxmin: 0.0
    kxmax: 0.3
    nkx: 2
```

| Block | Required | Fields | Description |
| --- | --- | --- | --- |
| `t` | yes | `tmin`, `tmax`, `nt` | Save times. All three are required — `tf-1d` does not fall back to the `grid` values. |
| `x` | yes | `xmin`, `xmax`, `nx` | Real-space interpolation grid for the saved fields. |
| `kx` | no | `kxmin`, `kxmax`, `nkx` | k-space diagnostic. Omit the block entirely to skip it. |

## `physics`

Both `ion` and `electron` blocks are required, even when a species is frozen
with `is_on: false`.

```yaml
physics:
  electron:
    is_on: true
    landau_damping: true
    mass: 1.0
    charge: -1.0
    T0: 1.0
    gamma: kinetic
    trapping:
      is_on: true
      model: zk
      kld: 0.3
      nuee: 1.0e-7
      nn: 8|8
  ion:
    is_on: false
    landau_damping: false
    mass: 1836.0
    charge: 1.0
    T0: 1.0
    gamma: 3
    trapping:
      is_on: false
      kld: 0.3
      nuee: 1.0e-9
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `is_on` | bool | — | Evolve this species. When false, `dn/dt = du/dt = dP/dt = 0`. |
| `landau_damping` | bool | — | Apply the tabulated Landau damping rate `ω_i(k)` to the momentum equation. |
| `mass` | float | — | Species mass normalized to `m_e`. |
| `charge` | float | — | Species charge normalized to `e` (electrons are `-1.0`). |
| `T0` | float | — | Initial temperature normalized to the reference `T0`. |
| `gamma` | int \| float \| str | — | Adiabatic index. The literal string `kinetic` uses the kinetic EPW dispersion table for the restoring force instead of a fixed `γ`. |
| `trapping` | block | — | Particle trapping model, see below. |

### `physics.<species>.trapping`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `is_on` | bool | — | Evolve the trapping variable `δ` and modify the damping term. |
| `kld` | float | — | `kλ_D` at which the trapping model is evaluated (sets `v_φ` and the model normalization). |
| `model` | `none` \| `zk` \| `delta` | `none` | Damping-reduction model. Only read when `is_on` is true. `delta` scales the damping by `1/(1 + δ²)`; `zk` uses the Zakharov–Karpman collision frequency (work in progress); `none` leaves the damping unmodified. |
| `nuee` | float | `null` | Electron–electron collision frequency normalized to `ωp0`. Required in practice when `is_on` is true — it sets the trapping growth-rate normalization. |
| `nn` | str | `null` | Legacy neural-network shape hint (e.g. `8|8`). Accepted for backwards compatibility with the shipped configs but not read by the solver. |

```{note}
`trapping.is_on: true` with no `model` key falls back to `none`, meaning `δ` is
evolved but does not feed back on the damping rate. Set `model` explicitly if
you want trapping to modify the physics.
```

```{warning}
`trapping.is_on: true` is currently **not runnable**. `ParticleTrapper` reads
the growth-rate network from `args["nu_g"]`, but nothing in the live code path
builds it from the `models` block, so the solve fails with `KeyError: 'nu_g'`
on the first step. Only `trapping.is_on: false` configs run end to end today.
```

## `drivers`

```yaml
drivers:
  ex:
    "0":
      k0: 0.3
      w0: 1.1598
      dw0: 0.0
      t_c: 80
      t_w: 100
      t_r: 20
      x_c: 600
      x_w: 800
      x_r: 80
      a0: 4.e-3
```

`drivers.ex` is a dictionary of independently-enveloped electrostatic drivers
keyed by a string index; their contributions are summed.

| Field | Type | Description |
| --- | --- | --- |
| `k0` | float | Wavenumber in `1/λ_D`. |
| `w0` | float | Angular frequency in `ωp0`. |
| `dw0` | float | Frequency offset added to `w0`. |
| `t_c`, `t_w`, `t_r` | float | Temporal envelope center, width, and rise/fall length. |
| `x_c`, `x_w`, `x_r` | float | Spatial envelope center, width, and rise/fall length. Use a very large `x_w` for a spatially uniform drive. |
| `a0` | float | Driver amplitude. The applied field is scaled by `|k0| * a0`, not `a0` alone. |

## `models`

Optional learned-closure specification used by the trapping growth rate
`nu_g` (and, when implemented, a damping model `nu_d`). Set `models: false` to
disable learned closures entirely.

```{warning}
This block is currently accepted and validated but not consumed — the live
solver never instantiates these networks. See the warning under
[`trapping`](#physicsspeciestrapping).
```

```yaml
models:
  file: models/weights.eqx     # or false for untrained weights
  nu_g:
    in_size: 3
    out_size: 1
    width_size: 8
    depth: 4
    activation: tanh
    final_activation: tanh
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | str \| bool | `false` | Path to serialized `equinox` weights, or `false` to start untrained. |
| `nu_g` | block | `null` | `equinox.nn.MLP` spec for the trapping growth rate. |
| `nu_d` | block | `null` | `equinox.nn.MLP` spec for a learned damping rate. |

Each MLP block takes `in_size`, `out_size`, `width_size`, `depth`, `activation`,
and an optional `final_activation`.
