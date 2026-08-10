# Vlasov-1D2V Configuration Reference

This document describes how to construct a configuration file for the `vlasov-1d2v`
solver, which advances $f(x, v_\parallel, v_\perp)$ in **cylindrical** velocity
space.

The configuration schema is the [Vlasov-1D schema](../vlasov1d/config.md) plus a
perpendicular velocity axis, so only the differences are documented here. Every
section not listed below (`units`, `density`, `mlflow`, `drivers`, ...) behaves
exactly as it does for `vlasov-1d`.

Example decks live in `configs/vlasov-1d2v/`:

| Deck | What it shows |
|------|---------------|
| `epw.yaml` | Driven EPW with the marginal-coefficient `dougherty` operator |
| `epw-cylindrical-landau.yaml` | The same wave with the full-geometry `cylindrical_landau` operator and per-channel weights |

## Velocity-space geometry

The perpendicular axis is cylindrical: $v_\perp \in (0, \texttt{vperp\_max})$ with
cell-centered points and integration weight

$$
w_\perp = 2\pi v_\perp \, \Delta v_\perp ,
\qquad
\int f \, d^3v = \sum_j f_j \, w_{\perp,j} \, \Delta v_\parallel .
$$

The **marginal**

$$
F(x, v_\parallel) = \int f \, w_\perp \, dv_\perp
$$

is the bridge to the 1D solver: it is initialized to exactly the `vlasov-1d`
initialization, saved in exactly the `vlasov-1d` layout, and — for the
marginal-coefficient collision operators — evolves with exactly the `vlasov-1d`
dynamics.

## grid

Adds two **required** fields to the [1D grid config](../vlasov1d/config.md#grid).
Neither has a default.

| Field | Type | Description |
|-------|------|-------------|
| `nvperp` | int | Number of perpendicular velocity cells |
| `vperp_max` | float | Upper bound of the perpendicular axis (the lower bound is always 0) |

```yaml
grid:
  dt: 0.1
  nx: 32
  xmin: 0.0
  xmax: 20.94
  nv: 512          # parallel axis: v_par in (-vmax, vmax)
  vmax: 6.4
  nvperp: 64       # perpendicular axis: v_perp in (0, vperp_max)
  vperp_max: 6.4
  tmin: 0.
  tmax: 1000.0
```

### Choosing `nvperp`

How much perpendicular resolution you need depends on the collision operator:

- The **marginal-coefficient** operators (`dougherty`, `dougherty_nodrag`,
  `lenard_bernstein`) have no $v_\perp$ dynamics at all — every $v_\perp$ slice is
  pushed by the same $v_\parallel$ operator. The perpendicular axis is then purely a
  quadrature axis, and can be coarse. Density is unaffected (the initializer
  normalizes $\sum_j M_\perp w_\perp = 1$ exactly); only the saved `pperp` carries
  the midpoint-rule bias, about 0.3% at $\Delta v_\perp = 0.25$.
- `cylindrical_landau` re-measures $(n, u, T)$ from $f$ on every call and feeds them
  back into its own coefficients, so the quadrature error enters the dynamics.
  Use $\Delta v_\perp \approx 0.1$, the resolution its validation gates were
  established at, and leave `moment_restoration` on.

## save

Field, scalar, and diagnostic saves are unchanged from `vlasov-1d`. Species
distribution saves come in two kinds, selected by which axes you list:

| Axes | Output | Rank | Notes |
|------|--------|------|-------|
| `{t, x, v}` | the marginal $F(x, v_\parallel)$, interpolated onto the requested sample points | 3 | Identical to a `vlasov-1d` distribution save, so the whole 1D analysis stack applies unchanged |
| `{t}` | the full $f(x, v_\parallel, v_\perp)$ at full resolution | 4 | Keep the cadence sparse |

```yaml
save:
  fields:
    t: {tmin: 0.0, tmax: 1000.0, nt: 1001}
  electron:
    marginal:                                  # rank-3 F(x, v_par)
      t: {tmin: 0.0, tmax: 1000.0, nt: 101}
      x: {xmin: 0.0, xmax: 20.94, nx: 32}
      v: {vmin: -6.4, vmax: 6.4, nv: 512}
    full:                                      # rank-4 f(x, v_par, v_perp)
      t: {tmin: 0.0, tmax: 1000.0, nt: 6}
```

The label (`marginal`, `full`) is yours to choose; it names the output stream
(`dist-electron.marginal.nc`). An `{t, x, v}` save must give the full `xmin`/`xmax`/`nx`
and `vmin`/`vmax`/`nv` triples — an empty `x: {}` does not build a sample axis.

Saved moments include `pperp` $= \int f\, v_\perp^2\, w_\perp\, dv_\perp\, dv_\parallel$
alongside the parallel `p`; with two perpendicular degrees of freedom,
$T_\perp = p_\perp / 2n$.

## diagnostics

`vlasov-1d2v` replaces the 1D `diag-*-dfdt` toggles with **cumulative** ones.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `diag-vlasov-cumulative` | bool | `False` | Accumulate the Vlasov term's contribution to $\partial F/\partial t$ |
| `diag-fp-cumulative` | bool | `False` | Accumulate the Fokker-Planck term's contribution to $\partial F/\partial t$ |

These are marginal `(nx, nv)` arrays holding a running **time integral**, not a
sampled rate. Difference them between save points to recover exact
interval-averaged rates — this is what avoids aliasing the $2\omega$ wave-particle
energy exchange that sampling a rate would produce. Enable them under
`diagnostics`, then add matching entries under `save` to write them out:

```yaml
diagnostics:
  diag-vlasov-cumulative: True
  diag-fp-cumulative: True

save:
  diag-vlasov-cumulative:
    t: {tmin: 0.0, tmax: 1000.0, nt: 101}
  diag-fp-cumulative:
    t: {tmin: 0.0, tmax: 1000.0, nt: 101}
```

## terms

Two restrictions relative to `vlasov-1d`:

| Field | Supported values |
|-------|------------------|
| `edfdv` | `exponential` only |
| `krook.is_on` | `False` only — the Krook operator is not implemented for this solver |

`krook` is still a **required** block (inherited from the 1D schema); give it
`is_on: False` and any valid profile objects.

### fokker_planck

| `type` | Model | Velocity geometry |
|--------|-------|-------------------|
| `dougherty` | Dougherty, coefficients from the marginal | $v_\parallel$ only |
| `dougherty_nodrag` | Diffusion of the deviation, coefficients from the marginal | $v_\parallel$ only |
| `lenard_bernstein` | Lenard-Bernstein, coefficients from the marginal | $v_\parallel$ only |
| `cylindrical_landau` | Linearized Landau/Coulomb operator | full $(v_\parallel, v_\perp)$ tensor |

The marginal-coefficient operators compute $(\bar v, \beta)$ from
$F(x, v_\parallel)$ and apply the resulting tridiagonal operator to every $v_\perp$
slice. Because the discrete energy-flux condition is linear in $f$ and the marginal
is a $w_\perp$-weighted sum of slices, this conserves $n$, $P_\parallel$, and
$E_\parallel$ to the same standard as the 1D operator. For a separable
$f = F(v_\parallel) M(v_\perp)$ the marginal dynamics are *exactly* the 1D
operator's. They accept the same
[`self_consistent_beta`](../vlasov1d/config.md#self_consistent_beta) sub-object as
`vlasov-1d`.

#### cylindrical_landau

The full-velocity-geometry linearized Coulomb operator,

$$
C[f] = \nu \, \nabla_v \cdot \left[ \mathbf{D}(v) \cdot M \nabla_v (f/M) \right],
\qquad
M = \exp\!\left(-|\mathbf{v} - \mathbf{u}|^2 / 2T\right),
$$

with the anisotropic test-particle tensor built about the bulk frame from the
erf-exact Rosenbluth-potential coefficients of a Maxwellian field species,

$$
\mathbf{D} = \alpha_\text{speed}\, \hat D_\parallel(s)\, \hat{\mathbf{s}}\hat{\mathbf{s}}
           + \alpha_\text{lorentz}\, \hat D_\perp(s) \left(\mathbf{I} - \hat{\mathbf{s}}\hat{\mathbf{s}}\right),
\qquad
s = |\mathbf{v} - \mathbf{u}| / \sqrt{T} .
$$

Writing the flux in the Einstein-relation form $M\nabla_v(f/M)$ means **each channel
separately annihilates the bulk Maxwellian**, and discretizing it with the geometric-mean
edge value of $M$ makes the *sampled* Maxwellian an exact discrete fixed point of any
channel mix at any `dt`. The bulk $(n, u, T)$ are re-measured from $f$ every call, so
the operator always relaxes toward the current bulk.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `channels.speed` | float | `1.0` | Weight $\alpha_\text{speed}$ of the speed-diffusion channel |
| `channels.lorentz` | float | `1.0` | Weight $\alpha_\text{lorentz}$ of the pitch-angle (Lorentz) channel |
| `moment_restoration` | bool | `True` | Restore the momentum/energy exchanged with the implicit field particles each step |
| `explicit_substeps` | int | `1` | Substeps for the explicit cross-diffusion terms |

**`channels`** exist so you can attribute an effect to one scattering process. Both
weights at `1.0` is the physical operator; setting one to `0.0` isolates the other.
At the $k = 0.3$ EPW resonance the Lorentz channel carries roughly 47% of the
projected marginal diffusion and the speed channel 53%.

**`moment_restoration` should stay on.** It is not just conservation bookkeeping:
because the operator re-linearizes about the measured moments, switching it off lets
the live-moment loop chase the $O(\Delta v_\perp^2)$ midpoint-rule bias in the measured
$T$, and the bulk drifts secularly (~5e-5 per collision time at
$\Delta v_\perp = 0.1$). With it on, $P_\parallel$ and $E$ are pinned exactly each
step and the loop closes.

**`explicit_substeps`** applies only to the cross-diffusion terms; both diagonal
diffusion sweeps are implicit batched tridiagonals. `1` is ample at
$\nu\,\Delta t \sim 10^{-4}$; raise it if you push $\nu$ or `dt` much higher.

```yaml
terms:
  field: poisson
  edfdv: exponential
  time: sixth
  fokker_planck:
    is_on: True
    type: cylindrical_landau
    channels:
      speed: 1.0
      lorentz: 1.0
    moment_restoration: True
    explicit_substeps: 1
    time:
      baseline: 1.0e-3
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
    time: {baseline: 1.0, bump_or_trough: bump, center: 0.0, rise: 25.0, slope: 0.0, bump_height: 0.0, width: 100000.0}
    space: {baseline: 1.0, bump_or_trough: bump, center: 0.0, rise: 25.0, slope: 0.0, bump_height: 0.0, width: 100000.0}
```

```{note}
The 2V-only Fokker-Planck keys (`channels`, `moment_restoration`,
`explicit_substeps`, `self_consistent_beta`) are read directly from the
configuration dictionary by the solver and are not fields of the validated pydantic
model. Config validation will therefore *not* catch a misspelled key here — it will
be silently ignored and the default used.
```
