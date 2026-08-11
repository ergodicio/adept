# Vlasov-2D Overview

Example decks live in `configs/vlasov-2d/`. To run one:

```bash
uv run run.py --cfg configs/vlasov-2d/base
```

The `vlasov-2d` solver evolves the 2D2V Vlasov–Maxwell system on a periodic 2D
spatial box. The distribution function `f(x, y, vx, vy, t)` is advanced via
operator splitting:

```
∂f/∂t + v · ∇_x f + (q/m)(E + v × B) · ∇_v f = C[f]
```

The electromagnetic fields `(Ex, Ey, Bz)` evolve under TE-mode Maxwell:

```
∂Ex/∂t =  c² ∂Bz/∂y − Jx
∂Ey/∂t = −c² ∂Bz/∂x − Jy
∂Bz/∂t = ∂Ex/∂y − ∂Ey/∂x
```

## Numerics

- **Streaming**: spectral exponential shift in (x, y) — exact for periodic
  velocity-independent advection along each axis.
- **Electric velocity push**: spectral exponential shift in (vx, vy); the two
  axes commute and are applied independently.
- **Magnetic velocity push**: exact 2D rotation of `f(vx, vy)` by angle
  `θ = −(q/m) Bz dt` at each `(x, y)`, applied with `interpax.interp2d` (cubic).
- **Maxwell**: Strang-split spectral solver (B-half, E-full with current J,
  B-half).
- **Collisions**: Dougherty Fokker–Planck (separable in vx, vy) and/or Krook
  relaxation to a local bi-Maxwellian.
- **Filtering**: optional Hou–Li exponential filter on any subset of
  `{x, y, vx, vy}`.

## Time-step ordering (one full dt)

1. `½ dt` x-streaming → `½ dt` y-streaming
2. Velocity push: `¼ dt` Ex push → `¼ dt` Ey push → `full dt` Bz rotation →
   `¼ dt` Ey push → `¼ dt` Ex push *(the four E-half steps add to `dt`)*
3. `½ dt` y-streaming → `½ dt` x-streaming
4. Maxwell update with `J = J_self + J_driver` evaluated at `t + dt/2`
5. Collisions + optional filter

## Boundary Conditions

| Axis / quantity | Condition | Notes |
|---|---|---|
| $x, y$ | **Periodic** | Streaming and the Maxwell solver are both spectral, so the 2D box is periodic by construction. |
| $v_x, v_y$ (electric push) | **Periodic** | The velocity push is a spectral exponential shift, so it carries the same wrap-around caveat as Vlasov-1D: keep $f \approx 0$ at the velocity edges. |
| $v_x, v_y$ (magnetic rotation) | Cubic interpolation | The $B_z$ rotation is applied with `interpax.interp2d`; points rotated in from outside the grid are interpolated, not wrapped. |
| $v_x, v_y$ (collisions) | **Zero-flux** | As in the 1D Dougherty operator. |

## Forcing and Drivers

The Maxwell update uses $J = J_\text{self} + J_\text{driver}$ evaluated at $t + dt/2$, so external
drivers enter as a prescribed current rather than as a field added after the fact. An optional
Hou-Li exponential filter can be applied to any subset of $\{x, y, v_x, v_y\}$, and Krook relaxation
toward a local bi-Maxwellian acts as a dissipative forcing term.

## What Gets Saved

**`binary/`**: `scalars-t=<t>.nc`, `<prefix>-shared-t=<t>.nc` (the shared EM fields),
`<prefix>-<species>-t=<t>.nc` (per-species moments), and `dist-<key>.nc` for each distribution save
block.

**`plots/`**: `plots/fields/`, `plots/scalars/`, and `plots/dists/` — space-time plots and lineouts
per field, scalar time series, and distribution snapshots.

`postprocess_time_min` is logged to MLflow as a metric.

Note that a 2D2V distribution is a rank-5 array; configure distribution saves sparsely in time or
the output will dominate the run.

## See also

- [Configuration reference](config.md)
- Template config: `configs/vlasov-2d/base.yaml`
