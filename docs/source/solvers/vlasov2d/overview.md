# Vlasov-2D Overview

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

## See also

- [Configuration reference](config.md)
- Template config: `configs/vlasov-2d/base.yaml`
