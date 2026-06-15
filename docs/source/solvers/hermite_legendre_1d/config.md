# Mixed Hermite-Legendre 1D Configuration Reference

This document describes how to construct a configuration file for the
`hermite-legendre-1d` solver, which implements the **mixed Hermite-Legendre
spectral method** for the 1D-1V electrostatic Vlasov-Poisson system (Issan,
Delzanno & Roytershteyn, arXiv:2606.12322).

The electron distribution is split `f = f0 + df`:

- `f0` (near-Maxwellian bulk) is expanded in the **asymmetrically-weighted (AW)
  Hermite** basis in velocity, with coefficients `C_n(x, t)`, `n = 0 .. Nh-1`.
- `df` (strongly non-Maxwellian features: beams, plateaus, filamentation) is
  expanded in the **Legendre** basis on a bounded velocity window `[v_a, v_b]`,
  with coefficients `B_m(x, t)`, `m = 0 .. Nl-1`.

The highest Hermite coefficient `C_{Nh-1}` feeds the Legendre modes (one-way
coupling), and both feed the self-consistent field through Poisson. The method is
most accurate, at fixed total velocity DOFs, when non-Maxwellian features are
localized in velocity.

**Normalization** (paper sec 2.1): time by `1/ω_pe`, space by the Debye length
`λ_D`, velocity by the electron thermal velocity `v_the`. A single electron species
is evolved against an immobile neutralizing ion background of density 1.

**Numerics.** Space is treated spectrally (Fourier, periodic domain); both
free-streaming operators are symmetric-tridiagonal in mode index and integrated
*exactly* via prediagonalized matrix exponentials. The E-field force, the Legendre
Dirichlet penalty, and the Hermite→Legendre coupling are advanced explicitly with
**Lawson-RK4**. (The paper uses an implicit-midpoint integrator for machine-precision
energy conservation; this module uses an explicit integrator — energy is then
conserved to the time-integrator's order, which converges with `dt`, while mass and
momentum remain conserved to machine precision.)

## Top-Level Structure

```yaml
solver: hermite-legendre-1d
mlflow: ...
units: ...
physics: ...
grid: ...
initialization: ...
save: ...
```

---

## physics

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Lx` | float | — | Domain length in x (normalized to `λ_D`) |
| `alpha` | float | — | AW-Hermite velocity **scale** parameter `α` (the benchmarks use `√2`) |
| `u` | float | `0.0` | AW-Hermite velocity **shift** parameter `u` |
| `v_a`, `v_b` | float | — | Legendre velocity-window bounds (`df` is resolved on `[v_a, v_b]`) |
| `gamma` | float | `0.5` | Penalty coefficient `γ` for the weak Legendre Dirichlet BC (`df(v_a)=df(v_b)=0`). Applied only to modes `m ≥ 3` to preserve conservation. |
| `nu_H` | float | `0.0` | Artificial (Lenard-Bernstein) Hermite collision rate `ν_H`. Keep small/zero so `f0` can feed `df` through the last Hermite moment. |
| `nu_L` | float | `0.0` | Artificial Legendre collision rate `ν_L`. Controls filamentation/recurrence in `df`. |
| `enforce_conservation` | bool | `true` | Zero the coupling integrals `J_{Nh,0}=J_{Nh,1}=J_{Nh,2}=0` so the discrete method conserves mass, momentum, and energy independent of `Nh` parity and of `α, u` (paper sec 3.4/4). |
| `field` | bool | `true` | Self-consistent Poisson field. Set `false` for the pure linear-advection test (`φ = 0`); the linear Hermite→Legendre closure flux still acts. |

The artificial collision operator (paper sec 2.5) uses the cubic spectrum
`col[n] = n(n-1)(n-2) / ((N-1)(N-2)(N-3))`, which is identically zero for
`n = 0, 1, 2` — so collisions never touch the mass/momentum/energy moments.

---

## grid

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Nx` | int | — | Number of Fourier modes in x |
| `Nh` | int | — | Number of AW-Hermite modes for `f0` (closure by truncation: `C_{Nh}=0`) |
| `Nl` | int | — | Number of Legendre modes for `df` (closure by truncation: `B_{Nl}=0`) |
| `tmax` | float | — | Final simulation time (normalized). Snapped to an exact multiple of `dt`. |
| `dt` | float | `0.01` | Timestep |

---

## initialization

Selects how the initial `C_n(x)` and `B_m(x)` coefficients are built.

| `type` | Parameters | Description |
|--------|-----------|-------------|
| `linear-advection` | `eps`, `mode` | `f0 = (1 + eps·cos(k x))/√(2π)·exp(-v²/2)`; `df = 0`. (`C_0 = n(x)/α`.) |
| `two-stream` | `eps`, `mode` | `f0 ∝ (1 + eps·cos(k x))·v²·exp(-v²/2)`: `C_0 = n(x)/α`, `C_2 = √2·C_0`; `df = 0`. |
| `bump-on-tail` | `eps`, `mode`, `n_beam`, `v_drift`, `v_th` | Bulk Maxwellian in `f0`; a drifting Gaussian beam `n_beam/(√(2π) v_th)·exp(-(v-v_drift)²/2v_th²)` projected onto Legendre as `df`. |
| `custom` | `hermite: {n: {base, eps, mode}}`, `df: {beams: [{amp, v_drift, v_th}], eps, mode}` | Generic Hermite coefficient profiles plus a beam/sum-of-Gaussians `df` projected onto Legendre. |

Here `k = 2π·mode/Lx`. The Legendre projection uses Gauss-Legendre quadrature.

---

## save

Standard ADEPT `save` block with `t: {nt: ...}` (or `tmin`/`tmax`/`nt`) sub-axes.

| Key | Contents |
|-----|----------|
| `fields` | Electric field `e(x,t)` and potential `phi(x,t)` |
| `hermite` | AW-Hermite-Fourier coefficient timeseries `Ck` (shape `nt × Nh × Nx`) |
| `legendre` | Legendre-Fourier coefficient timeseries `Bk` (shape `nt × Nl × Nx`) |
| `default` | Scalar invariants `mass`, `momentum`, `energy` (paper eqns 26, 28, 30-31) plus field energy and density extrema. Always added; the primary correctness gate. |

`post_process` writes netCDF binaries and spacetime/scalar plots, and reports the
relative drift of each invariant as the metrics `reldrift_{mass,momentum,energy}`.

---

## Example: two-stream instability

```yaml
solver: hermite-legendre-1d
mlflow: {experiment: hermite-legendre-1d, run: two-stream}
units: {normalizing_density: 1e20/cc, normalizing_temperature: 1keV}
physics:
  Lx: 12.566370614359172   # 4π
  alpha: 1.4142135623730951
  u: 0.0
  v_a: -2.5
  v_b: 2.5
  gamma: 0.5
  nu_H: 0.0
  nu_L: 1.0
  enforce_conservation: true
  field: true
grid: {Nx: 64, Nh: 85, Nl: 171, tmax: 35.0, dt: 0.01}
initialization: {type: two-stream, eps: 0.01, mode: 1}
save:
  fields: {t: {nt: 351}}
  legendre: {t: {nt: 71}}
```

See `configs/hermite-legendre-1d/` for the linear-advection, two-stream, and
bump-on-tail benchmark configurations.
