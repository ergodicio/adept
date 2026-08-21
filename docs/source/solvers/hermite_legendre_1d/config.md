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
*exactly* via prediagonalized matrix exponentials. The production `split` integrator
wraps the velocity-force update in exact half streaming/collision steps. Its Hermite
force solve is a bidiagonal recurrence. For the all-mode `gamma=0.5` penalty, the
Legendre force generator is skew-symmetric and prediagonalized once, so its
implicit-midpoint Cayley transform is applied with unit-modulus eigenvalue factors.
Other penalties use a lower-triangular plus rank-2 Woodbury solve. Neither path needs
a global Newton/GMRES iteration.

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
| `gamma` | float | `0.5` | Penalty coefficient `γ` for the weak Legendre Dirichlet BC (`df(v_a)=df(v_b)=0`). |
| `penalty_all_modes` | bool | integrator-dependent | Apply `gamma` to modes 0–2 as well as the high modes. Defaults to `true` for `split` (making the `gamma=0.5` force operator skew-symmetric) and when `enforce_conservation: false`; otherwise defaults to `false`. |
| `nu_H` | float | `0.0` | Artificial (Lenard-Bernstein) Hermite collision rate `ν_H`. Keep small/zero so `f0` can feed `df` through the last Hermite moment. |
| `nu_L` | float | `0.0` | Artificial Legendre collision rate `ν_L`. Controls filamentation/recurrence in `df`. |
| `enforce_conservation` | bool | `true` | Zero the coupling integrals `J_{Nh,0}=J_{Nh,1}=J_{Nh,2}=0`. With `split`, also apply the minimum-L2 correction to the six low `k=0` Hermite/Legendre coefficients after each step, restoring total mass, momentum, and energy. With other integrators this also defaults the penalty on modes 0–2 to zero. |
| `field` | bool | `true` | Self-consistent Poisson field. Set `false` for the pure linear-advection test (`φ = 0`); the linear Hermite→Legendre closure flux still acts. |

The artificial collision operator (paper sec 2.5) uses the cubic spectrum
`col[n] = n(n-1)(n-2) / ((N-1)(N-2)(N-3))`, which is identically zero for
`n = 0, 1, 2` — so collisions never touch the mass/momentum/energy moments.

**Choosing `Nh` (important).** Keep the Hermite basis *bulk-only* — structure inside
the Legendre window belongs to `df`. At large `Nh` (≳64) and saturation-scale fields,
the nonlinear force ladder (pump `~|E|·√(2n)/α`) outruns the cubic collision damping
in the mid-`n` window and a spurious `k=0` velocity-space cascade grows there at
~10× the physical rate, eventually destroying the run — at *any* practical `ν_H`
(0, 10, and 30 were all measured to fail on the bump-on-tail benchmark). A small
basis closes the window structurally: bump-on-tail with `Nh=32` reproduces the
`Nh=128` field observables to 3 digits with machine-precision energy conservation.

**Choosing the Legendre window (equally important).** Keep both boundaries outside
the evolving non-Maxwellian support and monitor `boundary_df_max`. In the production
bump-on-tail discriminator, `[4,15]` becomes contaminated and fails near `t=659`,
whereas `[2,18]` remains bounded through `t=700` with boundary occupancy about
`5.1e-5` and high-mode energy fraction about `5.3e-6`.

---

## grid

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Nx` | int | — | Number of Fourier modes in x |
| `Nh` | int | — | Number of AW-Hermite modes for `f0` (closure by truncation: `C_{Nh}=0`) |
| `Nl` | int | — | Number of Legendre modes for `df` (closure by truncation: `B_{Nl}=0`) |
| `tmax` | float | — | Final simulation time (normalized). Snapped to an exact multiple of `dt`. |
| `dt` | float | `0.01` | Timestep |
| `integrator` | str | `"lawson"` | Time integrator: `"split"` (structured Strang/Cayley, recommended), `"lawson"` (explicit Lawson-RK4), or `"imex"` (Lawson-RK4 + implicit Lorentz substep) — see below. |
| `split_field_iters` | int | `1` | (`split`) fixed-point midpoint-field corrector iterations. Increase to 2–3 for long, strongly nonlinear runs; `1` is the original predictor/corrector. |

### `integrator: split` (recommended)

The split path advances one timestep as:

1. an exact half-step of Hermite/Legendre streaming and diagonal collisions;
2. a full local velocity-force step in real `x`, using a predicted midpoint electric
   field, configurable fixed-point field correction, and implicit-midpoint Cayley
   transforms;
3. a second exact half-step of streaming and collisions.

The Hermite Cayley solve is an O(`Nx*Nh`) bidiagonal recurrence. With the default all-mode
`gamma=0.5` penalty, the Legendre force generator is skew-symmetric to roundoff; a
prediagonalized O(`Nx*Nl²`) Cayley update preserves the coefficient norm without the
ill-conditioned triangular factors that appear at high `Nl`. Non-skew penalty choices
fall back to the lower-triangular plus rank-2 Woodbury solve, also O(`Nx*Nl²`). No global
nonlinear solve or Krylov vectors are allocated. When `enforce_conservation: true`,
a minimum-L2 correction of `C_0..2(k=0)` and `B_0..2(k=0)` restores total mass, momentum,
and energy after the split step. The correction is disabled when an external field
driver is present so its physical work is not projected away.

The standard scalar output includes `boundary_df_a_max`, `boundary_df_b_max`, their
combined maximum `boundary_df_max`, `high_legendre_fraction`, `step_residual` (the
final fixed-point midpoint-field defect), and
`conservation_correction` (the L2 norm of the six-coefficient correction).

### `integrator: imex`

The stiffness that limits the explicit step is the `E·∂_v f` Lorentz force: in the
spectral velocity bases it is strictly lower-triangular (nilpotent for Hermite,
lower-triangular + a rank-2 penalty for Legendre) with operator norm `~Nl²/width·|E|`
— so explicit RK4's `|dt·‖L‖|≲2.8` limit tightens as modes/field grow. Setting
`integrator: imex` keeps free-streaming, collisions, and the Hermite→Legendre closure
flux in the explicit Lawson step, and advances the Lorentz force with an
**unconditionally stable frozen-E Backward-Euler substep** (a per-`x` triangular/dense
linear solve; first-order Lie split). This removes the CFL limit, letting two-stream
run at `dt ≈ 0.02` instead of `0.002`. Trade-offs: Backward Euler is mildly dissipative
and the split is first-order in `dt`, so for high-accuracy/conservation studies prefer
small-`dt` `lawson`; for robustness at large mode counts or large `Nx`, prefer `imex`.

**Choosing `dt`.** Free-streaming and collisions are integrated exactly, but the
explicit Lawson-RK4 treatment of the E-field force has a stability (CFL) limit that
tightens as the self-consistent field grows. For small-amplitude/linear runs (e.g.
driven Landau damping) `dt = 0.05` is fine; for nonlinear instabilities that saturate
to a large field (two-stream) a smaller step is needed — `dt ≈ 0.002` is stable and
converged for the two-stream benchmark. (The paper's `dt = 0.01` relies on its
unconditionally stable implicit-midpoint integrator; this explicit module trades that
for a smaller step and a much smaller memory footprint.) A run that goes `NaN` partway
through is the signature of `dt` above the CFL limit — halve it.

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

## drivers (optional)

An external longitudinal field `ex` can be applied to the velocity-space force (it
never enters the Poisson solve), e.g. to drive a resonant EPW for a Landau-damping
measurement — the analogue of the Vlasov-1D `ex` driver. Omit the `drivers` block for
self-consistent runs.

```yaml
drivers:
  ex:
    '0':                 # one entry per pulse
      k0: 0.4            # wavenumber
      w0: 1.285          # angular frequency (e.g. Re(omega) from the dispersion relation)
      dw0: 0.0           # frequency offset (added to w0)
      a0: 1.0e-3         # amplitude
      t_center: 20.0     # pulse: center / full width / rise(+fall) time
      t_width: 20.0
      t_rise: 5.0
      x_center: 7.85     # spatial envelope: center / width / rise (defaults span the box)
      x_width: 1.0e6
      x_rise: 1.0
```

The driver field is `E_drive(x,t) = Σ env(x,t)·(w0+dw0)·a0·sin(k0 x − (w0+dw0) t)` and
is saved as `de` in the `fields` group.

---

## save

Standard ADEPT `save` block with `t: {nt: ...}` (or `tmin`/`tmax`/`nt`) sub-axes.

| Key | Contents |
|-----|----------|
| `fields` | Electric field `e(x,t)`, potential `phi(x,t)`, and external driver field `de(x,t)` |
| `hermite` | AW-Hermite-Fourier coefficient timeseries `Ck` (shape `nt × Nh × Nx`) |
| `legendre` | Legendre-Fourier coefficient timeseries `Bk` (shape `nt × Nl × Nx`) |
| `default` | Scalar invariants `mass`, `momentum`, `energy` (paper eqns 26, 28, 30-31), field energy, density extrema, velocity-boundary occupancy, high-mode fraction, step residual, and conservation-correction norm. Always added; the primary correctness gate. |

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
