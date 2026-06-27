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
| `integrator` | str | `"lawson"` | Time integrator: `"lawson"` (explicit Lawson-RK4), `"imex"` (Lawson-RK4 + implicit Lorentz substep), or `"implicit"` (implicit midpoint, AD-JFNK) — see below. |
| `newton_iters` | int | `3` | (`implicit`) Newton iterations per step. |
| `gmres_restart`, `gmres_maxiter`, `gmres_tol` | int/int/float | `20`/`4`/`1e-8` | (`implicit`) matrix-free GMRES controls for the Newton linear solves. |
| `precondition` | bool | `true` | (`implicit`) use the streaming+collision operator as a physics-based GMRES preconditioner (see below). |

### `integrator: implicit` (implicit midpoint via AD-JFNK)

Advances the full RHS with the implicit-midpoint rule `y1 = y0 + dt·F((y0+y1)/2)`,
solved by **Jacobian-free Newton-Krylov**: each Newton linear system uses a matrix-free
GMRES whose Jacobian-vector products are *exact autodiff JVPs* (`jax.linearize`) — the
Jacobian is never assembled (memory is the state plus a few Krylov vectors). Implicit
midpoint is A-stable (no CFL at all) and conserves quadratic invariants, so it conserves
mass exactly and energy to the solve tolerance, and stays stable into the saturated /
long-time regime where both `lawson` and `imex` blow up (e.g. bump-on-tail). Cost: each
step does `newton_iters × (GMRES iterations) × (RHS evals)`, so it is the most expensive
per step — use it for the hard cases, not the cheap ones.

**Preconditioning** (`precondition: true`, default). The implicit operator's stiffness is
dominated by the skew streaming term, whose eigenvalues smear along the imaginary axis
(`~dt/2·α·k_max·√(2Nh)`) — the worst case for unpreconditioned GMRES, which then needs
many iterations and can fail to converge at large `dt`/`Nx` (Newton then injects energy).
The preconditioner `M = I − dt/2·(L_streaming + L_collision)` is block-diagonal in `k` and
tridiagonal in mode index, so `M⁻¹` is a cheap per-`k` tridiagonal solve that captures
exactly that stiff spectrum; GMRES on `M⁻¹A` then converges in a handful of iterations.
This is what makes large-`dt` implicit-midpoint runs practical.

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
