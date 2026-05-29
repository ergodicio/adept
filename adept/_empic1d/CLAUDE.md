# `_empic1d` — relativistic 1D2V electromagnetic PIC

Being built for **differentiable optimization of accelerator profiles**:
PWFA drive-beam current profile (fixed charge → transformer ratio) first, then
LWFA laser temporal profile (fixed energy). See the root `docs/ARCHITECTURE.md`
for how solvers plug into `ergoExo`.

## Physics model

- Fields `(E_x, E_y, B_z)`, momenta `(p_x, p_y)` — 1D, linear polarization.
  (Circular polarization would add `p_z, E_z, B_y` → a future 1D3V extension;
  the pusher is already written for general 3-vectors so it carries over.)
- **Pusher**: relativistic Higuera–Cary (Phys. Plasmas 24, 052104, 2017).
  Volume-preserving; correct E×B drift. `solvers/pushers/push.py`.
- **Longitudinal `E_x`**: Ampère (`∂_t E_x = -j_x`) with charge-conserving
  (Esirkepov-equivalent in 1D) current → Gauss's law exact, no global solve.
- **Transverse `(E_y, B_z)`**: Yee FDTD — NOT YET IMPLEMENTED (later increment).

## Conventions (read before touching the field code)

- We evolve **proper velocity** `u = γv` (velocity units), `γ = sqrt(1+|u|²/c²)`,
  `v = u/γ`. Momentum arrays are `(..., 3)` with components on the last axis.
- `qm = q/m` is the per-species charge-to-mass ratio. `c` is the speed of light
  in solver units (the cold equilibrium gives `ω_pe`).
- **Staggered grid** (Yee-compatible):
  - nodes `i` at `xmin + i·dx` carry charge density `ρ`
  - faces `i+½` at `xmin + (i+½)·dx` carry `E_x` and `j_x`
  - node `i` lies between faces `i-1` and `i`; Gauss is
    `(E_face[i] - E_face[i-1])/dx = ρ_node[i]`.
- Shape functions are **reused** from `_pic1d.solvers.pushers.shape`. That shared
  `deposit`/`gather` puts grid points at `(g+½)·dx + origin`, so:
  - gather face-centered `E_x`: pass `origin = xmin`
  - deposit charge to **nodes**: pass `origin = xmin - dx/2` (the half-cell shift
    in `charge_density_nodes`). Get this wrong and Gauss drifts.

## What's validated (tests in `tests/test_empic1d/`)

- `test_pusher.py` — single-particle analytic orbits: static-E energy gain
  (exact), pure-B energy conservation (machine precision), relativistic
  gyrofrequency `-(q/m)B/γ` (~1e-7), E×B drift, free streaming.
- `test_longitudinal.py` — Gauss's law preserved to ~1e-12 (x64); cold plasma
  oscillation rings at `ω_pe` (units chosen so `ω_pe = 1`).

## Build roadmap

- [x] **Inc 1** Higuera–Cary pusher + single-particle tests.
- [x] **Inc 2** Esirkepov current + Ampère `E_x`; Gauss + plasma-frequency tests.
- [ ] **Inc 3** PWFA: relativistic drive beam, profile via per-particle weights
  (linear in deposit ⇒ differentiable; fixed charge = `Σw` const), co-moving
  wake diagnostic, transformer ratio vs Bane–Chen–Wilson.
- [ ] **Inc 4** Differentiable PWFA optimization (optax over weights).
- [ ] **Inc 5+** LWFA: Yee `(E_y,B_z)` + `j_y` + laser injection + Mur ABC.

## Deferred scope (intentionally not here yet)

- No `datamodel.py` / `modules.py` / `_base_.py` dispatch yet — the `ergoExo`
  run path lands once there's a sim worth logging (Inc 3). Until then the solver
  is driven by `jax.lax.scan` directly in tests, kept physics-correct in
  isolation. `solvers/vector_field.py:longitudinal_step` is the reusable
  single-step kernel the eventual diffrax `ODETerm` will wrap.

## Gotchas

- The test suite enables `jax_enable_x64` via `tests/conftest.py`. Running a
  module function from a bare `python -c` does **not** load conftest → float32 →
  Gauss residual looks like ~1e-5 instead of ~1e-12. Not a bug; run under pytest.
