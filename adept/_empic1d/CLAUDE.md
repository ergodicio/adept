# `_empic1d` — relativistic 1D2V electromagnetic PIC

A complete relativistic 1D2V EM-PIC, built to enable **differentiable
optimization of accelerator profiles** (PWFA drive-beam current profile → its
transformer ratio; LWFA laser profile). ADEPT is a *frozen solver library*: this
module exposes differentiable kernels (driven by `jax.lax.scan`), and the actual
inverse-problem studies live in the downstream `wakefield-design` repo. There is
deliberately **no ergoExo/MLflow run path** here — that is not the surface
inverse problems use.

Two entry points in `solvers/vector_field.py`:
- `longitudinal_step` — electrostatic-via-Ampère path (no transverse fields).
  Cheaper; used by PWFA (no laser).
- `em_step` — full EM path (longitudinal + transverse Yee), coupled through the
  Higuera–Cary push. Used by LWFA; takes an optional `j_y_source` laser antenna.

## Physics model

- Fields `(E_x, E_y, B_z)`, momenta `(p_x, p_y)` — 1D, linear polarization.
  (Circular polarization would add `p_z, E_z, B_y` → a future 1D3V extension;
  the pusher is already written for general 3-vectors so it carries over.)
- **Pusher**: relativistic Higuera–Cary (Phys. Plasmas 24, 052104, 2017).
  Volume-preserving; correct E×B drift. `solvers/pushers/push.py`.
- **Longitudinal `E_x`**: Ampère (`∂_t E_x = -j_x`) with charge-conserving
  (Esirkepov-equivalent in 1D) current → Gauss's law exact, no global solve.
- **Transverse `(E_y, B_z)`**: Yee FDTD (`E_y` on nodes, `B_z` on faces),
  Faraday + Ampère in `field.py`; self-consistent `j_y` from particle `v_y`.
  Laser injection via a soft-source antenna in `laser.py` (`em_step`'s
  `j_y_source`). Periodic boundaries (no Mur ABC — use a large box + limited
  run time, as for the wake studies).

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
- `test_pwfa_wake.py` — a rigid relativistic drive beam's wake matches 1D linear
  cold-plasma theory `E_z(ξ)=∫_ξ^∞ n_b cos(k_p(ξ'-ξ))dξ'` (shape corr ~0.93,
  amplitude within ~7%, wavelength ≈ λ_p); symmetric-beam transformer ratio ≤ 2.
- `test_pwfa_optimize.py` — gradient flows through the PIC to the transformer
  ratio (fast); the full optimization (`slow`) shapes the beam weights to drive
  R from ~1.1 (symmetric) to ~5, beating the R ≤ 2 bound with a ramped profile.
- `test_em_fields.py` — vacuum EM wave `ω = c·k`; transverse EM wave in cold
  plasma `ω² = ω_pe² + c²k²` (validates the `j_y` coupling); laser soft-source
  pulse propagates at `c`.

## Multi-species + beam

`longitudinal_step` takes `state = {"species": {name: {x,u,w}}, "E": ...}` and
`species_params = {name: {"charge", "qm"}}`; charge/current sum over species, the
field is shared. A PWFA drive beam is just another (relativistic, heavy ⇒ rigid)
species. **Beam profile is set by per-particle weights on a fixed particle grid**
— deposition is linear in `w`, so the profile is a differentiable knob and fixed
charge is `Σw = const` (the Inc 4 optimization variables). Co-moving wake + the
differentiable transformer ratio live in `diagnostics.py`.

## Build roadmap

- [x] **Inc 1** Higuera–Cary pusher + single-particle tests.
- [x] **Inc 2** Esirkepov current + Ampère `E_x`; Gauss + plasma-frequency tests.
- [x] **Inc 3** PWFA drive beam (profile via per-particle weights), co-moving
  wake diagnostic + transformer ratio; wake validated vs 1D linear theory.
- [x] **Inc 4** Differentiable PWFA optimization (optax over weights at fixed
  charge) — backprop through the PIC drives R ~1.1 → ~5 with a ramped profile.
- [x] **Inc 5** EM half: transverse Yee `(E_y,B_z)` + `j_y` (`em_step`) + laser
  soft source (`laser.py`). Validated via vacuum/plasma dispersion + propagation.

ADEPT is now **frozen** — `_empic1d` is the last solver. Further work (LWFA
studies, multi-objective optimization, moving-window production) happens in the
downstream `wakefield-design` repo.

## Out of scope (by design)

- **No `ergoExo`/MLflow run path** (`datamodel.py` / `modules.py` / `_base_.py`
  dispatch). Inverse problems consume the kernels via `jax.lax.scan`
  (`longitudinal_step` / `em_step`), never the logging path.
- **No Mur ABC** — boundaries are periodic. Run LWFA in a large box and stop
  before the laser/wake wraps (a moving window is a downstream addition).

## Gotchas

- The test suite enables `jax_enable_x64` via `tests/conftest.py`. Running a
  module function from a bare `python -c` does **not** load conftest → float32 →
  Gauss residual looks like ~1e-5 instead of ~1e-12. Not a bug; run under pytest.
