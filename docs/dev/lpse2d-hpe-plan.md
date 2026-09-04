# Plan: Follett-style Hybrid Particle Evolution (HPE) for lpse2d

> **Status update (2026-09-04): 2-D implemented.** The `ny == 1` restriction assumed
> throughout Secs. 1-6 below is gone — see **Sec. 7** for the 2-D generalization
> (line resonance via per-direction projections), its validation and its measured cost.
> Sections 1-6 remain the quasi-1D design record.
>
> **Status update (2026-07-30): implemented** through M3 in `adept/_lpse2d/core/hpe.py`
> (+ integration per Sec. 3). Notable deviations/findings from implementation:
>
> 1. **Signed-k damping formula**: Eq. in Sec. 2.2 needs a $\mathrm{sgn}(k_x)$:
>    $\gamma_L = -\frac{\pi}{2}\frac{\omega_{p0}^3}{k_x^2}\,\mathrm{sgn}(k_x)\,\partial_v \hat f|_{v_\phi}$
>    so each propagation direction damps on its own tail.
> 2. **Gather upsampling** (`terms.hpe.gather_refine`, default 4): linear interpolation
>    of $E_x$ at $k\,dx \sim 1\text{--}2$ rad/cell attenuates the gathered field by
>    $\mathrm{sinc}^2(k\,dx/2)$ — 15–30% at SRS wavenumbers. The field is zero-padded in
>    k-space by 4× before the ifft (one cheap FFT per step), reducing this to ~1%.
> 3. **Per-k calibration lands at $C(k)\in[0.965, 1.013]$** over the resonant band —
>    the discrete operator is already accurate; the calibration mostly removes
>    binning bias (and doubles as a normalization audit).
> 4. **Trapping resonates at the Bohm–Gross $v_\phi$**, because the solver's envelope
>    rotation $e^{-i\delta\omega(k)t}$ shifts the physical wave to the BG phase
>    velocity. `omega_res: bohm_gross` (default) is therefore the consistent
>    extraction point; validation tests that freeze an envelope must rotate its
>    phase by $e^{-i\delta\omega t}$ or the plateau forms $\sim 0.5\,v_{te}$ below
>    the extraction point.
> 5. **Diagnostics**: the band-min damping ratio is shot-noise-limited (one clamped
>    mode reads 0), so the headline metric is `hpe_gamma_ratio_kpeak` — the ratio at
>    the resonant-band mode carrying the most EPW energy.
> 6. envelope-2d runs float32 by default (`run.py` enables x64 only for kinetic
>    solvers); the HPE module is dtype-agnostic and f32 phase/position errors are
>    negligible ($\omega_{p0} t$ resolution ~$2.7\times10^{-3}$ rad at 10 ps).
> 7. Tests live in `tests/test_lpse2d/test_hpe.py` (M0 free-streaming + bounce
>    frequency/carrier sign, M1 calibration, M3a linear closure, M3b O'Neil
>    flattening, end-to-end SRS smoke). Production config:
>    `configs/envelope-2d/srs-hpe.yaml`.

**Goal:** add the test-particle / self-consistent Landau damping module of Follett et al., *Phys. Plasmas* **24**, 102134 (2017) to the JAX lpse2d SRS solver, so that kinetic inflation (trapping-induced damping reduction) and hot-electron generation are captured. This targets the one remaining fluid-vs-PIC gap in the scan2 replication: all 63 LPSE-dead-but-OSIRIS-active points sit in the marginal-threshold / high-$T_e$ band where trapping physics decides the outcome (`srs-campaign/sims/lpse-srs-scan-test/NOTES.md`).

Status: **plan only — nothing implemented.** Branch context: `lpse2d/srs` @ `d3f8321` (clean).

---

## 1. The algorithm as specified in Follett 2017 (Secs. II.B–II.C)

1. **Test-particle push.** Fully relativistic trajectories integrated with a generalized Verlet scheme in the *real* electrostatic field, reconstructed by restoring the carrier:
   $$\tilde{\mathbf{E}}(\mathbf{x},t) = \tfrac{1}{2}\left[\mathbf{E}(\mathbf{x},t)\,e^{-i\omega_{pe}t} + \mathrm{c.c.}\right]$$
   The laser EM fields are **excluded** from the push (quiver excursion ≲ 1 nm at ICF intensities).
2. **Distribution accumulation.** A *spatially averaged* velocity distribution $\langle F_e\rangle(\mathbf{v},t)$ is accumulated on a histogram spanning $-c$ to $c$ ($100^3$ bins in 3-D).
3. **Landau damping feedback** (their Eq. 4):
   $$\gamma_L(\mathbf{k},t) = \frac{\pi\omega_{pe}^2}{k^2}\int d\mathbf{v}\,\frac{\partial \langle F_e\rangle(\mathbf{v},t)}{\partial \mathbf{v}}\,\delta(\omega_{pe}-\mathbf{k}\cdot\mathbf{v})$$
   evaluated per $k$-grid point (centered-difference $\partial F/\partial v$, interpolated onto the resonance surface), then applied in $k$-space exactly as the fluid code already does: $E(\mathbf{k},t+dt)=E(\mathbf{k},t)\,e^{-\gamma_L(\mathbf{k},t)\,dt}$.
4. **Feedback is Im-only.** The evolving damping is the *only* coupling back to the wave solver; the real part of the dispersion (nonlinear frequency shift) is *not* evolved — stated explicitly in Follett et al., PRL 2018 (frequency-detuning paper). This matters: HPE captures inflation-by-undamping but not trapping-induced detuning/autoresonance.
5. **Tail-only particles.** Electrons with $v < v_\phi^{\min}=\omega/k_{\max}$ can't resonate with any grid-resolvable EPW. With $k_{\max}\lambda_{De}\gtrsim 0.3$ (Landau cutoff resolved), $v_\phi^{\min}\lesssim 3.33\,v_{te}$; dropping the bulk saves ~100× in 3-D (98.9 % of a Maxwellian) while keeping all damping-relevant physics.
6. **Boundaries.** Particle boundaries thermalizing in $x$ (the gradient direction, matching the absorbing EPW boundary), periodic transverse (they thermalize 20 % of transverse crossings to mimic finite plasma; result insensitive to that fraction). No collisions (mfp at $3.33\,v_{te}$ ~100 µm > box, and the thermalizing walls re-equilibrate the bulk).
7. **Numbers used** (3-D, TPD): $10^8$ particles; field solver $dt = 10$ fs; particle substep $0.013$ fs ($\omega_{pe}\,dt_p \approx 0.035$); damping updated every 100 fs (statistical averaging window); HPE switched off for the first 9 ps (fluid steady state first), 2–3 ps to re-equilibrate after switch-on. 95 % of runtime is the particle push (they used 1 GPU via CUDA).
8. **Why bother:** their control run (velocity distribution evolved but damping *not* fed back) gave 84 % larger rms EPW amplitude and **50×** more hot electrons — the feedback is a first-order saturation mechanism, not a correction.

## 2. Mapping onto lpse2d — design decisions

The port targets the **quasi-1D SRS configuration** (`ny=1`) first, which is what the scan2 replication runs. Everything below is written 1D-first with the 2D generalization noted where it differs.

### 2.1 Envelope conventions (verified against the code)

- `state["epw"]` is $\phi_k$ (k-space potential), carrier at $\omega_{p0}$ = plasma frequency of the **envelope density** (`epw.py:534` detuning factor confirms). Real field for the push:
  $$\tilde{E}_x(x,t) = \mathrm{Re}\left[E_x(x,t)\,e^{-i\omega_{p0}t}\right]$$
  with $E_x$ from the existing `phi_k_to_e_fields` hook (`epw.py:424-458`). The local-density offset is already folded into the envelope by the density-gradient phase (`epw.py:684-691`), so restoring the single carrier $\omega_{p0}$ is exact — same situation as Follett (their box spans 0.19–0.27 $n_c$ against a single carrier).
- **Action item (M0):** verify the carrier *sign* convention against the MATLAB source before trusting the de-envelope; the physical test is that a seeded $+k$ EPW resonates particles at $v_\phi = +\omega_{p0}/k$, not $-$.
- Units: lengths µm, times ps, velocities µm/ps; $e$, $m_e$, $c$ from `cfg["units"]["derived"]` (`helpers.py:153-182`). Push in momentum per unit mass, $u = \gamma v$, so the relativistic update is division-free: $\dot{u} = -(e/m_e)\tilde{E}$, $\dot{x} = u/\gamma$, $\gamma = \sqrt{1+u^2/c^2}$.

### 2.2 1-D damping extraction

In 1-D the resonance surface is a point: for each grid $k_x \ne 0$,
$$v_\phi(k_x) = \frac{\omega_{p0}}{k_x} \quad (\text{signed — } \pm k \text{ modes damp on the } \pm v \text{ tails independently; essential for backscatter vs. rescatter}),$$
$$\gamma_L(k_x,t) = -\frac{\pi}{2}\frac{\omega_{p0}^3}{k_x^2}\,\left.\frac{\partial \hat{f}}{\partial v}\right|_{v_\phi(k_x)}, \qquad \int \hat{f}\,dv = 1,$$
with $\partial\hat f/\partial v$ by centered differences on the histogram and linear interpolation to $v_\phi$. Mask $|v_\phi|\ge c$ (no resonant particles → $\gamma_L=0$) and the $k=0$/zero-mask modes.

**Normalization by calibration, not derivation:** the constant in front is fixed by requiring that a freshly loaded Maxwellian tail reproduce the analytic rate already in `epw.py:9-30` (`calc_landau_damping_rate`) across $k\lambda_D \in [0.2, 0.45]$. This absorbs every bookkeeping factor at once (bin widths, tail-fraction weights, 1-D projection) and doubles as the M1 unit test. Note the analytic rate carries Bohm–Gross corrections while Follett's Eq. 4 resonates at bare $\omega_{pe}$; decide during M1 whether to evaluate at $v_\phi = \omega_{p0}/k$ or $\omega_{BG}(k)/k$ by which calibrates cleanly against the analytic curve.

### 2.3 Tail-only loading with blended damping

- Load $N_p$ particles from the Maxwellian tail $|v| > v_{\min}$ (config, default $2.5\,v_{te}$), uniform in $x$ weighted by the density profile (`cfg["grid"]["background_density"]`, available after `modules/base.py:84`). In 1-D the tail fraction at $2.5\sigma$ is ~1.2 %, so this is an ~80× effective-particle saving.
- The particle-based $\gamma_L$ is only valid where $|v_\phi(k)| > v_{\min}$; elsewhere keep the analytic rate:
  $$\gamma(k) = \begin{cases}\gamma_{\rm HPE}(k) & |v_\phi(k)| > v_{\min} + \Delta\\ \gamma_{\rm analytic}(k) & \text{otherwise}\end{cases}$$
  (small buffer $\Delta$ so the histogram edge never feeds the blend). All trapping-relevant modes have $v_\phi \sim 3\text{–}5\,v_{te}$, comfortably inside the particle range.
- Clamp $\gamma_{\rm HPE} \ge 0$ initially (inflation is *reduced* damping; a noisy negative rate would pump the field). Revisit if bump-on-tail gain turns out to matter.

### 2.4 Statistics: EMA histogram instead of interval updates

Follett updates $\gamma_L$ every 100 fs to average sampling noise. A JAX-friendlier equivalent with no step-counter branching: keep the histogram as a state variable updated every field step by exponential moving average,
$$H \leftarrow (1 - dt/\tau)\,H + (dt/\tau)\,H_{\rm inst}, \qquad \tau \sim 100\ \mathrm{fs\ (config)},$$
and recompute $\gamma_L(k)$ from $H$ every step (cost $O(N_p + n_v + n_x)$ — negligible next to the push). One-field-step lag in the feedback is far tighter than Follett's 100 fs lag.

### 2.5 Substepping the push

Within one field step the envelope is frozen; only the carrier oscillates. Sub-cycle with `lax.fori_loop` exactly like the light solver (`raman.py:151-163`), advancing the carrier phase analytically per substep:
$$n_{\rm sub} = \left\lceil \frac{\omega_{p0}\,dt}{\varepsilon} \right\rceil, \qquad \varepsilon \approx 0.05\ \text{(config; Follett ran } 0.035\text{)}.$$
For `srs.yaml` ($\lambda_0=351$ nm, envelope density 0.25 → $\omega_{p0} = 2.68$ rad/fs, $dt=1$ fs): $n_{\rm sub}\approx 54$. Integrator: relativistic KDK leapfrog (kick–drift–kick) — symplectic, matches "generalized Verlet". Gather by linear interpolation; reuse/adapt `_pic1d/solvers/pushers/shape.py::gather` (1-D, periodic → needs the non-periodic $x$ variant).

### 2.6 Particle boundaries

Thermalizing walls in $x$: particles crossing either $x$ boundary are re-injected with a fresh tail velocity (correct sign to re-enter) at the wall. Needs in-step RNG: thread `jax.random.fold_in(base_key, step_index)` with `step_index = round(t/dt)` — deterministic and resume-safe. ($ny=1$: no transverse particle motion at all, since the push uses only $E_x$; transverse boundaries become relevant only in the 2-D generalization.)

### 2.7 What stays out of scope (documented limitations)

- **No real-part feedback** (no nonlinear frequency shift / autoresonance) — same as LPSE. The Tran et al. 2020 fluid-trapping closure (`domain-knowledge/paper_notes/tran-2020-fluid-trapping.md`) provides exactly that piece; the two are complementary and could later be combined (HPE damping + Tran $\delta\omega_{NL}$).
- **No collisions** in the push (justified as in the paper; our boxes are ≤ tens of µm).
- **Not differentiable**: the histogram/resampling path breaks gradients. HPE runs are forward-only; the differentiable trapping path remains the tf1d-style ML closure. Use `stop_gradient` on the feedback if a config ever mixes HPE with optimization.

## 3. Code changes (concrete)

New module: **`adept/_lpse2d/core/hpe.py`** — `class HybridParticleEvolution(eqx.Module)` with
`load(cfg)` (initial $x, u$ arrays), `push(x, u, e_x_envelope, t)` (subcycled KDK, returns new $x,u$ + boundary re-injection), `histogram(u)`, `damping(H)` → $\gamma_L(k)$, and `__call__` orchestrating one field step.

Integration points (from the architecture audit):

| Where | Change |
|---|---|
| `modules/base.py:77-90` `init_state_and_args` | add `x_e (Np,)`, `u_e (Np,)`, `epw_hist (nv,)`, `gamma_L (nx, ny)` to the state dict (all real float64 — the `.view(float64)` at line 89 is a no-op for them). Must come *after* line 84 (`background_density`). |
| `vector_field.py:38` | leave new keys out of `complex_state_vars`. |
| `vector_field.py:21-51` `SplitStep.__init__` | construct `self.hpe` conditionally on `cfg["terms"]["hpe"]["active"]` (pattern of `self.raman` at lines 31-37). |
| `vector_field.py:108-123` `SplitStep.__call__` | after the epw update: `ex, ey = self.epw.phi_k_to_e_fields(new_y["epw"])`; push particles; EMA-update `epw_hist`; write `gamma_L`. |
| `epw.py:642-645` | replace the static `gamma_landau` with `y["gamma_L"]` when HPE is active (blend already baked into the array); while here, **fix the `terms.epw.damping.landau` no-op** (currently never read — damping is unconditionally on). |
| `helpers.py:920-939` `save_func` (fields) | special-case particle keys out of the spatial interpolator (shape `(Np,)` would crash it); optionally save `gamma_L` snapshots. |
| `helpers.py:1040-1072` default save | add scalar series: `f_hot_50keV`, `f_hot_100keV`, mean tail energy, $\min_k \gamma_{\rm HPE}/\gamma_{\rm analytic}$ (inflation-o-meter); add the `(nbins,)` energy histogram — requires the extra coord in `make_series_xarrays` (`helpers.py:754-757`, currently 1-D-in-t only). |
| `diagnostics.py:165-255` `series_metrics` | MLflow metrics named to match the OSIRIS scan2 set: `fhot_50keV`, `t_first_hot_e_50keV`, final damping-reduction factor. |
| `helpers.py:206-372` `get_derived_quantities` | parse/convert HPE config scalars (so they land in MLflow params). |
| `datamodel.py` + `docs/source/solvers/lpse2d/config.md` | schema + reference docs (mandatory per repo CLAUDE.md). Retire or repurpose the stale `trapping` table at `config.md:366-372`; `core/trapper.py` (dead code) is untouched by this work. |
| `vector_field.py:116` | restore the broken `drivers.E2` hook (`SpectralEPWSolver` has no `.driver`) — needed to drive a known EPW for M0/M1 validation. Alternatively seed `phi_k` directly in `init_state_and_args`. |

Config block:

```yaml
terms:
  hpe:
    active: true
    n_particles: 500000
    v_min: 2.5          # tail cutoff, units of vte
    substep_courant: 0.05   # wp0 * dt_particle
    tau_damping: 100fs      # EMA window for the histogram
    nv: 512                 # velocity bins spanning (-c, c)
    seed: 42
```

Adjoint note: HPE runs should use a plain forward solve (or `checkpoints` cut way down) — `RecursiveCheckpointAdjoint` snapshots the whole state and $2N_p$ extra floats per checkpoint adds up.

## 4. Milestones

**M0 — pusher infrastructure** (no coupling). `hpe.py` with gather + relativistic KDK; carrier-sign verification; tests: (a) energy conservation in a static field, (b) trapped-particle bounce frequency $\omega_b = \sqrt{e k E/m_e}$ in a frozen monochromatic wave vs. theory, (c) substep convergence in $\varepsilon$.

**M1 — damping extraction.** Histogram → $\gamma_L(k)$; **calibration unit test**: Maxwellian-loaded tail reproduces `calc_landau_damping_rate` to ≲5 % over $k\lambda_D \in [0.2, 0.45]$; convergence study in $N_p$, $n_v$, $\tau$. Decide $\omega_{p0}$ vs $\omega_{BG}$ resonance here.

**M2 — one-way coupling.** Particles pushed by the live SRS fields, feedback OFF. Hot-electron diagnostics wired end-to-end (series, metrics, plots); measure wall-clock overhead on CPU and on a blackbox GPU.

**M3 — feedback ON.** (a) Linear closure test: small-amplitude driven EPW damps at the analytic rate with HPE active (the loop reproduces linear theory). (b) Trapping test: large-amplitude EPW → O'Neil-style damping reduction, benchmarked against **adept's own vlasov1d** at matched $k\lambda_D$ and amplitude — an in-house kinetic ground truth Follett didn't have. (c) Follett's control experiment qualitatively: feedback off vs. on changes saturated EPW amplitude and hot-e yield in the expected direction (they saw 84 % / 50×).

**M4 — production validation.** Re-run a handful of scan2 points from the LPSE-dead-but-OSIRIS-active set (marginal $0.75\text{–}1.25\times$ threshold band + a $T_e=10$ keV point) with HPE on, on blackbox GPUs. Success criteria: reflectivity comes alive at points where linear-damping LPSE was dead; `fhot`/onset metrics within factor-of-a-few of the matching OSIRIS runs. Then decide $N_p$/$\tau$ defaults and whether to sweep the full 378-point grid.

## 5. Cost estimate

Per field step: $n_{\rm sub}\times N_p$ gather+kick ≈ $54 \times 5\times10^5 \approx 3\times10^7$ fused ops — trivial per-step on GPU. A 10-ps run at $dt=1$ fs is $10^4$ steps → $3\times10^{11}$ particle-substeps total; expect the push to dominate runtime exactly as in the paper (95 %). Consequences:
- HPE runs want the **GPU**, reversing the scan2 CPU-fleet strategy (quasi-1D fluid lpse2d was GPU-launch-latency-bound; HPE gives the GPU real work per step).
- Knobs if too slow: $\varepsilon \to 0.1$ (2×), $N_p \to 10^5$ (5×, at the cost of noisier $\gamma_L$ — lean on larger $\tau$), delay HPE activation until the fluid steady state (Follett's 9 ps trick) via a config `t_start`.

## 6. Risks / open questions

1. **Carrier sign & Bohm-Gross resonance choice** — resolved empirically in M0/M1 (both have crisp tests).
2. **Noise-driven feedback instability**: multiplicative $e^{-\gamma dt}$ with a noisy $\gamma$ could bias saturation levels. Mitigated by EMA + clamp ≥ 0 + M3(a) linear closure test.
3. **Spatially uniform $\langle f\rangle$ in a gradient**: box-averaged distribution mixes regions with different resonant $k$. Same approximation as the paper (accepted there over 0.19–0.27 $n_c$); our boxes are narrower in density.
4. **Missing frequency-shift physics** (Im-only feedback): if M4 shows onset comes alive but saturation amplitudes still disagree with OSIRIS, the Tran 2020 real-part closure is the designed follow-on, not a rewrite.
5. **diffrax memory** with particle state at $10^4$ steps — use forward-only solve for production HPE runs.

## 7. 2-D generalization — implemented 2026-09-04 (`lpse2d/hpe-2d`)

The `ny == 1` restriction is gone. `terms.hpe` now runs in a genuinely 2-D box; quasi-1D
is the `n_angles == 1` special case of the same code path, not a separate branch.

### 7.1 The one piece of new physics: a line resonance instead of a point

Follett's Eq. 4 integrates $\partial\langle F\rangle/\partial\mathbf{v}$ over the surface
$\omega = \mathbf{k}\cdot\mathbf{v}$. In 1-D that surface is the point $v_\phi = \omega/k$
and the rate is one `jnp.interp` of $\partial f/\partial v$. In 2-D it is a *line* in
$(v_x, v_y)$ — every velocity with $\mathbf{v}\cdot\hat k = \omega/|k|$ contributes.

The line integral is not new machinery: it is exactly the 1-D formula applied to the
**projection of $f$ onto the mode's own direction**,

$$P_{\hat k}(v) = \int d^2v'\, f(\mathbf{v}')\,\delta(\mathbf{v}'\cdot\hat k - v), \qquad \int P_{\hat k}\,dv = 1$$
$$\gamma_L(\mathbf{k}) = -\frac{\pi}{2}\frac{\omega_{p0}^3}{|k|^2}\left.\frac{\partial P_{\hat k}}{\partial v}\right|_{v = \omega_{\rm res}(|k|)/|k|}$$

so the implementation bins particles along `n_angles` directions spanning $[0,\pi)$ and
reads each mode's slope off the two bracketing directions (bilinear in angle and $v$).

**Why $[0,\pi)$ and not $[0,2\pi)$:** $P_{-\hat k}(v) = P_{\hat k}(-v)$, so the opposite
half-plane is reached by a sign $s = \pm 1$ on both the evaluation velocity and the
prefactor. That $s$ *is* the $\mathrm{sgn}(k_x)$ the 1-D code already carried — the 1-D
sign convention was the $d=1$ shadow of the projection-direction bookkeeping. Concretely
$\phi = \mathrm{atan2}(k_y,k_x)$, $\theta = \phi \bmod \pi$, $s = +1$ if
$\phi \bmod 2\pi < \pi$ else $-1$, and $v_\phi^{\rm signed} = s\,\omega_{\rm res}/|k|$.
For `ny == 1` this gives $\theta \equiv 0$, $s = \mathrm{sgn}(k_x)$, $v_\phi = \omega/k_x$ —
the original expressions, unchanged.

### 7.2 Implementation notes

- **Gather indices are static.** $v_\phi$ and each mode's angle come from the $k$-grid, so
  the angle index/weight and the velocity bin index/weight are all precomputed in
  `resonance_arrays`. Per step the extraction is 4 gathers per mode from the flattened
  `(n_angles+1, nv)` $\partial f/\partial v$ table — no `(n_modes, nv)` intermediate
  (which would be 100 MB on a 400x120 grid). The extra table row is the mirror of row 0
  ($\theta = \pi$), which closes the angular wrap without a special case.
- **One fused scatter for the histogram.** All angles are binned in a single
  `.at[].add()` over `Np * n_angles` samples with the bin index offset by `a * nv`,
  rather than `n_angles` separate histograms. This is what makes the cost nearly flat in
  `n_angles` (see 7.4) — the earlier `vmap(jnp.histogram)` prototype scaled linearly and
  cost as much as the push.
- **Tail cut is on the speed in 2-D** ($|v| > v_{\min}v_{te}$), not on $|v_x|$. The
  projection of that speed-truncated Maxwellian onto any axis equals the *true* 1-D
  Maxwellian exactly for $|v_\parallel| > v_{\min}v_{te}$ — i.e. everywhere the resonance
  reads it — and falls below it only inside the cutoff. The tail fraction changes
  accordingly: $\mathrm{erfc}(v_{\min}/\sqrt2)$ in 1-D, $\exp(-v_{\min}^2/2)$ in 2-D
  (`hpe.tail_fraction`, also used by `diagnostics.py` for the `fhot` weights).
- **Thermalizing wall.** The 2-D flux-weighted emission speed goes as
  $s^2 e^{-s^2/2v_{te}^2}$ (the polar area element adds a power of $s$ over the 1-D
  $s\,e^{-\ldots}$) and has no closed-form inverse CDF, so it is tabulated once at init.
  Direction is cosine-weighted about the inward normal, $\alpha = \arcsin(2U-1)$. The 1-D
  path keeps its exact closed form untouched.
- **Transverse boundary** is periodic (matching `terms.epw.boundary.y: periodic`, which
  is now enforced), with `y_thermal_frac` available for Follett's 20% re-thermalization.
- **State shapes**: `x_e`, `u_e` are `(Np, ndim)`; `epw_hist` is `(n_angles, nv)`. The
  `fhot` diagnostics now use $|u|$ rather than a per-component value — that was a real
  bug waiting for 2-D, harmless while `ndim == 1`.

### 7.3 Validation

`tests/test_lpse2d/test_hpe_2d.py`, 7 tests. The two load-bearing ones:

- **`test_2d_damping_is_isotropic_and_matches_analytic`** — an isotropic loaded tail must
  give a rate that (a) matches the analytic Landau rate over the band and (b) depends on
  $|k|$ alone. Binning the band by the direction of $\mathbf{k}$ and comparing per-angle
  means catches any error in the projection axis or the sign convention. Holds to 5% and
  12% respectively.
- **`test_2d_reduces_to_1d_for_axial_modes`** — the $k_y = 0$ column of a 2-D box equals
  the quasi-1D result (ratio 1.00 ± 0.05), i.e. this is a strict generalization.

Plus: `test_2d_projection_matches_direct_line_integral` compares the angle-interpolated
$\partial P/\partial v$ against a directly binned projection along the mode's own $\hat k$
for off-axis modes (15%); isotropic loading; 2-D free streaming with transverse wrap; a
$k_y$-only mode accelerating particles in $y$ and not $x$ (pins the $E_y$ gather branch);
and a 2-D end-to-end SRS smoke run through the save/plot/metrics path.

The 5 quasi-1D tests in `test_hpe.py` pass unchanged, as does the M3a linear closure and
the M3b O'Neil flattening.

### 7.4 Measured cost (CPU, 400x120 box, $N_p = 5\times10^5$, 54 substeps)

| | step |
|---|---|
| 1-D fluid | 0.47 ms |
| 1-D HPE | 49.3 ms (add-on 48.8) |
| 2-D fluid | 20.4 ms |
| 2-D HPE, `n_angles` 16 / 32 / 64 | 406 / 378 / 422 ms (add-on 385 / 357 / 402) |

So the 2-D add-on is **~8x the 1-D add-on** and ~20x the 2-D fluid step, and it is
**flat in `n_angles`** (the 10% spread is run-to-run noise) thanks to the fused scatter.
Scaling the 8x by the measured 3090 1-D add-on (+2.3 ms/step at $5\times10^5$, +8.2 at
$2\times10^6$) against a 15–50 ms/step 2-D fluid step puts a production 2-D HPE run at
roughly **2–4x** the cost of the same run without HPE.

Two consequences worth stating plainly: `n_angles` is an accuracy knob, not the cost knob
($N_p$ and `substep_courant` are); and because every projection consumes every particle,
**2-D needs no particle-count increase** over 1-D — the per-angle statistics are identical
at fixed $N_p$. The cost of 2-D is arithmetic, not statistical.

### 7.5 Still open

1. The box-averaged $\langle f\rangle$ limitation (risk 3 above) is untouched and is still
   the top follow-on — in 2-D it is now *two* averages (over $x$ and over $y$).
2. `gather_refine` applies in both directions, so the upsampled field is
   `(refine*nx, refine*ny)` complex per component. At `refine = 4` on a 2400x360 TPD grid
   that is ~220 MB; a per-direction refine would fix it if that ever binds.
3. TPD has not been exercised with HPE — the 2-D path is validated on SRS geometry, and
   TPD's off-axis resonance is exactly the case 2-D was built for, so it is the obvious
   first production target.
4. **Shot-noise clamping selects for itself** (found while validating the 2-D smoke
   run; mitigated, not eliminated). The rate is read from `df/dv` at one velocity bin
   per mode, and with finite particles some modes draw a slope steep enough that the
   `gamma >= 0` clamp sends them to exactly zero. A mode pinned at zero is *undamped*,
   so it grows relative to the band -- the error does not average out, it selects for
   itself. 2-D meets this far more often (6644 band modes vs 668 in the shipped smoke
   configs); in a 0.1 ps unsmoothed 2-D run the peak-energy band mode sat at exactly
   zero. `terms.hpe.hist_smooth` (binomial filter over the histogram before the slope,
   default 2 in 2-D, 0 in quasi-1D) is the mitigation, and is bias-free because `C(k)`
   is derived through the same operator. Measured frac of band modes clamped to zero,
   `hist_smooth` 0/1/2/4: 0.061/0.035/0.019/0.004 at 20k particles and
   0.0075/0.0021/0/0 at 200k, band-mean ratio 1.02-1.06 throughout. Worth revisiting
   whether the clamp itself should be softened (a floor at some small fraction of the
   analytic rate) rather than papered over with a filter.
5. **`hpe_gamma_ratio_kpeak` is noisier in 2-D**, and this is a diagnostic issue rather
   than a physics one. It reads a *single* mode (the band mode holding the most EPW
   energy), so it inherits that mode's shot noise — and a 2-D box has far more band modes
   than a quasi-1D one (6644 vs a few hundred in the smoke config), so the chance of
   landing on a mode whose noisy slope clamps to zero goes up. Measured at t = 0 on a
   freshly loaded Maxwellian in that box: the band-mean ratio converges correctly
   (1.058 / 1.050 / 1.002 at $N_p$ = 20k / 200k / 2M) while the fraction of band modes
   reading exactly zero falls 0.061 / 0.008 / 0.000. The *extraction* is unbiased; only
   the single-mode readout is fragile. An energy-weighted mean over the band would be a
   better headline metric, but changing it breaks one-to-one comparability with the
   logged 1-D scan2 runs, so it is left alone here. Production 2-D runs should use
   $N_p \gtrsim 5\times10^5$ (the default) and read the band statistics, not one mode.
