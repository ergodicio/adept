# ADEPT Codebase-Wide Physics & Conventions Audit

**Date:** 2026-07-16
**Scope:** every solver module and shared physics file in `adept/` — `_vlasov2d`, `_pic1d`, `_tf1d`, `_lpse2d`, `_spectrax1d`, `_hermite_poisson_1d`, `vfp1d`, `osiris`, `vlasov1d2v`, plus the shared `normalization.py`, `electrostatic.py`, `functions.py`, `driftdiffusion.py`, `utils.py`, and all tests/configs.
**Companion document:** `VLASOV1D_CONVENTIONS_AUDIT.md` (the `_vlasov1d` deep audit that established the root cause). This document extends that audit to the rest of the codebase and answers: *which solvers are immune?*
**Method:** six parallel independent module audits, each cross-checked against source; load-bearing claims re-verified by hand. No source files modified.

---

## 1. Executive summary

**The √2 bug is confined to the `electron_debye_normalization` consumers — and even there, only to the unit-conversion/reporting layer.** The dimensionless *dynamics* of every solver in the codebase are internally self-consistent. The full damage inventory of `normalization.py:91` ($v_0 = \sqrt{2T_0/m_e}$, $L_0 = \sqrt2\lambda_{De}$):

- `_vlasov1d`, `_vlasov2d`, `_pic1d`: logged physical units √2-wrong, physical temperature 2× the label, dimensional-string inputs √2-off, EM-branch $\hat c$ √2-small (details per module below).
- `_tf1d`: **independently re-implements the identical √2** in its own `write_units` (`_tf1d/modules.py:62`) — same symptom, separate code, needs its own fix.
- Everyone else: **immune** (see the table).

**One critical amendment to the fix prescription in the vlasov1d audit:** `vth_norm()` (`normalization.py:48–50`) must **NOT** be redefined. Grep-verified, it has exactly four callers, all in `vfp1d/`, which is built self-consistently on the $v_{th} \equiv \sqrt{2T/m}$ (most-probable-speed) convention — its initializer carries a compensating factor (`vfp1d/helpers.py:111`: $\alpha = \sqrt{3\Gamma(3/m)/(2\Gamma(5/m))} = 1$ at $m{=}2$, vs. `_vlasov1d`'s $\sqrt2$) so the realized variance is exactly $T/m$. Dropping the 2 in `vth_norm()` would do nothing for `_vlasov1d` (which never calls it) and would **silently halve VFP-1D's initialized temperature** while leaving its transport coefficients at $T_0$. The safe fix touches only `electron_debye_normalization` lines 91–92.

### 1.1 Immunity table (the answer to "which solvers are immune")

"Bug" = the current √2 in `normalization.py:91`. "Fix" = changing `electron_debye_normalization` to $v_0=\sqrt{T_0/m_e}$ (only; `vth_norm()` untouched).

| Solver | Working vth convention | Affected by bug NOW? | Affected by fix? | Notes |
|---|---|---|---|---|
| `_vlasov1d` | $\sqrt{T/m}$ (σ) | **YES** — logged units, physical T (2×), dimensional strings, EM $\hat c$; fixtures lock wrong values | Fix-safe for tests; must regenerate 3 fixtures | See companion doc |
| `_vlasov2d` | $\sqrt{T/m}$ (σ) | **YES (partial)** — same chain: logged units, physical T, EM $\hat c$; ES dynamics immune | **Fix-safe** — all 3 tests self-consistent, **no fixtures** | Cleanest of the affected: already fixed several vlasov1d bugs |
| `_pic1d` | $\sqrt{T/m}$ (σ) | **Logged units only** — dynamics immune (all inputs numeric) | **Fix-safe** — no fixtures, no test depends on it | Loader ≡ vlasov1d initializer convention |
| `_tf1d` | $\sqrt{T/m}$ (σ) | Not via `normalization.py` — but its **own private copy** of the √2 (`modules.py:62`) corrupts its logged units identically | Fix does **nothing** here; `modules.py:62` needs its own `2.0*T0 → T0` | Dynamics/tests fully decoupled from both |
| `_lpse2d` | $\sqrt{T/m}$ (σ), physical units | **IMMUNE** — never touches `normalization.py` | **IMMUNE** | Own ps/µm unit system; Bohm-Gross & Landau literature-exact |
| `_spectrax1d` | AW-Hermite $\alpha=\sqrt{2T/m}$ (self-consistent) | **IMMUNE** — zero imports of `normalization.py` | **IMMUNE** | α is a raw config float; physics identical to σ-convention |
| `_hermite_poisson_1d` | AW-Hermite $\alpha=\sqrt{2T/m}$ (self-consistent) | **IMMUNE** — zero imports | **IMMUNE** | Not even registered in ergoExo dispatch |
| `vfp1d` | $\sqrt{2T/m}$ (most-probable, self-consistent) | **IMMUNE** — uses `laser_normalization` ($v_0=c$), never `electron_debye_normalization` | **IMMUNE if** `vth_norm()` is left alone; **BROKEN if** `vth_norm()` is "fixed" | The reason the fix must not touch `vth_norm()` |
| `osiris` wrapper | OSIRIS native ($u_{th}=\sqrt{T/m}/c$, in-deck) | **IMMUNE** — `skin_depth_normalization` ($T_0$=None, $v_0=c$); no eV→uth conversion exists in the wrapper | **IMMUNE** | `vth_norm()` unreachable (would TypeError on $T_0$=None) |
| `vlasov1d2v` | $\sqrt{T/m}$ (σ), $m$≡1 | **IMMUNE by orphan status** — not in ergoExo dispatch, would KeyError before running | **IMMUNE** | Legacy code; carries many un-fixed bugs (§3.9) |

**Fully immune to both bug and fix:** `_lpse2d`, `_spectrax1d`, `_hermite_poisson_1d`, `osiris`, `vfp1d` (conditional on the fix not touching `vth_norm()`), and `vlasov1d2v` (by virtue of being unrunnable).
**Affected now:** `_vlasov1d`, `_vlasov2d`, `_pic1d` (via `normalization.py`) and `_tf1d` (via its private duplicate).
**Every solver's dimensionless dynamics are unaffected in shipped configs/tests** — the corruption is confined to logged units, physical interpretation of results, dimensional-string inputs, and the EM-branch $\hat c$.

### 1.2 Why no test ever caught the √2

The bug survived every test suite in the repository, and understanding why is itself a finding — the test suites, by construction, cannot see this class of error:

1. **Every physics test works entirely in dimensionless code units.** Landau damping, ion-acoustic, Bohm-Gross, two-stream, gyro — all set $k$, $\omega$, $T_0$, box sizes as numeric floats, which pass through `normalize()` untouched (`normalization.py:57–58`). The buggy conversion layer is simply never on the tested code path. A purely numeric run *is* a correct simulation under the σ-convention; only its physical labels are wrong, and no test reads the labels.
2. **The theory references the tests compare against live in the same code-unit world.** The kinetic roots come from `electrostatic.py` with `maxwellian_convention_factor=2` and $v_{th}=1$ — i.e., the reference is expressed in the engine's internal convention, so agreement validates internal consistency, not the dimensional dictionary. The tests pin *which* convention the engine uses (invaluable for this audit) but cannot detect that `normalization.py` speaks a different one.
3. **Where a test does touch the dimensional layer, it is self-referential.** `_vlasov2d`'s `test_em_dispersion` computes $\hat c$ from the same `electron_debye_normalization` the solver uses and places the driver at $\omega^2 = 1+\hat c^2 k^2$ with that same $\hat c$ — bug and reference move in lockstep, so it passes with the wrong $c$.
4. **The only tests that pin dimensional values pin the *wrong* ones.** The `_vlasov1d` `*_derived_config.yml` regression fixtures were generated under the buggy convention, so they actively *enforce* the √2 values (`c_light: 11.302`, `x0: 12.14 nm`, …) — a fix makes tests fail, not the bug.
5. **No solver has an absolute physical-units validation** (e.g., a Landau rate checked against a rate in Hz for a stated density and temperature, or an SRS resonance checked against a wavelength in nm), and none has a total-energy conservation diagnostic. Either would have caught the mismatch the first time a dimensional input mattered.

The general lesson: dimensionless-physics tests validate the engine; only a test that crosses the units boundary — dimensional input in, dimensional observable out, compared against an independent physical reference — can validate the normalization layer. The repo currently has zero such tests. Adding one per normalization entry point (a "round-trip units test") is the cheapest structural guard against recurrence, and belongs in the fix PR alongside item 6 of §4.

### 1.3 The three thermal-velocity conventions in the codebase

All are *internally* self-consistent within their modules; the mixing hazard is at module boundaries and in shared helpers:

| Convention | Definition | Used by |
|---|---|---|
| σ (RMS / std-dev) | $v_{th} = \sqrt{T/m}$, Maxwellian $e^{-v^2/2}$ at $T{=}1$, $L_0=\lambda_{De}$ | `_vlasov1d`, `_vlasov2d`, `_pic1d`, `_tf1d`, `_lpse2d`, `vlasov1d2v` dynamics; `electrostatic.py` (mcf=2 default); OSIRIS `uth` |
| Most-probable | $v_{th} = \sqrt{2T/m}$ | `vfp1d` (with compensating init factor); `vth_norm()`; **`electron_debye_normalization` (the outlier)**; `_tf1d/modules.py:62` (private duplicate, diagnostics-only) |
| AW-Hermite scale | $\alpha = \sqrt{2T/m}$ as basis parameter; represented Maxwellian still has variance $T/m$ | `_spectrax1d`, `_hermite_poisson_1d` (Schumer–Holloway / Parker–Dellar standard) |

---

## 2. Normalization wiring map

Grep-verified callers of each `normalization.py` entry point:

| Entry point | Definition | Callers |
|---|---|---|
| `electron_debye_normalization` | $v_0=\sqrt{2T_0/m_e}$ ← **the bug**, $L_0=v_0/\omega_{p0}$ | `_vlasov1d`, `_vlasov2d` (`modules.py:55`), `_pic1d` (`simulation.py:78`) |
| `laser_normalization` | $v_0=c$, $L_0=c/\omega_L$, $n_0=n_{crit}$, $T_0$ not self-consistent with $v_0$ (documented) | `vfp1d` (`base.py:17`) only |
| `skin_depth_normalization(_from_frequency)` | $v_0=c$, $L_0=c/\omega_{p0}$, $T_0$=None | `osiris` only |
| `vth_norm()` | $\sqrt{2T_0/m_0}/v_0$ — hard-codes most-probable | **`vfp1d` only** (grid.py:118, base.py:93, 98, 152) |
| `speed_of_light_norm()` | $c/v_0$ | `_vlasov1d`, `_vlasov2d`, `_pic1d` (√2-tainted); `osiris` (=1, exact) |
| `normalize(s, norm, dim)` | numeric passthrough; strings via $L_0$/τ/$v_0$/$T_0$ | vlasov family, `functions.py`, `vfp1d/helpers.py` profiles |
| — (no normalization import) | | `_tf1d` (own inline pint), `_lpse2d` (own unit system), `_spectrax1d`, `_hermite_poisson_1d` (raw α floats), `vlasov1d2v` (reads cfg keys nothing populates) |

Shared physics kernels:
- `electrostatic.py` — dispersion/Z-function utilities. Self-consistent under σ-convention: `maxwellian_convention_factor=2` (default, used by **every** caller in the repo) means $f_0\propto e^{-v^2/2v_{th}^2}$, $\xi=\omega/(\sqrt2 k v_{th})$. This is the reference that pins the working convention of `_vlasov1d`, `_vlasov2d`, `_pic1d`, `_tf1d`, `_spectrax1d`, `_hermite_poisson_1d` tests.
- `driftdiffusion.py` — Dougherty/LB kernel; measures $T=\langle(v-\bar v)^2\rangle$ (or spherical $\langle v^4\rangle/3\langle v^2\rangle$), relaxes to that width. Convention-agnostic and self-adapting; verified clean in both audits.
- `functions.py` — envelope/profile layer for the vlasov family. Numeric inputs pass through; dimensional strings go through $L_0$ (√2-tainted until the fix). One bug found (§3.10).
- `utils.py` — no physics content.

---

## 3. Per-module findings

Finding IDs continue the companion doc's F-series where the bug is a copy; new IDs are per-module.

### 3.1 `_vlasov2d` — affected now (units/EM), fix-safe, partially cleaned-up lineage

Convention: σ, identical to `_vlasov1d` (`helpers.py:62` `v_th=√(T0/mass)`, same super-Gaussian α; 2-D init correctly normalized, $\iint f\,d^2v = n$, equal widths in $v_x,v_y$). Landau test pins σ=1 via the shared kinetic roots. Pushers all unity-coefficient; TE-mode Maxwell curl signs verified; initial 2-D Poisson correct; magnetic rotation $\theta=-(q/m)B_z dt$ verified by the gyro test.

Bug-copy status vs `_vlasov1d`: **F5 fixed** (vbar divides by n, `fokker_planck.py:32`), **F6 fixed** (moments centered on $u=j/n$, `storage.py:100`), **F11 fixed** (`dx=(xmax-xmin)/nx`, `grid.py:60`), F7 N/A (no entropy diagnostic). **F4 still present** (Krook target $e^{-v_x^2/2}e^{-v_y^2/2}$ hard-coded, `fokker_planck.py:127–131`; latent, off in configs). **F10 still present** (`float(cfg.T0/v0x/v0y)`, `simulation.py:158–163`). **F8 present** (same α, latent at m=2).

√2 exposure: calls `electron_debye_normalization` (`modules.py:55`) → logged units √2-wrong (F2 analog), physical T 2× label (F1 analog), EM $\hat c = c/v_0$ √2-small (F3 analog). `test_em_dispersion` passes *because it is self-consistent, not correct* — it recomputes the same wrong $\hat c$ for the driver. **No regression fixtures exist**, so the fix requires no fixture regeneration; all three tests pass unchanged (the EM test moves in lockstep with the fix).

New (minor): `mean_KE` has the ½, field proxies $\langle E^2\rangle,\langle B^2\rangle$ don't (no combined conserved-energy diagnostic); `Txx/Tyy` are pressures ($n\cdot$variance) despite the T-name, and `T` omits the mass factor (fine for electrons); the separable per-axis Dougherty never isotropizes $T_x \leftrightarrow T_y$ (no cross-axis coupling operator exists in this module); dead config knobs `UnitsConfig.laser_wavelength`, `Z`.

### 3.2 `_pic1d` — dynamics immune, logged units affected, fix-safe

Convention: σ. The particle loaders (`helpers.py:34` quiet inverse-CDF, `:58` random rejection) use `v_thermal = np.sqrt(T0/mass)` — byte-for-byte the vlasov1d initializer convention. Pinned independently by `test_bohm_gross.py:34` (kinetic dielectric with vth=1) and `test_landau_damping.py:38–41` ($\omega^2=1+3k^2$, σ=1 Landau rate).

Verified clean: deposit/gather use the identical B-spline kernel (momentum-conserving, self-force-free); uniform loading deposits exactly $n=1$ in expectation ($w=n_0L/N$, partition of unity); Poisson $E_k=-i\rho_k/k$ unity prefactor; KDK leapfrog and Yoshida4 push coefficients exact; ponderomotive $(q/m)^2(-\tfrac12\partial_x a^2)$ matches vlasov1d's verified chain.

√2 exposure: `write_units` (`modules.py:44–57`) logs √2-wrong `v0, x0, c_light, box_length` (a 2000 eV epw run is physically 4000 eV). Dynamics immune: every shipped config and test input is numeric; loader reads raw `T0`. **No fixtures.** Fix is strictly an improvement — nothing breaks.

New findings:
- **P1 (confirmed):** energy diagnostics mix extensivity: `mean_KE = 0.5·m·Σw v²` is a box integral with the ½; `mean_e2 = mean(e²)` is a per-cell mean without ½ or dx. `mean_KE + mean_e2` is not conserved; no valid energy monitor exists.
- **P2 (confirmed):** quiet and random loaders truncate velocity differently for drifting species — quiet clips to $[-v_{max}, v_{max}]$ absolute (`helpers.py:38`), random clips to $[v_0-v_{max}, v_0+v_{max}]$ (`helpers.py:65`). Same config, two different realized distributions; quiet wrongly clips the high-v side of a drifting Maxwellian. Latent for shipped cold-beam decks.
- **P3 (confirmed):** `mean_KE/mean_p` are sums but named `mean_*` (naming, cf. F6-class).
- Inherited: F11 (`dx=xmax/nx` while particle wrap and placement correctly use $x_{max}-x_{min}$ — the field grid and particle box disagree for $x_{min}\ne0$), F10, F8.

### 3.3 `_tf1d` — decoupled from `normalization.py`, but carries a private copy of the √2

Convention: σ. Derived from the pushers: linearizing continuity + momentum ($-u\partial_x u - \frac{1}{n}\partial_x(p/m) - \frac{q}{m}E$) + adiabatic energy ($\gamma=3$) + Poisson gives $\omega^2 = 1+3k^2$, exactly what `test_resonance.py:19` asserts; the kinetic branch uses the shared roots (mcf=2). The Poisson sign ($E_k=+i\rho_k/k$, i.e. $\partial_x E=-\rho$) and the momentum sign ($-(q/m)E$) are *both* opposite the textbook and cancel — verified stable by derivation and by the passing test. Landau closure decays the field at exactly $2\,\mathrm{Im}\,\omega$, matching `test_landau_damping.py:67`.

**T1 (confirmed, the headline):** `_tf1d/modules.py:62` re-implements `v0 = √(2·T0/m_e)` inline in its own `write_units()` (with its own pint registry — no import of `normalization.py`). Every tf1d run logs `v0, x0, c_light, beta, box_length, sim_duration` √2-wrong and implies 2× the stated temperature. Damage confined to logged diagnostics (no dimensional-string inputs exist in tf1d; the EM `WaveSolver` branch is commented out; no fixtures). **Fixing `normalization.py` does not fix this** — `modules.py:62` needs its own `2.0*T0 → T0` in lockstep.

Other findings:
- **T2 (confirmed):** the kinetic-γ pressure closure (`pushers.py:195` `wr_corr=(wrs²-1)/k²` + γ forced to 1) yields $\omega^2 = (w_{rs}^2-1)T_0+1$ — exact only at $T_0=1$. Convention- and T₀-locked, same class as vlasov1d's Krook (F4). Latent (all configs use $T_0=1$).
- **T3 (confirmed):** `modules.py:78` `nuee_norm = nuee/wp0` — the *correct* Hz→code conversion — is computed and then **discarded** (never logged, never used). Meanwhile `physics.<species>.trapping.nuee` is an unrelated raw ML-input float. There is no collisional friction in the tf1d equations at all. (F9-class trap.)
- **T4 (confirmed):** `docs/source/usage/tf1d.md` momentum equation has a spurious $1/n$ on the E-force and mislabels the Poisson equation. Docs-only.
- **T5 (suspected, latent):** `EnergyStepper` evolves $p$ but advects/compresses $p/m$ — ambiguous mass normalization of the pressure variable; inconsistent with the momentum equation for mobile ions ($m=1836$). All configs have ions off.
- **T9 (minor):** `resonance_search.yaml` sets `ion.landau_damping: True` with `ion.is_on: False` (inert, misleading).

### 3.4 `_lpse2d` — fully immune; thermally clean; unrelated bugs found

Physical-units solver (ps/µm, Gaussian-cgs-derived) with its own `write_units` (`helpers.py:71`); zero imports of `normalization.py`. Convention: $v_{te} = c\sqrt{T_e/511}$ = $\sqrt{T_e/m_e}$ (σ) used consistently in the Bohm-Gross term ($e^{-i\,1.5\,v_{te}^2 k^2/\omega_{pe}\,dt}$, coefficient 3/2 correct for σ), the Landau damping rate (verified literature-exact: $\sqrt{\pi/8}(1+\tfrac32 k^2\lambda_D^2)(k\lambda_D)^{-3}e^{-3/2-1/(2k^2\lambda_D^2)}$), the sound speed, and the TPD threshold. Numerically cross-checked: the test config's driver $\omega = 1.5k^2v_{te}^2/\omega_{p0} = 19.7 \approx 20$ as configured. TPD/SRS coupling constants contain no thermal factor (convention-independent); laser $E_0=\sqrt{8\pi I/c}$ and WKB swelling $(1-n/n_c)^{-1/4}$ verified.

Findings (none √2-class):
- **L1 (confirmed):** logged `lambda_D = vte/w0` (`helpers.py:111`) divides by the **laser** frequency instead of $\omega_{pe}$ — factor 2 too small at $n_c/4$. Diagnostic only (dynamics compute $\omega_{pe}^2/k^2v_{te}^2$ directly).
- **L2 (confirmed, latent runtime bug):** the `E2` electrostatic-driver path calls `self.epw.driver(...)` (`core/vector_field.py:84`) but the active `SpectralEPWSolver` has no `driver` attribute (only the commented-out `SpectralPotential` does) → `AttributeError` for any config with an `E2` driver. Latent only because `test_epw_frequency` is currently disabled (`pass` body) — meaning **the module's dispersion conventions have no live regression test**.
- **L3 (confirmed):** `core/trapper.py` is dead code (never instantiated by `SplitStep`); it is a stale duplicate of `_tf1d`'s trapper. Note: it uses `electrostatic.py`, *not* `driftdiffusion.py` — the vlasov1d audit's cross-module note claiming lpse2d uses the Dougherty kernel is wrong for the current tree.
- **L4 (confirmed):** the σ-convention is undocumented in the lpse2d docs — the main *future* √2 risk here is someone "harmonizing" it onto `normalization.py`.
- **L5 (minor):** the Landau/Bohm-Gross formulas are duplicated byte-for-byte in `SpectralPotential` and `SpectralEPWSolver` (drift risk); `nu_coll`'s `/2` (amplitude vs energy rate) is correct but easy to double-count.

### 3.5 `_spectrax1d` and `_hermite_poisson_1d` — fully immune; self-consistent AW-Hermite basis

Both use the asymmetrically-weighted Hermite basis (Schumer–Holloway / Parker–Dellar) with scale $\alpha = \sqrt{2T/m}$ taken as a **raw config float** — zero imports of `normalization.py`. The represented Maxwellian has variance $\alpha^2/2 = T/m$: physically identical to the σ-convention solvers. Verified by derivation (streaming ladder $\alpha\sqrt{n/2}$ + force ladder $\sqrt{2n}/\alpha$ + Ampère/Poisson coupling → Langmuir $\omega=\omega_{pe}$ exactly, Bohm-Gross $\omega^2=1+3(k\lambda_{De})^2$ with $\lambda_{De}=\alpha/\sqrt2$) and by tests: both modules' Landau tests set $\alpha_e = \sqrt2\,k\lambda_D/k$ explicitly against the shared mcf=2 kinetic roots (2%/5% tolerance for hermite-poisson; a cross-module test pins the two modules' E-coupling equal to 1e-12).

The coincidence that $\alpha=\sqrt{2T/m}$ matches the `normalization.py:91` outlier is harmless — there is no code coupling in either direction. Both integrator paths (DoPri8 and Lawson-RK4 exponential operators) share identical ladder coefficients. The historical inverted E-coupling bug in hermite_poisson (`C[n+1]` vs `C[n-1]`) is fixed and regression-locked by three tests.

Findings (all diagnostic-label/minor): **S1** logged `lambda_D` is the *total* (ion-dominated) Debye length while `k_norm` uses electron-only and hard-codes mode 1 (`base_module.py:186,189`); **S2** the ion "temperature" diagnostic is a velocity variance missing the $m_i$ factor ($T_i/m_i$, misleading next to $T_e$; `storage.py:386–405`); **S3** shipped `landau-damping.yaml` has `Nn: 4` — far too few Hermite modes to resolve the damping it is named for (tests override to 512/32); `_spectrax1d/helpers.py` is an all-stub file not on any code path.

### 3.6 `vfp1d` — immune to the bug; the reason the fix must not touch `vth_norm()`

Uses `laser_normalization` (`base.py:17`): $v_0 = c$, $L_0 = c/\omega_L$, $n_0 = n_{crit}$, $T_0$ = reference eV (documented as not self-consistent with $v_0$). Velocity grid in units of $c$. Thermal convention: **most-probable, $v_{th} = \sqrt{2T/m}$**, deliberately and self-consistently:

- `vth_norm()` supplies $\sqrt{2T_0/m}/c$ (4 call sites, all vfp1d);
- the initializer's width factor `helpers.py:111` is $\alpha = \sqrt{3\Gamma(3/m)/(2\Gamma(5/m))}$ — note the extra `/2` vs the vlasov-family α — giving $\alpha=1$ at $m{=}2$ and realized variance exactly $T/m$ (the √2 and the /2 cancel);
- the v-grid extent `grid.py:118` `vmax = 8·vth_norm()/√2` = 8 standard deviations (the /√2 converts most-probable→σ);
- the IB coefficient `base.py:81` ($0.093373\,\lambda_{\mu m}^2/T_{keV}$ per $10^{15}$ W/cm²) normalizes $v_{osc}^2$ to $2T/m$, consistent;
- the temperature diagnostic `storage.py:317–322` uses the spherical variance $T = \langle v^4\rangle/3\langle v^2\rangle$ — reads the true temperature, no √2.

Collision operators ($\nu_{ee}$ coefficient anchored to $v_0=c$ via $r_e$, Lorentz e–i with Epperlein–Haines Z*, Rosenbluth I/J integrals) are temperature-convention-independent. **No internal √2 inconsistency exists.**

**C1 (confirmed, cross-module, high):** because items above are keyed to `vth_norm()` while the compensating α is a separate constant, redefining `vth_norm()` → $\sqrt{T_0/m}/v_0$ (as the vlasov1d audit's §5.1 originally suggested) silently initializes VFP-1D at **half** the intended temperature while all transport coefficients stay at $T_0$; the Spitzer/Epperlein–Haines gold test (`test_kappa_eh.py`, $\kappa\propto T^{5/2}$) would likely fail, and the pure-operator tests would *not* catch it. **The fix must be confined to `electron_debye_normalization:91–92`.** (The companion doc's §5.1 has been amended accordingly.)

Other findings: **C2** `base.py:118` stores the bound method `norm.vth_norm` instead of calling it (harmless repr-string in cfg; wrong); **C3** `storage.py:170` hard-codes $9.09\times10^{21}$ cm⁻³ per $n_c$ — that is $n_{crit}(351\,\mathrm{nm})$; wrong labeling for any other `laser_wavelength` (should derive from `norm.n0`); **C4** = F11 copy (`grid.py:71` `dx=xmax/nx`); **C6/C7 (suspected, latent — IB not in any shipped config):** the production inverse-bremsstrahlung wiring looks under-normalized — `w0_norm` is identically 1.0 under laser normalization, the Langdon-factor argument $Z^2 n_i/(\omega_0 v^3)$ carries no collision-frequency coefficient, and `vosc2_per_intensity` (thermal-speed² units) is consumed as $v_{osc}^2$ in grid ($c$) units — a $\sim(c/v_{th})^2$ discrepancy between the unit tests (where the grid unit *is* the thermal speed) and production. Recommend a Spitzer-IB validation run before enabling IB in production.

### 3.7 `osiris` wrapper — fully immune

Uses `skin_depth_normalization(_from_frequency)` ($v_0=c$, $T_0$=None). The wrapper drives a native OSIRIS deck: **no eV→`uth` conversion exists anywhere in the wrapper** — `uth` values pass through verbatim from the deck, so the T→v √2 trap cannot occur. Verified exact: $a_0 \to E_{peak} = a_0\omega_L m_e c/e \to I = E^2\epsilon_0 c/2$ (matches $I\lambda_{\mu m}^2 = 1.37\times10^{18}a_0^2$, test-confirmed); `c_light = beta = 1.0` identically; box lengths in true skin depths; `units.yaml` deliberately omits temperature-dependent keys (consistent with $T_0$=None). Temperature diagnostics (`plots.py:1004` $T=\sum u_{th,i}^2$ in $m_ec^2$ units; `:1059` momentum variance) carry no spurious factors. `vth_norm()` is unreachable (would TypeError on $T_0$=None).

**C8 (low, latent):** the adaptive-box feature's `reference_density = 0.25` "quarter-critical" (`density.py:75`) assumes the deck's $n_0 = n_{crit}$ (true for LPI decks with `omega0=1`); a deck normalized to a different density would silently mis-scale the density bounds (the *length* normalization stays correct).

### 3.8 Shared `electrostatic.py` and `functions.py`

`electrostatic.py`: self-consistent under the σ-convention throughout. $Z$ via Faddeeva, $Z' = -2(1+\xi Z)$ correct; `maxwellian_convention_factor=2` (default, used by every caller in the repo) ⇒ $f_0\propto e^{-v^2/2v_{th}^2}$, root returned as $\xi k v_{th}\sqrt{2} = \omega$. One cosmetic blemish (**T7**): the Newton `initial_root_guess` is ω-scaled ($\sqrt{\omega_p^2+3k^2v_{th}^2}$) but the root variable is ξ-scaled — converges anyway.

`functions.py`: numeric inputs pass through; `EnvelopeFunction` amplitudes (`baseline`, `bump_height`) are raw floats (the F9 collision-frequency trap); `SineFunction.wavenumber` correctly uses `dim="k"`. **T8 (suspected):** `LinearFunction`/`ExponentialFunction` normalize `val_at_center` — a *density* — with `dim="x"` (`functions.py:159–161, 178–180`); dimensionally wrong for any string input.

### 3.9 `vlasov1d2v` — orphaned legacy; a museum of the known bugs plus three new ones

Not registered in the ergoExo dispatch (`_base_.py:294–333` → NotImplementedError); reads `cfg["units"]["derived"]` keys nothing populates → cannot run. Never calls `normalization.py`. Convention: σ (init `exp(-(v_x^2+v_y^2)/2T_0)`, mass≡1; the super-Gaussian machinery is commented out so `m` is silently ignored).

Carries live copies of: **F5** (`fokker_planck.py:93,98` — vbar *and* the diffusion coefficient computed without ÷n), **F6 + F7** (`storage.py:152–158, 296`), **F10**, **F11** (`helpers.py:211`). F4 present but dead (Krook never instantiated).

New bugs (would matter if ever revived):
- **N1 (confirmed):** the density profile is computed for every basis and folded into the ion background (`helpers.py:287`), but the line applying it to $f$ (`helpers.py:83–85`) is **commented out** — electrons always start uniform, so non-uniform configs begin with spurious net charge and no actual density perturbation in $f$.
- **N2 (confirmed):** the explicit Dougherty substeps apply ν twice (`fokker_planck.py:134–135, 141–142`: `dfdt = nu*ddx(...)` then `f + dt*nu*dfdt`) — collisional relaxation runs at $\nu^2$.
- **N4 (confirmed):** `integrator.py:234` passes the **nu_ee** envelope args when building the **nu_ei** profile (copy-paste) — the configured nu_ei profile is silently ignored.
- **N3 (design note):** without collisions the $v_y$ dimension is dynamically inert (no $v_y$ force, streaming by $v_x$ only) — the entire $n_{vy}$ grid is dead compute except through the FP operators.
- Minor: default scalars broadcast over the $v_y$ axis only (`storage.py:292–295`).

Recommendation: either delete `vlasov1d2v` or quarantine it with a module-level comment; in its current state it is a trap for anyone who greps for reference implementations.

### 3.10 Cross-module systemic observations

1. **Driver amplitude conventions differ across the Vlasov family** (N5): `_vlasov1d` Ex driver $\propto \omega a_0$; `_vlasov2d` current source $\propto \omega^2 a_0$; `vlasov1d2v` $\propto |k| a_0$. Same config key, three physical meanings. Worth unifying or documenting per-solver.
2. **No solver has a valid total-energy conservation diagnostic.** vlasov1d/vlasov2d/pic1d all mix ½-factors and extensivity between kinetic and field terms. A per-solver conserved-energy scalar is the single cheapest guard that would have caught the √2 class of bugs.
3. **The F-series bugs are lineage-correlated:** `_vlasov2d` (newest refactor) fixed F5/F6/F11; `_pic1d` inherits F10/F11 by importing `_vlasov1d`'s grid/simulation; `vlasov1d2v` (oldest) has everything. When fixing `_vlasov1d`, fix the copies in the same PR: Krook targets (`_vlasov1d`, `_vlasov2d`, dead `vlasov1d2v`), `dx`/xmin (`_vlasov1d/grid.py:55`, `vfp1d/grid.py:71`, `vlasov1d2v/helpers.py:211`), raw-float T0/v0 (all three vlasov-family simulation.py files).
4. **Hz→code-unit collision-frequency conversion** exists correctly in exactly one place (`_tf1d/modules.py:78`) and is dead there; nowhere is it live. Every solver that takes a collision frequency takes it as a raw code-unit float with undocumented units.

---

## 4. Consolidated fix checklist (supersedes §5 of the vlasov1d audit where they differ)

**The √2 fix, safely scoped:**
1. `normalization.py:91–92`: $v_0 \to \sqrt{T_0/m_e}$, $x_0 \to \lambda_{De}$ — inside `electron_debye_normalization` **only**.
2. **Do NOT change `vth_norm()`** (`normalization.py:48–50`) — it is vfp1d-private and correct for vfp1d (C1). If its name is judged misleading, rename to `most_probable_speed_norm()`; do not change the value.
3. `_tf1d/modules.py:62`: `2.0*T0 → T0` (private duplicate, T1).
4. Regenerate the three `_vlasov1d` `*_derived_config.yml` regression fixtures (only vlasov1d has fixtures).
5. Fix the convention-locked Krook targets in the same PR (`_vlasov1d/…/fokker_planck.py:245`, `_vlasov2d/…/fokker_planck.py:127–131`): build from species T0/mass.
6. Re-validate: vlasov1d Landau + ion-acoustic (must pass unchanged), vlasov2d Landau/gyro/EM-dispersion (EM moves in lockstep), pic1d all three (unchanged), vfp1d `test_kappa_eh` (must pass unchanged — the sentinel that `vth_norm()` was left alone), numeric-driver SRS configs re-tuned per vlasov1d F3.
7. Document the per-solver conventions in each solver's docs page (σ vs most-probable vs AW-α; see §1.3) — the biggest residual risk is a future "harmonization" that imports the wrong convention into a currently-clean module (especially `_lpse2d`, L4).

**Independent bug fixes, by priority:**
- High (live, wrong physics if exercised): vfp1d IB normalization audit before production use (C6/C7); lpse2d `E2` driver path + re-enable `test_epw_frequency` (L2).
- Medium (live, wrong diagnostics): lpse2d logged `lambda_D` (L1); pic1d/vlasov2d/vlasov1d energy-diagnostic normalization + add conserved-energy scalars; spectrax ion-T mass factor (S2); vfp1d hard-coded $n_{crit}(351\,nm)$ (C3); pic1d quiet-loader drift truncation (P2).
- Low / hygiene: tf1d docs (T4) and dead `nuee_norm` (T3 — log it instead of dropping it); `functions.py` `val_at_center` dim (T8); vfp1d bound-method store (C2); spectrax `Nn:4` config (S3); delete or quarantine `vlasov1d2v` and lpse2d's dead trapper (L3); unify driver-amplitude conventions or document them (N5).

---

## 5. What was verified clean (highlights)

- All solver cores' dimensionless coefficients: vlasov1d/2d pushers and field solves, pic1d push/deposit/gather/Poisson, tf1d fluid system (including its double sign cancellation, verified by derivation), Hermite ladder algebra in both spectral modules (verified by derivation to the Langmuir and Bohm-Gross limits), lpse2d Bohm-Gross/Landau/TPD/SRS coefficients against literature, vfp1d collision coefficients and temperature diagnostics, osiris a0↔intensity conversions.
- The shared `driftdiffusion.py` Dougherty kernel (both audits): self-adapting, Buet factor-2 correctly absorbed, spherical temperature correct.
- `electrostatic.py`'s Z-function algebra and mcf=2 semantics — the single reference that consistently pins six modules' tests to the σ-convention.

---

*Companion deep-dive for `_vlasov1d`: `VLASOV1D_CONVENTIONS_AUDIT.md`. Method: six parallel independent module audits (one Opus agent per slice) with all load-bearing claims re-verified against source before inclusion. No source files were modified.*
