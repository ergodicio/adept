# Vlasov-1D Physics & Conventions Audit

**Date:** 2026-07-16
**Scope:** `adept/_vlasov1d/` (all files), plus the shared code it depends on: `adept/normalization.py`, `adept/driftdiffusion.py`, `adept/electrostatic.py`, and the `tests/test_vlasov1d/` suite and example configs.
**Trigger:** the discovery that `normalization.py:91` uses the $v_0 = \sqrt{2T_0/m}$ (most-probable-speed) convention while `_vlasov1d/helpers.py:69` uses $v_{th} = \sqrt{T_0/m}$ (RMS / standard-deviation convention). Neither has been changed yet; this audit determines which convention the module actually runs in, inventories every equation, and lists all inconsistencies found.

---

## 1. Executive summary

**The engine's working convention is $v_0 = \sqrt{T_0/m}$.** The distribution initializer, the Fokker–Planck operator, the Krook operator, the Landau-damping and ion-acoustic tests (the module's quantitative gold standards), and every numeric example config are all mutually consistent under: code velocity in units of the thermal *standard deviation* $\sqrt{T_0/m_e}$, code length unit $L_0 = \lambda_{De}$, code wavenumber $= k\lambda_{De}$, Maxwellian $\propto e^{-v^2/2}$ at $T=1$.

**`adept/normalization.py:91–92` is the sole outlier** (`electron_debye_normalization`: $v_0 = \sqrt{2T_0/m_e}$, $L_0 = \sqrt{2}\,\lambda_{De}$). Because numeric config inputs pass through `normalize()` untouched, this does **not** corrupt the dynamics of numeric-input runs — but it means:

- A run with `normalizing_temperature: 2000eV` and species `T0: 1.0` is physically a **4000 eV** plasma (factor 2 in temperature).
- Every dimensional **string** input converted with $L_0$ (box sizes in µm, gradient scale lengths, laser $k_0$) is off by $\sqrt{2}$.
- Every **logged** physical unit (`v0`, `x0`, `c_light`, `box_length`) is off by $\sqrt{2}$, and the regression fixtures currently lock in those wrong values.
- The EM wave speed $\hat c = c/v_0$ fed to the wave solver is $\sqrt2$ too small relative to the engine's thermal unit (with a partial cancellation for wavelength-specified drivers; see F3).

**Recommended fix direction:** change `normalization.py` to $v_0 = \sqrt{T_0/m_e}$ (so $L_0 = \lambda_{De}$), *not* the initializer. Fixing `helpers.py:69` instead (to $\sqrt{T_0/2m}$) would break the currently-passing Landau-damping and ion-acoustic tests and invalidate every existing config's driver $(k_0, \omega_0)$ values. Section 5 lists everything that must move in lockstep.

The collisionless solver core (`vector_field.py`, pushers) was verified **clean**: every coefficient is exactly unity under the declared normalization and is convention-independent. The convention bug lives entirely in the dimensional-conversion layer.

Beyond the $\sqrt2$ issue, the audit found several independent bugs and traps, listed in Section 4 (notably: FP `compute_vbar` missing $1/n$; storage central moments centered on $n u$ instead of $u$; a sign inconsistency between the two `-flogf` entropy diagnostics; the super-Gaussian $\alpha$ fixing the wrong moment for $m\ne2$; the Krook target hard-coded to $T=1$; and the collision-frequency normalization chain being entirely manual).

---

## 2. The two candidate conventions

With $\omega_{p0} = \sqrt{n_0 e^2/(\epsilon_0 m_e)}$, $\tau = 1/\omega_{p0}$ in both cases:

| Quantity | (a) `normalization.py` as written | (b) engine's actual convention |
|---|---|---|
| Velocity unit $v_0$ | $\sqrt{2T_0/m_e}$ (most-probable speed) | $\sqrt{T_0/m_e}$ (RMS / std-dev) |
| Length unit $L_0 = v_0/\omega_{p0}$ | $\sqrt{2}\,\lambda_{De}$ | $\lambda_{De}$ |
| Maxwellian at $\hat T = 1$ | $\propto e^{-v^2}$ (variance ½) | $\propto e^{-v^2/2}$ (variance 1) |
| Code wavenumber $\hat k$ | $\sqrt2\, k\lambda_{De}$ | $k\lambda_{De}$ |
| Bohm–Gross | $\hat\omega^2 = 1 + \tfrac{3}{2}\hat k^2$ | $\hat\omega^2 = 1 + 3\hat k^2$ |
| $\hat c = c/v_0$ (2000 eV) | 11.30 | 15.98 |

**Why the test suite never caught this:** every physics test (Landau, ion-acoustic, EM dispersion) works in dimensionless code units — numeric inputs bypass `normalize()` entirely (`normalization.py:57–58`), so the buggy conversion layer is never on the tested path, and the theory references (`electrostatic.py`, mcf=2, $v_{th}=1$) are expressed in the engine's internal convention, so agreement validates internal consistency rather than the physical-units dictionary. Worse, the only tests that *do* pin dimensional values — the `*_derived_config.yml` regression fixtures — were generated under the buggy convention and therefore lock the wrong values in. No test crosses the units boundary (dimensional input in, dimensional observable out, against an independent physical reference), which is precisely the kind of test that must accompany the fix (see `ADEPT_CONVENTIONS_AUDIT.md` §1.2 for the full analysis).

Evidence pinning convention (b) as the working one, strongest first:

1. **Landau-damping test** (`tests/test_vlasov1d/test_landau_damping.py:24`) calls `electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, k)` — i.e. $\omega_{pe}=1$, $v_{th}=1$ with `maxwellian_convention_factor=2` (`adept/electrostatic.py:79,115,98`), which is the kinetic dielectric for $f_0 \propto e^{-v^2/2v_{th}^2}$, $\xi = \omega/(\sqrt2 k v_{th})$. The measured $\omega(k)$ matches these roots to 2 decimals only if the simulated Maxwellian has $\sigma = 1$. (E.g. $k=0.3 \Rightarrow \omega = 1.1598$, which is exactly the `w0` in `resonance.yaml` and `epw.yaml`; under convention (a) the resonance would sit near 1.07 and the test would fail.)
2. **Ion-acoustic test** (`test_ion_acoustic_wave.py`) uses $c_s^2 = ZT_e/m_i$ and $\omega^2 = k^2c_s^2/(1+k^2\lambda_D^2)$ with $\lambda_D = 1$ in code units — asserting $L_0 = \lambda_{De}$.
3. **Initializer** (`helpers.py:69–76`), **Krook target** (`fokker_planck.py:245`), and **Dougherty stationary state** (`fokker_planck.py:51,96` + `driftdiffusion.py`) all build/assume $e^{-v^2/(2T/m)}$, variance $T/m$.
4. Every numeric example config (`epw.yaml`, `resonance.yaml`, `wavepacket.yaml`, `bump-on-tail.yaml`) uses driver $(k_0,\omega_0)$ pairs consistent only with $\hat k = k\lambda_{De}$ and $\sigma = \sqrt{T/m}$.

---

## 3. Equation inventory

### 3.1 Normalization layer (`adept/normalization.py`)

| Location | Expression | Meaning | Convention note |
|---|---|---|---|
| :88 | $\omega_{p0} = \sqrt{n_0e^2/\epsilon_0 m_e}$, $\tau = 1/\omega_{p0}$ | time unit | standard |
| **:91** | $v_0 = \sqrt{2T_0/m_e}$ | velocity unit | **outlier — convention (a)** |
| :92 | $x_0 = v_0/\omega_{p0} = \sqrt2\lambda_{De}$ | length unit | inherits $\sqrt2$ |
| :48–50 | `vth_norm()` $= \sqrt{2T_0/m_0}/v_0 = 1$ | "thermal velocity" | convention (a); **never called** in `_vlasov1d` |
| :52–53 | `speed_of_light_norm()` $= c/v_0$ | $\hat c$ | inherits $\sqrt2$ (F3) |
| :37–38 | NRL $\log\Lambda_{ee}$ | Coulomb log | uses reference $T_0$ |
| :45 | $\nu_{ee} = 2.91\times10^{-6}\, n_{cc}\log\Lambda\, T_{eV}^{-3/2}$ Hz | NRL e–e rate | **diagnostic only**, never converted to code units (F9) |
| :56–74 | `normalize()`: x/L0, t/τ, v/v0, T/T0, k·L0 | dim→code | numeric inputs pass through untouched; `dim="temp"` and `dim="v"` branches are dead code for `_vlasov1d` |

### 3.2 Initialization & grids (`helpers.py`, `grid.py`, `modules.py`, `simulation.py`, `datamodel.py`)

| Location | Expression | Meaning | Convention note |
|---|---|---|---|
| helpers.py:65–66 | `dv = 2 vmax/nv`; cell-centered `vax` on $[-v_{max}+dv/2,\ v_{max}-dv/2]$ | v-grid | edges at $\pm v_{max}$; half-cell center offset |
| **helpers.py:69** | $v_{th} = \sqrt{T_0/m}$ | thermal width | **convention (b)** |
| helpers.py:72 | $\alpha = \sqrt{3\,\Gamma(3/m)/\Gamma(5/m)}$ ($=\sqrt2$ at $m{=}2$) | super-Gaussian width factor | fixes $\langle v^4\rangle/\langle v^2\rangle = 3v_{th}^2$; variance $=v_{th}^2$ **only** at $m=2$ (F8) |
| helpers.py:74–76 | $f \propto \exp[-|(v-v_d)/(\alpha v_{th})|^m]$ | init EDF; $m{=}2$: $e^{-(v-v_d)^2/(2T_0/m)}$ | variance $T_0/m$ — convention (b) |
| helpers.py:80,84 | `f /= sum(f)·dv`; `f *= n_prof` | $\int f\,dv = n(x)$ | midpoint rule, consistent with all moments |
| grid.py:55 | `dx = xmax/nx` | cell width | **ignores `xmin`** (F11) |
| grid.py:59–60 | `dt = min(dt, 0.95·dx/c)`, $c = 1/\beta$ | EM CFL | $\hat c$ inherits F1's $\sqrt2$ |
| grid.py:65–66,75 | `nt = int(tmax/dt+1)`; `t = linspace(0, dt·nt, nt)` | time axis | overshoot ≤ dt; `grid.t` spacing ≠ dt (F12) |
| grid.py:74,77 | cell-centered x; `kx = 2π·fftfreq(nx, d=dx)` | spectral grid | inherits F11's dx for `xmin≠0` |
| modules.py:112 | `box_length = (xmax−xmin)·L0` | box size in µm | uses `xmin` correctly (unlike grid.py:55); value carries F1's $\sqrt2$ |
| simulation.py:213–214 | `v0 = float(cfg.v0)`; `T0 = float(cfg.T0)` | species drift & temperature | bare code floats, bypass `normalize()` (F10) |
| simulation.py:74–76 | $a_0 = a_{0,std}\cdot(c/v_0)$ | intensity → quiver velocity in $v_0$ units | internally consistent with pusher (§3.3) |
| simulation.py:79–85 | $\hat k_0 = k_{phys}L_0$, $\hat\omega_0 = \omega_{phys}\tau$ | wavelength driver → code units | $\hat k_0$ carries F1's $\sqrt2$; $\hat\omega_0/\hat k_0 = \hat c$ ✓ |
| simulation.py:87 | `dw0 = 0.0  # ???` | frequency offset placeholder | unresolved (F12) |

### 3.3 Solver core (`solvers/vector_field.py`, `solvers/pushers/vlasov.py`, `solvers/pushers/field.py`) — **verified clean**

The dimensionless system implemented, with every coefficient exactly unity under the declared normalization (and independent of the (a)/(b) choice, since $v_0$ cancels):

$$\partial_t f_s + v\,\partial_x f_s + \frac{\hat q_s}{\hat m_s}\left(E - \frac{\hat q_s}{2 \hat m_s}\partial_x a^2\right)\partial_v f_s = C[f_s]$$
$$\partial_x E = \sum_s \hat q_s \int f_s\,dv \ (+\ \text{static ion background}), \qquad \partial_t E = -\sum_s \hat q_s \int v f_s\,dv$$
$$\partial_t^2 a = \hat c^2\,\partial_x^2 a - n_e\, a + S$$

| Location | Code | Check |
|---|---|---|
| vlasov.py:210–214, 220 | exact spectral shift $f(x - v\,dt)$ | coefficient $v_0\tau/L_0 = 1$ ✓ |
| vlasov.py:83–87, 127–129 | `force = q·e + (q²/m)·pond; accel = force/m` | $\hat q/\hat m$ E-push; ponderomotive $(\hat q^2/\hat m^2)$ ✓ |
| vector_field.py:411 | `pond = −0.5·∂ₓ(a²)` | exact instantaneous $-\frac{1}{2m}\partial_x p_\perp^2$ with $p_\perp = qA$; the $c/v_0$ scaling of $a_0$ (simulation.py:76) makes $a$ the quiver velocity in $v_0$ units — **no hidden $\sqrt2$; verified exact** |
| field.py:203–224 | $E_k = -i\rho_k/k$ | Poisson prefactor $e^2n_0/(\epsilon_0 m_e\omega_{p0}^2) = 1$ ✓ |
| field.py:263–264, 280 | $E^{n+1} = E^n - dt\,j$ | Ampère prefactor 1 ✓ |
| field.py:337–343 | $\Delta E_k = -i\frac{q}{k}\int dv\, f_k(e^{-ikv\,dt}-1)$ | exact (Hamiltonian) Ampère along free streaming ✓ |
| field.py:146–153 | leapfrog wave eq., plasma term $-n_e a$ | dispersion $\hat\omega^2 = \hat n_e + \hat c^2\hat k^2$ ✓; $n_e$ is electron-only (~$m_e/m_i$ approx., F12) |
| vector_field.py:292–301, 341 | $n_e = -\,\hat q_e \int f_e\,dv$, time-centered | sign keyed to electron charge $-1$ ✓ |
| field.py:21–26 | $E_x$ driver $= (\omega_0{+}\delta\omega)\,a_0\sin(k_0x - \omega t)$ | vector-potential amplitude convention ($E = -\partial_t A$); matches docs (config.md:312) ✓ |
| field.py:67–69, 78–80 | point source $F_0 = 2\omega\hat c\,a_0$ | reproduces amplitude $a_0$ in vacuum; plasma value larger by $k_{vac}/k_{plasma}$, documented in docstring ✓ |

Charge-sign conventions were checked across Poisson / Ampère / wave-equation and the force term: consistent, no compensating double-error. Current density correctly has **no** mass factor in code (but see docstring bug F12).

### 3.4 Collisions (`solvers/pushers/fokker_planck.py`, `adept/driftdiffusion.py`)

Operator: $\partial_t f = \nu\,\partial_v[(v - \bar v)f + T\,\partial_v f]$ (Lenard–Bernstein/Dougherty, Buet notation $\beta = 1/2T$, $D = 1/2\beta = T$, drift $C = 2\beta D(v-\bar v) = (v-\bar v)$).

| Location | Code | Check |
|---|---|---|
| driftdiffusion.py:101ff (`discrete_temperature`) | $T = \sum f(v-\bar v)^2 dv \,/\, \sum f\,dv$ | full 2nd central moment, no factor 2, no $m$ — variance convention, matches init ✓ |
| driftdiffusion.py:170,180 | $\beta_{init} = 1/2T$; $f_{mx} = e^{-\beta(v-\bar v)^2}$ | stationary state $e^{-(v-\bar v)^2/2T}$ ✓ |
| driftdiffusion.py:333–334 | $C = 2\beta D\,(v_{edge} - \bar v)$ | $2\beta D = 1$ exactly ✓ |
| driftdiffusion.py (flux/Chang–Cooper, implicit solve) | conservative central & positivity-preserving fluxes; $(I - dt\,\nu L)f^{n+1} = f^n$ | numerics only; $\nu\,dt$ dimensionless ✓ |
| driftdiffusion.py:20–25 | Buet "extra factor of 2" note | correctly absorbed into $\beta = 1/2T$; **no stray factor survives** (verified) |
| **fokker_planck.py:81** | `compute_vbar` $= \sum f\,v\,dv$ | returns $n\bar u$, **not** $\bar u$ — missing $1/n$ (F5) |
| **fokker_planck.py:245** | Krook target $f_{mx} \propto e^{-v^2/2}$ | hard-coded variance 1 = ($T{=}1$, $m{=}1$, convention (b)) — convention-locked (F4) |
| fokker_planck.py:262–266 | $f \to f e^{-\nu dt} + n(x) f_{mx}(1 - e^{-\nu dt})$ | BGK; conserves $n$, not momentum/energy (by design) |

Key property: the **Dougherty operator is self-adapting** — it measures $T = \langle(v-\bar v)^2\rangle$ from $f$ and relaxes toward exactly that width, so it preserves whatever convention the initializer used and needs no change under a convention fix. The **Krook operator does not** — its width is a compile-time constant (F4).

### 3.5 Diagnostics & storage (`storage.py`)

| Location | Code | Meaning | Note |
|---|---|---|---|
| :134–135 | $\int(\cdot)\,dv$ = `sum·dv` | midpoint rule | consistent with init normalization ✓ |
| :139 | `n` $= \int f\,dv$ | density | ✓ |
| :140 | `v` $= \int v f\,dv$ | **raw first moment $= n\bar u$**, labeled "v" | flux, not velocity (F6) |
| :141–142 | `p` $= \int (v - n\bar u)^2 f\,dv$ | "pressure" | centered on $n\bar u$, not $\bar u$ (F6); for $n{=}1$ Maxwellian at $T_0$: `p` $= T_0$ under convention (b) ✓, would read $2T_0$ under (a) |
| :143 | `q` $= \int (v-n\bar u)^3 f\,dv$ | heat flux | same centering issue |
| :144 | `-flogf` $= \int f\log|f|\,dv$ | "entropy" | **missing minus sign** (F7) |
| :307 | `mean_-flogf` $= \langle\int -|f|\log|f|\,dv\rangle$ | entropy | has the minus; opposite sign to :144 (F7) |
| :303–306, 308 | `mean_P` $=\langle\int v^2f\rangle$, `mean_j` $=\langle\int vf\rangle$, `mean_n`, `mean_q` $=\langle\int v^3 f\rangle$, `mean_f2` | raw moments | `mean_j` and field "v" are the same integrand under different names (F6) |
| :311–312 | `mean_de2` $=\langle de^2\rangle$, `mean_e2` $=\langle e^2\rangle$ | field energy proxies | no ½; kinetic `mean_P` also lacks ½, consistently — internal conservation OK, but no combined energy monitor exists (F12) |
| :154, :313 | `pond` $= -\tfrac12\partial_x a^2$ | ponderomotive | ½ present and correct ✓ |

---

## 4. Findings

Ordered by severity. **Status: CONFIRMED** = derivation and code verified; **SUSPECTED** = probable issue, evidence stated.

### F1 — CONFIRMED (root cause): `normalization.py` $v_0$ is $\sqrt2$ larger than the engine's velocity unit

`normalization.py:91` ($v_0 = \sqrt{2T_0/m_e}$) vs. the engine convention $\sqrt{T_0/m_e}$ established by `helpers.py:69`, the collision operators, the dispersion tests, and all configs (§2). Since the physical temperature of the simulated plasma is $T_{phys} = m_e\,\sigma_{code}^2\,v_0^2 = T_{0,code}\cdot(m_e v_0^2)$:

- **Under (a) as written: $T_{phys} = 2\,T_{0,code}\,T_{0,ref}$.** A `normalizing_temperature: 2000eV`, `T0: 1.0` run is a 4000 eV plasma.
- $L_0 = \sqrt2\lambda_{De}$, so all `dim="x"`/`dim="k"` string conversions carry a spurious $\sqrt2$ (box sizes, gradient scale lengths in `datamodel.py:159–177`, laser $k_0$ at `simulation.py:81`).
- The species-config `T0` parameter is effectively in units of $2T_{0,ref}$ — half the naïve expectation.

The dimensionless dynamics of numeric-input runs are unaffected (the solver core never sees $v_0$); the damage is to the *physical interpretation* of every run and to dimensional-string inputs.

### F2 — CONFIRMED (propagation): logged units and regression fixtures lock in the $\sqrt2$ values

`write_units()` (`modules.py:119–141`) logs `v0`, `x0`, `box_length`, `c_light`, `nuee` computed under convention (a). For the 2000 eV resonance case the fixtures record `c_light: 11.302`, `v0: 2.652e7 m/s`, `x0: 12.14 nm` (`tests/test_vlasov1d/test_config_regression/resonance_derived_config.yml:125–132`, same in the fokker-planck and multispecies fixtures); the convention-(b) truth values are $c/v_{th} = 15.98$, $v_{th} = 1.876\times10^7$ m/s, $\lambda_{De} = 8.585$ nm. Additionally `nuee` is evaluated at $T_{0,ref}$ while the plasma actually simulated is at $2T_{0,ref}$, so the logged collisionality does not describe the simulated plasma. Any fix of F1 must regenerate these fixtures.

### F3 — CONFIRMED: EM branch — $\hat c$ is $\sqrt2$ too small; partial cancellation hides it for wavelength drivers only

$\beta = 1/\hat c$ with $\hat c = c/v_0$ (`modules.py:58,121`) feeds the wave solver and the CFL limit. Under (a), $\hat c = c/(\sqrt2 v_{th})$ — $\sqrt2$ smaller than the engine's thermal unit warrants. For **wavelength-specified** drivers the $\sqrt2$'s cancel in the product $\hat c\hat k = \frac{c}{\sqrt2 v_{th}}\cdot\sqrt2 k\lambda_{De}$ — which is why `test_em_dispersion.py` and `srs.yaml`'s EM branch behave physically. For **numeric** AKW drivers (`srs-debug-small.yaml`: `k0: 1.0, w0: 2.79` used as-is at `simulation.py:59`), $\hat k$ is *not* rescaled while $\hat c$ still carries the $\sqrt2$, so the EM dispersion $\hat\omega^2 = \hat n_e + \hat c^2\hat k^2$ is $\sqrt2$-inconsistent with the ES branch's $\hat k = k\lambda_{De}$ interpretation. Mixed-input SRS runs are therefore internally inconsistent between branches.

### F4 — CONFIRMED: Krook target Maxwellian is hard-coded to $e^{-v^2/2}$ (convention-locked, $T{=}1$, electron grid only)

`fokker_planck.py:245`. Three separate problems: (i) it bakes in convention (b) with variance exactly 1, so it must be changed **in lockstep** with any convention fix or it will spuriously heat/cool by 2× in temperature; (ii) even today, it ignores the species `T0` and `mass` — any species initialized at $T_0 \ne 1$ (e.g. `twostream.yaml`, $T_0 = 0.2$) is dragged toward $T = 1$; (iii) it is built on the electron grid and applied only to `"electron"` (`fokker_planck.py:171`). It is a fixed-target BGK operator: conserves density only, not momentum or energy. Currently `is_on: False` in shipped configs, so latent.

### F5 — CONFIRMED: `Dougherty.compute_vbar` is missing the $1/n$ normalization

`fokker_planck.py:81` returns $\int v f\,dv = n\bar u$ while its docstring claims "Mean velocity ⟨v⟩". `discrete_temperature` (`driftdiffusion.py:101ff`) *does* divide by $\int f\,dv$ — the two moments are asymmetric. Consequence: the drag centers on $n\bar u$, the operator relaxes toward $e^{-\beta(v - n\bar u)^2}$, and **momentum is not conserved where $n(x) \ne 1$**; $T_{target}$ is also biased by the wrong centering. Negligible for the shipped EPW/SRS configs ($n \approx 1 \pm 10^{-4}$), real for bump-on-tail / large density perturbations with collisions on. Fix: divide by $\int f\,dv$.

### F6 — CONFIRMED: storage central moments centered on $n\bar u$, not $\bar u$; misleading names

`storage.py:140–143`: the field moment `"v"` is $\int v f\,dv = n\bar u$ (a flux), and `p`, `q` are centered on it. Only exact when $n = 1$; biased for `nlepw-ic.yaml` (10% density perturbation) and `bump-on-tail.yaml`. Same integrand is named `"v"` in field moments but `"mean_j"` in scalars (`storage.py:304`) — one of the labels is wrong; `j` is the honest name (or divide by $n$ and keep "v"). Note this is the same class of bug as F5, appearing independently in two places.

### F7 — CONFIRMED: the two `-flogf` entropy diagnostics have opposite signs

Field moment `storage.py:144` computes $+\int f\log|f|\,dv$ (no minus, uses $f$); scalar `storage.py:307` computes $-\int|f|\log|f|\,dv$. For $f > 0$ these are exact negatives. The label `-flogf` matches the scalar; the field version is sign-flipped.

### F8 — SUSPECTED (intent unclear): super-Gaussian $\alpha$ fixes $\langle v^4\rangle/\langle v^2\rangle$, not the variance, for $m \ne 2$

`helpers.py:72`: $\alpha = \sqrt{3\Gamma(3/m)/\Gamma(5/m)}$ normalizes the kurtosis ratio $\langle v^4\rangle/\langle v^2\rangle = 3v_{th}^2$ for all $m$, but the variance equals $v_{th}^2$ only at $m = 2$. Numerically the realized variance is $\{1.240, 1.371, 1.449\}\times T_0/m$ for $m = \{3,4,5\}$ — so the measured `p`/`n` moment will *not* equal the input `T0` for super-Gaussian species. May be an intentional flat-top-EDF temperature definition; if so it should be documented, because the `T0` config docstring ("Temperature") and the second-moment diagnostic disagree with it. All shipped configs use $m = 2$, where it is exact.

### F9 — CONFIRMED (gap/trap): collision-frequency normalization chain is entirely manual

- `approximate_ee_collision_frequency` returns **Hz** and is only *logged* (`modules.py:119,132`); nothing converts it to code units or feeds it to the FP operator.
- The FP/Krook rate magnitudes (`baseline`, `bump_height`) are taken as **raw floats** (`functions.py:97–98`), not passed through `normalize()` — users must supply $\nu$ already in code units ($1/\omega_{p0}$), which is nowhere documented (`config.md` lists `baseline` with no units).
- The correct conversion is $\hat\nu = \nu_{ee}[\mathrm{Hz}]\cdot\tau = \nu_{ee}/\omega_{p0}$ with **no** $2\pi$ (the NRL rate is a true s⁻¹ rate); `_tf1d/modules.py:78` has this pattern (`nuee_norm = nuee/wp0`), `_vlasov1d` has no analog. A user thinking in cyclic frequency is one step from a silent $2\pi$ error.
- The LB/Dougherty $\nu$ and the NRL $\nu_{ee}$ agree only up to an O(1) factor — identifying them silently is itself an approximation worth a docs note.
- $\nu$ is a prescribed space-time envelope; it does not track the local $n(x)/T(x)^{3/2}$ (limitation, not a bug).

Suggested: log a `nuee_norm` alongside `nuee`, and document `baseline`'s units.

### F10 — CONFIRMED: species `T0` and drift `v0` bypass the unit machinery

`simulation.py:213–214` take `float(cfg.T0)`, `float(cfg.v0)`; `datamodel.py:20–21` type them as bare floats. The `normalize(dim="temp")` and `dim="v"` branches are dead code for `_vlasov1d`. Consequences: `T0: "500 eV"` is impossible (only the global `normalizing_temperature` is dimensional), and the drift `v0` is expressed in units of the reference $v_0$ while the thermal width is in $\sigma$ units — under convention (a) these differ by $\sqrt2$ *within the same distribution*, a genuine user trap (drift "1.0" is not "one thermal width").

### F11 — CONFIRMED (latent): `dx = xmax/nx` ignores `xmin`

`grid.py:55` should be `(xmax − xmin)/nx`; the x-axis itself (`grid.py:74`) spans $[x_{min}, x_{max}]$ with the wrong spacing when $x_{min} \ne 0$, and the `kx` grid (`grid.py:77`) and the semi-Lagrangian interpolation period (`pushers/vlasov.py:31`, `period = xmax`) inherit the error. `modules.py:112` uses the correct $(x_{max}-x_{min})$, highlighting the discrepancy. All shipped configs use `xmin: 0.0`, so latent but real.

### F12 — Minor items (confirmed, low impact)

1. **`AmpereSolver` class docstring** (`field.py:230`) claims $j = \sum_s (q_s/m_s)\int vf_s\,dv$; the code (`field.py:263–264`) correctly omits $1/m_s$. Docstring bug only.
2. **Energy diagnostics lack ½ and a total**: `mean_P`, `mean_e2`, `mean_de2` are $2\times$ the respective energies (consistently, so `mean_P + mean_e2` is still conserved), and no diagnostic sums kinetic + field energy. A dedicated conservation monitor would have caught F1 earlier.
3. **`dw0 = 0.0  # ???`** placeholder at `simulation.py:87` for the intensity/wavelength driver.
4. **Dead config knob**: `GridConfig.c_light` (`datamodel.py:73`; set in `wavepacket.yaml`) is never read — `c_light`/`beta` are always recomputed from the normalization.
5. **`grid.t` axis spacing**: `nt = int(tmax/dt + 1)`, `tmax = dt·nt` overshoots the request by up to ~dt and `linspace(0, tmax, nt)` has spacing $\ne$ dt. Cosmetic (saves use their own axes).
6. **EM plasma term is electron-only** (`vector_field.py:292–301`, `field.py:152`): ions omitted from the transverse current ($\sim m_e/m_i$ — fine, but an asymmetry vs. Poisson/Ampère which include all species).
7. **Point-source amplitude** uses the vacuum $k = \omega/c$ (`field.py:67`); realized plasma amplitude larger by $k_{vac}/k_{plasma}$ — already documented in the class docstring.
8. **Half-cell axis convention**: `vax`/`x` hold cell *centers* while the extents name the *edges* ($\pm v_{max}$) — keep in mind when labeling axes.

### Verified correct (checked because they looked suspicious, and passed)

- The entire collisionless solver core: all coefficients unity, signs consistent (§3.3).
- The ponderomotive chain: the $-\tfrac12\partial_x a^2$, the $(q^2/m^2)$ factor, and the $c/v_0$ scaling of $a_0$ combine *exactly* — no cycle-average ½ missing, no hidden $\sqrt2$.
- The Buet "factor of 2" in the Dougherty operator: correctly absorbed by $\beta = 1/2T$; equilibrium width equals the measured variance exactly.
- The $E_x$ driver amplitude $E = \omega a_0$: consistent vector-potential convention, matches the docs.
- Midpoint-rule ($\mathrm{sum}\cdot dv$) integration: used uniformly in init, moments, and field solves.

---

## 5. Recommended resolution and lockstep checklist

**Adopt convention (b) globally**: in `electron_debye_normalization`, set $v_0 = \sqrt{T_0/m_e}$ (so $L_0 = \lambda_{De}$), and redefine `vth_norm()` accordingly. This leaves the initializer, all collision operators, all tests, and all numeric configs untouched and correct, and makes the logged physical units true.

Things that must change together / be re-verified:

1. `normalization.py:91–92` ($v_0$, $x_0$) — **and nothing else in that file. Do NOT redefine `vth_norm()` (:48–50)**: the codebase-wide audit (`ADEPT_CONVENTIONS_AUDIT.md`, finding C1) found that `vth_norm()` has exactly four callers, all in `vfp1d/`, which is self-consistently built on the $\sqrt{2T/m}$ convention (its initializer carries a compensating factor). Changing `vth_norm()` does nothing for `_vlasov1d` (never called here) and would silently halve VFP-1D's initialized temperature.
2. Regenerate the three `*_derived_config.yml` regression fixtures (they lock `c_light`, `v0`, `x0` at the $\sqrt2$ values).
3. Re-check every config that uses **dimensional string** inputs (`srs.yaml`: `xmax: 100um`, gradient scale length `200um`, laser wavelength/intensity) — their physical meaning shifts by $\sqrt2$ (they become *correct*; the previously-inferred physical parameters of past runs were what was wrong).
4. `c_light`/`beta` changes for all EM runs: re-validate `test_em_dispersion.py`, the point-source amplitude test, and numeric-driver SRS configs (`srs-debug-small.yaml` `w0`/`k0` values were presumably tuned under the old $\hat c$ — see F3).
5. Krook target (`fokker_planck.py:245`) — no change needed under (b), but fix its $T_0$/species handling anyway (F4) so it stops being convention-locked.
6. Docs: state the normalization explicitly in `docs/source/solvers/vlasov1d/` (velocity unit $=\sqrt{T_0/m_e}$, $L_0 = \lambda_{De}$, $\hat k = k\lambda_{De}$, Maxwellian $e^{-v^2/2}$ at $T{=}1$), and document `baseline` units (F9) and the drift-`v0` units (F10).
7. ~~Audit sibling solvers for the same $v_0$ dependency~~ **Done — see `ADEPT_CONVENTIONS_AUDIT.md`** (codebase-wide audit with per-solver immunity verdicts). Summary: `electron_debye_normalization` is consumed only by `_vlasov1d`, `_vlasov2d`, `_pic1d` — all σ-convention, all fix-safe, and only `_vlasov1d` has fixtures to regenerate. `vfp1d` genuinely runs in $\sqrt{2T/m}$ units but via `laser_normalization` + `vth_norm()`, so it is untouched by the scoped fix (see amended item 1). `_tf1d` has a private inline duplicate of the √2 (`_tf1d/modules.py:62`) that must be fixed separately. All other modules are immune.

Independent bug fixes (any order, no convention coupling): F5 (`compute_vbar` $1/n$), F6 (moment centering + naming), F7 (`-flogf` sign), F11 (`dx` with `xmin`), F12.1 (docstring), F12.2 (add an energy-conservation scalar).

### Suggested verification runs after any fix

- Landau damping and ion-acoustic tests must still pass unchanged (they pin convention (b)).
- A Dougherty-collisions run with a strong density perturbation: check $\partial_t\int v f\,dv \approx 0$ (exercises F5).
- An SRS wavelength-driver run: confirm backscatter resonance moves to the physically correct location once `c_light` and the µm conversions change together.
- Compare the second-moment temperature diagnostic against `T0` for an $m = 4$ super-Gaussian to decide F8's intended convention.

---

*Audit method: four parallel independent reviews (initialization/grids/derived quantities; solver core; collisions; diagnostics/docs/tests/configs), followed by cross-checking of all load-bearing claims against the source. No source files were modified.*
