# ADEPT Research Notes

Append-only checkpoints for scientific model development and validation.

## 2026-09-03 22:29:49 PDT — Envelope-2D IAW parity scope

- **Checkpoint ID:** 20260904T052949Z-92eb82df
- **State:** PLANNED
- **Objective:** Integrate the remaining `lpse-matlab` physics needed for the requested full TPD+SRS workflow, including particle feedback, laser feedback, and ion-acoustic waves.
- **USER (verbatim):** "can you make a PR to integrate the missing features from ../lpse-matlab into the envelope2d solver? we want the full set of TPD+SRS w/ the particle tracker and laser feedback and IAWs (am i missing anything?)"
- **Action or change:** Audited ADEPT `origin/main` against `/Users/archis/Dev/code/ergodic/lpse-matlab` at commit `3bf8b0b`. Current ADEPT main already contains TPD, SRS/Raman light, evolved-pump depletion feedback, Follett-style hybrid test-particle evolution with damping feedback, reproducible noise, SRS diagnostics, broadband/speckle drivers, and shifted-band source anti-aliasing. The planned implementation scope is the missing IAW subsystem and its coupling to light/EPW fields, with focused parity tests and an example deck.
- **Provenance:** ADEPT commit `25719e727ca1fa0d46a00ede1e2054ae3ec105d8`; branch `codex/lpse2d-iaw-parity`; MATLAB source commit `3bf8b0b`.
- **Observation:** No result yet. MATLAB exposes an IAW density perturbation `Nelf` and velocity-divergence field `W`, split-step evolution, acoustic/convective terms, collisional and Landau damping, boundary damping, and a ponderomotive light-intensity source. Existing ADEPT `origin/main` has no IAW state or solver.
- **Evidence or artifacts:** Vault mirror `Notes/adept/2026-09-03-222949-codex-lpse2d-iaw-parity-92eb82df.md`.
- **Interpretation:** The named TPD, SRS, particle, and laser-feedback capabilities do not need to be reimplemented; IAW parity is the substantive missing feature. Additional parity considerations to preserve are source anti-aliasing, reproducible noise, diagnostics, broadband colors, and speckles.
- **Next:** Port and test the MATLAB IAW split-step path on top of current `origin/main`, then run the LPSE test suite and open a PR.

## 2026-09-03 22:50:18 PDT — Scope expansion: reciprocal TPD pump depletion

- **Checkpoint ID:** 20260904T055018Z-2e66ec98
- **State:** PLANNED
- **Objective:** Add TPD pump depletion to the IAW parity work so every enabled parametric instability feeds back on the evolved pump.
- **USER (verbatim):** "i think we need TPD pump depletion too right?" Follow-up implementation constraint: "no dont use the old stuff, just ditch it and rewrite it"
- **Action or change:** Expanded `terms.light.pump_depletion` from SRS-only to all active TPD/SRS sources. The TPD term is a clean implementation constructed as the discrete reciprocal of the current potential-form TPD operator; the abandoned `origin/update/lpse/pump` prototype is not incorporated.
- **Provenance:** Current Envelope-2D TPD operator in `adept/_lpse2d/core/epw.py`; MATLAB TPD source in `m201805_matlabLpse_v11.m`; no TPD pump feedback exists in the MATLAB reference.
- **Observation:** For coupling terms alone, the new pump RHS and existing EPW RHS conserve `|E0|^2 + (wp0/w0)|E_epw|^2` to floating-point tolerance. Focused TPD-depletion and IAW tests pass (9 tests).
- **Evidence or artifacts:** `tests/test_lpse2d/test_tpd_depletion.py`; `tests/test_lpse2d/test_iaw.py`; `configs/envelope-2d/tpd-srs-iaw.yaml`.
- **Interpretation:** Reciprocal TPD feedback closes the above-threshold fluid energy path. It does not make the existing quasi-1D HPE tracker angle-resolved; 2D kinetic feedback remains a separate model extension.
- **Next:** Run the full Envelope-2D regression suite, finalize documentation and mirrored notes, then open the PR.

## 2026-09-03 23:09:13 PDT — Envelope-2D parity implementation completed

- **Checkpoint ID:** 20260904T060913Z-1cea910a
- **State:** COMPLETED
- **Objective:** Deliver verified IAW evolution and reciprocal TPD pump depletion on top of the merged TPD/SRS/HPE feature set.
- **Action or change:** Added IAW state/evolution, MATLAB-order damping and boundaries, ponderomotive drive, EPW/pump/Raman density feedback, IAW diagnostics, clean reciprocal TPD pump feedback, TPD-only flux diagnostics, dynamic-light stability setup, a combined 2D example deck, and an explicit TPD/HPE dimensionality guard.
- **Provenance:** Commit `1cea910`; branch `codex/lpse2d-iaw-parity`; MATLAB IAW reference `m201805_matlabLpse_v11.m` at repository commit `3bf8b0b`.
- **Observation:** Full non-slow Envelope-2D regression: 53 passed, 1 skipped, 3 deselected in 938.91 s. Final focused IAW/TPD suite: 12 passed in 4.76 s. Ruff and compile checks passed.
- **Evidence or artifacts:** Pull request https://github.com/ergodicio/adept/pull/363; `adept/_lpse2d/core/iaw.py`; `tests/test_lpse2d/test_iaw.py`; `tests/test_lpse2d/test_tpd_depletion.py`; `configs/envelope-2d/tpd-srs-iaw.yaml`.
- **Interpretation:** The 2D fluid model now composes TPD, SRS, pump feedback, and IAWs. The existing quasi-1D HPE tracker composes with SRS and IAWs, but a genuinely combined TPD+HPE model remains blocked on a multidimensional particle/feedback formulation and is rejected explicitly.
- **Next:** Review and merge PR #363; design `(x, y, p_x, p_y)` HPE and an angle-resolved damping estimator as a separate follow-up if TPD kinetic feedback is required.

## 2026-09-04 07:32:05 PDT — Expand HPE to a global 2D particle ensemble

- **Checkpoint ID:** 20260904T143205Z-6b54cbe1
- **State:** PLANNED
- **Objective:** Extend the HPE tracker on PR #363 from `(x, p_x)` to `(x, y, p_x, p_y)` so TPD and SRS can use kinetic damping feedback in a 2D box.
- **USER (verbatim):** "let's make the particle tracker 2D. i thought the particles are box averaged and not at each point?"
- **Action or change:** Reopened the kinetic-model scope. The planned design keeps one global macro-particle ensemble; particle positions exist only for field gathering, while the feedback distribution remains box averaged. Direction-dependent damping will be derived from projections of the global 2D velocity distribution along each retained wavevector rather than from per-cell particle populations.
- **Provenance:** Commit `8f9ff7690fff915a3146f74e4dd68551b5c5ef37`; branch `codex/lpse2d-iaw-parity`; PR https://github.com/ergodicio/adept/pull/363.
- **Observation:** No result yet. Current HPE stores one position and momentum component per particle, gathers only `Ex`, and forms one global `f(vx)` histogram that is broadcast across `ky`.
- **Evidence or artifacts:** `adept/_lpse2d/core/hpe.py`; vault mirror `Notes/adept/2026-09-03-222949-codex-lpse2d-iaw-parity-92eb82df.md`.
- **Interpretation:** The user's box-averaging point removes the need for a particle population at every spatial point, but oblique TPD modes still require 2D particle orbits and a directional velocity-space projection.
- **Next:** Design the directional damping estimator and state layout, implement the 2D gather/push/feedback path with quasi-1D compatibility, and validate isotropic initialization plus oblique-mode response.

## 2026-09-04 08:02:48 PDT — Box-averaged 2D2V HPE completed

- **Checkpoint ID:** 20260904T150248Z-2d2f0e6d
- **State:** COMPLETED
- **Objective:** Make the HPE particle tracker genuinely 2-D while retaining one spatially averaged ensemble for TPD and SRS damping feedback.
- **Action or change:** Rewrote the HPE core as a dual 1D1V/2D2V implementation. In a 2-D box each macro-particle carries `(x, y, p_x, p_y)`, gathers spectrally refined `Ex` and `Ey` by bilinear interpolation, and contributes to one bank of box-wide projected velocity histograms. Thirty-two oriented projections over `[0, 2 pi)` are interpolated onto every `(kx, ky)` resonance; opposite directions remain distinct. Removed the HPE+TPD setup rejection, generalized hot-electron/damping diagnostics, and enabled HPE in the full TPD+SRS+IAW example deck.
- **Provenance:** Branch `codex/lpse2d-iaw-parity`; PR https://github.com/ergodicio/adept/pull/363; implementation in `adept/_lpse2d/core/hpe.py`.
- **Observation:** The complete non-slow Envelope-2D suite passes with isolated local tracking and a headless plot backend: 60 passed, 1 skipped, 3 deselected in 261.85 s. Focused HPE/TPD/IAW verification passed 20 tests before the final directional test was added; the final full suite includes that test. Ruff lint/format and Python compilation pass. A combined one-step test advances TPD, SRS, reciprocal pump depletion, IAWs, and 2D2V HPE together with finite state and diagnostics.
- **Evidence or artifacts:** `tests/test_lpse2d/test_hpe.py`; `tests/test_lpse2d/test_tpd_depletion.py`; `configs/envelope-2d/tpd-srs-iaw.yaml`; `docs/source/solvers/lpse2d/config.md`.
- **Interpretation:** The user's box-averaging understanding is correct: positions are retained only for local force gathering, not to form a particle population per mesh point. Directional Landau feedback comes from Radon-like projections of the single global distribution, so histogram state scales as `n_angles * nv`, independent of `nx * ny`. The remaining HPE limitation is Im-only feedback (no trapping-induced nonlinear frequency shift), not transverse dimensionality.
- **Next:** Push the update to PR #363 and use production TPD runs to converge `n_particles`, `n_angles`, gather refinement, and the damping EMA window on GPU.

## 2026-09-04 09:18:28 PDT — Correct 2D absorbing-wall reinjection

- **Checkpoint ID:** 20260904T161828Z-f10c2eed
- **State:** PLANNED
- **Objective:** Remove the review-blocking bias in the 2D HPE thermal-wall sampler before declaring the combined kinetic model ready.
- **USER (verbatim):** "ok can you fix"
- **Action or change:** Accepted the review finding on PR #363. The current 2D boundary code draws the radially truncated box distribution isotropically and flips the normal sign; that is not the incoming flux distribution. The replacement will sample wall-frame velocities from `p(r, theta) proportional to r^2 exp(-r^2 / (2 vte^2)) cos(theta)` for `r > v_min`, using an exact truncated-Maxwell inverse CDF and `sin(theta)` uniform on `[-1, 1]`.
- **Provenance:** Commit `b7787b84321d7391d5d3a1fab7304a0987810ed0`; branch `codex/lpse2d-iaw-parity`; PR https://github.com/ergodicio/adept/pull/363; review state `CHANGES_REQUESTED` by `ergodic-ludwig` at 2026-09-04 08:07:29 PDT.
- **Observation:** No corrected result yet. The existing sampler preserves inward direction and radial-tail support but omits the normal-velocity flux weighting, so repeated absorbing-wall crossings can bias the global directional histograms.
- **Evidence or artifacts:** `adept/_lpse2d/core/hpe.py`; PR review on #363; vault mirror `Notes/adept/2026-09-03-222949-codex-lpse2d-iaw-parity-92eb82df.md`.
- **Interpretation:** For an isotropic 2D Maxwellian at a planar wall, multiplying by inward normal speed and transforming to polar wall coordinates gives a separable radial truncated-Maxwell law and angular `cos(theta)` law. A stationarity/distribution test should lock both factors.
- **Corrects:** Checkpoint `20260904T150248Z-2d2f0e6d`, which marked the 2D HPE implementation complete before independent boundary-law review.
- **Next:** Implement the exact flux sampler, test its radial CDF and angular moments plus inward wall mapping, rerun focused/full regressions, and push a review correction.

## 2026-09-04 09:29:09 PDT — Flux-weighted HPE wall correction completed

- **Checkpoint ID:** 20260904T162909Z-cbd107a4
- **State:** COMPLETED
- **Objective:** Correct 2D HPE thermal-wall reinjection so absorbing boundaries preserve the intended box-averaged tail distribution.
- **Action or change:** Replaced isotropic sign-flip reinjection with the exact planar-flux sampler. The radial speed is drawn from the Maxwell law `p(r) proportional to r^2 exp(-r^2/(2 vte^2))` truncated to `[v_min, 0.99c]` by inverse-survival bisection; the wall angle uses `sin(theta)` uniform on `[-1,1]`, equivalent to `p(theta)=cos(theta)/2`. The wall-frame sample is rotated onto all four faces with inward normal velocity. The initial 2D tail loader now also uses the exact doubly truncated Rayleigh inverse rather than clipping an unbounded draw at `0.99c`, and configuration rejects a cutoff at or above that cap.
- **Provenance:** Based on commit `b7787b84321d7391d5d3a1fab7304a0987810ed0`; branch `codex/lpse2d-iaw-parity`; PR https://github.com/ergodicio/adept/pull/363.
- **Observation:** The boundary-law test verifies the radial probability-integral transform has uniform mean/variance within tolerances `0.004`/`0.002`, verifies `mean(sin(theta))=0` and `var(sin(theta))=1/3` within `0.005`/`0.004`, enforces radial support, and checks inward mapping on all four faces. Focused HPE/TPD/IAW suite: 22 passed, 3 deselected in 20.05 s. Full non-slow Envelope-2D suite with local MLflow and `MPLBACKEND=Agg`: 61 passed, 1 skipped, 3 deselected in 257.71 s. Ruff lint/format and Python compilation pass.
- **Evidence or artifacts:** `adept/_lpse2d/core/hpe.py`; `tests/test_lpse2d/test_hpe.py::test_2d_wall_reinjection_is_flux_weighted`; `docs/source/solvers/lpse2d/config.md`.
- **Interpretation:** The review-blocking distribution bias is removed and its separable radial/angular law is locked by a statistical test. Repeated planar-wall reinjection now samples the flux distribution associated with the same retained 2D Maxwellian tail used at initialization.
- **Corrects:** Checkpoint `20260904T150248Z-2d2f0e6d`; completes planned correction `20260904T161828Z-f10c2eed`.
- **Next:** Commit and push the correction to PR #363, report the derivation/test to the reviewer, and wait for re-review.

## 2026-09-05 21:48:02 PDT — Vlasov-1D Strang and degree-7 remap

- **Checkpoint ID:** 20260906T044802Z-79ad3623
- **State:** PLANNED
- **Objective:** Add opt-in `time: strang` and `edfdv: lagrange7` to upstream Vlasov-1D and open a PR.
- **USER (verbatim):** "okay can you make a PR for bullet 1. im not too worried about bullet 2 or 3. we dont need to get into the weeds of the numerics. i'm trying to improve the modeling of the turbulence problem"
- **USER (verbatim), scope clarification:** "dont need to address turbulence in this repo at all. i menat in the ../vp-turbulence project, but that is downstream and this is upstream and you dont need to worry about it"
- **Action or change:** Implement spectral half-streaming, midpoint-field full velocity push, and spectral half-streaming; add an eight-point degree-7 Lagrange velocity interpolation with nonperiodic boundaries, multispecies support, and the existing sharding interface. Keep legacy options and all production configs unchanged.
- **Provenance:** Base `origin/main` commit `4c49e93`; branch `codex/vlasov1d-strang-lagrange7`; isolated worktree `/private/tmp/adept-strang-lagrange7`.
- **Observation:** No implementation result yet. Validation will cover interpolation, midpoint forcing, field diagnostics, and solver integration; no turbulence campaign or broad convergence study is in scope.
- **Evidence or artifacts:** Vault mirror `Notes/adept/2026-09-05-214802-codex-strang-lagrange7-79ad3623.md`.
- **Interpretation:** The requested modeling outcome remains unmeasured; this task delivers upstream capabilities only.
- **Next:** Implement, run focused tests and formatting checks, and open the PR.

## 2026-09-05 21:56:51 PDT — Upstream Strang and degree-7 implementation verified

- **Checkpoint ID:** 20260906T045651Z-7a5c55d3
- **State:** COMPLETED
- **Objective:** Deliver opt-in upstream Vlasov-1D time stepping and velocity remapping.
- **Action or change:** Added `time: strang` for Poisson/Poisson-Boltzmann with midpoint forcing, one velocity remap, and endpoint field diagnostics. Added `edfdv: lagrange7` using an eight-point stencil, floor-filled nonperiodic boundaries, per-species grids/forces, and shared interpolation-pusher sharding. Added focused tests and API documentation. Existing production configs and downstream projects were not changed.
- **Provenance:** Base `4c49e93`; branch `codex/vlasov1d-strang-lagrange7`; Python 3.14 / JAX 0.9.0.1 on local CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
- **Observation:** 35 tests passed across `test_velocity_lagrange7.py`, `test_strang.py`, `test_velocity_cubic_spline.py`, `test_multispecies_pushers.py`, and `test_asymmetric_velocity_grid.py`. Checks include interpolation/boundaries, gradients, multispecies forces, midpoint impulse, final-field synchronization, all three velocity pushers with both Strang field solvers, and combined x/v sharding. Pre-commit hooks and `git diff --check` passed before final notebook update.
- **Material commands:** `python -m pytest tests/test_vlasov1d/test_velocity_lagrange7.py tests/test_vlasov1d/test_strang.py tests/test_vlasov1d/test_velocity_cubic_spline.py -q` (26 passed); `python -m pytest tests/test_vlasov1d/test_multispecies_pushers.py tests/test_vlasov1d/test_asymmetric_velocity_grid.py -q` (8 passed); after adding the combined sharding check, `python -m pytest tests/test_vlasov1d/test_strang.py::test_strang_sharded_matches_serial -q` (1 passed). All used the shared ADEPT development environment, `MPLBACKEND=Agg`, and the four-device setting above.
- **Evidence or artifacts:** `tests/test_vlasov1d/test_velocity_lagrange7.py`; `tests/test_vlasov1d/test_strang.py`; vault mirror `Notes/adept/2026-09-05-214802-codex-strang-lagrange7-79ad3623.md`.
- **Interpretation:** These are implementation checks, not a turbulence result or an extensive convergence study. The new interpolation is unlimited and does not renormalize boundary losses. Strang's second-order scope is the collisionless electrostatic update; existing collision/filter/transverse coupling is unchanged.
- **Next:** Publish the branch and open the requested upstream PR. Downstream adoption is outside this task.
