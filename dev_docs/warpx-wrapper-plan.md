# WarpX wrapper + SRS diagnostics parity — plan

*2026-08-12. Branch: `warpx-wrapper` (cut from `osiris-wrapper` @ b7d8fde). Goal: 1:1
comparison of WarpX against existing OSIRIS SRS results (scan2, srs-Ln100-Te4-5x
campaigns), with a WarpX wrapper built the way the OSIRIS wrapper is built — but
WarpX-native throughout: no translation of OSIRIS decks, no imitation of OSIRIS file
layouts. Parity is defined at the **physics-metric level**, not the file level.*

WarpX itself is installed and smoke-tested on Perlmutter (dev @ 72280884a,
`/global/common/software/m4490/philt/warpx`, runbook in `warpx-sw/README.md`).

---

## 1. Design principles

1. **WarpX-native problem setup.** The SRS problem is specified in SI from physical
   parameters ($T_e$, $L_n$, $I$, $\lambda_0$, $n/n_c$ at a reference point) using
   WarpX's own machinery: `parse_density_function` for the exponential profile, the
   laser antenna for the pump, PML + particle absorption at boundaries. We match the
   OSIRIS runs on *physics* (same $T_e$, $L_n$, $I\lambda^2$, box length in μm, ppc,
   resolution in skin depths) — never by transliterating deck values.
2. **WarpX-native diagnostics first.** Prefer reduced diagnostics (step-cadence
   scalars/histograms) and openPMD over recreating OSIRIS dump patterns. Where a parity
   metric needs the *same estimator* as the OSIRIS pipeline (e.g. the boundary-slab
   Riemann split), compute it from openPMD field dumps at matched cadence — but also log
   the cheaper/better native estimator as a cross-check (e.g. Poynting flux every step).
3. **Same seam as OSIRIS.** adept (`adept/warpx/`) knows how to run and read WarpX;
   the LPI layer knows what runs mean. Nothing SRS-specific goes in `adept/warpx/`.
4. **Shared metric code = credible parity.** The `osiris_lpi` analysis modules already
   take `xr.DataArray`s (not paths) in every hot path. The WarpX side supplies
   DataArrays in the same conventions and reuses the *identical* metric functions —
   same formulas, same fit windows, same thresholds. Parity by construction.

## 2. Architecture

### 2.1 `adept/warpx/` (this branch) — mirrors `adept/osiris/` module-for-module

| Module | Role | vs OSIRIS wrapper |
|---|---|---|
| `base.py` | `BaseWarpX(ADEPTModule)`: lifecycle, `write_units()`, registry key `solver: warpx` | same contract; `get_derived_quantities` from `amr.n_cell`, `geometry.prob_lo/hi`, `max_step`/`stop_time` |
| `deck.py` | WarpX inputs = flat ParmParse namespace → parse/render/override/flatten is a **flat dict**, far simpler than the OSIRIS ordered-sections parser. Roundtrip invariant kept. `warpx_used_inputs` archived for provenance | replace wholesale; no integer-index override machinery |
| `runner.py` | subprocess driver: `srun -n N warpx.<dim>d inputs [key=value overrides]`; binary precedence `warpx.binary` → `WARPX_BIN_<dim>D` → `WARPX_BIN`; tee/tail; error classification tuned to AMReX (`amrex::Abort`, assertion failures, **non-zero exit codes** — unlike OSIRIS, WarpX fails loudly, so drop the exit-0 fuzz logic; keep salvage-if-output-exists) | port with re-tuned classification; ramdisk staging optional later (openPMD/ADIOS2 is already efficient) |
| `io.py` | openPMD reader (`openpmd-api`): `list_diagnostics`, `load_series`/`open_series` (lazy, `t_indices`), reduced-diag `.txt` parsers → **emit the same `binary/<diag>.nc` NetCDF contract** the plot/regen layer already consumes | rewrite reader; keep NetCDF artifact contract so `plots`/`regen` port cheaply |
| `units.py` / `write_units` | SI in, normalized out: derive $\omega_{p0}$, $c/\omega_{p0}$, $u = p/m_ec$ from a **reference density** in the manifest (not from the deck — WarpX has no $n_0$). All DataArrays handed to analysis are converted to these code units so metric code is shared; SI originals preserved in the NetCDFs' attrs | inverted direction: SI → normalized, instead of normalized-only |
| `post.py`, `plots.py` | metrics (SI energy: $\varepsilon_0 E^2/2 + B^2/2\mu_0$), general canned plots against the NetCDF contract (field names `Ex..Bz`, axes labeled both SI and normalized) | port; swap field-name tables and energy formulas |
| `regen.py` | `python -m adept.warpx.regen <dir>` offline regeneration | port pattern |
| `tests/test_warpx/` | checked-in inputs decks; binary-free tests that self-skip; `cpu-tests.yaml` paths filter (**easy to forget — CI silently skips otherwise**) | port pattern |

Streaming (`stream.py`): defer. openPMD/ADIOS2 BP5 with file-per-iteration
(`openpmd_encoding = f`) already gives incremental, per-iteration output; the OSIRIS
streaming layer existed to tame `MS/*.h5` sprawl. Revisit only if artifact-size or
postproc-latency pressure shows up.

### 2.2 LPI layer — the `warpx-lpi` repo (ergodicio/warpx-lpi)

`WarpxLPI(BaseWarpX)` subclass + a thin adapter, in its own repo mirroring
osiris-lpi's layout (`run.py`, `warpx_lpi/`, `sims/<campaign>/`, `tests/`,
`package = false`, adept as editable path dep). Decided 2026-08-12; scaffold exists.
Reusing the `osiris_lpi` metric modules across repos requires making osiris-lpi
installable (drop `package = false` + ship `osiris_lpi` as a package) or extracting
the shared analysis — decide at M3; the modules are already DataArray-in, so either
is small. The adapter implements exactly the surface `collect_srs` needs:

- `list_diagnostics(run_dir)`, `load_series/open_series` → DataArrays with dims
  `(t, x…)`, `(t, x, p)` in code units ($t$ in $1/\omega_{p0}$, $x$ in $c/\omega_{p0}$,
  $u = p/m_ec$, fields normalized so $I_0 = (a_0\omega_0)^2/2$)
- `physical_axis(da, dim, it)`, `sim_box_bound(da, dim, upper)`
- `transverse_field_boundary_slabs(run_dir, guard_cells, window_cells)` → same dict
  shape (Riemann split of $E_y/B_z$, $E_z/B_y$ at 3-cell edge slabs)

With those in place, `laser_budget`, `hot_electrons`, `epw_growth`, `em_spectrum`,
`energy_spectrum`, `fp1_fit`, `timeseries` run **unchanged** → identical MLflow metric
names (`laser_reflectivity`, `hot_e_flux_frac_fwd_50keV`, `t_first_hot_e_50keV_*`,
`epw_growth_rate`, `fp1_supergauss_m_bulk`, `T_hot_keV`, seg/noise/fullrun variants)
and the same `timeseries/srs_timeseries.nc` schema. Small refactor allowed in
`osiris_lpi` where a module currently calls `adept.osiris.io` directly: inject the io
functions (or dispatch on a `run_dir` marker) — keep it minimal and covered by the
existing osiris tests.

## 3. Diagnostics: WarpX-native sources for every parity family

| Parity family (OSIRIS source) | WarpX-native source | Notes |
|---|---|---|
| Boundary R/T/A budget, EM spectra, spectrogram (`e2/b3`,`e3/b2` slabs from full-grid FLD dumps) | openPMD full-field dumps at matched cadence (1D fields are tiny) → same slab Riemann split; **plus** `FieldProbe` lines at both edges every step, and the Poynting-flux reduced diagnostic (verify exact name, `FieldPoyntingFlux`) for step-cadence $R(t)$/$T(t)$ | step-cadence probes beat OSIRIS time resolution — log native version as `*_probe` cross-check metrics |
| EPW energy $W(t) = \tfrac12\int E_x^2\,dx$ (`FLD/e1`) | `FieldReduction` reduced diag: `reduction_type = integral`, `reduced_function = Ex*Ex` — every step, nearly free (verify parser syntax) | growth fits get far more points than dump-cadence OSIRIS |
| Hot-e $x$–$\gamma$ energy-flux (`x1gl_q1`, log-γ, deposit $KE\cdot v_1$) | `ParticleHistogram2D`: abscissa1 $= x$, abscissa2 $= \log_{10}(\gamma - 1)$ (log-spaced bins via the function), `value_function` $= KE\cdot v_x$ signed | covers the true 50 keV threshold natively (the OSIRIS linear-γ axis couldn't); match 1024 bins, γmax=200 |
| $f(x, p_1)$ phase space (`p1x1`) | `ParticleHistogram2D`: $x$ vs $u_x$, matched bin ranges (ps_pmin=−0.5, ps_pmax=2.4 → same in $u$) | feeds distribution lineouts, `fwd_frac_above_50keV`, fp1 super-gaussian fit, p1x1 fallback flux |
| 1-D γ spectrum (`gl`) → `T_hot_keV`, `baseline_hot` | `ParticleHistogram` with $\log_{10}(\gamma-1)$ bin function | text output, trivial to parse |
| Species/field energy history (`HIST/`) → `energy_drift_frac`, species-KE plot | `ParticleEnergy` + `FieldEnergy` reduced diags | native analog of HIST, every-N-steps |
| Field maps, $\omega$–$k$, `field_k_t` | openPMD field dumps (cadence-matched) | general adept plot set consumes the NetCDFs |
| **New, no OSIRIS analog:** true escaping hot-e flux | `BoundaryScrapingDiagnostics` / ParticleBoundaryBuffer at $x$ boundaries | log as `hot_e_flux_frac_*_scraped` — a cleaner physical estimator than the in-box slab deposit; part of the comparison story, not a parity blocker |

Deck-side requirements carried over as *behaviors*, not deck lines: laser-off gap
before turn-on (~9 field-dump intervals) so `*_noise`/`*_denoised` metrics work;
dump cadence ≈ every 10.7 $\omega_p^{-1}$ for the parity estimators (fp1's final-100-dump
and onset's first-50-dump windows are dump-count-sensitive — keep cadence matched);
momentum bins resolving $u_{thr} = 0.4531$.

Verify early (M0): exact reduced-diag names/syntax at our pinned commit
(`FieldReduction`, `FieldProbe`, `ParticleHistogram2D` output format/openPMD-vs-text,
Poynting-flux diag existence) — 30 minutes against the WarpX docs/source we have built.

## 4. Physics matching for the 1:1

Reference point: **srs-Ln100-Te4** (both collisionless and collisional variants exist
at 5x duration; MLflow exps 188939/188941; raw dumps on Perlmutter scratch — analyze
in place). Match in SI: same $T_e$, $L_n$, $I$, $\lambda_0$, density range ($n/n_c$
window), box length, simulation duration; same ppc, same $\Delta x$ in skin depths,
CFL-comparable $\Delta t$; Yee/FDTD solver first (matching OSIRIS's FDTD — PSATD is a
later, separate comparison); matched particle shape order (check OSIRIS deck
`interpolation`; WarpX `algo.particle_shape`); current smoothing off or matched.

Expectations to hold ourselves to (from the SRS literature notes): linear growth rate,
saturated-window $R$/$T$/$A$, hot-e flux fractions, $T_{hot}$, onset time should agree;
instantaneous traces after saturation will not (different noise seeds), and thresholds
are ppc-sensitive — compare windowed averages and, near thresholds, small ensembles.

Collisions: the OSIRIS coll runs used its binary-collision package; WarpX has its own
binary-collision module — a physics comparison of the two operators is part of the
payoff, but start the 1:1 with the **collisionless** pair to isolate PIC-core parity.

## 5. Milestones

- **M0 — verify diag primitives** (small): confirm reduced-diagnostic names/output
  formats at commit 72280884a; run a 10-minute 1D test with all candidate diags on a
  shared GPU slice; decide FieldProbe vs dump-slab detail.
  **Source-verification DONE 2026-08-14 → `warpx-m0-diag-verification.md`.** All
  primitives exist (incl. `FieldPoyntingFlux`); headline corrections: PML aborts in
  1D (use `absorbing_silver_mueller`), Maxwellian is `maxwellian` + `ux_std` (no
  `maxwell_boltzmann`/eV inputs), ParticleHistogram2D uses `_abs`/`_ord` params and
  openPMD-mesh output, no `gamma` parser variable. The on-GPU 10-minute smoke run
  remains to be done at M1 wrap-up.
- **M1 — wrapper skeleton** (core): `adept/warpx/` deck+runner+base, `solver: warpx`
  registry, two-stream-style smoke deck, tests (self-skipping without binary), CI
  paths filter. Runs end-to-end on Perlmutter inside an allocation, logs to MLflow.
  **DONE 2026-08-14** (deck/runner/base/post + registry + tests + CI + docs;
  `WarpxLPI` pass-through stub + smoke campaign in warpx-lpi). Live end-to-end run
  verified on a Perlmutter debug-QOS GPU node (job 56960877, binary
  `build_pm_gpu/bin/warpx.1d`): 100 steps in 0.57 s, zero WarpX warnings, openPMD
  h5 dumps + FieldEnergy/ParticleEnergy reduced diags on disk, MLflow exp
  `warpx-lpi-smoke` (188980) run FINISHED with wall_time_s/exit_code/final_step
  metrics and inputs/warpx_used_inputs/reducedfiles/units.yaml artifacts.
- **M2 — io + general plots**: openPMD→NetCDF layer, units (SI→normalized), general
  canned plots (spacetime, lineouts, ω–k, energy traces), `regen`. Reduced-diag
  parsers land here.
  **DONE 2026-08-14.** `adept/warpx/io.py` reads openPMD-h5 with plain h5py (no
  openpmd-api dep) and emits the *same* `binary/<diag>.nc` contract the OSIRIS
  wrapper writes — keys (`FLD/e1`, `RAW/<sp>`, `HIST/energy`, `REDUCED/<name>`),
  dims `(t, x1)`, code units from `CodeUnits(n0)`, OSIRIS attrs — via the
  handedness-preserving 1D cyclic map `(z,x,y)→(1,2,3)` (`E_z→e1`, `E_x→e2`,
  `E_y→e3`, `B_z→b1`, …), so `adept.osiris.plots.save_canned_plots` renders WarpX
  runs **unchanged** (spacetime/lineouts/ω–k/currents/Riemann field_decomp/energy
  traces); `HIST/energy.nc` is built from FieldEnergy+ParticleEnergy in the exact
  `load_hist_energy` schema so conservation plots + `energy_drift_frac` work too.
  `plots.py` adds per-reduced-diag trace figures; `post.collect` gains the
  conversion, plots, energy metrics, and the `completed_steps_frac` exit-0
  tripwire; `regen.py` mirrors the OSIRIS CLI (and converts a raw `diags/` dir in
  place, recovering units via `io.code_units_for_run`). Verified against committed
  fixtures = the real M1 smoke-run output (`tests/test_warpx/fixtures/smoke-1d/`,
  2.6 MB): 48 tests incl. hand-checked unit conversions (p1 spread = deck
  `ux_std`, macro-charge sum = profile mean, staggering at dz/2).
- **M3 — SRS deck + parity postproc**: WarpX-native 1D SRS deck (antenna, exponential
  profile, PML, laser-off gap, full diagnostic set); `warpx_lpi` adapter +
  `WarpxLPI`; `collect_srs` running unchanged on a WarpX run producing the full
  metric set + `srs_timeseries.nc` + plots.
  **DONE 2026-08-14.** Because M2 fixed the `binary/` contract, the "adapter"
  collapsed into the io layer: `adept/warpx/io.py` now converts
  ParticleHistogram2D openPMD histories → `PHA/<name>/<species>.nc` in the
  OSIRIS phase-space conventions (edge-style axes; count deposits as
  charge-signed `q·f` with `Σ f·d(axis) = n/n0`; flux deposits — value
  functions in `m_e c^3`-reduced form — normalized so the ordinate integral is
  a flux in `n0 m_e c^3` units), 1-D ParticleHistogram txt → `PHA` spectra,
  and BoundaryScraping → `SCRAPED/<sp>/<edge>.nc`; deck-named diagnostics
  (`p1x1`, `x1log_gamma_q1`, `log_gamma`) then drive `collect_srs`'s dispatch
  **unchanged** via `adept.osiris.io`. osiris-lpi was made installable
  (hatchling, `osiris_lpi` package only) and warpx-lpi depends on it
  (editable path dep) — decision (1) resolved as "installable". `WarpxLPI`
  runs `collect_srs` on `td/binary` (drive params from `write_units`:
  a0, ω0/ωp0, t_unit_ps) + the native cross-checks in `warpx_lpi/native.py`
  (`*_poynting` R/T from FieldPoyntingFlux, `*_native` EPW fit from
  FieldReduction, `*_scraped` wall flux). Deck
  `warpx-lpi/sims/srs-Ln100-Te4-1d/inputs-1d`: scan2-matched physics (§4)
  with all diagnostics at OSIRIS cadence (60 steps = 10.68/ω0). Verified on
  committed real fixtures (`tests/test_warpx/fixtures/srs-mini-1d`, a real
  Perlmutter run of the shrunk deck; 56 adept + 9 warpx-lpi + 99 osiris-lpi
  tests) and live on debug QOS: the 2 ps / 4×-threshold smoke logged the full
  parity set (`epw_growth_rate` 0.00352 with the native fit agreeing to 0.3%,
  `T_hot_keV` 73, `hot_e_flux_frac_fwd_50keV` 0.79, R/T/A budget, onset,
  `srs_timeseries.nc`, full plot set). Gotchas: `amr.n_cell = 3594` needs
  `blocking_factor = 2` (default 8 aborts); ParticleHistogram2D defaults to
  bp5 — pin `<name>.openpmd_backend = h5` — and leaves an *empty* companion
  `.txt` (skip it; it once broke the HIST builder); its openPMD iteration key
  is the plain step (not step+1) with a correct `time` attr; thermal particle
  walls leave the scraping buffers empty (parity walls beat the scraped
  estimator — revisit at M4). Antenna calibration flag for M4a:
  `laser_incident_measured_frac` = 1.76 — the antenna embedded in
  0.18 n0 plasma radiates ~1.76× the vacuum-normalized I0.
- **M4 — validation ladder**: (a) vacuum laser propagation: `laser_incident_measured_frac`≈1,
  $R$≈0, $T$≈1; (b) linear EPW/Landau or two-stream growth vs OSIRIS same-physics run;
  (c) the srs-Ln100-Te4 collisionless 1:1 — side-by-side $R(t)$, $W(t)$ growth rate,
  hot-e flux, spectra, onset, $T_{hot}$. Deliverable: comparison report/plots.
- **M5 — campaign readiness**: parsl scan driver pattern (as osiris-lpi
  `scan_pack_parsl.py`), sbatch template reuse from `warpx-sw`, then optionally a
  scan2-subset replication and the collisional comparison.

M1–M3 are mostly mechanical given how cleanly the OSIRIS wrapper decomposes; the
physics risk concentrates in M4 (ppc/noise sensitivity, laser-injection differences —
OSIRIS antenna vs WarpX antenna ramp shapes need care to match $I(t)$ envelopes).

## 6. Open questions (defaults chosen, flag if wrong)

1. ~~`warpx_lpi` inside the osiris-lpi repo vs a new repo~~ — **decided: new repo
   `ergodicio/warpx-lpi`** (2026-08-12). Follow-on choice at M3: make osiris-lpi
   installable vs extract shared analysis package.
2. Native inputs files as deck truth (default; flat overrides + `warpx_used_inputs`
   provenance, matches the wrapper philosophy) vs PICMI-generated input — PICMI stays
   available for interactive work either way.
3. Streaming/ramdisk layer deferred (default) — revisit on evidence.
