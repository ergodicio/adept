# WarpX Configuration Reference

This document describes how to construct a configuration file for the `warpx` solver — and how
that YAML manifest relates to the native WarpX inputs file it wraps.

Like the OSIRIS wrapper (and unlike the native adept solvers), the WarpX wrapper does not define
the physics in YAML. The **native WarpX inputs file is the source of truth**: the manifest points
at an inputs file, optionally patches it with `overrides`, and configures how the run is executed
and logged. WarpX inputs are SI throughout and form a flat [AMReX ParmParse](https://warpx.readthedocs.io/en/latest/usage/parameters.html)
namespace, so overrides are a flat `key: value` mapping rather than the OSIRIS section machinery.

## Top-Level Structure

```yaml
solver: warpx           # required, dispatch key

mlflow:
  experiment: my-experiment   # required
  run: my-run                 # required

warpx:
  deck: decks/srs-1d          # required — the native WarpX inputs file
  reference_density: 9.05e21  # cm^-3; fixes the wp0/skin-depth normalization
  binary: /path/to/warpx.1d   # optional, see binary resolution
  mpi_ranks: 1
  overrides:
    amr.n_cell: 512
```

## warpx

| Field | Type | Description |
|-------|------|-------------|
| `deck` | string | Path to the native WarpX inputs file, repo-relative or absolute (required). This file is parsed, optionally patched by `overrides`, re-rendered to `inputs` in the run directory, and every key is logged to MLflow. |
| `reference_density` | string \| float | Physical reference density that fixes the normalization used for MLflow comparability and (in M2+) for converting diagnostics to code units. A pint-parsable string (`"9.05e21 /cc"`) or a bare number interpreted as cm^-3 (the OSIRIS `simulation.n0` convention). Falls back to the deck's `my_constants.n0` interpreted as SI m^-3. |
| `binary` | string | Path to the built WarpX executable. Optional — see [Binary resolution](#binary-resolution). |
| `mpi_ranks` | int | `1` runs the binary directly; `>1` launches `<mpi_launcher> -n N` (default `1`) |
| `mpi_launcher` | string | Launcher for `mpi_ranks > 1` (default `srun`; use `mpirun` locally) |
| `extra_mpi_args` | list[string] | Extra arguments passed to the launcher, e.g. `["--gpus-per-task=1"]` |
| `run_root` | string | Parent directory for per-run working dirs (default `./checkpoints`) |
| `overrides` | mapping | Flat deck patches applied before rendering. See [below](#overrides-patching-the-deck). |

> **Note on `run_root`:** as with the OSIRIS wrapper, the default sits inside `checkpoints/`
> deliberately — sync scripts exclude it, so in-flight and finished WarpX outputs survive a sync.

### Binary resolution

The runner resolves the WarpX executable in this order:

1. `warpx.binary` in the manifest
2. `WARPX_BIN_<dim>D` environment variable (e.g. `WARPX_BIN_1D`), where the dimensionality is
   read from the deck's `geometry.dims` (falling back to the length of `amr.n_cell`)
3. `WARPX_BIN` environment variable

### Overrides: patching the deck

`overrides` is a flat mapping from full ParmParse keys to values, applied to the parsed deck
before rendering:

```yaml
warpx:
  overrides:
    amr.n_cell: 512
    max_step: 2000
    electrons.density_function: "n0*exp(z/Ln)"   # base name resolves to the (x,y,z) key
    laser1.e_max: 3.66e10
```

- A key may be given without its parser-argument spec (`electrons.density_function` matches
  `electrons.density_function(x,y,z)`) when unambiguous.
- Unknown keys are appended — WarpX ignores unused parameters, and a new key (e.g. an extra
  diagnostic) is a legitimate override.
- List values are rendered space-separated, matching ParmParse syntax.
- **Multi-valued parameters must be YAML lists, not strings.** The override's YAML type
  selects the ParmParse type: `warpx.numprocs: [2, 16]` renders as `= 2 16` (an int
  array), while `warpx.numprocs: "2 16"` is a *string* and renders **quoted** (`= "2 16"`,
  one token — strings with whitespace must be re-quoted to survive the parse/render round
  trip), which makes WarpX abort at startup inside `ReadParameters`
  (`queryArrWithParser`). Nothing validates an override's type against the deck value it
  replaces; the run fails at launch with the `ReadParameters` abort in the error. A
  non-zero exit is only salvaged (logged, exit code recorded as a metric, post-processing
  run on the partial data) when non-empty files exist under the diagnostic paths the deck
  configures — `warpx_used_inputs` and `Backtrace.*`, which a startup abort leaves in the
  run dir, do not count.

The post-override deck is what runs, is logged key-by-key to MLflow (under `deck.*`), and is
archived as the `inputs` artifact; WarpX's own `warpx_used_inputs` is archived too for provenance.

## Units

`write_units` derives the same canonical scales the other adept solvers log (`wp0`, `tp0`, `n0`,
`x0` = skin depth, `v0` = c) from `reference_density`, plus the physical laser drive scales
(`w_laser`, `laser_wavelength`, `laser_a0`, `laser_intensity`) from the first laser's
`wavelength` and `e_max`. WarpX decks are SI, so this is an SI → normalized derivation — the
inverse direction of the OSIRIS wrapper.

## output

Optional block controlling post-processing, mirroring the OSIRIS wrapper:

| Field | Type | Description |
|-------|------|-------------|
| `diagnostics_to_log` | list[string] | Whitelist of diagnostics to convert/upload, matched on the contract key (`FLD/e1`) or its leaf name (`e1`). Default: everything. **Gates conversion, not just upload — see the warning below.** |
| `v_th` | float | Electron thermal velocity (units of c) for the Langmuir/Bohm–Gross overlay on ω–k plots |
| `omega_k_zoom` | float \| null | `(k, ω)` half-width for the equal-aspect lower ω–k panel (`null` → full Nyquist) |
| `overlay_density` | float | Density (units of `reference_density`) at which dispersion overlays are evaluated (`ω_p = sqrt(n)`) |
| `bam` | bool | Shade the beam-acoustic-mode band on ω–k plots (needs `v_th`) |

```{warning}
`diagnostics_to_log` is passed straight to
{func}`adept.warpx.io.save_run_datasets` as its `diagnostics=` argument, so a
non-whitelisted diagnostic is **never written into `binary/`** — it is not
merely withheld from the MLflow upload. Every downstream consumer reads
`binary/`, so using this field to keep bulk off the tracking server also
disables the analyses that depend on it.

Excluding the full-field maps (`FLD/e1`…`b3`, `DENSITY/*`) on a **2D** run
silently removes: the `srs2d` bundle figures (F1–F9 — `bundle2d` is on by
default and simply gets no input), `energy_vs_time` /
`energy_components_vs_time` / `epw_energy_vs_time`, the non-native
`epw_growth_rate`, and the k-t spectrogram. It also makes `warpx_lpi.native`
fall back to the 1-D Poynting normalization — it reads the transverse box
width off a converted 2-D `FLD/*.nc` — yielding a spurious
`laser_reflectivity_poynting` ≈ −1.

Unaffected: `PHA/*`, `REDUCED/*`, `HIST/*`, and the FieldProbe line series
`FLD/<comp>-line-*`, which are one-dimensional in space. Note that each line
series is its own contract key, so whitelisting `e2` does **not** carry
`e2-line-x2-0024` along with it.
```

```{note}
The boundary-light products — the dump-cadence laser energy budget
(`laser_reflectivity`, `laser_transmissivity`, `laser_absorbed_frac`), the EM
boundary-light spectrum, and its spectrogram — are **absent on any 2D run
regardless of this whitelist**. They are built by
{func}`adept.osiris.plots.transverse_field_boundary_slabs`, which loads a
candidate field only when `ser.ndim == 2`; a converted 2-D map is
`(t, x2, x1)`, so no pair is ever formed and the caller reports *"no transverse
field pairs (need e2/b3 or e3/b2)"*. Widening the whitelist will not fix this.
```

## Post-processing artifacts

After the run, `post.collect` converts the WarpX output into the same per-diagnostic NetCDF
contract the OSIRIS wrapper emits, in **code units** fixed by `reference_density` (time in
`1/ω_p`, length in `c/ω_p`, fields in `m_e c ω_p / e`, current in `e n_0 c`; SI passthrough
when no reference density is available). The 1D axis mapping is the handedness-preserving
cyclic relabeling `(z, x, y) → (1, 2, 3)`, so `E_z → e1` (longitudinal), `E_x → e2`,
`E_y → e3`, and the OSIRIS sign conventions (including the left/right Riemann pairs) carry
over:

```
binary/FLD/e1.nc …           stacked (t, x1) field series, OSIRIS naming
binary/DENSITY/<sp>/charge.nc  rho_<species> in e·n0 units (when dumped)
binary/RAW/<species>.nc      long-form particle dumps (x1, p1–p3 in m_s c, ene = γ−1,
                             q = signed macro-charge, w = openPMD weighting)
binary/PHA/<name>/<sp>.nc    ParticleHistogram2D phase spaces / ParticleHistogram
                             spectra in OSIRIS phase-space conventions (see below)
binary/SCRAPED/<sp>/<edge>.nc  BoundaryScraping buffers, long-form + t_scraped
binary/REDUCED/<name>.nc     native SI reduced-diagnostic tables
binary/HIST/energy.nc        OSIRIS energy-history schema from FieldEnergy+ParticleEnergy
plots/…                      the OSIRIS canned plot set + reduced/<name>.png traces
```

### Phase-space histograms (`PHA/`)

`ParticleHistogram2D` reduced diagnostics (openPMD dirs under
`diags/reducedfiles/<name>/`) and 1-D `ParticleHistogram` tables become
OSIRIS-style phase spaces keyed `PHA/<name>/<species>` — so naming a
reduced diagnostic after the OSIRIS phase space it mirrors (`p1x1`,
`x1log_gamma_q1`, `log_gamma`) makes downstream OSIRIS-convention consumers
(e.g. `osiris_lpi.collect_srs`) dispatch on it unchanged. The run's rendered
`inputs` deck drives the conversion:

- the ordinate dim is named from `histogram_function_ord`: `log10(...)` →
  `gamma` (the OSIRIS log-γ axis), bare `uz`/`ux`/`uy` → `p1`/`p2`/`p3`;
- `value_function = w` (a count deposit) is stored as the charge-signed
  density `(q/|q|)·f` normalized so `Σ f·d(axis) = n(x)/n0` — the OSIRIS
  cartesian phase-space convention;
- any other value function is treated as a flux deposit and must be written
  in the `m_e c^3`-reduced form (e.g. `w*(g-1)*(uz/g)` for the OSIRIS
  `x1gl_q1` energy-flux deposit `KE·v1`); the stored field integrates over
  the ordinate to a flux in `n0 m_e c^3` units — the same unit as
  `I0 = (a0 ω0)^2/2`;
- axes carry the adept-OSIRIS *edge-style* labels (`linspace(min, max, n)`
  over the bin range) and the deck's `geometry.prob_lo/hi` become
  `sim.XMIN/XMAX`.

### Conversion: one walk, one slab

Both the field records and the ParticleHistogram2D phase spaces are converted
**a dump at a time**, through the same `StreamWriter` the OSIRIS drainer uses.
Two properties follow, and both are load-bearing at production cadences:

*Peak memory is one slab plus a ~1 MiB write batch, not the stacked history.*
A production `x1log_gamma_q1` (11705 dumps × 1000 × 1024) is a 47.9 GB cube,
and the eager `load_particle_histogram2d` → `series_to_dataset` path held two
copies of it. The field path stacked a list of float64 slabs — ~6.3 GB per
component at 88k dumps of 3594 cells, and far worse in 2D.

*Each dump is opened once, not once per record.* openPMD bundles every record
of a dump into one file, but the converter used to re-walk the whole tree per
component. At a field cadence giving a 2 ω₀ Nyquist (a dump every 7–9 steps at
`dt = 0.178/ω₀`, so ~88k dumps over a 23 ps run) that is 614k opens against
88k files; at the ~6.5 ms/file measured on Perlmutter's Lustre the redundant
six-sevenths alone cost ~57 min per simulation. Everything a record needs —
axes, code-unit scaling, `binary/` key, storage-order transpose — comes from
the first dump, so `diagnostics_to_log` now also skips records *before* they
are read rather than loading and discarding them.

`StreamWriter` chunks each variable `(chunk_t, …)` with `chunk_t` sized to
~1 MiB (1 dump per chunk at production bin counts), so the per-dump reads in
`osiris_lpi` are chunk-aligned; the batch path's default h5netcdf chunking
spread one time slice over 2016 chunks. `load_field_series` and
`load_particle_histogram2d` still return whole histories for interactive use
and remain the reference implementations the streamed output is tested
against.

Because the contract matches, `adept.osiris.io.list_diagnostics` / `load_series` and the
OSIRIS canned plots read these files unchanged. Logged metrics include `final_step`,
`completed_steps_frac` (the exit-0 early-termination tripwire — WarpX exits 0 on
`break_signals` and silently ignores typo'd parameters), `field/efield/bfield_energy_final`
(code units, from the FieldEnergy reduced diagnostic), and `energy_drift_frac`.

Regenerate the plot set offline from the saved NetCDFs (or convert a raw run directory
in place):

```
python -m adept.warpx.regen <run-or-binary-dir> [--out DIR] [--v-th 0.0885] [...]
```

## Status

M1 (wrapper skeleton): deck parsing/overrides/logging, subprocess runner with
salvage-on-partial-output, units, provenance upload. M2 (io + plots): openPMD → NetCDF
conversion to the OSIRIS `binary/` contract, code-units conversion, reduced-diagnostic
parsers, canned plots, `regen`. M3 (SRS parity): phase-space histogram / boundary-scraping
conversion (`PHA/`, `SCRAPED/`) feeding the `osiris_lpi` SRS analyses unchanged; the SRS
deck and the `WarpxLPI` adapter live in the `warpx-lpi` repo. See
`dev_docs/warpx-wrapper-plan.md` on the `warpx-wrapper` branch for the plan of record.
