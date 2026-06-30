# OSIRIS adept module — usage overview

## File structure

```
adept/                                          (repo root)
├── run.py                                      ← CLI entry (--cfg path, no .yaml suffix)
├── adept/
│   ├── _base_.py                               ← dispatcher: `osiris` solver branch
│   └── osiris/                                 OSIRIS wrapper package
│       ├── __init__.py                         lazy export of BaseOsiris
│       ├── base.py        ◀── BaseOsiris(ADEPTModule): __init__ parses deck,
│       │                      __call__ runs OSIRIS, post_process delegates
│       ├── deck.py        ◀── namelist parser/renderer/merger
│       │                      parse_deck(text), parse_deck_file(path),
│       │                      render_deck(sections), merge_overrides(...),
│       │                      deck_to_flat_dict(...) for MLflow params
│       ├── runner.py      ◀── subprocess driver
│       │                      run_osiris(deck_text, binary=…, mpi_ranks=…),
│       │                      discover_binary(...), OSIRIS-error detection
│       ├── post.py        ◀── post-run collection: final-step HDF5 copy,
│       │                      NetCDF export, scalar metrics
│       ├── io.py          ◀── HDF5/NetCDF readers + dataset save/load
│       ├── plots.py       ◀── canned plot set (save_canned_plots)
│       └── regen.py       ◀── regenerate plots offline from saved NetCDFs
├── configs/
│   └── osiris/                                 example manifests
│       ├── twostream-1d.yaml                   full run (deck tmax=100)
│       └── twostream-1d-short.yaml             tmax=1.0 smoke
└── tests/
    └── test_osiris/
        ├── decks/two-stream-1d                 in-repo example deck (manifests point here)
        ├── test_deck_roundtrip.py              namelist parser round-trip
        ├── test_runner.py                      subprocess runner + discover_binary
        ├── test_units.py                       units.yaml derivation
        ├── test_post_netcdf.py                 post → NetCDF export
        ├── test_io_and_plots.py                io readers + plotting
        ├── test_diagnostics_plots.py           per-diagnostic plots
        └── test_plots_new_views.py             field-decomp / phase-space views
```

## End-to-end data flow

```
 YAML manifest                      OSIRIS native deck
 (configs/osiris/*.yaml)            (tests/test_osiris/decks/… or any path)
        │                                    │
        └──────────► run.py --cfg ◄──────────┘
                          │
   ── setup phase ────────┼──────────────────────────────────────────────
   ergoExo.setup ──► BaseOsiris.__init__:
                       parse_deck_file()
                       merge_overrides()                  (mutates sections in place)
                       cfg["deck"] = deck_to_flat_dict(merged sections)
                          │
                          ▼
                     log_params ──► MLflow                ← logs the POST-override deck
                          │
   ── run phase ──────────┼──────────────────────────────────────────────
   ergoExo(modules) ─► BaseOsiris.__call__:
                       render_deck(merged sections) ──► run_dir/os-stdin
                       runner.run_osiris(mpirun…)   ──► run_dir/MS/{FLD,PHA,…}
                          │
                          ▼
                     post.collect ──► td/ ──► MLflow
```

The deck logged to MLflow is the one **after** `merge_overrides` is applied — the
same merged sections that `render_deck` later writes to `os-stdin` — so the logged
params always match what OSIRIS actually ran.

## How to run

First, point the runner at your built OSIRIS binary. The runner resolves it in
this order: `osiris.binary` in the manifest → `OSIRIS_BIN_<dim>D` env var (e.g.
`OSIRIS_BIN_1D`) → `OSIRIS_BIN` env var. The example manifests omit `osiris.binary`,
so set the env var once per shell:

```bash
export OSIRIS_BIN_1D=/path/to/osiris-1D.e        # per-dim, preferred
# export OSIRIS_BIN=/path/to/osiris.e            # or a single default for all dims
```

Then run from the repo root (`run.py` appends `.yaml` to `--cfg`, so omit the suffix):

```bash
uv run run.py --cfg configs/osiris/twostream-1d-short      # smoke
uv run run.py --cfg configs/osiris/twostream-1d            # full
uv run mlflow ui --backend-store-uri file://$(pwd)/mlruns  # browse
```

## Manifest schema

```yaml
solver: osiris                                    # required, dispatch key

mlflow:
  experiment: osiris-pic1d-twostream              # required
  run: cold-equal-beams                           # required

osiris:
  deck: tests/test_osiris/decks/two-stream-1d     # required: source of truth (repo-relative)
  # binary: /path/to/osiris-1D.e                  # optional: overrides the env-var default
  #                                               #   (OSIRIS_BIN_<dim>D → OSIRIS_BIN)
  mpi_ranks: 1                                    # 1 → direct, >1 → mpirun -n N
  run_root: ./checkpoints                         # parent of per-run dirs (default)
  # NOTE: the default sits inside checkpoints/ deliberately — sync-up.sh
  # rsyncs with --delete but excludes checkpoints/, so in-flight and finished
  # OSIRIS outputs survive a sync. If you point run_root anywhere outside an
  # excluded directory, a sync-up invocation will delete those outputs.
  # Nothing deletes run dirs automatically (post-processing only copies out
  # of them), so clean them up manually on occasion — though $PSCRATCH's
  # purge policy will take care of stale ones eventually.
  extra_mpi_args: ["--oversubscribe"]             # optional, passed to mpirun

  stream_convert: true                            # optional (default true): convert MS/
                                                  #   HDF5 dumps to binary/*.nc concurrently
                                                  #   with the run; set false for the old
                                                  #   batch conversion at job end
  stream_poll_s: 10.0                             # optional: watcher poll interval (s)

  density:                                        # optional: adaptive box sizing (1D)
    gradient_scale_length: 300um                  #   target L_n; scales the box so the
                                                  #   deck's density ramp realizes this L
    # min: 0.225                                  #   nmin (n_c units); default: deck profile.fx
    # max: 0.275                                  #   nmax (n_c units); default: deck profile.fx
    # reference_density: 0.25                     #   density (n_c units) where L is defined

  overrides:                                      # optional: applied before render
    time: {tmax: 50.0}                            #   merge into the (one) time block
    grid: {nx_p: [256]}                           #   refresh an array key
    species:                                      #   indexed for repeated sections:
      0: {num_par_x: [512]}                       #     species 1: bump ppc
      1: {ufl: [-2.0, 0.0, 0.0]}                  #     species 2: change drift

output:
  diagnostics_to_log: null                        # null = all; or [e1, charge, …]
  v_th: 0.1                                         # optional: overlays the Bohm–Gross
                                                    #   Langmuir branch on ω–k plots
  omega_k_zoom: 4.0                                 # (k, ω) half-width [ω_p] for the
                                                    #   equal-aspect lower ω–k panel
                                                    #   (clamped to Nyquist); null = full
```

Override keys can use the **base name** (`nx_p`) or the **exact key** (`nx_p(1:1)`). Indexed `{0: …, 1: …}` form addresses occurrences of repeated sections (`species`, `udist`, `profile`, `spe_bound`, `diag_species`, `zpulse`, …) in source order.

## Adaptive box sizing from a density gradient scale length

`osiris.density` (1D decks) scales the simulation box so the deck's linear density
ramp realizes a target gradient scale length `L`, mirroring how adept's `_lpse2d`
and `kinetic_srs` solvers size their grids. This is the **inverse** of holding the
box fixed and steepening the ramp: here the density range (`nmin`/`nmax`) is held
and the box length follows `L`.

For a linear ramp `n(x): nmin → nmax`, the local scale length is
`L(x) = n(x)/(dn/dx) = n(x)·ramp_span/(nmax−nmin)`, so requiring `L(n_ref) = L` at the
reference density `n_ref` (default the quarter-critical surface, `n_c/4 = 0.25`)
fixes the ramp span — the same relation adept uses, `ramp_span = L/0.25·(nmax−nmin)`.

The result is a single spatial scale factor `s` applied to **every** length in the
deck: `space.xmin`/`space.xmax`, all `profile.x` arrays, and all `diag_species`
phase-space windows (`ps_xmin`/`ps_xmax`). `grid.nx_p` is scaled by `s` too — holding
the cell size `dx` fixed (rounded up to a multiple of `node_conf.node_number(1)` for
even domain decomposition). Time (`dt`, `tmax`) is untouched, so the CFL ratio is
preserved.

- Activates only when `osiris.density.gradient_scale_length` is present (decks
  otherwise run with their hand-set box, unchanged). Runs **after** `overrides`, so
  it supersedes any `space.xmax` override.
- `gradient_scale_length` takes a unit string (`300um`, converted via the deck's
  `simulation.n0`/`omega_p0`) or a bare number already in `c/wp0` units.
- `min`/`max` default to the ramp's interior `profile.fx` endpoints; if given, they
  are written into the primary `profile.fx`.
- The computed quantities (`box_norm`, `nx`, `scale_factor`, …) are logged under
  `osiris.density.derived.*`.
- Multi-dimensional decks raise `NotImplementedError`. Drive positions (e.g. a
  `zpulse` spatial center) are **not** rescaled — boundary antennas like the SRS
  deck's `antenna_array` have no position to scale.

## What lands in MLflow

| Kind        | Content                                                                          |
| ----------- | -------------------------------------------------------------------------------- |
| Params      | every deck key, flattened — `deck.grid.nx_p_1:1`, `deck.species_0.ufl_1:3.0`, …  |
| Params      | the manifest itself — `solver`, `osiris.deck`, `output.diagnostics_to_log`, …    |
| Metrics     | `wall_time_s`, `exit_code`, `field_energy_final`, `final_iter`, `run_time`, `postprocess_time` |
| Artifacts   | `config.yaml`, `derived_config.yaml`, `units.yaml` (adept stock)                 |
| Artifacts   | `os-stdin` (rendered OSIRIS deck), `stdout.log`, `stderr.log`                    |
| Artifacts   | `binary/<FLD\|PHA\|…>/<diag>.nc` — one xarray netCDF per diagnostic, holding the full `(t, …)` time history (replaces the raw h5 dumps) |
| Artifacts   | `plots/…` — canned PNGs (see below)                                              |

## Concurrent H5 → NetCDF conversion (`osiris.stream_convert`)

The `binary/*.nc` series are converted **during** the run by default
(`osiris.stream_convert: true`). The alternative — set it `false` — is the old
behavior: build them **after** the run, where `post.collect` reads each
diagnostic's whole `(t, …)` history into memory and writes it in one shot. For
runs that dump a field every few steps (hundreds of thousands of tiny HDF5
files) that end-of-run pass is slow (a cold Lustre re-read of every file) and
memory-hungry (the whole stacked series at once), which is why streaming is the
default.

With streaming on, a best-effort background thread
(`adept/osiris/stream.py::StreamConverter`), spawned by `run_osiris` alongside
the OSIRIS subprocess, polls `MS/` every `stream_poll_s` seconds and appends
each completed dump into `<run_dir>/binary/<diag>.nc` as it lands. Effects:

- **I/O overlaps compute** — the conversion latency is hidden behind the
  (hours-long) PIC run, and each dump is read while still warm in the page
  cache instead of re-opened cold at the end.
- **Bounded memory** — both the watcher and the at-job-end fallback build the
  NetCDF a dump at a time (`load_grid_h5` → one `(t, …)` slot), so peak
  conversion RAM is a single dump rather than the full series.

Mechanics and guarantees:

- Each NetCDF uses an **unlimited `t` dimension grown one slot per dump**, so it
  ends up sized to exactly the dumps produced — no pre-sized guess, no trailing
  fill to trim on early termination. Compression (zlib + shuffle) and chunking
  are preserved, so the files match the batch path's `binary/*.nc` contract and
  every downstream reader (`load_series_nc`, the plotting code, `regen`) is
  unchanged.
- A dump is only read once the **next** dump exists (OSIRIS has moved on),
  avoiding partial-file reads; the final sweep at job end picks up the last one.
- **Failure-isolated:** any error in the converter is logged and never touches
  the OSIRIS run. The standard batch conversion is the safety net — `post.collect`
  reuses whatever the watcher finished and stream-builds (still memory-bounded)
  any grid diagnostic it did not reach. RAW (particle) diagnostics, whose
  per-dump particle count varies, always use the batch concat path.
- **Restart-safe:** a `binary/<diag>.nc` reopened after a checkpoint restart
  resumes after the slots already on disk.

This addresses the I/O wall-time and conversion-memory problems only; the
heavy `pcolormesh`/`fft2` plotting cost at *plot* time is independent and
handled by the plotting changes, not here. See
`osiris-lpi/postproc-performance.md` for the full analysis.

## Canned plots (`plots/` artifacts)
These canned plots focus on 1D simulations.

`post.collect` renders a standard plot set via `adept/osiris/plots.py::save_canned_plots`. All labels are emitted as proper LaTeX (`$\omega$`, `$c/\omega_p$`, …).

| Path                                          | What it shows |
| --------------------------------------------- | ------------- |
| `spacetime/<diag>.png`, `spacetime_log/<diag>.png` | `(t, x)` heatmap of each FLD diagnostic (lin + log) |
| `lineouts/<diag>.png`                         | value-vs-`x` snapshots at sampled times |
| `omega_k/<diag>.png`                          | 2-D FFT `(k, ω)` dispersion — full Nyquist range on top, plus an equal-aspect square window below where `ω = k` is drawn at a true 45° |
| `currents/spacetime.png`, `currents/lineouts.png` | combined `J_x/J_y/J_z` (`j1/j2/j3`) views |
| `moments/<species>/…`                         | per-species density-moment spacetime + lineouts |
| `profiles/<species>/density.png`              | density profile vs `x` (final snapshot + late-time mean) |
| `profiles/<species>/temperature.png`          | temperature profile vs `x`, from `uth1/2/3` or `T11/22/33` moments (omitted if neither is dumped) |
| `phasespace/<species>/<ps>.png`, `phasespace_evolution/…` | `(x, p)` phase-space heatmaps |
| `field_decomp/<comp>.png`                     | left/right-going transverse `E` (vacuum Riemann split `(e2±b3)/2`, `(e3∓b2)/2`), spacetime + `ω–k` |
| `energy_vs_time.png`, `energy_components_vs_time.png`, `total_energy_vs_time.png` | field / kinetic energy traces |

> **Note on `field_decomp/`.** The left/right split is exact only in vacuum or a uniform non-dispersive medium (`|E| = |B|` for a pure travelling wave). In a plasma the EM wave is dispersive, so the split is approximate — useful for direction, but cross-check the dispersion before reading the residual as physical counter-propagating power. The longitudinal `e1` is electrostatic and is intentionally excluded.

> **SRS-specific plots & metrics live in osiris-lpi.** The laser-energy budget (reflected / transmitted / absorbed over time: the `energy_budget.png` + `laser_energy_budget.txt` artifacts and the `laser_reflectivity` / `laser_transmissivity` / `laser_absorbed_frac` metrics) and the distribution-function lineouts (`distribution_lineouts/`: `f(p)` averaged over the four domain quarters and the whole box, for the last dump and the last-1/8 average, in linear / log / `δf` views) are **not** produced by adept's general OSIRIS wrapper. They live in the [osiris-lpi](https://github.com/ergodicio/osiris-lpi) repo — `osiris_lpi.OsirisLPI` subclasses `BaseOsiris` and adds them in `post_process`, reading the drive `antenna.a0`/`antenna.omega0` from the deck. Regenerate them offline from saved NetCDFs with `python -m osiris_lpi.regen`.

## Programmatic use

```python
from adept import ergoExo
import yaml

cfg = yaml.safe_load(open("configs/osiris/twostream-1d-short.yaml"))
cfg["osiris"]["overrides"] = {"time": {"tmax": 2.0}}

exo = ergoExo()
modules = exo.setup(cfg)                          # parse + log params
solver_result, post, run_id = exo(modules)        # run + log metrics/artifacts
print(run_id, post["metrics"])
```

## Adding new metrics

Edit `adept/osiris/post.py:collect`. The h5 dump for each diagnostic is at `run_dir/MS/<TYPE>/<diag_name>/<diag_name>-NNNNNN.h5`. Helpers `_latest_h5`, `_field_energy_from_dump`, and `_walk_diag_dirs` are already factored out. Anything you add to the returned `metrics` dict shows up in MLflow automatically.

## Adding a new test problem

Native-deck-as-truth: just write the deck, point a manifest at it, run. No code changes.

```bash
cp my-new.deck tests/test_osiris/decks/
cp configs/osiris/twostream-1d.yaml configs/osiris/my-new.yaml
$EDITOR configs/osiris/my-new.yaml          # change deck path + mlflow.run
uv run run.py --cfg configs/osiris/my-new
```
