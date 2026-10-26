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
│       │                      (load_series ±t_indices, lazy open_series, series_len)
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

The full manifest schema — the `osiris:` section (deck path, binary resolution, MPI, staging,
`overrides` deck-patching semantics, `density` adaptive box sizing) and the `output:` section — is
documented in the [Configuration Reference](config.md). Minimal example:

```yaml
solver: osiris

mlflow:
  experiment: osiris-pic1d-twostream
  run: cold-equal-beams

osiris:
  deck: tests/test_osiris/decks/two-stream-1d
  mpi_ranks: 1

output:
  diagnostics_to_log: null
```

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

## Ramdisk staging (`osiris.stage_root`)

OSIRIS has **no asynchronous diagnostic I/O**: every dump is a synchronous
barrier in the time loop, so on a parallel filesystem (Lustre) the ranks stall
on each collective write, and at full dump cadence that stall can halve compute
utilization. `stage_root` works around it without trimming cadence: point it at
a fast **node-local ephemeral filesystem** (a `/dev/shm` ramdisk on NERSC GPU
nodes — those nodes have no local disk, so RAM-backed `tmpfs` is the only
node-local option) and OSIRIS's synchronous writes hit RAM (microseconds)
instead of Lustre. The slow durable write is moved off the critical path onto
the background drainer — effectively the async I/O OSIRIS itself lacks.

```yaml
osiris:
  run_root: /pscratch/.../checkpoints   # durable (Lustre)
  stage_root: /dev/shm/osiris           # fast scratch (ramdisk); requires stream_convert
  stream_convert: true
```

Mechanics:

- OSIRIS runs with its working directory on `stage_root/<run-name>`; all `MS/`
  dumps land in RAM. The durable run directory under `run_root` is created as
  usual, and `run_dir` in the result (hence everything post-processing reads) is
  always that **durable** directory — so `post.collect` is unchanged.
- The `StreamConverter` runs in **drain mode**: each completed dump is mirrored
  to `run_root/<run-name>/MS/<diag>/` and then **deleted from the ramdisk**, so
  the RAM high-water mark is bounded by the poll interval × production rate, not
  the whole run. Grid diagnostics are additionally streamed to the durable
  `binary/*.nc` as before; RAW (particle) diagnostics, which can't be
  slot-streamed, are mirrored as HDF5 so the batch path still finds them.
- `binary/*.nc` and `stdout.log`/`stderr.log` are written straight to the
  durable dir (by the driver, off OSIRIS's critical path), so they survive even
  if the run is killed. At job end the drainer's final sweep flushes the
  held-back last dump of each diagnostic; the runner then folds any stragglers
  (`HIST/`, `RE/`, …) from the scratch into the durable dir and **reclaims the
  ramdisk**.

Caveats:

- **RAM budget.** The `tmpfs` counts against node memory *alongside* the
  simulation. Bounded by the drain keeping up with production (Lustre bandwidth
  ≫ dump rate, so it normally does); for very long full-cadence runs, keep
  `stream_poll_s` small and leave headroom.
- **Crash window.** Whatever is still on the ramdisk when a node dies is lost
  (`tmpfs` is volatile) — logs and already-mirrored dumps are safe on Lustre;
  only the few in-flight dumps are at risk. If you enable OSIRIS
  checkpoint/restart, keep those files off the ramdisk.
- **Single-node only** — `/dev/shm` is per-node. Fine for the 1-GPU / single-node
  SRS runs; multi-node would need per-node drains and non-shared-file dumps.

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

> **Memory: phase-space plots read slices, not the whole history.** A saved `(t, …)` series can be tens of GB uncompressed — `x1gamma_q1` is ~48 GB in memory (11705 × 1000 × 1024), ~0.19 GB gzipped on disk (sparse, ~250×) — so `io.load_series(path)` decompresses all of it (at 16 sims/node this OOM'd production). `save_canned_plots` therefore loads only what each plot needs: `io.load_series(path, t_indices=…)` for the final dump + the evolution panels (`_evolution_t_indices`), and it decimates field heatmaps to render resolution before `pcolormesh`. A new plotter over a large series must do the same — `load_series(..., t_indices=…)` for a few slices, or `with io.open_series(path) as da:` (lazy; `da.isel(t=it)` reads one dump on demand) when it walks the whole time axis reducing one dump at a time. A stray `da.values` / `da.sum()` over `t`, or a **deep** `da.copy()`, pulls the full series into RAM (`_decorate` copies shallow for this reason).

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

Native-deck-as-truth: just write the deck, point a manifest at it, run. No code changes — see
[Adding a new test problem](config.md#adding-a-new-test-problem) in the Configuration Reference.
