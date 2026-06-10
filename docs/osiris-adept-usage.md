# OSIRIS adept module — usage overview

## File structure

```
adept/                                          (repo root, ~/Desktop/adept/adept/)
├── run.py                                      ← existing CLI entry (unchanged)
├── adept/
│   ├── _base_.py                               ← MODIFIED: dispatcher gained `osiris` branch
│   └── osiris/                                 ← NEW package
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
│       └── post.py        ◀── post-run collection
│                              final-step HDF5 copy, optional MS/ tarball,
│                              scalar metrics
├── configs/
│   └── osiris/                                 ← NEW: example manifests
│       ├── twostream-1d.yaml                   full 30-ω_p run
│       ├── twostream-1d-short.yaml             tmax=1.0 smoke
│       └── twostream-1d-uploadall.yaml         tmax=0.5 + ms.tar.gz
└── tests/
    └── test_osiris/                            ← NEW
        ├── test_deck_roundtrip.py              15 parser tests
        └── test_runner.py                       5 runner tests
```

## End-to-end data flow

```
 YAML manifest                  OSIRIS native deck
 (configs/osiris/*.yaml)        (any path you point at)
        │                                │
        ▼                                ▼
   run.py --cfg …            parse_deck_file()  ──┐
        │                                          │
        ▼                                          ▼
   ergoExo.setup ──► BaseOsiris.__init__ ──► merge_overrides ──► render_deck
                                  │                                   │
                                  ▼                                   ▼
                       cfg["deck"] = flat_dict          run_dir/os-stdin
                                  │                                   │
                                  ▼                                   ▼
                        log_params → MLflow         runner.run_osiris(mpirun…)
                                                                      │
                                                                      ▼
                                                          run_dir/MS/{FLD,PHA,…}
                                                                      │
                                                                      ▼
                                                       post.collect → td/ → MLflow
```

## How to run

```bash
conda activate adept
cd ~/Desktop/adept/adept
python run.py --cfg configs/osiris/twostream-1d-short    # smoke
python run.py --cfg configs/osiris/twostream-1d          # full
mlflow ui --backend-store-uri file://$(pwd)/mlruns       # browse
```

## Manifest schema

```yaml
solver: osiris                                    # required, dispatch key

mlflow:
  experiment: osiris-pic1d-twostream              # required
  run: cold-equal-beams                           # required

osiris:
  deck: /path/to/native/deck                      # required: source of truth
  binary: /path/to/osiris-1D.e                    # or OSIRIS_BIN / OSIRIS_BIN_<dim>D
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
  dist_cells: 150                                   # right-boundary cells averaged for
                                                    #   the phase-space f(p) lineouts
  omega_k_zoom: 4.0                                 # (k, ω) half-width [ω_p] for the
                                                    #   equal-aspect lower ω–k panel
                                                    #   (clamped to Nyquist); null = full
```

Override keys can use the **base name** (`nx_p`) or the **exact key** (`nx_p(1:1)`). Indexed `{0: …, 1: …}` form addresses occurrences of repeated sections (`species`, `udist`, `profile`, `spe_bound`, `diag_species`, `zpulse`, …) in source order.

## What lands in MLflow

| Kind        | Content                                                                          |
| ----------- | -------------------------------------------------------------------------------- |
| Params      | every deck key, flattened — `deck.grid.nx_p_1:1`, `deck.species_0.ufl_1:3.0`, …  |
| Params      | the manifest itself — `solver`, `osiris.binary`, `output.diagnostics_to_log`, …  |
| Metrics     | `wall_time_s`, `exit_code`, `field_energy_final`, `final_iter`, `run_time`, `postprocess_time` |
| Artifacts   | `config.yaml`, `derived_config.yaml`, `units.yaml` (adept stock)                 |
| Artifacts   | `os-stdin` (rendered OSIRIS deck), `stdout.log`, `stderr.log`                    |
| Artifacts   | `binary/<FLD\|PHA\|…>/<diag>.nc` — one xarray netCDF per diagnostic, holding the full `(t, …)` time history (replaces the raw h5 dumps) |
| Artifacts   | `plots/…` — canned PNGs (see below)                                              |

## Canned plots (`plots/` artifacts)

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
| `distribution_lineouts/<species>/<ps>.png`    | `f(p)` averaged over the rightmost `dist_cells` cells, overlaid at sampled times — stacked linear `f`, `\|f\|` log, and `δf = f - f_M` panels |
| `field_decomp/<comp>.png`                     | left/right-going transverse `E` (vacuum Riemann split `(e2±b3)/2`, `(e3∓b2)/2`), spacetime + `ω–k` |
| `energy_vs_time.png`, `energy_components_vs_time.png`, `total_energy_vs_time.png` | field / kinetic energy traces |
| `energy_budget.png` + `laser_energy_budget.txt` | reflected / transmitted / absorbed laser power vs time (stacked-area), with scalar `R`/`T`/absorbed in the `.txt` — emitted only when the drive `a0`/`omega0` are known |

> **Note on `field_decomp/`.** The left/right split is exact only in vacuum or a uniform non-dispersive medium (`|E| = |B|` for a pure travelling wave). In a plasma the EM wave is dispersive, so the split is approximate — useful for direction, but cross-check the dispersion before reading the residual as physical counter-propagating power. The longitudinal `e1` is electrostatic and is intentionally excluded.

> **Laser energy budget.** Reflected = left-going Poynting flux at the left boundary, transmitted = right-going flux at the right boundary (from the same Riemann split), compared to the incident intensity `I₀ = (a0·ω0)²/2` from the drive. Scalars are the boundary flux averaged over the last 25 % of the run (the saturated phase) ÷ `I₀`; `absorbed = 1 − R − T`. The antenna sits at the lower-`x` boundary, so its source cell is skipped. The same dispersive-medium caveat as `field_decomp/` applies — the split is approximate where the boundaries sit in plasma. `R`/`T`/absorbed are also logged to MLflow as `laser_reflectivity` / `laser_transmissivity` / `laser_absorbed_frac`.

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
cp my-new.deck ~/Desktop/pic/projects/whatever/
cp configs/osiris/twostream-1d.yaml configs/osiris/my-new.yaml
$EDITOR configs/osiris/my-new.yaml          # change deck path + mlflow.run
python run.py --cfg configs/osiris/my-new
```
