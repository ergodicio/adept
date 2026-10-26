# OSIRIS Configuration Reference

This document describes how to construct a configuration file for the `osiris` solver — and how
that YAML manifest relates to the native OSIRIS input deck it wraps.

Unlike the native adept solvers, the OSIRIS wrapper does not define the physics in YAML. The
**native OSIRIS deck is the source of truth**: the manifest points at a deck file, optionally
patches it with `overrides`, and configures how the run is executed, converted, and logged. See the
[usage overview](overview.md) for the end-to-end data flow.

## Top-Level Structure

```yaml
solver: osiris          # required, dispatch key

mlflow:
  # Experiment tracking (required)

osiris:
  # Deck location, binary, MPI, staging, overrides

output:
  # Post-processing / plotting options
```

## mlflow

| Field | Type | Description |
|-------|------|-------------|
| `experiment` | string | MLflow experiment name (required) |
| `run` | string | MLflow run name (required) |

## osiris

| Field | Type | Description |
|-------|------|-------------|
| `deck` | string | Path to the native OSIRIS deck, repo-relative or absolute (required). This file is parsed, optionally patched by `overrides`/`density`, re-rendered to `os-stdin`, and every key is logged to MLflow. |
| `binary` | string | Path to the built OSIRIS executable. Optional — see [Binary resolution](#binary-resolution). |
| `mpi_ranks` | int | `1` runs the binary directly; `>1` launches `mpirun -n N` (default `1`) |
| `extra_mpi_args` | list[string] | Extra arguments passed to `mpirun`, e.g. `["--oversubscribe"]` |
| `run_root` | string | Parent directory for per-run working dirs (default `./checkpoints`) |
| `stage_root` | string | Optional fast ephemeral filesystem (e.g. `/dev/shm/osiris`) to run OSIRIS on, draining dumps to `run_root` in the background. Requires `stream_convert: true`. See [Ramdisk staging](overview.md#ramdisk-staging-osirisstage_root). |
| `stream_convert` | bool | Convert `MS/` HDF5 dumps to `binary/*.nc` concurrently with the run (default `true`); `false` restores the batch conversion at job end. See [the overview](overview.md#concurrent-h5--netcdf-conversion-osirisstream_convert). |
| `stream_poll_s` | float | Watcher poll interval in seconds (default `10.0`) |
| `stream_workers` | int | Number of drain-worker subprocesses to shard diagnostics across (default `0` = drain in-thread; env fallback `ADEPT_STREAM_WORKERS`). Use when OSIRIS out-produces the single drain loop (small fast boxes at high dump cadence): each worker owns a disjoint set of diagnostics, so per-file conversion cost parallelizes while every NetCDF keeps one writer. Workers are light (no jax import); the converter falls back to in-thread draining if the pool cannot start. |
| `density` | mapping | Adaptive box sizing from a target gradient scale length (1D). See [below](#density-adaptive-box-sizing). |
| `overrides` | mapping | Deck patches applied before rendering. See [below](#overrides-patching-the-deck). |

> **Note on `run_root`:** the default sits inside `checkpoints/` deliberately — `sync-up.sh` rsyncs
> with `--delete` but excludes `checkpoints/`, so in-flight and finished OSIRIS outputs survive a
> sync. Pointing `run_root` outside an excluded directory exposes those outputs to deletion on the
> next sync. Nothing deletes run dirs automatically (post-processing only copies out of them), so
> clean them up manually on occasion.

### Binary resolution

The runner resolves the OSIRIS executable in this order:

1. `osiris.binary` in the manifest
2. `OSIRIS_BIN_<dim>D` environment variable (e.g. `OSIRIS_BIN_1D`), where the dimensionality is
   read from the deck
3. `OSIRIS_BIN` environment variable

The example manifests omit `osiris.binary`, so set the env var once per shell:

```bash
export OSIRIS_BIN_1D=/path/to/osiris-1D.e        # per-dim, preferred
# export OSIRIS_BIN=/path/to/osiris.e            # or a single default for all dims
```

### overrides: patching the deck

`osiris.overrides` merges values into the parsed deck sections **before** the deck is rendered to
`os-stdin` and logged to MLflow — the logged params always match what OSIRIS actually ran.

```yaml
osiris:
  overrides:
    time: {tmax: 50.0}                #   merge into the (one) time block
    grid: {nx_p: [256]}               #   refresh an array key
    species:                          #   indexed form for repeated sections:
      0: {num_par_x: [512]}           #     species 1: bump particles-per-cell
      1: {ufl: [-2.0, 0.0, 0.0]}      #     species 2: change drift
```

- Keys can use the **base name** (`nx_p`) or the **exact deck key** (`nx_p(1:1)`).
- Sections that repeat in a deck (`species`, `udist`, `profile`, `spe_bound`, `diag_species`,
  `zpulse`, …) are addressed with the indexed `{0: …, 1: …}` form, indexing occurrences in
  **source order** (0-based).
- Array values are given as YAML lists (`[256]`, `[-2.0, 0.0, 0.0]`).

### density: adaptive box sizing

`osiris.density` (1D decks only) scales the simulation box so the deck's linear density ramp
realizes a target gradient scale length $L_n$, mirroring how adept's `_lpse2d` and `kinetic_srs`
solvers size their grids. The density range is held fixed and the box length follows $L_n$.

| Field | Type | Description |
|-------|------|-------------|
| `gradient_scale_length` | string or float | Target $L_n$. A unit string (`300um`, converted via the deck's `simulation.n0`/`omega_p0`) or a bare number already in $c/\omega_{p0}$ units. Activates the feature. |
| `min` | float | $n_{min}$ in $n_c$ units (default: the ramp's interior `profile.fx` endpoint) |
| `max` | float | $n_{max}$ in $n_c$ units (default: from `profile.fx`) |
| `reference_density` | float | Density (in $n_c$ units) where $L_n$ is defined (default `0.25`, the quarter-critical surface) |

For a linear ramp $n(x): n_{min} \to n_{max}$ the local scale length is
$L(x) = n(x)/(dn/dx) = n(x) \cdot \Delta x_{ramp}/(n_{max}-n_{min})$, so requiring
$L(n_{ref}) = L_n$ fixes the ramp span: $\Delta x_{ramp} = (L_n/n_{ref})(n_{max}-n_{min})$.

Behavior:

- A single spatial scale factor $s$ is applied to **every** length in the deck: `space.xmin`/
  `space.xmax`, all `profile.x` arrays, and all `diag_species` phase-space windows
  (`ps_xmin`/`ps_xmax`). `grid.nx_p` is scaled by $s$ too — holding the cell size $dx$ fixed
  (rounded up to a multiple of `node_conf.node_number(1)` for even domain decomposition). Time
  (`dt`, `tmax`) is untouched, so the CFL ratio is preserved.
- Activates only when `gradient_scale_length` is present; decks otherwise run with their hand-set
  box, unchanged. Runs **after** `overrides`, so it supersedes any `space.xmax` override.
- If `min`/`max` are given, they are written into the primary `profile.fx`.
- The computed quantities (`box_norm`, `nx`, `scale_factor`, …) are logged under
  `osiris.density.derived.*`.
- Multi-dimensional decks raise `NotImplementedError`. Drive positions (e.g. a `zpulse` spatial
  center) are **not** rescaled — boundary antennas like the SRS deck's `antenna_array` have no
  position to scale.

## output

| Field | Type | Description |
|-------|------|-------------|
| `diagnostics_to_log` | list or null | `null` logs all diagnostics; or a list like `[e1, charge]` to restrict which `binary/*.nc` series are uploaded |
| `v_th` | float | Optional: overlays the Bohm–Gross Langmuir branch on $\omega$–$k$ plots |
| `omega_k_zoom` | float or null | $(k, \omega)$ half-width in $\omega_p$ units for the equal-aspect lower $\omega$–$k$ panel (clamped to Nyquist); `null` shows the full range |

## Complete example

```yaml
solver: osiris

mlflow:
  experiment: osiris-pic1d-twostream
  run: cold-equal-beams

osiris:
  deck: tests/test_osiris/decks/two-stream-1d
  mpi_ranks: 1
  overrides:
    time: {tmax: 50.0}

output:
  diagnostics_to_log: null
```

Run it from the repo root (`run.py` appends `.yaml` to `--cfg`, so omit the suffix):

```bash
uv run run.py --cfg configs/osiris/twostream-1d-short      # smoke
uv run run.py --cfg configs/osiris/twostream-1d            # full
```

## Adding a new test problem

Native-deck-as-truth: write the deck, point a manifest at it, run. No code changes.

```bash
cp my-new.deck tests/test_osiris/decks/
cp configs/osiris/twostream-1d.yaml configs/osiris/my-new.yaml
$EDITOR configs/osiris/my-new.yaml          # change deck path + mlflow.run
uv run run.py --cfg configs/osiris/my-new
```
