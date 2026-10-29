# ADEPT

[![Docs](https://github.com/ergodicio/adept/actions/workflows/docs.yaml/badge.svg)](https://ergodicio.github.io/adept/)
[![Tests](https://github.com/ergodicio/adept/actions/workflows/cpu-tests.yaml/badge.svg)](https://github.com/ergodicio/adept/actions/workflows/cpu-tests.yaml)

![ADEPT](./docs/source/adept-logo.png)

ADEPT is an **A**utomatic **D**ifferentiation **E**nabled **P**lasma **T**ransport code.

The solvers are written in [JAX](https://github.com/jax-ml/jax) and time-integrated with
[diffrax](https://github.com/patrick-kidger/diffrax), so every simulation is differentiable end-to-end and runs on
CPU or GPU. Runs are logged to [MLflow](https://mlflow.org/).

## Solvers

| `solver:` key | Description | Docs |
| --- | --- | --- |
| `vlasov-1d` | 1D1V Vlasov-Poisson/Maxwell with Fokker-Planck collisions (`vlasov-1d-iaw` adds kinetic ions with Boltzmann electrons) | [overview](docs/source/solvers/vlasov1d/overview.md) · [config](docs/source/solvers/vlasov1d/config.md) |
| `vlasov-1d2v` | 1D2V in cylindrical velocity space, with a full-geometry Coulomb collision operator | [overview](docs/source/solvers/vlasov1d2v/overview.md) · [config](docs/source/solvers/vlasov1d2v/config.md) |
| `vlasov-2d` | 2D2V Vlasov-Maxwell | [overview](docs/source/solvers/vlasov2d/overview.md) · [config](docs/source/solvers/vlasov2d/config.md) |
| `vfp-1d` | Vlasov-Fokker-Planck electron transport (spherical harmonic expansion) | [overview](docs/source/solvers/vfp1d/overview.md) · [config](docs/source/solvers/vfp1d/config.md) |
| `envelope-2d` | 2D laser-plasma envelope solver (TPD, SRS) | [overview](docs/source/solvers/lpse2d/overview.md) · [config](docs/source/solvers/lpse2d/config.md) |
| `spectrax-1d` | 1D Hermite-Fourier Vlasov-Maxwell (`hermite-epw-1d` and `hermite-maxwell-1d` are reduced entry points) | [config](docs/source/solvers/spectrax1d/config.md) |
| `hermite-legendre-1d` | 1D1V mixed Hermite-Legendre electrostatic Vlasov-Poisson | [config](docs/source/solvers/hermite_legendre_1d/config.md) |
| `pic-1d` | 1D electrostatic particle-in-cell | — |
| `tf-1d` | 1D warm two-fluid with kinetic closures | [overview](docs/source/solvers/tf1d/overview.md) · [config](docs/source/solvers/tf1d/config.md) |

Example configs for each live in [`configs/`](configs).

## Installation

ADEPT uses [uv](https://docs.astral.sh/uv/). Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### As a dependency

```bash
uv add "adept @ git+https://github.com/ergodicio/adept.git"
```

or, into an existing environment:

```bash
uv pip install "git+https://github.com/ergodicio/adept.git"
```

### For development

```bash
git clone https://github.com/ergodicio/adept.git
cd adept
uv sync --extra dev
uv run pre-commit install
```

`uv sync` creates `.venv/` and installs from `uv.lock`, so you get the exact versions CI uses. Python 3.11+ is
required. Prefix commands with `uv run` and you never need to activate the environment.

For an NVIDIA GPU, sync the `gpu` extra instead, which pulls `jax[cuda12]`:

```bash
uv sync --extra gpu
```

## Usage

```bash
uv run run.py --cfg configs/vlasov-1d/epw
```

The `--cfg` path is given **without** the `.yaml` extension. This runs the simulation defined in the config and
stores the output to an `mlflow` server.

Unless you have separately deployed an `mlflow` server somewhere (or set `MLFLOW_TRACKING_URI`), it simply writes
files using the mlflow specification to the current working directory. To access and visualize the results, it is
easiest to use the UI from the browser by running `uv run mlflow ui` from the same directory and opening
http://localhost:5000.

There are other ways to use ADEPT, notably as part of a neural network training pipeline that leverages
differentiable simulation. In reference [1], neural networks are trained to learn forcing functions that drive the
system towards previously unseen behavior. In reference [2], neural networks are trained to help bridge the
micro-macro physics gap in multiphysics simulations.

## Tests

```bash
uv run pytest
```

Long-running parameter sweeps are marked `slow` and skipped by default; run them with `uv run pytest -m slow`.

## Docs

https://ergodicio.github.io/adept/

## Examples

https://github.com/ergodicio/adept-notebooks

## Contributing guide

The contributing guide is in development but for now, just make an issue / pull request and we can go from there :)

Linting and formatting are handled by `ruff` via pre-commit. If you have run `uv run pre-commit install`, it runs on
every commit; otherwise, run it manually before pushing:

```bash
uv run pre-commit run --all-files
```

If you change dependencies in `pyproject.toml`, refresh the lockfile with `uv lock` and commit it — CI installs with
`--frozen`.

## Citation

If you are using this package for your research, please cite the following

```
A. Joglekar and A. Thomas, “ADEPT - automatic differentiation enabled plasma transport,”
ICML - SynS & ML Workshop (https://syns-ml.github.io/2023/contributions/), 2023

```

## References

[1] A. S. Joglekar & A. G. R. Thomas. "Unsupervised discovery of nonlinear plasma physics using differentiable kinetic simulations." J. Plasma Phys. 88, 905880608 (2022).

[2] A. S. Joglekar and A. G. R. Thomas, “Machine learning of hidden variables in multiscale fluid simulation,” Mach. Learn.: Sci. Technol., vol. 4, no. 3, p. 035049, Sep. 2023, doi: 10.1088/2632-2153/acf81a.

## Execution monitoring

`ergoExo.setup`, forward execution, and `val_and_grad` publish best-effort lifecycle
telemetry to the configured MLflow server, in the dedicated `continuum-run-monitor`
experiment (override with `ADEPT_MONITOR_EXPERIMENT`). Each setup plus first invocation has a
separate tracking run tagged `monitor.record_type=execution` and
`monitor.schema_version=1`; repeated invocations create distinct execution records.
`monitor.scientific_run_id` links to the scientific run, whose existing ID and artifacts
remain the public return value. Monitoring never changes the active MLflow run.

Set `ADEPT_MONITOR_LAUNCH_ID` to a launch registry UUID to associate scheduler
submission and execution. Otherwise ADEPT generates an ID and a closed singleton launch registry record;
each subsequent invocation gets its own implicit launch. `monitor.execution_id`
is unique per attempt. On Slurm, records include job, step, and cluster from the environment.
`ADEPT_MONITOR_OWNER=auto` (the default) suppresses nonzero MPI/PMI/`RANK`
workers but treats Slurm tasks as independent simulations. Use `rank0` for a
single distributed simulation launched directly with `srun` (also checks
`SLURM_PROCID`), `all` for independent Parsl workers even if rank variables are
inherited, or `none` to disable reporting in a process. Choose explicitly for
ambiguous task layouts to avoid duplicate or missing records.

`monitor.phase` moves through setup, executing, postprocessing, uploading, and a
terminal succeeded or failed state. `monitor.outcome` remains running until
execution, postprocessing, and configured artifact uploads finish. Exhausted
artifact upload retries mark the execution failed while preserving the existing
simulation return behavior. `export=False` requires no result artifact upload.
JAX work is synchronized before advancing beyond execution. Setup alone does not
mean success, even if the scientific MLflow run temporarily reports FINISHED.

A single process-wide daemon publishes telemetry through a bounded queue and
reports `monitor.heartbeat_at` every 30 seconds (UTC Unix seconds).
`monitor.progress_at` records actual phase transitions, not numerical timestep
progress; a heartbeat only proves that the reporting process is alive. Terminal
records include `monitor.completed_at`. Tracking errors are best effort and do
not fail valid simulation work. All monitoring network calls run in the daemon;
completion flush waits at most 0.1 seconds. Normal interpreter exit drains queued
reports for at most five seconds; terminal tags use one MLflow batch request.
A forked child resets the inherited queue and starts its own publisher. A blocked tracking request can delay
other telemetry, and queue overflow drops telemetry rather than simulation work.
After setup the phase returns to setup while awaiting execution. Process termination, SIGKILL, tracking outages,
and setup-only processes can leave nonterminal records: an external watchdog must
reconcile stale heartbeats with scheduler state. Telemetry does not install signal
handlers or infer numerical progress inside compiled kernels.
