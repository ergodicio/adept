# Usage

## Installation

ADEPT uses [uv](https://docs.astral.sh/uv/). Clone the repository and sync the environment:

```bash
git clone https://github.com/ergodicio/adept.git
cd adept
uv sync --extra dev
```

This creates `.venv/` and installs the exact versions pinned in `uv.lock`. Python 3.11+ is required. For an NVIDIA
GPU, use `uv sync --extra gpu` instead, which pulls `jax[cuda12]`.

To use ADEPT as a dependency of another project:

```bash
uv add "adept @ git+https://github.com/ergodicio/adept.git"
```

---

## Run an Example

The most common use case for ADEPT is a simple forward simulation that can be run from the command line. For example, to run a 1D1V Vlasov simulation of a driven electron plasma wave:

```bash
uv run run.py --cfg configs/vlasov-1d/epw
```

The input parameters are provided in `configs/vlasov-1d/epw.yaml`. Note that `--cfg` takes the path *without* the
`.yaml` extension.

### Access the Output

The output will be saved and made accessible via MLFlow. To access it:

1. Launch an mlflow server via running `uv run mlflow ui` from the command line
2. Open a web browser and navigate to http://localhost:5000
3. Click on the experiment name to see the results

---

## Solver-Specific Guides

Each solver has a single page covering the equations it solves, its boundary conditions and forcing,
what it writes out, and how to run it. They are listed under [Available Solvers](solvers.md).

Shared initialization options — density profiles, tanh flat-tops, super-Gaussians — are documented
once:

```{toctree}
:maxdepth: 2

usage/initialization
```

---

## Running on AWS Batch

```{toctree}
:maxdepth: 2

usage/cloud
```
