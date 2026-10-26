# Developer Guide

This page is for when you want to do something other than a plain forward simulation:

```bash
uv run run.py --cfg configs/<solver>/<config>
```

That command is a thin wrapper. It loads the YAML, constructs an `ergoExo`, and calls it. If you
want to train a model through a simulation, run a parameter scan in-process, or embed a solver in a
larger program, you work with `ergoExo` and `ADEPTModule` directly.

## The two classes

ADEPT separates the numerical solvers from experiment management:

1. **`ADEPTModule`** is the base class for a solver. It owns the physics — units, grids, initial
   state, the `diffrax` terms, and post-processing.
2. **`ergoExo`** is the harness. It creates the MLflow run, calls the module's lifecycle methods in
   the right order, and logs configuration, parameters, and artifacts.

This decoupling is what makes adding a solver cheap: implement the lifecycle methods and the harness
handles everything else.

## Typical usage

```python
from adept import ergoExo

exo = ergoExo()
modules = exo.setup(cfg)
sol, post_out, run_id = exo(modules)
```

To resume an existing MLflow run:

```python
exo = ergoExo(mlflow_run_id=run_id)
modules = exo.setup(cfg)
sol, post_out, run_id = exo(modules)
```

To supply your own solver rather than dispatching on `cfg["solver"]`:

```python
exo = ergoExo()
modules = exo.setup(cfg, adept_module=MyCustomModule)
sol, post_out, run_id = exo(modules)
```

## The setup sequence

`exo.setup(cfg)` resolves `cfg["solver"]` to an `ADEPTModule` subclass and then calls its lifecycle
methods in this order, logging an artifact after most of them:

| Call | What it does | Artifact logged |
|---|---|---|
| — | dump the raw config | `config.yaml` |
| `write_units()` | build the normalization constants and physical quantities | `units.yaml` |
| `get_derived_quantities()` | compute scalar quantities derived from the config | `derived_config.yaml`, plus MLflow params |
| `get_solver_quantities()` | compute array-valued quantities (grids, initial distributions) | `array_config.pkl` |
| `init_state_and_args()` | build the initial state and the solver arguments (usually drivers) | — |
| `init_diffeqsolve()` | assemble the `diffrax` terms, solver, and `SaveAt` | — |
| `init_modules()` | construct any trainable `eqx.Module`s | — |

The split between `get_derived_quantities` and `get_solver_quantities` exists because the former
produces scalars that are worth logging to MLflow as parameters, while the latter produces arrays
that are not.

`init_modules()` returns the dict of trainable modules, which is what `setup` hands back to you.
Calling `exo(modules)` then runs the solve, calls `post_process()`, and logs the results.

## Taking gradients

Because the whole solve is JAX, you can differentiate through it. `ergoExo.val_and_grad(modules)`
returns the value and the gradient with respect to the parameters of the trainable modules:

```python
exo = ergoExo()
modules = exo.setup(cfg)
val, grad, (sol, post_out, run_id) = exo.val_and_grad(modules)
```

The value and the L2 norm of the gradient are logged to MLflow automatically.

This requires the module to implement `vg()` — the base class raises `NotImplementedError`, usually
because there is no metric defined to differentiate. Subclass and implement it to get gradients.
This is the mechanism behind the machine-learned-closure work in the
[Two-Fluid 1D](solvers/tf1d/overview.md#machine-learned-closures) solver.

## Adding a solver

1. Create `adept/_mysolver/` with `modules.py` (the `ADEPTModule` subclass), `datamodel.py` (a
   pydantic config model), and `solvers/vector_field.py` (the ODE right-hand side).
2. Re-export the module class from `adept/mysolver.py`.
3. Add a branch to `_get_adept_module_` in `adept/_base_.py` keyed on your `solver:` string.
4. Add example configs under `configs/my-solver/` and tests under `tests/test_mysolver/`.
5. Add a paths filter and a test job for the solver in `.github/workflows/cpu-tests.yaml`. CI only runs the
   suites for solvers it detects as changed, so without both the new tests never run.
6. Add `docs/source/solvers/mysolver/overview.md` and `config.md`, and list them in
   `docs/source/solvers.md` and the toctrees in `docs/source/index.rst`.

## Libraries

Two open-source libraries carry most of the weight:

1. **MLflow** as the experiment manager, for logging parameters, metrics, and artifacts
2. **Diffrax** for the time integration
