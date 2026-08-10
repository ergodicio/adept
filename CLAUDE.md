# ADEPT

The ADEPT repo is a collection of solvers for differentiable plasma physics.

- @docs/ARCHITECTURE.md
- @docs/RUNNING_A_SIM.md

## Branch names

Name the branch after the solver being added or modified: `<type>/<solver>-<short-description>`.
Existing examples: `feat/vlasov1d-dougherty-nodrag`, `fix/vlasov1d-v0-units`, `feat/empic1d-relativistic-em-pic`.
Commit subjects carry the same scope: `feat(vlasov1d): ...`.
For repo-wide work with no solver (CI, docs, packaging), use the area instead: `ci/...`, `docs/...`, `chore/...`.

## Everything is built on ergoExo

Solvers are never invoked directly. `ergoExo` (`adept/_base_.py`) owns the run: it creates the MLflow run and
calls lifecycle methods on an `ADEPTModule` subclass in a fixed order. When adding or modifying a solver, work
inside that contract rather than around it.

Order — `ergoExo#_setup_()`, then `ergoExo#__call__()`:

1. `write_units()` — physical and normalizing quantities; dumped to `units.yaml`
2. `get_derived_quantities()` — cfg-derived **scalars**; mutates `self.cfg`, which is then logged to MLflow
3. `get_solver_quantities()` — grids and other **arrays**; mutates `self.cfg`, NOT logged
4. `init_state_and_args()` — sets `self.state` and `self.args` (initial conditions, drivers)
5. `init_diffeqsolve()` — sets `self.diffeqsolve_quants` / `self.time_quantities`; consumes `get_save_func()`
6. `init_modules()` — returns the dict of trainable `eqx.Module`s
7. `__call__(trainable_modules, args)` — runs `diffrax.diffeqsolve`
8. `post_process(run_output, td)` — datasets, plots, metrics

Consequences that bite:
- Anything a later step reads must be built by an earlier one. A new term needing a coefficient array cannot
  first appear in `init_state_and_args()` if nothing put it in `cfg` during `get_solver_quantities()`.
- The scalar/array split between steps 2 and 3 is load-bearing — arrays leaking into step 2 break param logging.
- Gradients go through `vg()`; the base class raises unless the module implements a metric.
- `ergoExo#_get_adept_module_()` is the registry: a hand-written branch mapping the `solver:` string to a class,
  with no autodiscovery. One module directory can register several keys (`vlasov-1d` → `BaseVlasov1D`,
  `vlasov-1d-iaw` → `IAWTurbulence1D`), so add a key per runnable module, not per directory.

`docs/source/dev_guide.md` has the step-by-step for adding a solver — follow it rather than reconstructing the
list. The step easiest to skip is the paths filter and job in `.github/workflows/cpu-tests.yaml`: CI only runs
the suites for solvers it detects as changed, so a new `tests/test_<solver>/` without them silently never runs.

## Pint units (jpu)

`UREG.foo` (e.g. `UREG.c`, `UREG.m_e`) returns a **Unit**, not a Quantity — you cannot call `.to()` on a Unit. A Quantity multiplied or divided by a Unit produces a Quantity (e.g. `351.0 * UREG.um` is a dimensional Quantity). Convention: only use `.to()` when reading string inputs or storing final outputs, not in intermediate calculations. Extract `.magnitude` only from dimensionless Quantities.

## Documentation

Docs are filed under `docs/`.

When adding a new configuration option, you MUST remember to update the corresponding reference documentation.
These are organized by solver and live under `docs/source/solvers/<solver>/config.md`.
