# ADEPT

The ADEPT repo is a collection of solvers for differentiable plasma physics.

- @docs/ARCHITECTURE.md
- @docs/RUNNING_A_SIM.md

## Pint units (jpu)

`UREG.foo` (e.g. `UREG.c`, `UREG.m_e`) returns a **Unit**, not a Quantity — you cannot call `.to()` on a Unit. A Quantity multiplied or divided by a Unit produces a Quantity (e.g. `351.0 * UREG.um` is a dimensional Quantity). Convention: only use `.to()` when reading string inputs or storing final outputs, not in intermediate calculations. Extract `.magnitude` only from dimensionless Quantities.

## Documentation

Docs are filed under `docs/`.

When adding a new configuration option, you MUST remember to update the corresponding reference documentation.
These are organized by solver and live under `docs/source/solvers/<solver>/config.md`.
