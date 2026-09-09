# Legacy API compatibility

`ergoExo` and `ADEPTModule` remain supported and are not deprecated in this release.
The compatibility façade preserves their setup, return values, MLflow lifecycle,
configuration artifacts, and solver-specific post-processing while solver internals
move onto the explicit architecture.

## Current routing

Supported TF1D and electrostatic PIC1D forward runs are prepared through
`SimulationSpec` and the solver registry, then executed by `run_prepared`. The façade
converts the structured result back into the historical
`{"solver result": diffrax.Solution}` shape before invoking the existing post-processor.
This routing does not change the three-value return from `ergoExo.__call__`.

The façade opts into prepared execution only when preparation reproduces the legacy
initial state exactly. It uses the legacy path when any compatibility-sensitive input
is present, including:

- a custom `ADEPTModule`;
- explicit `args` or non-empty trainable modules;
- replacement of the module state or stored arguments after setup;
- TF1D learned trapping closures or unsupported save layouts;
- PIC1D transverse or stochastic drivers, off-grid saves, or an initialization that
  differs from the legacy seeded state;
- a solver without an explicit façade adapter, including Vlasov1D and LPSE2D today; or
- `ergoExo.val_and_grad`, which continues to call `ADEPTModule.vg`.

No failed prepared solve is silently rerun. Fallback decisions happen before numerical
execution. After setup or a run, `exo.execution_backend` is either `"prepared"` or
`"legacy"`; `exo.compatibility_fallback_reason` explains a legacy selection.

## New integrations

New code should use `SimulationSpec`, `solver_registry.prepare`, and `run_prepared`
directly. Existing applications can keep using `ergoExo` while solvers migrate. The
next compatibility slices will add builders for the primary Vlasov1D and LPSE2D paths
before removal or deprecation of the legacy API is considered.
