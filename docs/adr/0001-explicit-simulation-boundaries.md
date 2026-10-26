# ADR 0001: Explicit simulation and runtime boundaries

- Status: Accepted
- Date: 2026-09-02
- Issues: [#349](https://github.com/ergodicio/adept/issues/349), [#350](https://github.com/ergodicio/adept/issues/350), [#352](https://github.com/ergodicio/adept/issues/352), [#353](https://github.com/ergodicio/adept/issues/353), [#357](https://github.com/ergodicio/adept/issues/357)

## Context

`ergoExo` currently owns solver selection, mutable preparation, JIT compilation,
MLflow lifecycle, artifact export, and result analysis. Each solver implements those
steps by subclassing `ADEPTModule`. This public API remains widely used, but the
bound-method JIT boundary can capture configuration and state invisibly, and merely
importing ADEPT loads numerical and tracking dependencies.

ADEPT needs a logging-free preparation API and an explicit numerical boundary without
requiring all existing solvers and downstream projects to migrate at once.

## Decision

ADEPT will use the following vocabulary and ownership boundaries:

- `SimulationSpec` is versioned user intent. It contains a stable solver name and
  solver configuration, but no tracking configuration, live clients, credentials, or
  runtime resources.
- `SolverRegistry` maps stable names to logging-free `SolverBuilder`s. Registrations
  may be lazy so inspecting specifications does not import numerical implementations.
- `SolverBuilder.prepare(spec, *, key)` deterministically returns a
  `PreparedSimulation` and performs no tracking, filesystem writes, or process-global
  configuration.
- `PreparedSimulation` explicitly owns a program, initial parameters, state, inputs,
  manifest, analyzer, and solver capabilities. Changes produce a replaced value rather
  than mutating closure-captured fields.
- `JaxProgram(params, state, inputs, key)` is the transformed numerical boundary. A
  stable `RawResult` separates final state, observations, times, status, and stats.
  `ContinuousSystem.rhs` exposes a true derivative, while `DiscreteSystem.step`
  exposes a complete next-state map; Diffrax and scan adapters execute each kind.
- `Objective(result, params, inputs)` is a pure, composable transform-side loss. It
  returns an `ObjectiveResult` containing a scalar loss plus stable metrics and
  auxiliary PyTrees. ADEPT's standard value-and-gradient helper differentiates only
  explicitly selected `params`; complementary runtime values remain fixed `inputs`.
- A host-side executor turns a `RawResult` into a materialized `Result`. An `Analyzer`
  turns that result into a `Report` of metrics and artifact descriptions. Tracking,
  artifact upload, checkpointing, plotting, scheduling, and external processes remain
  host-side services.

The dependency direction is:

```text
serializable spec/run plan
          |
          v
registry -> builder -> prepared simulation -> JAX program -> raw result
                           |                                |
                           +-> manifest/capabilities        v
                                                executor -> result
                                                             |
                                                             v
                                               analyzer -> report
                                                             |
                                                             v
                                              tracker / artifact sink
```

Specification and registry modules use only the Python standard library. Importing
them must not import JAX, Equinox, MLflow, solver modules, or initialize a runtime.
Concrete builders and executors own those imports at their execution boundary.

## Compatibility

This is a strangler migration, not a flag-day replacement:

1. The new contracts and registry land alongside the existing lifecycle.
2. Representative continuous and discrete solvers gain new builders and numerical
   entry points with legacy/new parity tests.
3. Host-side tracker and analyzer services reproduce existing run and artifact
   behavior.
4. `ergoExo` and `ADEPTModule` become compatibility façades over migrated paths.
5. Unmigrated and caller-supplied legacy modules retain an explicit legacy fallback.
6. Deprecation begins only after downstream pilot migrations and a published removal
   policy.

During the transition, the façade preserves documented setup, call, gradient, output,
configuration-snapshot, nested/resumed run, and custom-module behavior. Arbitrary
mutation inside transformed execution will not be emulated on the new path; callers
will receive an actionable migration error or use the legacy fallback.

## Error policy

- Invalid specifications fail before builder loading.
- Unknown solver names report the available registry keys.
- Registry collisions fail instead of silently replacing a builder.
- Capability and resource mismatches fail during executor preflight, before a solve.
- Tracking and artifact failures are host-side and follow an explicit strict or
  best-effort policy; they never alter numerical results silently.

## Consequences

The change is additive and does not reroute legacy simulations. It introduces some
temporary duplication while solvers migrate, but gives every migration a parity-tested
rollback path. `tf-1d` and electrostatic `pic-1d` are the first builder pilots; their
legacy modules remain intact. Keeping the contract layer dependency-free also enables
future external and scheduler submitters to inspect plans without importing JAX or
MLflow.
