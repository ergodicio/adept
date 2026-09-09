# ADEPT Architecture

ADEPT is incrementally moving toward explicit, logging-free preparation and a pure
numerical transform boundary. [ADR 0001](adr/0001-explicit-simulation-boundaries.md)
defines the target contracts and compatibility policy. `tf-1d` and electrostatic
`pic-1d` are the first registered builder pilots. Their legacy classes remain available
as parallel fallback paths while downstream workflows migrate.

## Explicit preparation and execution

`SimulationSpec` adapts existing YAML without retaining its `mlflow` section. A
registered builder validates and resolves that intent into immutable preparation data:

```python
import equinox as eqx
import jax

from adept import SimulationSpec, solver_registry

spec = SimulationSpec.from_legacy_config(config)
prepared = solver_registry.prepare(spec, key=42)


def run(program, params, state, inputs, key):
    return program(params, state, inputs, key)


result = eqx.filter_jit(run)(
    prepared.program,
    prepared.params,
    prepared.state,
    prepared.inputs,
    jax.random.key(42),
)
```

Pure objectives and explicit parameter selection are documented in the
[explicit-program guide](source/usage/explicit_programs.md). `ObjectiveResult` keeps a
scalar loss, metrics, and auxiliary values inside a stable PyTree. The standard
`value_and_grad` helper differentiates only `params`; complementary fixed values remain
in `state` and `inputs`. `WeightedSumObjective` and `L2Penalty` provide basic
composition without adding tracking or host callbacks to the numerical graph.

The transformed boundary always receives five explicit PyTrees: `program`, `params`,
`state`, runtime `inputs`, and `key`. Pass those fields individually; the manifest,
analyzer, units, configuration models, tracking clients, and paths are host-side data
and do not enter JAX transforms. Use `partition_parameters` to select controls from a
runtime PyTree and keep the complementary leaves frozen, then call ADEPT's
`value_and_grad` helper. Fixed arrays belong in the program, state, or inputs—not in
the differentiation target. `vmap` is valid only when the builder advertises
`capabilities.batchable`.

Every numerical call returns the same `RawResult` named-tuple schema:

```text
RawResult(
    final_state,   # complete state after the solve/rollout
    observations,  # saved in-memory diagnostics; () when none are requested
    times,         # observation times; an empty array when there are none
    status,        # numerical completion status
    stats,         # numerical counters and auxiliary solver statistics
)
```

The `tf-1d` pilot uses the continuous `DiffraxProgram`; the electrostatic `pic-1d`
pilot uses the discrete `ScanProgram`. Both are parity-tested against their legacy
numerical maps. PIC transverse (`ey`) and stochastic forcing, and TF learned trapping
closures, still produce an actionable error directing callers to `ergoExo`.

`RunPlan` carries JSON-safe solver intent, seed, resources, run identity, and service
references across an executor boundary. The initial `LocalExecutor` validates declared
solver and executor capabilities before loading a builder, bootstraps x64 before JAX
import, and exposes explicit submission, status, cancellation, and result retrieval.
See the [run-plan guide](source/usage/run_plans.md) for the supported local adapters
and the incremental scope.

## Legacy lifecycle

- ADEPT solvers are packaged into `ADEPTModule`s and run via the `ergoExo`
  Code pointer: `adept/_base_.py`
- The `ergoExo` manages creation of an MLflow "run" and calls lifecycle methods on the `ADEPTModule` to perform logging of configuration, parameters, and run artifacts.
  Code pointer: `adept/_base_.py: ergoExo#_setup_()`
- The different `ADEPTModule`s are defined in subdirectories of `adept`, for example `_vlasov1d`, `_lpse2d`, etc.
- `ADEPTModule`s wrap a `diffrax` differential equation solver. The RHS of the ODE (often a discretized PDE) is typically found in a file named `vector_field.py`, in the class `VectorField`. For example, the Vlasov1D RHS is defined in `adept/_vlasov1d/solvers/vector_field.py`.

## Module Documentation

See the [full documentation](https://ergodicio.github.io/adept/) for detailed solver guides.

Quick links to configuration references:
- [Vlasov-1D Config](source/solvers/vlasov1d/config.md)
- [Vlasov-1D2V Config](source/solvers/vlasov1d2v/config.md)
- [VFP-1D Config](source/solvers/vfp1d/config.md)
- [Vlasov-2D Config](source/solvers/vlasov2d/config.md)
- [LPSE-2D Config](source/solvers/lpse2d/config.md)
- [Spectrax-1D Config](source/solvers/spectrax1d/config.md)
- [Hermite-Legendre-1D Config](source/solvers/hermite_legendre_1d/config.md)
- [PIC-1D Config](source/solvers/pic1d/config.md)
- [Two-Fluid-1D Config](source/solvers/tf1d/config.md)
- [OSIRIS Wrapper Config](source/solvers/osiris/config.md)
