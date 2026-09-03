# Explicit programs and objectives

ADEPT is introducing a logging-free numerical API alongside `ergoExo`. It is opt-in:
existing solver entry points and output dictionaries are unchanged. The currently
registered pilots are `tf-1d` and electrostatic `pic-1d`.

The numerical boundary has five explicit values:

```text
program(params, state, inputs, key) -> RawResult
```

- `params` contains only the leaves selected for differentiation.
- `state` is the evolving initial state.
- `inputs` contains fixed runtime forcing, scenarios, and targets.
- `key` is the PRNG key for this run.
- `RawResult` contains the final state, observations, times, status, and solver statistics.

Keeping fixed arrays in `state` or `inputs` prevents them from becoming accidental
gradient targets. None of these calls start MLflow, write files, or mutate their
arguments.

## Forward-only execution

Prepare from an existing configuration, then pass the numerical fields individually
through the transform:

```python
import equinox as eqx
import jax

from adept import SimulationSpec, solver_registry

prepared = solver_registry.prepare(
    SimulationSpec.from_legacy_config(config),
    key=42,
)


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

The manifest, analyzer, configuration model, paths, and tracking clients remain on the
host and must not be passed to `jit`, `grad`, or `vmap`.

## Differentiated execution

An `Objective` returns a scalar loss plus stable metric and auxiliary PyTrees. A scalar
callable can be adapted with `CallableObjective`:

```python
import jax.numpy as jnp

from adept import CallableObjective, partition_parameters, value_and_grad

runtime_values = eqx.combine(prepared.params, prepared.inputs)
selector = jax.tree.map(lambda _: False, runtime_values)
selector = eqx.tree_at(
    lambda tree: tree["drivers"]["ex"]["0"]["a0"],
    selector,
    True,
)
partition = partition_parameters(runtime_values, selector)
params = partition.trainable
inputs = partition.frozen

objective = CallableObjective(
    lambda result, params, inputs: jnp.mean(
        result.observations["x"]["electron"]["u"][-1] ** 2
    ),
    metric_name="electron_flow_energy",
)

run = eqx.filter_jit(value_and_grad)(
    prepared.program,
    objective,
    params,
    prepared.state,
    inputs,
    jax.random.key(42),
)

loss = run.objective.loss
metrics = run.objective.metrics
gradients = run.gradients
raw_result = run.simulation
```

`value_and_grad` differentiates only its `params` argument. Solver-specific `vg()`
methods are not needed on this path.

## Selecting parameters and freezing other values

The pilot builders keep driver values in `inputs` by default. Select trainable leaves
with a boolean PyTree of the same structure. The complementary output keeps every
unselected value fixed:

```python
from adept import partition_parameters

runtime_values = eqx.combine(prepared.params, prepared.inputs)
selector = jax.tree.map(lambda _: False, runtime_values)
selector = eqx.tree_at(
    lambda tree: tree["drivers"]["ex"]["0"]["a0"],
    selector,
    True,
)
partition = partition_parameters(runtime_values, selector)

params = partition.trainable
inputs = partition.frozen
```

Here `a0` is the only differentiable leaf. Other arrays—including `w0`, the driver
envelope, initial state, and any target data—stay explicit but frozen. Replacing a
selected or frozen value with another value of the same shape and dtype reuses the
compiled executable.

Objectives compose without putting logging in the JAX graph:

```python
from adept import L2Penalty, WeightedSumObjective

objective = WeightedSumObjective(
    (data_objective, L2Penalty()),
    weights=(1.0, 1e-4),
    names=("fit", "regularization"),
)
```

## Batched execution

For a builder that advertises `prepared.capabilities.batchable`, batch the explicit
runtime values rather than closing over scenarios:

```python
if not prepared.capabilities.batchable:
    raise ValueError("This solver has not declared batched execution support")

batched_run = eqx.filter_jit(
    eqx.filter_vmap(run, in_axes=(None, 0, 0, 0, 0))
)
results = batched_run(prepared.program, params_batch, state_batch, inputs_batch, keys)
```

The two initial solver builders do not yet advertise general batching. The generic
program contract is tested under `vmap`; each solver must validate its own state and
observation layout before enabling the capability.

## Migrating `trainable_modules` and `vg()` callers

`LegacyVGAdapter` temporarily preserves the numerical call shape expected by older
optimization loops:

```python
from adept import LegacyVGAdapter

adapter = LegacyVGAdapter(
    prepared.program,
    objective,
    prepared.state,
    inputs,
    jax.random.key(42),
)
(loss, output), gradients = eqx.filter_jit(adapter.vg)(params)
```

The adapter emits a migration warning and returns `output["solver result"]` alongside
the structured objective. It does not create an MLflow run or emulate mutation of
captured module attributes. New code should call `value_and_grad` directly. A later
compatibility façade will route supported `ergoExo` behavior through these contracts;
until then, existing `ergoExo` and `ADEPTModule` callers continue on the legacy path.
