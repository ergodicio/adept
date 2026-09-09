# Run plans and local execution

`RunPlan` is the wire-format boundary between simulation intent and an execution
backend. It combines a `SimulationSpec`, deterministic seed, resource requirements,
tracked-run identity, and references to runtime services. Plans contain no JAX
arrays, live clients, callables, or credentials and round-trip through canonical
JSON:

```python
from adept import (
    AcceleratorKind,
    LocalExecutor,
    ResourceRequirements,
    RunPlan,
    RunRequest,
    ServiceReference,
    SimulationSpec,
)

plan = RunPlan(
    simulation=SimulationSpec.from_legacy_config(config),
    seed=42,
    resources=ResourceRequirements(accelerator=AcceleratorKind.CPU),
    run=RunRequest(experiment="local", run_id="example-42"),
    artifact_sink=ServiceReference("directory", {"root": "./adept-results"}),
)

payload = plan.to_json()
restored = RunPlan.from_json(payload)

with LocalExecutor() as executor:
    completed = executor.execute(restored)
```

`LocalExecutor.validate` checks the solver's import-light registry declaration,
topology, precision, requested features, and tracker/artifact adapters before it
loads the solver builder. Submission passes a serialized copy to the worker, which
performs JAX precision bootstrap before importing a JAX-dependent builder. The
prepared solver's actual capabilities are checked again before numerical execution.

The local adapter supports null or MLflow tracking and null, directory, or MLflow
artifact sinks. MLflow modules are imported only when an MLflow reference is
selected. A directory sink implies the `artifact-access` requirement and is
preflighted by the host runtime before the solve.

## Capability requirements

`ResourceRequirements` describes placement, precision, accelerator kind, host count,
devices per host, and optional features. Multi-host placement automatically requires
`distributed-jax`. Other typed features cover shared durable storage, rank-zero I/O,
checkpointing, differentiability, batching, and artifact access. Unsupported
requirements fail together in one actionable preflight error.

Built-in solver capabilities are registered without importing their builders. This
lets an executor discover that TF1D and PIC1D require x64 and configure a fresh JAX
worker before solver import. If JAX was already initialized with x64 disabled, local
execution fails instead of silently running at the wrong precision.

Service configuration may contain locations such as directory roots and tracking
URIs. Credential-like fields are rejected. Configure credentials in the worker
environment, an AWS profile, or the backend's normal credential provider instead of
embedding them in a plan.

## Submission lifecycle

`submit` returns an `ExecutionHandle`. `status`, `cancel`, and `result` operate on
that explicit handle; `execute` is the blocking submit-and-result convenience. A
local run can be cancelled while queued, but an already-running JAX call is not
interruptible.

Checkpoint store references and policies are serialized in schema version 2. Active
policies request the `checkpointing` capability; multi-host checkpointing also
requests shared durable storage and rank-zero I/O. See [Versioned checkpoints](checkpoints.md)
for the initial store contract and current executor-integration boundary.

This is the first executor slice. Parsl submission, collective multi-host JAX,
external-process execution, retry/chunk policy, and managed checkpoint scheduling
will use the same serialized plan and executor lifecycle in follow-up work.
