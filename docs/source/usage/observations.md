# Observation planning and result materialization

The numerical program retains only observations declared before execution. An
`ObservationPlan` names each observation, gives it a finite time or step schedule,
declares its output schema and placement, and sets a retention policy. The plan
reports its expected retained bytes and rejects a run that exceeds its configured
budget before the solver starts.

The TF1D and PIC1D builders adapt their existing `save` configuration into this
contract. TF1D's continuous adapter translates the plan into private Diffrax
`SaveAt` values. PIC1D records the same kind of plan directly in its discrete scan;
it does not use Diffrax for observation handling.

## Defining a bounded plan

Observation functions are pure JAX callables with the signature
`function(time, state, inputs)`. Schema inference uses JAX abstract evaluation, so
it does not execute a solve or materialize an array:

```python
import equinox as eqx
import jax.numpy as jnp

from adept import ObservationPlan, ObservationReduction, ObservationSchedule
from adept.core.observations_jax import infer_observation_spec


class MeanDensity(eqx.Module):
    def __call__(self, time, state, inputs):
        del time, inputs
        return state["density"]


schedule = ObservationSchedule.every_steps(10, start=0, stop=1000)
spec = infer_observation_spec(
    "mean-density",
    MeanDensity(),
    schedule,
    t=0.0,
    state=initial_state,
    inputs=runtime_inputs,
    reduction=ObservationReduction.MEAN,
)
plan = ObservationPlan((spec,), max_retained_bytes=64 * 1024**2)

print(plan.to_dict())
print(plan.estimated_retained_bytes)
```

Schedules must be finite and strictly increasing. A discrete program also rejects
times that do not lie on its step grid. `ObservationRetention.LAST` keeps only the
last requested sample. `ObservationPlacement` and `ObservationCollective` declare
whether outputs remain sharded or replicated and what communication the observation
performs; executor capability negotiation will enforce those declarations as that
layer is introduced.

## Named device results

Programs return observations and their coordinates under matching names:

```python
device_fields = result.observations["fields"]
device_times = result.times["fields"]
```

These values remain JAX arrays and may remain sharded. They can be used by an
objective inside `jit` or `grad` without crossing the host boundary.

## Explicit host materialization

Materialization is never hidden in a postprocessor. Transfer a `RawResult`
explicitly to every host or to rank zero:

```python
from adept import MaterializationTarget

host_result = result.materialize(MaterializationTarget.ALL_HOSTS)
rank_zero_result = result.materialize(MaterializationTarget.RANK_ZERO)
```

For non-fully-addressable global JAX arrays, every process must enter the call so the
required collective can complete. The rank-zero form returns `None` on other ranks
after that collective. Returned numerical leaves are NumPy arrays.

`run_prepared` performs this explicit transfer before calling the host-side analyzer
when the prepared simulation has an observation plan. It preserves the original
device `raw_result` and also returns the host tree as `materialized_result`.

Durable streaming and multi-host rank policy belong to the later executor and
checkpoint integrations; they are not inferred from an observation function or a
Diffrax save buffer.
