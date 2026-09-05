# Versioned checkpoints

A checkpoint is durable solver state. It is separate from observation buffers,
diagnostic reports, and MLflow artifacts: save the carry needed to continue the
numerical program directly.

`CheckpointStore` defines `save`, `restore`, `validate`, `list`, and `latest`
operations. `NullCheckpointStore` explicitly disables persistence.
`LocalCheckpointStore` writes numeric PyTree leaves to a local directory without
pickle:

```python
import jax
import jax.numpy as jnp

from adept import CheckpointCompatibility, CheckpointMetadata, LocalCheckpointStore

store = LocalCheckpointStore("./checkpoints")
store.preflight()

metadata = CheckpointMetadata.from_manifest(
    prepared.manifest,
    checkpoint_id="step-1000",
    program=spec.solver,
    program_version=spec.schema_version,
    simulation_time=1.0,
    step=1000,
    chunk_id="chunk-10",
)
pending = store.save(carry, metadata)
saved = pending.wait()  # Required even when the selected store completes synchronously.
assert saved is not None

target = jax.tree.map(jnp.zeros_like, carry)
restored = store.restore(
    saved,
    target,
    compatibility=CheckpointCompatibility.from_metadata(saved.metadata),
)
```

The restore target supplies the PyTree container structure and desired JAX device or
sharding placement. Paths, shapes, and dtypes must match; logical sharding may change
when the target layout is compatible. Program identity, compatibility version, and
configuration and structural fingerprints can be checked before restoration.

Metadata also records simulation time, step and chunk identity, original logical
sharding, per-leaf checksums, and code and package versions copied from the run
manifest. Unknown metadata versions fail explicitly.

## Atomic local commits

The local store writes state and canonical JSON metadata into a private temporary
directory, synchronizes the files, adds a commit marker, and atomically renames the
directory into place. Only then is the `latest` pointer replaced atomically. A failed
save therefore cannot replace the previous latest checkpoint, and incomplete
temporary directories are neither listed nor restorable. Committed state is checksum
validated with NumPy pickle loading disabled.

`preflight` probes the selected directory before numerical work so missing or
unwritable storage fails with a checkpoint-specific error.

## Run-plan policy

Checkpoint cadence and resume intent cross execution boundaries as data:

```python
from adept import CheckpointPolicy, RunPlan, ServiceReference

plan = RunPlan(
    simulation=spec,
    checkpoint_store=ServiceReference("directory", {"root": "./checkpoints"}),
    checkpoint_policy=CheckpointPolicy(
        every_steps=1000,
        save_on_completion=True,
        resume_from="latest",
    ),
)
```

An enabled policy implies the `checkpointing` executor capability. Multi-host plans
also require shared durable storage and rank-zero I/O coordination. The initial
`LocalExecutor` does not yet advertise checkpointing, so it rejects an enabled policy
instead of silently running without saves. Executor-managed cadence and restore,
Orbax storage, cross-layout sharded tests, multi-host commit coordination, retention,
and metadata migrations remain follow-up work in issue #355.
