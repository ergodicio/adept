# Host-side tracking and artifacts

Prepared simulations can run without MLflow and without putting tracking clients,
paths, or credentials inside the JAX program. `run_prepared` owns the host boundary:
it executes and synchronizes the numerical result, calls the analyzer, records metrics,
verifies artifacts, and only then marks the run finished.

## Untracked and directory-backed runs

`NullTracker` disables telemetry. `DirectoryArtifactSink` retains analyzer artifacts
under a directory named for the explicit run ID:

```python
import jax

from adept import (
    DirectoryArtifactSink,
    NullTracker,
    RunRequest,
    SimulationSpec,
    run_prepared,
    solver_registry,
)

prepared = solver_registry.prepare(
    SimulationSpec.from_legacy_config(config),
    key=42,
)
completed = run_prepared(
    prepared,
    key=jax.random.key(42),
    request=RunRequest(experiment="untracked", run_id="local-example"),
    tracker=NullTracker(),
    artifact_sink=DirectoryArtifactSink("./adept-results"),
)

raw_result = completed.raw_result
report = completed.report
```

This path does not import or contact MLflow. Artifact destinations are preflighted
before numerical execution. Files and directories are copied through a staging path,
hashed, and read back for verification before the run can finish successfully.

## Returning metrics and artifacts from an analyzer

Analyzers return a `Report`; they do not call MLflow or upload files themselves:

```python
from dataclasses import replace

from adept import Artifact, MetricEvent, Report


class EnergyAnalyzer:
    def analyze(self, result, manifest):
        del manifest
        energy = float(result.observations["energy"][-1])
        return Report(
            result={"final_energy": energy},
            metrics=(MetricEvent({"final_energy": energy}),),
            artifacts=(Artifact("./plots/energy.png", artifact_path="plots"),),
        )


prepared = replace(prepared, analyzer=EnergyAnalyzer())
```

Artifacts are local file or directory references. The selected `ArtifactSink` decides
where they are stored. Upload or verification failures are strict: the tracker is told
that the run failed, and the exception is returned to the caller.

## MLflow without active-run state

The MLflow adapters use `MlflowClient` and pass the `RunHandle` to every operation.
They never use `mlflow.set_experiment`, `mlflow.start_run`, or the process-global active
run. This makes concurrent parent and child runs independent:

```python
from adept import (
    MLflowArtifactSink,
    MLflowTracker,
    RunRequest,
    run_prepared,
)

tracker = MLflowTracker(tracking_uri="https://tracking.example")
artifacts = MLflowArtifactSink(tracking_uri="https://tracking.example")

completed = run_prepared(
    prepared,
    key=key,
    request=RunRequest(experiment="tpd-scan", name="angle-18"),
    tracker=tracker,
    artifact_sink=artifacts,
)
```

Experiment creation is race-tolerant. When another worker creates the experiment
first, ADEPT resolves its explicit ID and creates the run there; it never silently
falls back to MLflow's Default experiment.

To resume an existing run, pass its ID in `RunRequest(run_id=...)`. The adapter
validates that the run is active (not deleted), preserves its existing parent, and
explicitly transitions it back to `RUNNING` before returning the handle. A process
failure during resumed execution therefore leaves the run visibly incomplete rather
than retaining a stale terminal status.

For an MLflow service behind ADEPT's `/ajax-api/2.0` reverse-proxy route, configure the
compatibility behavior only on the adapter:

```python
tracker = MLflowTracker(
    tracking_uri="https://tracking.example",
    rest_api_path_prefix="/ajax-api/2.0",
)
```

Credentials remain in the environment or normal MLflow/AWS configuration. They must
not be placed in `RunRequest`, `RunManifest`, or a serialized simulation specification.

## Failure policy

Tracking is strict by default. For disposable progress telemetry, use
`FailurePolicy.BEST_EFFORT`; tracking errors are then returned in
`HostRunResult.tracking_errors` while a successful numerical result is preserved:

```python
from adept import FailurePolicy

completed = run_prepared(
    prepared,
    key=key,
    tracker=tracker,
    artifact_sink=DirectoryArtifactSink("./adept-results"),
    tracking_failure_policy=FailurePolicy.BEST_EFFORT,
)
```

Best-effort applies only to tracker telemetry. Artifact writes and verification remain
strict because a run with missing declared outputs must not appear successfully
archived.

The existing `ergoExo` and `ADEPTModule` entry points retain their current MLflow
behavior during this phase. They will move onto these services through the later
compatibility façade rather than changing behavior in place.
