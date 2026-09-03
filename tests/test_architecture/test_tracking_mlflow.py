from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mlflow

from adept import Artifact, MetricEvent, RunRequest, RunStatus
from adept.core.tracking_mlflow import MLflowArtifactSink, MLflowTracker


def test_mlflow_tracker_uses_explicit_parent_child_runs_and_verified_artifacts(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    tracker = MLflowTracker(tracking_uri=tracking_uri)
    request = RunRequest(experiment="concurrent-runs", name="parent")
    tracker.preflight(request)
    parent = tracker.start(request)

    def run_child(index):
        child = tracker.start(
            RunRequest(
                experiment="concurrent-runs",
                name=f"child-{index}",
                parent=parent,
            )
        )
        tracker.log_metrics(child, (MetricEvent({"loss": float(index)}, step=index),))
        tracker.finish(child, RunStatus.FINISHED)
        return child

    with ThreadPoolExecutor(max_workers=4) as executor:
        children = tuple(executor.map(run_child, range(4)))

    artifact_file = tmp_path / "result.txt"
    artifact_file.write_text("durable result")
    sink = MLflowArtifactSink(tracking_uri=tracking_uri)
    sink.preflight()
    receipt = sink.put(parent, Artifact(artifact_file, artifact_path="reports"))
    sink.verify(parent, receipt)

    artifact_directory = tmp_path / "snapshot"
    artifact_directory.mkdir()
    (artifact_directory / "manifest.json").write_text('{"complete": true}')
    directory_receipt = sink.put(parent, Artifact(artifact_directory, artifact_path="snapshots"))
    sink.verify(parent, directory_receipt)
    tracker.finish(parent, RunStatus.FINISHED)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("concurrent-runs")
    assert experiment is not None
    assert experiment.experiment_id != "0"
    assert client.get_run(parent.run_id).info.status == "FINISHED"
    for index, child in enumerate(children):
        run = client.get_run(child.run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["mlflow.parentRunId"] == parent.run_id
        assert run.data.metrics["loss"] == float(index)
    assert receipt.path == "reports/result.txt"
    assert directory_receipt.path == "snapshots/snapshot"
    assert mlflow.active_run() is None
