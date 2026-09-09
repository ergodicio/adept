from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mlflow
import pytest
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils import rest_utils

from adept import Artifact, MetricEvent, RunHandle, RunRequest, RunStatus
from adept.core.tracking_mlflow import MLflowArtifactSink, MLflowTracker


def _v2_endpoints(client):
    return {endpoint for endpoint, _method in client._tracking_client.store._METHOD_TO_INFO.values()}


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


def test_mlflow_resume_reopens_terminal_run_and_rejects_deleted_run(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    tracker = MLflowTracker(tracking_uri=tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("resume-runs")

    parent = client.create_run(experiment_id, run_name="parent")
    terminal = client.create_run(
        experiment_id,
        run_name="terminal",
        tags={"mlflow.parentRunId": parent.info.run_id},
    )
    client.set_terminated(terminal.info.run_id, status="FINISHED")
    resumed = tracker.start(
        RunRequest(
            experiment="resume-runs",
            run_id=terminal.info.run_id,
        )
    )

    assert resumed.run_id == terminal.info.run_id
    assert resumed.parent_run_id == parent.info.run_id
    assert client.get_run(resumed.run_id).info.status == "RUNNING"

    with pytest.raises(ValueError, match="belongs to parent"):
        tracker.start(
            RunRequest(
                experiment="resume-runs",
                run_id=terminal.info.run_id,
                parent=RunHandle("different-parent", "mlflow"),
            )
        )

    deleted = client.create_run(experiment_id, run_name="deleted")
    client.delete_run(deleted.info.run_id)
    with pytest.raises(ValueError, match="cannot resume deleted"):
        tracker.start(
            RunRequest(
                experiment="resume-runs",
                run_id=deleted.info.run_id,
            )
        )


def test_prefixed_and_ordinary_mlflow_clients_keep_independent_route_tables():
    original_prefix = rest_utils._REST_API_PATH_PREFIX
    original_routes = dict(RestStore._METHOD_TO_INFO)
    original_endpoints = {endpoint for endpoint, _method in original_routes.values()}
    ordinary_before = mlflow.tracking.MlflowClient(tracking_uri="https://ordinary-before.example")

    prefixed = MLflowTracker(
        tracking_uri="https://prefixed.example",
        rest_api_path_prefix="/ajax-api/2.0",
    )._get_client()
    differently_prefixed = MLflowTracker(
        tracking_uri="https://other-prefix.example",
        rest_api_path_prefix="/other-api/2.0",
    )._get_client()
    ordinary_after = mlflow.tracking.MlflowClient(tracking_uri="https://ordinary-after.example")

    assert all(endpoint.startswith("/ajax-api/2.0") for endpoint in _v2_endpoints(prefixed))
    assert all(endpoint.startswith("/other-api/2.0") for endpoint in _v2_endpoints(differently_prefixed))
    for ordinary in (ordinary_before, ordinary_after):
        assert _v2_endpoints(ordinary) == original_endpoints
    assert rest_utils._REST_API_PATH_PREFIX == original_prefix
    assert RestStore._METHOD_TO_INFO == original_routes
