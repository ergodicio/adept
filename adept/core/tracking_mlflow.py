"""Lazy MLflow adapters using explicit clients and run identifiers."""

from __future__ import annotations

import importlib
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .tracking import (
    Artifact,
    ArtifactReceipt,
    MetricEvent,
    RunHandle,
    RunRequest,
    RunStatus,
    _hash_path,
    _validate_source,
)


def _configure_rest_api_path_prefix(prefix: str) -> None:
    """Contain ADEPT's reverse-proxy compatibility patch in this adapter."""

    if not prefix.startswith("/") or prefix.endswith("/"):
        raise ValueError("rest_api_path_prefix must start, but not end, with '/'")

    rest_utils = importlib.import_module("mlflow.utils.rest_utils")
    service_module = importlib.import_module("mlflow.protos.service_pb2")
    store_module = importlib.import_module("mlflow.store.tracking.rest_store")
    vars(rest_utils)["_REST_API_PATH_PREFIX"] = prefix
    vars(rest_utils)["_TRACE_REST_API_PATH_PREFIX"] = f"{prefix}/mlflow/traces"
    store_module.RestStore._METHOD_TO_INFO = rest_utils.extract_api_info_for_service(
        service_module.MlflowService,
        prefix,
    )


class _MLflowClientProvider:
    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        rest_api_path_prefix: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.rest_api_path_prefix = rest_api_path_prefix
        self._provided_client = client
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        if self._client is None:
            if self.rest_api_path_prefix is not None:
                _configure_rest_api_path_prefix(self.rest_api_path_prefix)
            tracking = importlib.import_module("mlflow.tracking")
            self._client = tracking.MlflowClient(
                tracking_uri=self.tracking_uri,
                registry_uri=self.registry_uri,
            )
        return self._client


class MLflowTracker(_MLflowClientProvider):
    """MLflow tracker that never reads or writes fluent active-run state."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        rest_api_path_prefix: str | None = None,
        experiment_create_retries: int = 5,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            rest_api_path_prefix=rest_api_path_prefix,
            client=client,
        )
        if experiment_create_retries < 1:
            raise ValueError("experiment_create_retries must be at least one")
        self.experiment_create_retries = experiment_create_retries

    def preflight(self, request: RunRequest) -> None:
        client = self._get_client()
        client.search_experiments(max_results=1)
        if request.run_id is not None:
            client.get_run(request.run_id)

    def _get_or_create_experiment(self, name: str) -> str:
        client = self._get_client()
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            return str(experiment.experiment_id)

        create_error: Exception | None = None
        try:
            return str(client.create_experiment(name))
        except Exception as error:
            # A concurrent process may have won creation. Resolve the explicit id
            # rather than falling back to MLflow's process-global default experiment.
            create_error = error

        for attempt in range(self.experiment_create_retries):
            experiment = client.get_experiment_by_name(name)
            if experiment is not None:
                return str(experiment.experiment_id)
            if attempt + 1 < self.experiment_create_retries:
                time.sleep(0.05 * 2**attempt)
        assert create_error is not None
        raise create_error

    def start(self, request: RunRequest) -> RunHandle:
        client = self._get_client()
        parent_run_id = request.parent.run_id if request.parent is not None else None
        if request.parent is not None and request.parent.backend != "mlflow":
            raise ValueError("an MLflow child run requires an MLflow parent handle")

        if request.run_id is not None:
            run = client.get_run(request.run_id)
            if str(run.info.lifecycle_stage).lower() == "deleted":
                raise ValueError(f"cannot resume deleted MLflow run {request.run_id!r}")

            existing_parent_run_id = run.data.tags.get("mlflow.parentRunId")
            if parent_run_id is not None and existing_parent_run_id != parent_run_id:
                raise ValueError(
                    f"MLflow run {request.run_id!r} belongs to parent {existing_parent_run_id!r}, not {parent_run_id!r}"
                )
            parent_run_id = existing_parent_run_id

            client.update_run(run.info.run_id, status=RunStatus.RUNNING.value)
            reopened = client.get_run(run.info.run_id)
            if reopened.info.status != RunStatus.RUNNING.value:
                raise RuntimeError(f"MLflow run {request.run_id!r} did not reopen: status is {reopened.info.status!r}")
            return RunHandle(
                run_id=str(reopened.info.run_id),
                backend="mlflow",
                parent_run_id=parent_run_id,
            )

        experiment_id = self._get_or_create_experiment(request.experiment)
        tags = dict(request.tags)
        if parent_run_id is not None:
            tags["mlflow.parentRunId"] = parent_run_id
        run = client.create_run(
            experiment_id=experiment_id,
            tags=tags,
            run_name=request.name,
        )
        return RunHandle(
            run_id=str(run.info.run_id),
            backend="mlflow",
            parent_run_id=parent_run_id,
        )

    def log_metrics(self, handle: RunHandle, events: Sequence[MetricEvent]) -> None:
        if handle.backend != "mlflow":
            raise ValueError("MLflowTracker requires an MLflow run handle")
        if not events:
            return

        metric_type = importlib.import_module("mlflow.entities").Metric
        now_ms = int(time.time() * 1000)
        metrics = [
            metric_type(
                key=name,
                value=value,
                timestamp=event.timestamp_ms if event.timestamp_ms is not None else now_ms,
                step=event.step,
            )
            for event in events
            for name, value in event.values.items()
        ]
        self._get_client().log_batch(handle.run_id, metrics=metrics, synchronous=True)

    def finish(self, handle: RunHandle, status: RunStatus, *, error: str | None = None) -> None:
        if handle.backend != "mlflow":
            raise ValueError("MLflowTracker requires an MLflow run handle")
        status = RunStatus(status)
        if status is RunStatus.RUNNING:
            raise ValueError("finish requires a terminal run status")

        client = self._get_client()
        tag_error: Exception | None = None
        if error is not None:
            try:
                client.set_tag(handle.run_id, "adept.error", error[:5000], synchronous=True)
            except Exception as caught:
                tag_error = caught
        client.set_terminated(handle.run_id, status=status.value)
        if tag_error is not None:
            raise tag_error


class MLflowArtifactSink(_MLflowClientProvider):
    """Verified MLflow artifact storage for explicit run handles."""

    def preflight(self) -> None:
        self._get_client().search_experiments(max_results=1)

    def validate(self, handle: RunHandle) -> None:
        if handle.backend != "mlflow":
            raise ValueError("MLflowArtifactSink requires an MLflow run handle")
        client = self._get_client()
        client.get_run(handle.run_id)
        # Exercise the configured artifact repository before numerical work. This
        # catches missing S3 credentials or an unreadable artifact root while the
        # run can still be failed cleanly without losing computed outputs.
        client.list_artifacts(handle.run_id)

    @staticmethod
    def _target_path(artifact: Artifact) -> str:
        return str(PurePosixPath(artifact.artifact_path or "") / Path(artifact.source).name)

    def put(self, handle: RunHandle, artifact: Artifact) -> ArtifactReceipt:
        if handle.backend != "mlflow":
            raise ValueError("MLflowArtifactSink requires an MLflow run handle")
        source = Path(artifact.source).expanduser()
        _validate_source(source)
        target_path = self._target_path(artifact)
        client = self._get_client()

        if source.is_dir():
            client.log_artifacts(handle.run_id, str(source), artifact_path=target_path)
        else:
            client.log_artifact(handle.run_id, str(source), artifact_path=artifact.artifact_path)

        artifact_uri = str(client.get_run(handle.run_id).info.artifact_uri).rstrip("/")
        size, digest = _hash_path(source)
        return ArtifactReceipt(
            path=target_path,
            uri=f"{artifact_uri}/{target_path}",
            is_directory=source.is_dir(),
            size_bytes=size,
            sha256=digest,
        )

    def verify(self, handle: RunHandle, receipt: ArtifactReceipt) -> None:
        if handle.backend != "mlflow":
            raise ValueError("MLflowArtifactSink requires an MLflow run handle")
        with tempfile.TemporaryDirectory() as destination:
            downloaded = Path(
                self._get_client().download_artifacts(
                    handle.run_id,
                    receipt.path,
                    dst_path=destination,
                )
            )
            if not downloaded.exists():
                raise FileNotFoundError(f"MLflow artifact upload is missing: {receipt.path}")
            if downloaded.is_dir() is not receipt.is_directory:
                raise OSError(f"MLflow artifact type changed after upload: {receipt.path}")
            size, digest = _hash_path(downloaded)
            if receipt.size_bytes != size or receipt.sha256 != digest:
                raise OSError(f"MLflow artifact verification failed: {receipt.path}")


__all__ = ["MLflowArtifactSink", "MLflowTracker"]
