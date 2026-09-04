from __future__ import annotations

from types import SimpleNamespace

import pytest

from adept import (
    Artifact,
    ArtifactReceipt,
    DirectoryArtifactSink,
    ExecutionKind,
    FailurePolicy,
    MetricEvent,
    PreparedSimulation,
    RawResult,
    Report,
    RunHandle,
    RunManifest,
    RunRequest,
    RunStatus,
    SolverCapabilities,
    run_prepared,
)
from adept.core.tracking_mlflow import MLflowTracker


class ReadyValue:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.ready = False

    def block_until_ready(self) -> None:
        self.ready = True
        self.events.append("synchronize")


class Program:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, params, state, inputs, key):
        del params, state, inputs, key
        self.events.append("execute")
        return RawResult(ReadyValue(self.events), (), (), "ok", {})


class Analyzer:
    def __init__(self, events: list[str], artifacts: tuple[Artifact, ...] = ()) -> None:
        self.events = events
        self.artifacts = artifacts

    def analyze(self, result, manifest):
        del manifest
        assert result.final_state.ready
        self.events.append("analyze")
        return Report(
            result={"ready": True},
            metrics=(MetricEvent({"energy": 2.5}, step=4),),
            artifacts=self.artifacts,
        )


class RecordingTracker:
    def __init__(self, events: list[str], *, fail: str | None = None) -> None:
        self.events = events
        self.fail = fail
        self.logged = ()

    def preflight(self, request):
        del request
        self.events.append("tracker-preflight")
        if self.fail == "preflight":
            raise RuntimeError("tracker preflight failed")

    def start(self, request):
        self.events.append("start")
        if self.fail == "start":
            raise RuntimeError("tracker start failed")
        return RunHandle(
            run_id=request.run_id or "run-1",
            backend="recording",
            parent_run_id=request.parent.run_id if request.parent is not None else None,
        )

    def log_metrics(self, handle, events):
        del handle
        self.events.append("metrics")
        if self.fail == "metrics":
            raise RuntimeError("tracker metrics failed")
        self.logged = tuple(events)

    def finish(self, handle, status, *, error=None):
        del handle, error
        self.events.append(f"finish-{status.value.lower()}")
        if self.fail == "finish":
            raise RuntimeError("tracker finish failed")


class RecordingSink:
    def __init__(self, events: list[str], *, fail: str | None = None) -> None:
        self.events = events
        self.fail = fail

    def preflight(self):
        self.events.append("artifact-preflight")
        if self.fail == "preflight":
            raise RuntimeError("artifact preflight failed")

    def put(self, handle, artifact):
        del handle, artifact
        self.events.append("artifact-put")
        if self.fail == "put":
            raise RuntimeError("artifact upload failed")
        return ArtifactReceipt("output.txt", "memory://output.txt", False, 1, "digest")

    def validate(self, handle):
        del handle
        self.events.append("artifact-validate")
        if self.fail == "validate":
            raise RuntimeError("artifact handle mismatch")

    def verify(self, handle, receipt):
        del handle, receipt
        self.events.append("artifact-verify")
        if self.fail == "verify":
            raise RuntimeError("artifact verification failed")


def _prepared(events: list[str], analyzer: Analyzer) -> PreparedSimulation:
    return PreparedSimulation(
        program=Program(events),
        params={},
        state={},
        inputs={},
        manifest=RunManifest(raw_config={}, resolved_config={}, structural_fingerprint="sha256:test"),
        analyzer=analyzer,
        capabilities=SolverCapabilities(ExecutionKind.DISCRETE),
    )


def _direct(prepared, key):
    return prepared.program(prepared.params, prepared.state, prepared.inputs, key)


def test_host_run_orders_synchronization_analysis_persistence_and_terminal_status(tmp_path):
    artifact_file = tmp_path / "output.txt"
    artifact_file.write_text("result")
    events: list[str] = []
    tracker = RecordingTracker(events)
    sink = RecordingSink(events)
    prepared = _prepared(events, Analyzer(events, (Artifact(artifact_file),)))

    result = run_prepared(
        prepared,
        key="key",
        request=RunRequest(experiment="test", run_id="run-1"),
        tracker=tracker,
        artifact_sink=sink,
        execute=_direct,
    )

    assert events == [
        "artifact-preflight",
        "tracker-preflight",
        "start",
        "artifact-validate",
        "execute",
        "synchronize",
        "analyze",
        "metrics",
        "artifact-put",
        "artifact-verify",
        "finish-finished",
    ]
    assert result.report.result == {"ready": True}
    assert result.tracking_errors == ()
    assert result.artifacts[0].path == "output.txt"
    assert tracker.logged[0].values.keys() == {
        "adept.run_time_seconds",
        "adept.analysis_time_seconds",
    }
    assert tracker.logged[1].values == {"energy": 2.5}


def test_best_effort_tracker_failure_preserves_successful_result():
    events: list[str] = []
    tracker = RecordingTracker(events, fail="metrics")
    prepared = _prepared(events, Analyzer(events))

    result = run_prepared(
        prepared,
        key="key",
        tracker=tracker,
        tracking_failure_policy=FailurePolicy.BEST_EFFORT,
        execute=_direct,
    )

    assert result.report.result == {"ready": True}
    assert result.tracking_errors == ("tracker metrics: RuntimeError: tracker metrics failed",)
    assert events[-1] == "finish-finished"


def test_strict_tracker_failure_marks_run_failed():
    events: list[str] = []
    tracker = RecordingTracker(events, fail="metrics")
    prepared = _prepared(events, Analyzer(events))

    with pytest.raises(RuntimeError, match="tracker metrics failed"):
        run_prepared(prepared, key="key", tracker=tracker, execute=_direct)

    assert events[-1] == "finish-failed"
    assert "finish-finished" not in events


@pytest.mark.parametrize("failure", ["put", "verify"])
def test_artifact_failure_is_strict_and_cannot_finish_successfully(tmp_path, failure):
    artifact_file = tmp_path / "output.txt"
    artifact_file.write_text("result")
    events: list[str] = []
    tracker = RecordingTracker(events)
    prepared = _prepared(events, Analyzer(events, (Artifact(artifact_file),)))

    with pytest.raises(RuntimeError, match="artifact"):
        run_prepared(
            prepared,
            key="key",
            tracker=tracker,
            artifact_sink=RecordingSink(events, fail=failure),
            execute=_direct,
        )

    assert events[-1] == "finish-failed"
    assert "finish-finished" not in events


def test_artifact_preflight_fails_before_run_creation_or_execution():
    events: list[str] = []
    prepared = _prepared(events, Analyzer(events))

    with pytest.raises(RuntimeError, match="artifact preflight failed"):
        run_prepared(
            prepared,
            key="key",
            tracker=RecordingTracker(events),
            artifact_sink=RecordingSink(events, fail="preflight"),
            execute=_direct,
        )

    assert events == ["artifact-preflight"]


def test_artifact_handle_mismatch_marks_run_failed_before_execution():
    events: list[str] = []
    prepared = _prepared(events, Analyzer(events))

    with pytest.raises(RuntimeError, match="handle mismatch"):
        run_prepared(
            prepared,
            key="key",
            tracker=RecordingTracker(events),
            artifact_sink=RecordingSink(events, fail="validate"),
            execute=_direct,
        )

    assert "execute" not in events
    assert events[-1] == "finish-failed"


def test_directory_sink_round_trips_and_detects_corruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.txt").write_text("correct")
    sink = DirectoryArtifactSink(tmp_path / "artifacts")
    handle = RunHandle("run-1", "null")

    sink.preflight()
    receipt = sink.put(handle, Artifact(source, artifact_path="reports/final"))
    sink.verify(handle, receipt)

    assert receipt.path == "reports/final/source"
    assert receipt.size_bytes == len("correct")
    stored = tmp_path / "artifacts" / "run-1" / receipt.path / "result.txt"
    assert stored.read_text() == "correct"
    stored.write_text("corrupt")
    with pytest.raises(IOError, match="verification failed"):
        sink.verify(handle, receipt)


@pytest.mark.parametrize(
    "artifact_path",
    [
        "../escape",
        r"..\escape",
        r"nested\..\escape",
        r"C:\escape",
        "C:/escape",
        "C:escape",
        r"\escape",
        "nested//escape",
        "nested/./escape",
        "nested/escape/",
    ],
)
def test_artifact_rejects_non_normalized_cross_platform_paths(tmp_path, artifact_path):
    source = tmp_path / "output.txt"
    source.write_text("result")

    with pytest.raises(ValueError, match="relative POSIX"):
        Artifact(source, artifact_path=artifact_path)


def test_directory_sink_rejects_unsafe_destination_paths(tmp_path):
    source = tmp_path / "output.txt"
    source.write_text("result")

    with pytest.raises(ValueError, match="filesystem-safe"):
        DirectoryArtifactSink(tmp_path / "artifacts").put(
            RunHandle("../escape", "null"),
            Artifact(source),
        )


def test_directory_sink_checks_resolved_containment_before_writing(tmp_path):
    source = tmp_path / "output.txt"
    source.write_text("result")
    artifact_root = tmp_path / "artifacts"
    run_root = artifact_root / "run-1"
    outside = tmp_path / "outside"
    run_root.mkdir(parents=True)
    outside.mkdir()
    try:
        (run_root / "reports").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="outside its run directory"):
        DirectoryArtifactSink(artifact_root).put(
            RunHandle("run-1", "null"),
            Artifact(source, artifact_path="reports"),
        )

    assert not (outside / source.name).exists()


def test_mlflow_experiment_creation_race_resolves_explicit_winning_id():
    class RacingClient:
        def __init__(self):
            self.experiment = None
            self.created_with = None

        def get_experiment_by_name(self, name):
            del name
            return self.experiment

        def create_experiment(self, name):
            del name
            self.experiment = SimpleNamespace(experiment_id="17")
            raise RuntimeError("RESOURCE_ALREADY_EXISTS")

        def create_run(self, experiment_id, tags, run_name):
            del tags, run_name
            self.created_with = experiment_id
            return SimpleNamespace(info=SimpleNamespace(run_id="run-17"))

    client = RacingClient()
    tracker = MLflowTracker(client=client)

    handle = tracker.start(RunRequest(experiment="concurrent"))

    assert handle.run_id == "run-17"
    assert client.created_with == "17"
    assert client.created_with != "0"


def test_run_request_and_metric_event_are_defensive():
    tags = {"team": "plasma"}
    values = {"loss": 1.0}
    request = RunRequest(tags=tags)
    event = MetricEvent(values)
    tags["team"] = "changed"
    values["loss"] = 2.0

    assert request.tags == {"team": "plasma"}
    assert event.values == {"loss": 1.0}
    with pytest.raises(TypeError, match="numeric, not bool"):
        MetricEvent({"bad": True})


def test_run_status_is_explicitly_terminal():
    assert RunStatus.FINISHED.value == "FINISHED"
    assert RunStatus.FAILED.value == "FAILED"
