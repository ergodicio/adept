"""Offline runner tests: no MLflow connection and no numerical simulation."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from adept import ArtifactReceipt, RunHandle
from adept.core.tracking import _hash_path
from examples.farsight_comparison.cases import get_case
from examples.farsight_comparison.run import S3VerifiedArtifactSink, _execute_timed, _services, farsight_config


class FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.reads = []

    def get_object(self, *, Bucket, Key):
        self.reads.append((Bucket, Key))
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        return [{"Contents": [{"Key": key} for key in reversed(self.objects) if key.startswith(Prefix)]}]


def test_s3_file_verification_uses_streaming_boto3():
    payload = b"simulation artifact"
    client = FakeS3({"experiment/run/artifacts/scalars.nc": payload})
    sink = S3VerifiedArtifactSink(s3_client=client)
    receipt = ArtifactReceipt(
        path="scalars.nc",
        uri="s3://bucket/experiment/run/artifacts/scalars.nc",
        is_directory=False,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    sink.verify(RunHandle("test", "mlflow"), receipt)
    assert client.reads == [("bucket", "experiment/run/artifacts/scalars.nc")]


def test_s3_directory_hash_matches_runtime_contract(tmp_path):
    (tmp_path / "a.nc").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.nc").write_bytes(b"bbb")
    size, digest = _hash_path(tmp_path)
    client = FakeS3({"data/a.nc": b"a", "data/nested/b.nc": b"bbb", "unrelated": b"ignore"})
    sink = S3VerifiedArtifactSink(s3_client=client)
    receipt = ArtifactReceipt(path="data", uri="s3://bucket/data", is_directory=True, size_bytes=size, sha256=digest)
    sink.verify(RunHandle("test", "mlflow"), receipt)
    assert [key for _, key in client.reads] == ["data/a.nc", "data/nested/b.nc"]


def test_s3_hash_mismatch_fails():
    sink = S3VerifiedArtifactSink(s3_client=FakeS3({"data": b"wrong"}))
    receipt = ArtifactReceipt(path="data", uri="s3://bucket/data", is_directory=False, size_bytes=3, sha256="0" * 64)
    with pytest.raises(OSError, match="verification failed"):
        sink.verify(RunHandle("test", "mlflow"), receipt)


def test_tracking_is_never_silently_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="never silently disable"):
        _services()


def test_farsight_case_config_matches_shared_definition():
    case = get_case("two-stream", nx=16, nv=32, tmax=1.0, dt=0.05, frame_dt=0.5)
    config = farsight_config(case)
    assert config["grid"]["xmax"] == case.length
    assert config["initial"] == case.farsight_initial()
    assert config["numerical"]["remesh_every"] == 1
    assert config["save"]["distribution"]["every_steps"] == 10


def test_benchmark_executes_identical_closure_three_times(tmp_path):
    import jax.numpy as jnp

    initial = jnp.array([1.0])
    starts = []

    def execute():
        starts.append(initial)
        return initial + 1

    result = _execute_timed(execute, benchmark=True, steps=20, directory=tmp_path)
    assert float(result[0]) == 2
    assert len(starts) == 3 and all(value is initial for value in starts)
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["executions"] == 3
    assert len(timing["warm_seconds_per_step"]) == 2
    assert timing["steps_per_execution"] == 20


@pytest.fixture
def offline_services(monkeypatch):
    from examples.farsight_comparison import run

    class Tracker:
        tracking_uri = "file:///unused"

        def __init__(self):
            self.finished = []

        def _get_client(self):
            return self

        def get_experiment_by_name(self, name):
            return object()

        def preflight(self, request):
            pass

        def start(self, request):
            return RunHandle("early-id", "mlflow")

        def set_tag(self, *args):
            pass

        def log_param(self, *args):
            pass

        def finish(self, handle, status, **kwargs):
            self.finished.append(status.value)

    class Sink:
        fail = False

        def __init__(self):
            self.uploads = []

        def preflight(self):
            pass

        def validate(self, handle):
            pass

        def put(self, handle, artifact):
            if self.fail:
                raise OSError("offline")
            self.uploads.append(artifact)
            return artifact

        def verify(self, handle, receipt):
            pass

    tracker, sink = Tracker(), Sink()
    monkeypatch.delenv("NERSC_HOST", raising=False)
    monkeypatch.setattr(run, "_services", lambda _: (tracker, sink))
    monkeypatch.setattr(run, "_provenance", lambda _: {"source_archive_sha256": "test", "git_commit": "test-sha"})
    return run, tracker, sink


def test_failed_solve_records_run_id_and_marks_failed_even_if_error_upload_fails(
    tmp_path, monkeypatch, offline_services
):
    run, tracker, sink = offline_services

    def fail(*args):
        assert json.loads((tmp_path / "case" / "run.json").read_text())["run_id"] == "early-id"
        sink.fail = True
        raise ArithmeticError("invalid panels")

    monkeypatch.setattr(run, "_run_farsight", fail)
    with pytest.raises(ArithmeticError, match="invalid panels"):
        run.run_one({"solver": "farsight", "case": "two-stream", "output": tmp_path / "case"})
    assert tracker.finished == ["FAILED"]
    summary = json.loads((tmp_path / "case" / "run.json").read_text())
    assert summary["status"] == "FAILED" and summary["run_id"] == "early-id"


@pytest.mark.parametrize("failure_stage", ["initial-summary", "failure-summary"])
def test_summary_disk_failure_preserves_original_error_and_terminates_tracking(
    tmp_path, monkeypatch, offline_services, failure_stage
):
    run, tracker, sink = offline_services
    write_json = run._write_json
    quota_error = OSError("disk quota exceeded")
    solve_error = ArithmeticError("invalid panels")

    def fail_summary(path, value):
        if path.name == "run.json" and (failure_stage == "initial-summary" or value["status"] == "FAILED"):
            raise quota_error
        write_json(path, value)

    def fail_solve(*args):
        raise solve_error

    monkeypatch.setattr(run, "_write_json", fail_summary)
    monkeypatch.setattr(run, "_run_farsight", fail_solve)
    expected = quota_error if failure_stage == "initial-summary" else solve_error
    with pytest.raises(type(expected)) as caught:
        run.run_one({"solver": "farsight", "case": "two-stream", "output": tmp_path / "case"})
    assert caught.value is expected
    assert tracker.finished == ["FAILED"]
    assert any("Failure summary persistence also failed" in note for note in caught.value.__notes__)
    # A failed rewrite must not re-upload an old RUNNING summary as failure evidence.
    summary_uploads = [artifact for artifact in sink.uploads if Path(artifact.source).name == "run.json"]
    assert len(summary_uploads) == (0 if failure_stage == "initial-summary" else 1)


def test_analyzer_does_not_upload_provisional_run_summary(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr

    from adept import MetricEvent, Report
    from adept.farsight1d.__main__ import FileAnalyzer
    from examples.farsight_comparison.run import DiagnosticFileAnalyzer

    (tmp_path / "run.json").write_text(json.dumps({"run_id": "test", "status": "RUNNING"}))
    dataset = xr.Dataset({"mass": ("t", [1.0]), "c2": ("t", [1.0])}, coords={"t": [0.0]})

    class Analyzer:
        def analyze(self, result, manifest):
            return Report(result={"scalars": dataset}, metrics=(MetricEvent({"final_mass": 1.0}),))

    result = SimpleNamespace(final_state={"valid": np.asarray(True)})
    manifest = SimpleNamespace(to_dict=dict)
    report = DiagnosticFileAnalyzer(FileAnalyzer(Analyzer(), tmp_path), tmp_path).analyze(result, manifest)
    filenames = {Path(artifact.source).name for artifact in report.artifacts}
    assert filenames == {"scalars.nc", "manifest.json", "metrics.json", "final_state.npz"}
    assert report.result["scalars"] is dataset
    assert dict(report.metrics[0].values) == {"final_mass": 1.0}
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "RUNNING"


def test_amr_and_tree_options_are_resolved_and_validated():
    case = get_case("two-stream", nx=8, nv=16)
    options = {
        "amr": True,
        "amr_max_level": 2,
        "amr_min_level": 1,
        "amr_max_panels": 512,
        "amr_atol": 0.025,
        "amr_rtol": 0.1,
        "amr_max_gap_fraction": 0.001,
        "quadrature": "simpson",
        "remesh_every": 2,
        "field_solver": "treecode",
        "tree_degree": 12,
        "tree_theta": 0.2,
        "tree_leaf_size": 16,
    }
    config = farsight_config(case, **options)
    assert config["amr"] == {
        "enabled": True,
        "max_level": 2,
        "min_level": 1,
        "max_panels": 512,
        "atol": 0.025,
        "rtol": 0.1,
        "max_gap_fraction": 0.001,
    }
    assert config["numerical"]["treecode"] == {"degree": 12, "theta": 0.2, "leaf_size": 16}
    assert config["numerical"]["quadrature"] == "simpson"
    assert config["numerical"]["remesh_every"] == 2
    with pytest.raises(ValueError, match="max_panels"):
        farsight_config(case, **{**options, "amr_max_panels": 16})
    with pytest.raises(ValueError, match="Unknown FARSIGHT"):
        farsight_config(case, amr_typo=True)


def test_amr_pair_changes_only_field_solver_and_allows_independent_eulerian_resolution(tmp_path):
    from examples.farsight_comparison.scan import build_parser, build_tasks

    args = build_parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--cases",
            "two-stream",
            "--amr-pair",
            "--nx",
            "8",
            "--nv",
            "16",
            "--amr-max-level",
            "2",
            "--amr-max-panels",
            "512",
            "--eulerian-nx",
            "128",
            "--eulerian-nv",
            "256",
        ]
    )
    tasks = build_tasks(vars(args), tmp_path)
    assert [task["solver"] for task in tasks] == ["farsight", "farsight", "eulerian-softened", "eulerian-poisson"]
    assert [task["field_solver"] for task in tasks[:2]] == ["direct", "treecode"]
    differing = {key for key in tasks[0] if tasks[0][key] != tasks[1][key]}
    assert differing == {"field_solver", "name", "output"}
    assert all(task["amr"] for task in tasks[:2])
    assert all(task["nx"] == 128 and task["nv"] == 256 for task in tasks[2:])


def test_task_file_defaults_make_ordered_amr_only_pair(tmp_path):
    from examples.farsight_comparison.scan import build_parser, build_tasks

    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "defaults": {
                    "case": "two-stream",
                    "nx": 8,
                    "nv": 16,
                    "amr": True,
                    "amr_max_panels": 512,
                    "amr_max_level": 2,
                },
                "tasks": [{"field_solver": "direct"}, {"field_solver": "treecode"}],
            }
        )
    )
    options = vars(build_parser().parse_args(["--output", str(tmp_path / "outputs"), "--task-file", str(path)]))
    tasks = build_tasks(options, tmp_path / "outputs")
    assert len(tasks) == 2 and all(task["solver"] == "farsight" for task in tasks)
    assert "farsight-amr-direct" in tasks[0]["name"]
    assert "farsight-amr-treecode" in tasks[1]["name"]
    assert tasks[0]["nx"] == 8 and tasks[0]["amr_max_panels"] == 512
    path.write_text(json.dumps([{"case": "two-stream", "output": "../escape"}]))
    with pytest.raises(ValueError, match="relative subdirectory"):
        build_tasks(options, tmp_path / "outputs")


def test_failed_analysis_retains_raw_evidence_and_reraises(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr

    from examples.farsight_comparison.run import DiagnosticFileAnalyzer

    failure = ArithmeticError("invalid panels")

    class RejectingAnalyzer:
        def analyze(self, result, manifest):
            raise failure

    result = SimpleNamespace(
        final_state={
            "valid": np.array(False),
            "active_panels": np.array(8),
            "f": np.array([1.0, np.nan]),
            "capacity_exceeded": np.array(True),
        },
        observations={"scalars": {"valid": np.array([True, False]), "active_panels": np.array([4, 8])}},
        times={"scalars": np.array([0.0, 0.05])},
    )
    with pytest.raises(ArithmeticError) as caught:
        DiagnosticFileAnalyzer(RejectingAnalyzer(), tmp_path).analyze(result, None)
    assert caught.value is failure
    evidence = json.loads((tmp_path / "failure_diagnostics.json").read_text())
    assert evidence["status"] == "FAILED"
    assert evidence["first_invalid_saved_time"] == 0.05
    assert evidence["final_scalars"]["valid"] is False
    assert evidence["final_state_nonfinite_counts"]["f"] == 1
    with np.load(tmp_path / "failed_final_state.npz") as saved:
        assert not saved["valid"] and np.isnan(saved["f"][-1])
    with xr.open_dataset(tmp_path / "failed_scalars.nc") as saved:
        assert saved.attrs["status"] == "FAILED"
        assert saved.valid.values.tolist() == [1, 0]
