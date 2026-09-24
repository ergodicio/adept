"""Offline runner tests: no MLflow connection and no numerical simulation."""

import hashlib
import io
import json

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


def test_failed_solve_records_run_id_and_marks_failed_even_if_error_upload_fails(tmp_path, monkeypatch):
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

        def preflight(self):
            pass

        def validate(self, handle):
            pass

        def put(self, handle, artifact):
            if self.fail:
                raise OSError("offline")
            return artifact

        def verify(self, handle, receipt):
            pass

    tracker, sink = Tracker(), Sink()
    monkeypatch.delenv("NERSC_HOST", raising=False)
    monkeypatch.setattr(run, "_services", lambda _: (tracker, sink))
    monkeypatch.setattr(run, "_provenance", lambda _: {"source_archive_sha256": "test", "git_commit": "test-sha"})

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
