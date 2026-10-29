"""Lifecycle tests use client doubles; numerical kernels are irrelevant here."""

import time
from types import SimpleNamespace
from typing import ClassVar

import pytest

from adept import _monitor


class Client:
    def __init__(self):
        self.runs = []

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def create_run(self, experiment_id, tags):
        self.runs.append(dict(tags))
        return SimpleNamespace(info=SimpleNamespace(run_id=str(len(self.runs))))

    def set_tag(self, run_id, key, value):
        self.runs[int(run_id) - 1][key] = value

    def log_batch(self, run_id, tags):
        for tag in tags:
            self.set_tag(run_id, tag.key, tag.value)

    def set_terminated(self, run_id, status):
        self.set_tag(run_id, "status", status)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ADEPT_MONITOR_LAUNCH_ID", "launch")
    client = Client()
    monkeypatch.setattr(_monitor.mlflow.tracking, "MlflowClient", lambda **kwargs: client)
    for key in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"):
        monkeypatch.delenv(key, raising=False)
    return client


class Simulation:
    mlflow_run_id = None
    cfg: ClassVar = {"mlflow": {"experiment": "test"}}

    @_monitor.monitored
    def setup(self, cfg):
        self.mlflow_run_id = "scientific"

    @_monitor.monitored
    def __call__(self, fail=False, upload=False):
        self._monitor.phase("postprocessing")
        if fail:
            raise ValueError("simulation failed")
        self._monitor.upload_failed = upload
        return 42


def test_repeated_calls_and_launch_identity(client, monkeypatch):
    monkeypatch.setenv("ADEPT_MONITOR_LAUNCH_ID", "launch")
    sim = Simulation()
    sim.setup(sim.cfg)
    _monitor._messages.join()
    assert client.runs[0]["monitor.outcome"] == "running"
    assert sim() == 42
    assert sim() == 42
    assert len(client.runs) == 2
    assert client.runs[0]["monitor.execution_id"] != client.runs[1]["monitor.execution_id"]
    for run in client.runs:
        assert run["monitor.launch_id"] == "launch"
        assert run["monitor.outcome"] == "succeeded"
        assert run["monitor.scientific_run_id"] == "scientific"
        assert "monitor.completed_at" in run


@pytest.mark.parametrize("upload", [False, True])
def test_execution_and_upload_failures(client, upload):
    sim = Simulation()
    sim.setup(sim.cfg)
    if upload:
        sim(upload=True)
    else:
        with pytest.raises(ValueError):
            sim(fail=True)
    assert client.runs[0]["monitor.outcome"] == "failed"
    assert sim._monitor.stop.is_set()


def test_heartbeat_does_not_imply_progress(client):
    monitor = _monitor.ExecutionMonitor("test", interval=0.005)
    _monitor._messages.join()
    first = client.runs[0]["monitor.heartbeat_at"]
    deadline = time.monotonic() + 3
    while client.runs[0]["monitor.heartbeat_at"] == first and time.monotonic() < deadline:
        time.sleep(0.005)
    monitor.finish("succeeded")
    assert client.runs[0]["monitor.heartbeat_at"] != first
    assert "monitor.progress_at" not in client.runs[0]


def test_telemetry_failure_and_worker_ownership(client, monkeypatch):
    monkeypatch.setenv("ADEPT_MONITOR_OWNER", "rank0")
    monkeypatch.setenv("SLURM_PROCID", "1")
    sim = Simulation()
    sim.setup(sim.cfg)
    assert sim() == 42
    assert not client.runs
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setattr(_monitor.mlflow.tracking, "MlflowClient", lambda **kwargs: 1 / 0)
    sim.setup(sim.cfg)
    assert sim() == 42


def test_setup_failure_is_terminal(client):
    class Broken(Simulation):
        @_monitor.monitored
        def setup(self, cfg):
            raise RuntimeError("setup failed")

    with pytest.raises(RuntimeError):
        Broken().setup(Simulation.cfg)
    assert client.runs[0]["monitor.outcome"] == "failed"


def test_ergoexo_real_mlflow_lifecycle(tmp_path, monkeypatch):
    """Real MLflow store, real wrappers, tiny solver double; no numerical compilation."""
    import adept._base_ as base
    from adept import ergoExo

    mlflow = base.mlflow
    monkeypatch.setenv("ADEPT_MONITOR_EXPERIMENT", "monitor-integration")
    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    uploads = []
    try:
        mlflow.set_experiment("science")
        with mlflow.start_run() as outer:
            exo = ergoExo(mlflow_nested=True, parent_run_id=outer.info.run_id)

            def setup(cfg, td, module):
                exo.ran_setup = True
                exo.adept_module = SimpleNamespace(
                    post_process=lambda output, path: {}, vg=lambda modules, args: ((1.0, {}), {})
                )
                return {}

            monkeypatch.setattr(exo, "_setup_", setup)
            monkeypatch.setattr(exo, "_execute_simulation", lambda modules, args: {})
            monkeypatch.setattr(base, "filter_jit", lambda fn: fn)
            monkeypatch.setattr(base, "robust_log_artifacts", lambda td: uploads.append(td) or False)
            cfg = {"mlflow": {"experiment": "science", "run": "tiny"}}
            exo.setup(cfg)
            assert mlflow.active_run().info.run_id == outer.info.run_id
            exo(export=False)
            assert mlflow.active_run().info.run_id == outer.info.run_id
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("monitor-integration")
            records = client.search_runs(
                [experiment.experiment_id], filter_string="tags.`monitor.record_type` = 'execution'"
            )
            assert len(records) == 1
            assert records[0].data.tags["monitor.outcome"] == "failed"  # setup upload exhausted
            assert records[0].data.tags["monitor.scientific_run_id"] == exo.mlflow_run_id
            exo(export=False)
            exo.val_and_grad(export=False)
            records = client.search_runs(
                [experiment.experiment_id], filter_string="tags.`monitor.record_type` = 'execution'"
            )
            assert len(records) == 3
            assert sorted(r.data.tags["monitor.outcome"] for r in records) == ["failed", "succeeded", "succeeded"]
            assert len({r.data.tags["monitor.execution_id"] for r in records}) == 3
            assert len(uploads) == 1
            assert mlflow.active_run().info.run_id == outer.info.run_id
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_blocked_tracking_never_blocks_simulation_or_spawns_per_execution_threads(client, monkeypatch):
    import threading

    entered = threading.Event()
    release = threading.Event()

    def blocked(**kwargs):
        entered.set()
        release.wait(timeout=5)
        return client

    monkeypatch.setattr(_monitor.mlflow.tracking, "MlflowClient", blocked)
    sim = Simulation()
    sim.setup(sim.cfg)
    assert entered.wait(timeout=1)
    started = time.monotonic()
    try:
        assert sim() == 42
        assert sim() == 42
        assert time.monotonic() - started < 0.5
        assert len([t for t in threading.enumerate() if t.name == "adept-monitor"]) == 1
        assert sim._monitor.stop.is_set()
    finally:
        release.set()
        _monitor._messages.join()


def test_artifact_retry_exhaustion_reports_false(monkeypatch):
    from adept import utils

    def fail(directory):
        raise OSError("upload failed")

    monkeypatch.setattr(utils.mlflow, "log_artifacts", fail)
    assert utils.robust_log_artifacts("unused", retries=2, delay=0) is False
    monkeypatch.setattr(utils.mlflow, "log_artifacts", lambda directory: None)
    assert utils.robust_log_artifacts("unused", retries=2, delay=0) is True


def test_standalone_creates_closed_singleton_launch(client, monkeypatch):
    monkeypatch.delenv("ADEPT_MONITOR_LAUNCH_ID", raising=False)
    sim = Simulation()
    sim.setup(sim.cfg)
    sim()
    sim()
    assert len(client.runs) == 4
    executions = [r for r in client.runs if r["monitor.record_type"] == "execution"]
    launches = [r for r in client.runs if r["monitor.record_type"] == "launch"]
    assert {r["monitor.launch_id"] for r in executions} == {r["monitor.launch_id"] for r in launches}
    assert len({r["monitor.launch_id"] for r in executions}) == 2
    assert all(r["monitor.registration_closed"] == "true" and r["monitor.expected_executions"] == "1" for r in launches)


def _publisher_subprocess(script, timeout=12):
    import subprocess
    import sys

    # Load the actual publisher without importing numerical libraries: the test
    # exercises process/thread lifetime rather than JAX initialization.
    prefix = f"""
import importlib.util, sys, types, time, os
from pathlib import Path
sys.modules['adept'] = types.ModuleType('adept')
entities = types.ModuleType('mlflow.entities')
class RunTag:
    def __init__(self, key, value): self.key, self.value = key, value
entities.RunTag = RunTag
sys.modules['mlflow.entities'] = entities
tracking = types.SimpleNamespace(get_tracking_uri=lambda: 'unused')
sys.modules['adept.patched_mlflow'] = tracking
spec = importlib.util.spec_from_file_location('adept._monitor', {_monitor.__file__!r})
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
os.environ['ADEPT_MONITOR_OWNER'] = 'all'
os.environ['ADEPT_MONITOR_LAUNCH_ID'] = 'launch'
"""
    return subprocess.run(
        [sys.executable, "-c", prefix + script], timeout=timeout, capture_output=True, text=True, check=False
    )


def test_normal_interpreter_exit_flushes_delayed_terminal_batch(tmp_path):
    output = tmp_path / "terminal.txt"
    result = _publisher_subprocess(f"""
class Client:
    def get_experiment_by_name(self, name):
        time.sleep(.15)
        return types.SimpleNamespace(experiment_id='1')
    def create_run(self, experiment, tags):
        time.sleep(.15)
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id='execution'))
    def log_batch(self, run_id, tags):
        time.sleep(.15)
        values = {{tag.key: tag.value for tag in tags}}
        if values.get('monitor.outcome') == 'succeeded':
            Path({str(output)!r}).write_text(str(values))
    def set_terminated(self, *args, **kwargs): time.sleep(.15)
tracking.tracking = types.SimpleNamespace(MlflowClient=lambda **kwargs: Client())
execution = monitor.ExecutionMonitor()
execution.phase('executing')
execution.finish('succeeded')
""")
    assert result.returncode == 0, result.stderr
    assert "monitor.completed_at" in output.read_text()
    assert "succeeded" in output.read_text()


def test_unavailable_tracking_has_bounded_interpreter_shutdown():
    started = time.monotonic()
    result = _publisher_subprocess("""
def blocked(**kwargs):
    time.sleep(30)
tracking.tracking = types.SimpleNamespace(MlflowClient=blocked)
execution = monitor.ExecutionMonitor()
execution.finish('failed')
""")
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 8


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="requires POSIX fork")
def test_fork_restarts_publisher_with_fresh_queue(tmp_path):
    output = tmp_path / "child.txt"
    result = _publisher_subprocess(f"""
monitor._submit(lambda: None)
monitor._messages.join()
pid = os.fork()
if pid == 0:
    monitor._submit(lambda: Path({str(output)!r}).write_text('published'))
    monitor._messages.join()
    os._exit(0)
_, status = os.waitpid(pid, 0)
assert status == 0
""")
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "published"


def test_ownership_for_independent_slurm_tasks_and_distributed_runs(monkeypatch):
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK", "ADEPT_MONITOR_OWNER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "9")
    assert _monitor.reporting_owner()  # independent Slurm task/Parsl manager
    monkeypatch.setenv("ADEPT_MONITOR_OWNER", "rank0")
    assert not _monitor.reporting_owner()
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("ADEPT_MONITOR_OWNER", "all")
    assert _monitor.reporting_owner()  # explicitly independent workers
    monkeypatch.setenv("ADEPT_MONITOR_OWNER", "none")
    assert not _monitor.reporting_owner()
