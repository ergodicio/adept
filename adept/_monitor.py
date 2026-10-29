"""Best-effort execution telemetry, independent of MLflow's fluent run stack."""
# ruff: noqa: BLE001 -- telemetry errors must never interrupt simulation work.

import atexit
import functools
import logging
import os
import queue
import threading
import time
import uuid

from mlflow.entities import RunTag

from . import patched_mlflow as mlflow


def reporting_owner():
    """Explicit ownership for task farms versus a distributed single simulation."""
    owner = os.environ.get("ADEPT_MONITOR_OWNER", "auto")
    if owner == "all":
        return True
    if owner == "none":
        return False
    keys = ["OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"]
    if owner == "rank0":
        keys.append("SLURM_PROCID")
    return all(os.environ.get(key, "0") == "0" for key in keys)


# One daemon for the process: an unresponsive server cannot create a thread leak
# per execution, and no tracking I/O runs on the numerical application's thread.
_messages = queue.Queue(maxsize=1024)
_worker_lock = threading.Lock()
_worker = None
_active = set()


def _reset_after_fork():
    global _messages, _worker_lock, _worker, _active
    _messages = queue.Queue(maxsize=1024)
    _worker_lock = threading.Lock()
    _worker = None
    _active = set()


def _drain_at_exit(timeout=5):
    # Queue.join has no timeout. Wait on its condition with a shared deadline.
    deadline = time.monotonic() + timeout
    with _messages.all_tasks_done:
        while _messages.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            _messages.all_tasks_done.wait(timeout=remaining)


atexit.register(_drain_at_exit)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _publish_loop():
    while True:
        try:
            callback = _messages.get(timeout=1)
        except queue.Empty:
            callback = None
        if callback is not None:
            try:
                callback()
            except Exception:
                logging.getLogger(__name__).warning("ADEPT monitoring publisher unavailable")
            finally:
                _messages.task_done()
        now = time.monotonic()
        for monitor in tuple(_active):
            if monitor.done:
                _active.discard(monitor)
            elif now >= monitor.next_heartbeat:
                monitor._update(heartbeat_at=time.time())
                monitor.next_heartbeat = now + monitor.interval


def _submit(callback):
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = threading.Thread(target=_publish_loop, daemon=True, name="adept-monitor")
            _worker.start()
    try:
        _messages.put_nowait(callback)
    except queue.Full:
        pass  # Bounded memory even during tracking outages.


class ExecutionMonitor:
    def __init__(self, parent_run_id=None, interval=30):
        self.pid = os.getpid()
        self.warned = False
        self.done = False
        self.upload_failed = False
        self.run_id = None
        self.stop = threading.Event()
        self.interval = interval
        self.next_heartbeat = time.monotonic() + interval
        self.parent_run_id = parent_run_id
        # Capture launch context on the calling thread; workers may change env later.
        self.environment = dict(os.environ)
        self.tracking_uri = mlflow.get_tracking_uri()
        if reporting_owner():
            _submit(self._initialize)

    def _initialize(self):
        try:
            experiment = self.environment.get("ADEPT_MONITOR_EXPERIMENT", "continuum-run-monitor")
            self.client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            exp = self.client.get_experiment_by_name(experiment)
            experiment_id = exp.experiment_id if exp else self.client.create_experiment(experiment)
            tags = {
                "monitor.schema_version": "1",
                "monitor.record_type": "execution",
                "monitor.execution_id": str(uuid.uuid4()),
                "monitor.launch_id": self.environment.get("ADEPT_MONITOR_LAUNCH_ID") or str(uuid.uuid4()),
                "monitor.phase": "setup",
                "monitor.outcome": "running",
                "monitor.heartbeat_at": str(time.time()),
            }
            if self.parent_run_id:
                tags["monitor.scientific_run_id"] = self.parent_run_id
            if self.environment.get("SLURM_JOB_ID"):
                tags["monitor.scheduler"] = "slurm"
                for source, target in (
                    ("SLURM_JOB_ID", "job_id"),
                    ("SLURM_STEP_ID", "step_id"),
                    ("SLURM_CLUSTER_NAME", "cluster"),
                ):
                    if source in self.environment:
                        tags[f"monitor.{target}"] = self.environment[source]
            self.run_id = self.client.create_run(experiment_id, tags=tags).info.run_id
            if not self.environment.get("ADEPT_MONITOR_LAUNCH_ID"):
                launch_tags = {
                    "monitor.schema_version": "1",
                    "monitor.record_type": "launch",
                    "monitor.launch_id": tags["monitor.launch_id"],
                    "monitor.registration_closed": "true",
                    "monitor.expected_executions": "1",
                    "monitor.created_at": str(time.time()),
                    "monitor.phase": "running",
                }
                for key in ("monitor.scheduler", "monitor.job_id", "monitor.step_id", "monitor.cluster"):
                    if key in tags:
                        launch_tags[key] = tags[key]
                self.client.create_run(experiment_id, tags=launch_tags)
            _active.add(self)
        except Exception:
            self._warn()

    def _warn(self):
        if not self.warned:
            self.warned = True
            logging.getLogger(__name__).warning("ADEPT execution monitoring unavailable; simulation continues")

    def _update(self, **tags):
        if self.run_id is None:
            return
        try:
            self.client.log_batch(
                self.run_id, tags=[RunTag(f"monitor.{key}", str(value)) for key, value in tags.items()]
            )
        except Exception:
            self._warn()

    def update(self, **tags):
        _submit(lambda: self._update(**tags))

    def phase(self, phase):
        self.update(phase=phase, progress_at=time.time())

    def finish(self, outcome):
        self.stop.set()
        self.done = True
        completed_at = time.time()
        flushed = threading.Event()

        def publish():
            try:
                self._update(phase=outcome, completed_at=completed_at, heartbeat_at=completed_at, outcome=outcome)
                if self.run_id:
                    self.client.set_terminated(self.run_id, status="FINISHED" if outcome == "succeeded" else "FAILED")
            except Exception:
                self._warn()
            finally:
                _active.discard(self)
                flushed.set()

        _submit(publish)
        flushed.wait(timeout=0.1)


def monitored(method):
    """One record covers setup and the first call; subsequent calls get new records."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        setup = method.__name__ == "setup"
        monitor = getattr(self, "_monitor", None)
        if monitor is not None and monitor.pid != os.getpid():
            monitor = None
        if setup and monitor is not None and not monitor.done:
            monitor.finish("failed")
        if setup or monitor is None or monitor.done:
            monitor = self._monitor = ExecutionMonitor(self.mlflow_run_id)
        try:
            if not setup:
                monitor.phase("executing")
            result = method(self, *args, **kwargs)
            if setup:
                monitor.update(scientific_run_id=self.mlflow_run_id)
                monitor.phase("setup")
            else:
                monitor.finish("failed" if monitor.upload_failed else "succeeded")
            return result
        except BaseException:
            monitor.finish("failed")
            raise

    return wrapped
