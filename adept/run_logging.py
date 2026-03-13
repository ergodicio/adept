import atexit
import os
import sys
import tempfile
import threading

from mlflow.tracking import MlflowClient

_multi_tee_stdout: "MultiTeeStream | None" = None
_multi_tee_stderr: "MultiTeeStream | None" = None


class MultiTeeStream:
    """Wraps a stream and writes every byte to it plus a list of open files."""

    def __init__(self, stream):
        self.stream = stream
        self._files: list = []
        self._lock = threading.Lock()

    def add_file(self, f):
        with self._lock:
            self._files.append(f)

    def remove_file(self, f):
        with self._lock:
            try:
                self._files.remove(f)
            except ValueError:
                pass

    def write(self, data):
        self.stream.write(data)
        with self._lock:
            for f in self._files:
                try:
                    f.write(data)
                    f.flush()
                except Exception:
                    pass

    def flush(self):
        self.stream.flush()
        with self._lock:
            for f in self._files:
                try:
                    f.flush()
                except Exception:
                    pass

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class AdeptLogger:
    """
    Captures stdout/stderr to a per-run log file and periodically uploads it
    to MLflow as an artifact.

    Usage is automatic when running through ergoExo — no user action required.
    The first ergoExo created in a process owns the MultiTeeStream setup.
    Nested ergoExos (mlflow_nested=True) add their own file to the shared
    multi-tee and get a log slice covering only their own setup+call window.
    """

    def __init__(self, upload_interval: int = 60):
        self.upload_interval = upload_interval
        self.run_ids: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_owner = False
        self._stopped = False
        self.log_path: str | None = None
        self._log_file = None

    def start(self):
        """Open log file and attach to the process-wide MultiTeeStream."""
        global _multi_tee_stdout, _multi_tee_stderr

        log_dir = os.environ.get("BASE_TEMPDIR") or tempfile.mkdtemp()
        self.log_path = os.path.join(log_dir, "run.log")
        self._log_file = open(self.log_path, "w", buffering=1)  # line-buffered

        if _multi_tee_stdout is None:
            _multi_tee_stdout = MultiTeeStream(sys.stdout)
            _multi_tee_stderr = MultiTeeStream(sys.stderr)
            sys.stdout = _multi_tee_stdout
            sys.stderr = _multi_tee_stderr
            self._is_owner = True
            atexit.register(self.stop)

        _multi_tee_stdout.add_file(self._log_file)
        _multi_tee_stderr.add_file(self._log_file)

    def attach(self, run_id: str):
        """Register a MLflow run_id and start background upload thread."""
        self.run_ids.append(run_id)
        if self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._upload_loop, daemon=True)
            self._thread.start()

    def _upload_loop(self):
        while not self._stop_event.wait(self.upload_interval):
            self._upload()

    def _upload(self):
        if not self.run_ids or self.log_path is None:
            return
        client = MlflowClient()
        for run_id in self.run_ids:
            try:
                client.log_artifact(run_id, self.log_path)
            except Exception:
                pass

    def stop(self):
        """Final upload, detach log file from multi-tee, optionally restore streams."""
        global _multi_tee_stdout, _multi_tee_stderr

        if self._stopped:
            return
        self._stopped = True

        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5)

        self._upload()

        if _multi_tee_stdout is not None and self._log_file is not None:
            _multi_tee_stdout.remove_file(self._log_file)
        if _multi_tee_stderr is not None and self._log_file is not None:
            _multi_tee_stderr.remove_file(self._log_file)

        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass

        if self._is_owner:
            if _multi_tee_stdout is not None:
                sys.stdout = _multi_tee_stdout.stream
            if _multi_tee_stderr is not None:
                sys.stderr = _multi_tee_stderr.stream
            _multi_tee_stdout = None
            _multi_tee_stderr = None
