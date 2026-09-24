"""Tracked, reproducible FARSIGHT/Eulerian comparison runs.

Invoke with ``python -m examples.farsight_comparison.run --help``.  The campaign
driver must pre-create the experiment before dispatching concurrent workers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from adept import Artifact, MetricEvent, MLflowArtifactSink, MLflowTracker, RunRequest, RunStatus


class S3VerifiedArtifactSink(MLflowArtifactSink):
    """Use boto3 streaming reads, not MLflow downloads, to verify S3 artifacts."""

    def __init__(self, *, s3_client=None, **kwargs):
        super().__init__(**kwargs)
        self._s3_client = s3_client

    def verify(self, handle, receipt):
        parsed = urlparse(receipt.uri)
        if parsed.scheme != "s3":
            return super().verify(handle, receipt)
        if handle.backend != "mlflow":
            raise ValueError("MLflowArtifactSink requires an MLflow run handle")
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3")
        bucket, key = parsed.netloc, unquote(parsed.path.lstrip("/"))
        keys = [key]
        if receipt.is_directory:
            prefix = key.rstrip("/") + "/"
            pages = self._s3_client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            keys = sorted(
                obj["Key"] for page in pages for obj in page.get("Contents", []) if not obj["Key"].endswith("/")
            )
        digest, size = hashlib.sha256(), 0
        for object_key in keys:
            if receipt.is_directory:
                relative = object_key[len(prefix) :].encode()
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
            body = self._s3_client.get_object(Bucket=bucket, Key=object_key)["Body"]
            try:
                while chunk := body.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
            finally:
                body.close()
        if size != receipt.size_bytes or digest.hexdigest() != receipt.sha256:
            raise OSError(f"S3 artifact verification failed: {receipt.path}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _services(tracking_uri=None):
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be configured; comparison runs never silently disable tracking")
    kwargs = {"tracking_uri": uri}
    if urlparse(uri).scheme in {"http", "https"}:
        kwargs["rest_api_path_prefix"] = "/ajax-api/2.0"
    return MLflowTracker(**kwargs), S3VerifiedArtifactSink(**kwargs)


def ensure_experiment(name="farsight-comparison", *, tracking_uri=None) -> str:
    """Call once in the driver before launching any concurrent run workers."""
    tracker, _ = _services(tracking_uri)
    return tracker._get_or_create_experiment(name)


def _git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _provenance(directory: Path):
    """Capture only scoped source and allowlisted runtime facts, never the env."""
    import jax

    repo = Path(__file__).resolve().parents[2]
    versions = {}
    for package in ("adept", "jax", "jaxlib", "equinox", "diffrax", "interpax", "numpy", "mlflow", "boto3"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    hashes = {}
    session_state = {}
    session_root = os.environ.get("EC_SESSION_ROOT")
    if session_root:
        state = Path(session_root) / "source-state"
        if state.is_file():
            session_state = dict(
                line.split("=", 1)
                for line in state.read_text().splitlines()
                if line.partition("=")[0] in {"version", "base_sha", "dirty", "workspace_fingerprint", "synced_at"}
            )
    with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        # This bounded source snapshot includes untracked comparison code, which
        # git diff alone would omit in an isolated development session.
        for path in sorted(Path(__file__).parent.iterdir()):
            if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml"}:
                relative = path.relative_to(repo).as_posix()
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.write(path, relative)
        patch = _git(repo, "diff", "HEAD", "--", "adept", "examples/farsight_comparison")
        if patch is None and os.environ.get("EC_SESSION_WORKDIR"):
            session_patch = Path(os.environ["EC_SESSION_WORKDIR"]) / "source.patch"
            if session_patch.is_file():
                # The session patch can contain unrelated work. Retain only the
                # solver and comparison source that this run actually uses.
                patch = "".join(
                    block
                    for block in re.split(r"(?=^diff --git )", session_patch.read_text(), flags=re.MULTILINE)
                    if block.startswith(("diff --git a/adept/", "diff --git a/examples/farsight_comparison/"))
                )
        archive.writestr("source.patch", patch or "")
        if session_state:
            archive.writestr("session-source-state.json", json.dumps(session_state, indent=2))
    info = {
        "git_commit": _git(repo, "rev-parse", "HEAD") or session_state.get("base_sha"),
        "git_status_scoped": _git(repo, "status", "--porcelain", "--", "adept", "examples/farsight_comparison"),
        "session_source_state": session_state,
        "source_sha256": hashes,
        "source_archive_sha256": hashlib.sha256((directory / "source.zip").read_bytes()).hexdigest(),
        "versions": versions,
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "devices": [{"platform": dev.platform, "kind": dev.device_kind, "id": dev.id} for dev in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }
    _write_json(directory / "provenance.json", info)
    return info


def _execute_timed(call, *, benchmark, steps, directory):
    """Every call starts from the same closed-over initial state; never advance it."""
    import jax

    durations = []
    for _ in range(3 if benchmark else 1):
        started = time.perf_counter()
        result = call()
        jax.block_until_ready(result)
        durations.append(time.perf_counter() - started)
    memory = []
    for device in jax.devices():
        statistics = device.memory_stats() or {}
        memory.append(
            {
                "platform": device.platform,
                "kind": device.device_kind,
                "id": device.id,
                **{
                    key: int(statistics[key])
                    for key in ("bytes_limit", "peak_bytes_in_use", "bytes_in_use")
                    if key in statistics
                },
            }
        )
    timing = {
        "steps_per_execution": steps,
        "executions": len(durations),
        "first_execution_seconds_including_compile": durations[0],
        "warm_execution_seconds": durations[1:],
        "warm_seconds_per_step": [value / steps for value in durations[1:]],
        "identical_initial_state_each_execution": True,
        "device_memory_after_executions": memory,
        "memory_note": (
            "Device allocator statistics; peak is the process high-water mark, not an isolated single-run measurement."
        ),
        "note": (
            "Compilation is included only in the first execution; preparation, analysis and artifact IO are excluded."
        ),
    }
    _write_json(directory / "timing.json", timing)
    print(json.dumps({"timing": timing}), flush=True)
    return result


def farsight_config(case, *, field_solver="direct", chunk_size=64):
    """Build the fixed-panel configuration for the common analytic case."""
    return case.to_farsight_config(field_solver=field_solver, chunk_size=chunk_size)


def _run_farsight(case, config, directory, handle, tracker, sink, benchmark):
    import jax

    from adept import SimulationSpec, run_prepared, solver_registry
    from adept.core.runtime import _default_execute
    from adept.farsight1d.__main__ import FileAnalyzer

    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=42)
    client = tracker._get_client()
    client.set_tag(handle.run_id, "adept.structural_fingerprint", prepared.manifest.structural_fingerprint)
    prepared = replace(prepared, analyzer=FileAnalyzer(prepared.analyzer, directory))

    def execute(prepared, key):
        return _execute_timed(
            lambda: _default_execute(prepared, key),
            benchmark=benchmark,
            steps=round(case.tmax / case.dt),
            directory=directory,
        )

    completed = run_prepared(
        prepared,
        key=jax.random.key(42),
        request=RunRequest(run_id=handle.run_id),
        tracker=tracker,
        artifact_sink=sink,
        execute=execute,
    )
    return {key: value for event in completed.report.metrics for key, value in event.values.items()}


def _run_eulerian(case, config, directory, handle, tracker, sink, benchmark):
    from adept import ergoExo
    from adept import patched_mlflow as mlflow
    from examples.farsight_comparison.eulerian import BenchmarkVlasov1D

    # Legacy ergoExo uses fluent tracking; the early-created run makes both host
    # lifecycles refer to exactly one run, even if setup fails.
    mlflow.set_tracking_uri(tracker.tracking_uri)

    class TimedExo(ergoExo):
        def _execute_simulation(self, modules, args):
            execute = super()._execute_simulation
            return _execute_timed(
                lambda: execute(modules, args),
                benchmark=benchmark,
                steps=round(case.tmax / case.dt),
                directory=directory,
            )

    exo = TimedExo(mlflow_run_id=handle.run_id)
    modules = exo.setup(cfg=config, adept_module=BenchmarkVlasov1D)
    _, post, _ = exo(modules)
    return post.get("metrics", {})


def run_one(task: dict) -> dict:
    """A serializable Parsl worker entry point; raises on solve or logging failure."""
    host = socket.gethostname()
    if os.environ.get("NERSC_HOST") and not host.startswith("nid"):
        raise RuntimeError(f"Refusing numerical work on NERSC non-compute host {host!r}")
    import jax

    jax.config.update("jax_enable_x64", True)
    from examples.farsight_comparison.cases import get_case

    solver = task["solver"]
    if solver not in {"farsight", "eulerian-softened", "eulerian-poisson"}:
        raise ValueError(f"Unknown comparison solver {solver!r}")
    case = get_case(
        task["case"],
        **{key: task[key] for key in ("nx", "nv", "tmax", "dt", "frame_dt", "epsilon") if task.get(key) is not None},
    )
    directory = Path(task["output"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    phase = task.get("phase", "pilot")
    experiment = task.get("experiment", "farsight-comparison")
    name = task.get("name") or f"{case.name}-{solver}-{phase}-{case.nx}x{case.nv}"
    field_model = "poisson" if solver == "eulerian-poisson" else "farsight-softened"
    if solver == "farsight":
        config = farsight_config(
            case, field_solver=task.get("field_solver", "direct"), chunk_size=task.get("chunk_size", 64)
        )
    else:
        from examples.farsight_comparison.eulerian import build_eulerian_config

        config = build_eulerian_config(
            case,
            field_model=field_model,
            output_dir=directory,
            experiment=experiment,
            run_name=name,
        )
    _write_json(directory / "case.json", asdict(case))
    _write_json(directory / "config.json", config)
    tracker, sink = _services(task.get("tracking_uri"))
    client = tracker._get_client()
    if client.get_experiment_by_name(experiment) is None:
        raise RuntimeError(f"Experiment {experiment!r} must be pre-created in the driver")
    request = RunRequest(
        experiment=experiment,
        name=name,
        tags={
            "comparison.case": case.name,
            "comparison.solver": solver,
            "comparison.phase": phase,
            "comparison.field_model": field_model,
            "comparison.benchmark": str(bool(task.get("benchmark", False))),
            "comparison.eulerian_x": "spectral",
            "comparison.eulerian_v": "cubic-spline",
        },
    )
    tracker.preflight(request)
    sink.preflight()
    handle = tracker.start(request)
    summary = {
        "run_id": handle.run_id,
        "experiment": experiment,
        "name": name,
        "case": case.name,
        "solver": solver,
        "field_model": field_model,
        "phase": phase,
        "status": "RUNNING",
        "output": str(directory),
    }
    _write_json(directory / "run.json", summary)
    print(json.dumps(summary), flush=True)
    try:
        sink.validate(handle)
        provenance = _provenance(directory)
        client.set_tag(handle.run_id, "comparison.source_sha256", provenance["source_archive_sha256"])
        if provenance["git_commit"]:
            client.set_tag(handle.run_id, "mlflow.source.git.commit", provenance["git_commit"])
        for key, value in asdict(case).items():
            client.log_param(handle.run_id, f"comparison.{key}", value)
        for file in ("run.json", "case.json", "config.json", "provenance.json", "source.zip"):
            receipt = sink.put(handle, Artifact(directory / file, artifact_path="comparison"))
            sink.verify(handle, receipt)
        started = time.perf_counter()
        run = _run_farsight if solver == "farsight" else _run_eulerian
        metrics = run(case, config, directory, handle, tracker, sink, bool(task.get("benchmark", False)))
        summary.update(status="FINISHED", total_time_seconds=time.perf_counter() - started, metrics=metrics)
        summary["timing"] = json.loads((directory / "timing.json").read_text())
        timing_metrics = {
            "comparison.first_execution_seconds_including_compile": summary["timing"][
                "first_execution_seconds_including_compile"
            ]
        }
        for index, duration in enumerate(summary["timing"]["warm_seconds_per_step"], 1):
            timing_metrics[f"comparison.warm_{index}_seconds_per_step"] = duration
        tracker.log_metrics(handle, [MetricEvent(timing_metrics)])
        _write_json(directory / "run.json", summary)
        # Explicitly verify the common outputs for both runtime paths; legacy
        # ergoExo's retry helper alone does not raise when every upload fails.
        for path in sorted(directory.iterdir()):
            if path.is_file():
                receipt = sink.put(handle, Artifact(path, artifact_path="comparison"))
                sink.verify(handle, receipt)
        tracker.finish(handle, RunStatus.FINISHED)
    except Exception as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        _write_json(directory / "run.json", summary)
        try:
            receipt = sink.put(handle, Artifact(directory / "run.json", artifact_path="comparison"))
            sink.verify(handle, receipt)
        except Exception as tracking_error:  # noqa: BLE001 - retain the original solve failure and both causes
            error.add_note(f"Failure artifact upload also failed: {type(tracking_error).__name__}: {tracking_error}")
        try:
            tracker.finish(handle, RunStatus.FAILED, error=summary["error"])
        except Exception as tracking_error:  # noqa: BLE001 - retain the original solve failure and both causes
            error.add_note(f"Failure status update also failed: {type(tracking_error).__name__}: {tracking_error}")
        raise
    print(json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("two-stream", "nlepw"))
    parser.add_argument("--solver", required=True, choices=("farsight", "eulerian-softened", "eulerian-poisson"))
    for name in ("nx", "nv"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("tmax", "dt", "frame-dt", "epsilon"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--experiment", default="farsight-comparison")
    parser.add_argument("--name")
    parser.add_argument("--phase", default="pilot", choices=("pilot", "movie"))
    parser.add_argument(
        "--benchmark", action="store_true", help="Time compilation and two warmed identical-initial-state executions"
    )
    parser.add_argument("--field-solver", choices=("direct", "treecode"), default="direct")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--tracking-uri", help="Defaults to MLFLOW_TRACKING_URI")
    args = vars(parser.parse_args(argv))
    ensure_experiment(args["experiment"], tracking_uri=args["tracking_uri"])
    run_one(args)


if __name__ == "__main__":
    main()
