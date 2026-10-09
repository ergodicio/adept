"""Run a list of deck comparisons concurrently -- one worker per GPU on a node, or N workers
on a CPU host -- each job a ``python -m adept._lpse2d.parity run`` subprocess.

A job file is YAML::

    defaults:              # optional, merged under every job
      experiment: lpse-parity
      tags: {session: plan2-run-1}
    jobs:
      - deck: test_010                       # shipped deck name or a path to lpse.parms
        run: test_010/n1-seed/2              # MLflow run name (plan 2: <deck>/<item>/<seed>)
        overrides: {terms: {epw: {source: {noise_seed: 2}}}}
        windows: 0.3-0.9,0.9-1.5             # optional, else DEFAULT_WINDOWS[deck]
        tags: {item: N.1, variable: seed}    # optional MLflow tags
        no_log: false                        # optional: compare but do not log the comparison

The driver writes one log per job under ``--log-dir`` (``<slug>.log``, the slug being the
run name with ``/`` -> ``--``), the run's ``config.yaml`` / ``series.nc`` / ``fields.nc`` under
``--out-root/<slug>/`` and a ``summary.json`` next to the logs with the exit code, wall time,
MLflow run id and every ``key: value`` metric line the run printed. ``--skip-done`` skips a
job whose summary entry already has ``rc == 0`` (idempotent re-launch after an interruption).

With ``--gpus 0,1,2,3`` each worker pins ``CUDA_VISIBLE_DEVICES`` to one device; without it
the workers share whatever devices the process sees (a CPU host).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from .harness import merge_overrides

METRIC_LINE = re.compile(r"^  ([A-Za-z0-9_.]+): (\S+)$")
RUN_ID_LINE = re.compile(r"^mlflow run ([0-9a-f]{32})")


def slug(run_name: str) -> str:
    return run_name.replace("/", "--").replace(" ", "_")


def load_jobs(path: str | Path) -> list[dict]:
    """The job list of a batch file with ``defaults`` merged under each job (a job's own
    keys win; ``overrides`` and ``tags`` merge recursively)."""
    import yaml

    with open(path) as fi:
        spec = yaml.safe_load(fi) or {}
    defaults = spec.get("defaults") or {}
    jobs = []
    for raw in spec.get("jobs") or []:
        job = json.loads(json.dumps(defaults))  # deep copy
        for k, v in raw.items():
            if k in ("overrides", "tags") and isinstance(v, dict) and isinstance(job.get(k), dict):
                merge_overrides(job[k], v)
            else:
                job[k] = v
        if "deck" not in job or "run" not in job:
            raise ValueError(f"every job needs 'deck' and 'run': {raw}")
        jobs.append(job)
    return jobs


def job_command(job: dict, out_root: str | Path | None, python: str = sys.executable) -> list[str]:
    """The ``parity run`` argument vector for one job."""
    cmd = [python, "-m", "adept._lpse2d.parity", "run", str(job["deck"]), "--run", str(job["run"])]
    if job.get("overrides"):
        cmd += ["--overrides", json.dumps(job["overrides"])]
    if job.get("windows"):
        cmd += ["--windows", str(job["windows"])]
    if job.get("experiment"):
        cmd += ["--experiment", str(job["experiment"])]
    for k, v in (job.get("tags") or {}).items():
        cmd += ["--tag", f"{k}={v}"]
    if job.get("no_log"):
        cmd.append("--no-log")
    if job.get("full"):
        cmd.append("--full")
    if out_root is not None:
        cmd += ["--out", str(Path(out_root) / slug(job["run"]))]
    return cmd


def parse_log(path: Path) -> dict:
    """The MLflow run id and the printed metrics of a finished job log."""
    out: dict = {"run_id": None, "metrics": {}}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = RUN_ID_LINE.match(line)
        if m:
            out["run_id"] = m.group(1)
            continue
        m = METRIC_LINE.match(line)
        if m:
            try:
                out["metrics"][m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def run_batch(
    jobs: list[dict],
    log_dir: str | Path,
    out_root: str | Path | None = None,
    workers: int = 1,
    gpus: list[str] | None = None,
    skip_done: bool = False,
    dry_run: bool = False,
    python: str = sys.executable,
    env: dict | None = None,
) -> dict[str, dict]:
    """Run ``jobs`` on ``workers`` concurrent subprocesses; returns the summary
    ``{run name: {rc, seconds, run_id, metrics, log}}`` (also written to ``summary.json``)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.json"
    summary: dict[str, dict] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    lock = threading.Lock()
    todo: queue.Queue = queue.Queue()
    for job in jobs:
        name = str(job["run"])
        if skip_done and summary.get(name, {}).get("rc") == 0:
            print(f"[batch] skip (done): {name}", flush=True)
            continue
        todo.put(job)
    if gpus:
        workers = max(workers, len(gpus))

    def worker(index: int):
        while True:
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            name = str(job["run"])
            cmd = job_command(job, out_root, python)
            job_env = dict(os.environ if env is None else env)
            if gpus:
                job_env["CUDA_VISIBLE_DEVICES"] = str(gpus[index % len(gpus)])
            log = log_dir / f"{slug(name)}.log"
            where = f"gpu {job_env['CUDA_VISIBLE_DEVICES']}" if gpus else f"worker {index}"
            print(f"[batch] start {name} on {where} -> {log}", flush=True)
            if dry_run:
                print("        " + " ".join(shlex.quote(c) for c in cmd), flush=True)
                rc, seconds = 0, 0.0
            else:
                t0 = time.time()
                with open(log, "w") as fo:
                    fo.write("# " + " ".join(shlex.quote(c) for c in cmd) + "\n")
                    fo.flush()
                    try:
                        rc = subprocess.call(cmd, stdout=fo, stderr=subprocess.STDOUT, env=job_env)
                    except OSError as exc:  # interpreter missing / not executable: record, do not die
                        fo.write(f"[batch] could not start: {exc}\n")
                        rc = -1
                seconds = time.time() - t0
            parsed = parse_log(log) if not dry_run else {"run_id": None, "metrics": {}}
            entry = {"rc": rc, "seconds": round(seconds, 1), "log": str(log), **parsed, "deck": job["deck"]}
            with lock:
                summary[name] = entry
                summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
            status = "OK" if rc == 0 else f"FAILED rc={rc}"
            ratios = {k: round(v, 3) for k, v in parsed["metrics"].items() if "ratio" in k}
            print(f"[batch] {status} {name} {seconds / 60:.1f} min run={parsed['run_id']} {ratios}", flush=True)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, workers))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return summary


def print_summary(summary: dict[str, dict]):
    for name, e in summary.items():
        ratios = " ".join(
            f"{k.replace('_energy_growth_ratio_', ':')}={v:.3f}" for k, v in e["metrics"].items() if "ratio" in k
        )
        print(f"{name:40s} rc={e['rc']} {e['seconds'] / 60:5.1f} min {str(e.get('run_id'))[:8]} {ratios}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adept._lpse2d.parity batch", description=__doc__)
    ap.add_argument("jobs", help="YAML job file")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-root", help="directory for the per-run config.yaml / series.nc / fields.nc")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", help="comma-separated CUDA device indices, one worker each")
    ap.add_argument("--only", help="comma-separated run names (or substrings) to run")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    jobs = load_jobs(args.jobs)
    if args.only:
        keys = args.only.split(",")
        jobs = [j for j in jobs if any(k in str(j["run"]) for k in keys)]
    gpus = args.gpus.split(",") if args.gpus else None
    summary = run_batch(
        jobs,
        args.log_dir,
        args.out_root,
        workers=args.workers,
        gpus=gpus,
        skip_done=args.skip_done,
        dry_run=args.dry_run,
    )
    print_summary({j["run"]: summary[j["run"]] for j in jobs if j["run"] in summary})
    return 0 if all(summary.get(j["run"], {}).get("rc") == 0 for j in jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
