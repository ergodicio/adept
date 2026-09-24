"""One-worker, one-GPU comparison campaign using an existing allocation.

Run this driver ON a compute node (e.g. session.sh exec). LocalProvider does
not submit Slurm jobs or place work into an allocation by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path


def require_compute_node():
    host = socket.gethostname()
    if os.environ.get("NERSC_HOST") and not host.startswith("nid"):
        raise RuntimeError(f"Refusing LocalProvider execution on NERSC non-compute host {host!r}")
    return host


def _task(task):
    import gc
    import os
    import socket

    host = socket.gethostname()
    if os.environ.get("NERSC_HOST") and not host.startswith("nid"):
        raise RuntimeError(f"Refusing worker execution on NERSC non-compute host {host!r}")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from examples.farsight_comparison.run import run_one

    try:
        return run_one(task)
    finally:
        import jax

        jax.clear_caches()
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New campaign output directory")
    parser.add_argument("--cases", nargs="+", choices=["two-stream", "nlepw"], default=["two-stream", "nlepw"])
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=["farsight", "eulerian-softened", "eulerian-poisson"],
        default=["farsight", "eulerian-softened", "eulerian-poisson"],
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nv", type=int, default=128)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--frame-dt", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--experiment", default="farsight-comparison")
    parser.add_argument("--phase", choices=["pilot", "movie"], default="movie")
    args = parser.parse_args(argv)
    host = require_compute_node()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Precreate/resolve the experiment before submitting any workers. Creation
    # races can leave an experiment with a broken artifact root on our server.
    from adept import patched_mlflow as mlflow

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        client.create_experiment(args.experiment)

    import parsl
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="comparison",
                max_workers_per_node=1,
                cores_per_worker=4,
                # Exactly one worker inherits Slurm's GPU visibility unchanged.
                # Replacing a physical UUID/index with "0" can escape that mapping.
                provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
            )
        ],
        run_dir=str(output / "runinfo"),
        retries=0,
        strategy="none",
    )
    tasks = []
    for case in args.cases:
        for solver in args.solvers:
            task = {k: v for k, v in vars(args).items() if k not in ("cases", "solvers", "output")}
            task.update(
                case=case,
                solver=solver,
                output=str(output / f"{case}-{solver}"),
                name=f"{case}-{solver}-{args.phase}-{args.nx}x{args.nv}",
            )
            tasks.append(task)
    (output / "tasks.json").write_text(json.dumps({"host": host, "tasks": tasks}, indent=2) + "\n")
    print(json.dumps({"event": "dispatch", "host": host, "count": len(tasks), "output": str(output)}), flush=True)
    summaries = []
    failures = []
    with parsl.load(config):
        run_task = parsl.python_app(_task, executors=["comparison"])
        futures = [(task, run_task(task)) for task in tasks]
        for task, future in futures:
            try:
                summary = future.result()
            except Exception as error:  # noqa: BLE001 - persist each failed future, then exit nonzero
                summary = {"case": task["case"], "solver": task["solver"], "status": "FAILED", "error": str(error)}
                failures.append(summary)
            summaries.append(summary)
            (output / "summaries.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n")
            print(json.dumps({"event": "task_finished", **summary}, default=str), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} comparison task(s) failed; see summaries.json")


if __name__ == "__main__":
    main()
