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


def build_parser():
    from examples.farsight_comparison.run import add_farsight_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New campaign output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--task-file", type=Path, help="JSON list, or {defaults: {...}, tasks: [...]}; task order is retained"
    )
    mode.add_argument(
        "--amr-pair",
        action="store_true",
        help=(
            "For each case run identical AMR configs with direct then treecode, followed by selected Eulerian controls"
        ),
    )
    parser.add_argument("--cases", nargs="+", choices=["two-stream", "nlepw"], default=["two-stream", "nlepw"])
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=["farsight", "eulerian-softened", "eulerian-poisson"],
        default=["farsight", "eulerian-softened", "eulerian-poisson"],
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nv", type=int, default=128)
    parser.add_argument(
        "--eulerian-nx", type=int, help="Independent Eulerian resolution in generated scans (default: nx)"
    )
    parser.add_argument(
        "--eulerian-nv", type=int, help="Independent Eulerian resolution in generated scans (default: nv)"
    )
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--frame-dt", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--experiment", default="farsight-comparison")
    parser.add_argument("--phase", default="movie", help="Tracking phase label, e.g. pilot, movie, or amr-pilot")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tracking-uri")
    add_farsight_arguments(parser)
    return parser


def build_tasks(options, output):
    """Resolve ordered, uniquely located run tasks without submitting anything."""
    from examples.farsight_comparison.cases import FARSIGHT_OPTION_DEFAULTS, get_case
    from examples.farsight_comparison.run import farsight_config, method_label

    run_keys = {
        "case",
        "solver",
        "nx",
        "nv",
        "tmax",
        "dt",
        "frame_dt",
        "epsilon",
        "experiment",
        "phase",
        "name",
        "benchmark",
        "tracking_uri",
        *FARSIGHT_OPTION_DEFAULTS,
    }
    shared = {key: value for key, value in options.items() if key in run_keys}
    entries = []
    if options.get("task_file"):
        document = json.loads(Path(options["task_file"]).read_text())
        defaults = {}
        if isinstance(document, dict):
            if set(document) - {"defaults", "tasks"}:
                raise ValueError("Task file accepts only 'defaults' and 'tasks' at its top level")
            defaults = document.get("defaults", {})
            document = document.get("tasks")
        if not isinstance(defaults, dict) or not isinstance(document, list):
            raise TypeError("Task file requires an object of defaults and a list of task objects")
        if not document:
            raise ValueError("Task file must provide a nonempty list of task objects")
        for task in document:
            if not isinstance(task, dict):
                raise TypeError("Every task must be an object")
            entry = {"solver": "farsight", **shared, **defaults, **task}
            unknown = set(entry) - run_keys - {"output"}
            if unknown:
                raise ValueError(f"Unknown task options: {sorted(unknown)}")
            entries.append(entry)
    else:
        for case in options["cases"]:
            if options.get("amr_pair"):
                entries.extend(
                    {**shared, "case": case, "solver": "farsight", "amr": True, "field_solver": field}
                    for field in ("direct", "treecode")
                )
            for solver in options["solvers"]:
                if options.get("amr_pair") and solver == "farsight":
                    continue
                entry = {**shared, "case": case, "solver": solver}
                if solver.startswith("eulerian"):
                    for axis in ("nx", "nv"):
                        if options.get(f"eulerian_{axis}") is not None:
                            entry[axis] = options[f"eulerian_{axis}"]
                entries.append(entry)
    seen_outputs = set()
    output = Path(output).resolve()
    for index, task in enumerate(entries):
        if task.get("solver") not in {"farsight", "eulerian-softened", "eulerian-poisson"}:
            raise ValueError(f"Unknown task solver {task.get('solver')!r}")
        if not isinstance(task.get("phase", "pilot"), str) or not task.get("phase", "pilot").strip():
            raise ValueError("Task phase must be a nonempty string")
        case = get_case(
            task.get("case"),
            **{
                key: task[key] for key in ("nx", "nv", "tmax", "dt", "frame_dt", "epsilon") if task.get(key) is not None
            },
        )
        if task["solver"] == "farsight":
            farsight_config(case, **{key: task[key] for key in FARSIGHT_OPTION_DEFAULTS if task.get(key) is not None})
        method = method_label(task)
        relative = Path(task.pop("output", f"{index:02d}-{case.name}-{method}"))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
            raise ValueError("Per-task output must be a relative subdirectory within the campaign output")
        destination = (output / relative).resolve()
        if not destination.is_relative_to(output) or destination in seen_outputs:
            raise ValueError("Every task requires a unique campaign-local output directory")
        seen_outputs.add(destination)
        task["output"] = str(destination)
        task.setdefault("name", f"{case.name}-{method}-{task.get('phase', 'pilot')}-{case.nx}x{case.nv}")
    return entries


def main(argv=None):
    args = build_parser().parse_args(argv)
    host = require_compute_node()
    output = args.output.expanduser().resolve()
    tasks = build_tasks(vars(args), output)
    output.mkdir(parents=True, exist_ok=False)

    # Precreate/resolve the experiment before submitting any workers. Creation
    # races can leave an experiment with a broken artifact root on our server.
    from examples.farsight_comparison.run import ensure_experiment

    for experiment, tracking_uri in {(task["experiment"], task.get("tracking_uri")) for task in tasks}:
        ensure_experiment(experiment, tracking_uri=tracking_uri)

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
            except Exception as error:  # Persist each failed future, then exit nonzero.
                summary = {"case": task["case"], "solver": task["solver"], "status": "FAILED", "error": str(error)}
                failures.append(summary)
            summaries.append(summary)
            (output / "summaries.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n")
            print(json.dumps({"event": "task_finished", **summary}, default=str), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} comparison task(s) failed; see summaries.json")


if __name__ == "__main__":
    main()
