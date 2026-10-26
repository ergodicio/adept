"""Run repeated strong-scaling measurements with an MPI launcher template.

Placements have the form nodes:ranks. For Slurm, pass an explicit job ID,
--overlap, --nodes={nodes}, --ntasks={ranks}, and
--ntasks-per-node={ranks_per_node}. This script never allocates resources.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--launcher", default="mpiexec -n {ranks}")
    parser.add_argument("--placements", nargs="+", default=["1:1", "1:2", "1:4"])
    parser.add_argument("--nx", type=int, default=32768)
    parser.add_argument("--nv", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-state", type=Path)
    parser.add_argument("--host-mpi", action="store_true")
    parser.add_argument("--expect-cuda", action="store_true")
    parser.add_argument("--velocity", choices=["natural", "cubic"], default="natural")
    parser.add_argument("--layout", choices=["transpose", "halos"], default="transpose")
    parser.add_argument("--integrator", choices=["strang", "sixth"], default="strang")
    parser.add_argument("--cells", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--v-grid", type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--length", type=float, default=12.566370614359172)
    parser.add_argument("--vmax", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--max-mass-change",
        type=float,
        default=1e-8,
        help="negative records conservation diagnostically without an absolute cutoff",
    )
    args = parser.parse_args()
    placements = []
    for placement in args.placements:
        nodes, ranks = map(int, placement.split(":"))
        if nodes < 1 or ranks < 1 or ranks % nodes:
            parser.error("placements must have positive nodes/ranks and equal ranks per node")
        placements.append((nodes, ranks))
    if args.repeats < 1:
        parser.error("repeats must be positive")
    provenance = args.source_state.read_text() if args.source_state else None
    # Exclusive create prevents accidentally replacing an earlier measurement.
    with args.output.open("x") as output:
        for nodes, ranks in placements:
            for repeat in range(args.repeats):
                command = [
                    *shlex.split(args.launcher.format(nodes=nodes, ranks=ranks, ranks_per_node=ranks // nodes)),
                    str(args.binary.resolve()),
                    "--nx",
                    str(args.nx),
                    "--nv",
                    str(args.nv),
                    "--steps",
                    str(args.steps),
                    "--warmup",
                    str(args.warmup),
                    "--mode",
                    "landau",
                ]
                if args.host_mpi:
                    command.append("--host-mpi")
                command.extend(
                    [
                        "--velocity",
                        args.velocity,
                        "--layout",
                        args.layout,
                        "--integrator",
                        args.integrator,
                        "--nu",
                        str(args.nu),
                        "--length",
                        str(args.length),
                        "--vmax",
                        str(args.vmax),
                        "--dt",
                        str(args.dt),
                    ]
                )
                if args.cells:
                    command.append("--cells")
                for key in ["input", "v_grid", "background"]:
                    value = getattr(args, key)
                    if value:
                        command.extend(["--" + key.replace("_", "-"), str(value)])
                completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
                if completed.returncode:
                    raise RuntimeError(f"{' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
                row = json.loads(completed.stdout.strip().splitlines()[-1])
                if row["nodes"] != nodes or row["ranks"] != ranks:
                    raise RuntimeError(f"placement mismatch: {row}")
                if args.expect_cuda and row["backend"] != "cuda":
                    raise RuntimeError("expected the CUDA backend")
                if (args.max_mass_change >= 0 and abs(row["mass_relative_change"]) > args.max_mass_change) or not row[
                    "finite"
                ]:
                    raise RuntimeError(f"invalid physical output: {row}")
                row.update(repeat=repeat, command=command, rank_log=completed.stderr, source_state=provenance)
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {key: row[key] for key in ["nodes", "ranks", "repeat", "step_ms", "spline_ms", "mpi_ms"]}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
