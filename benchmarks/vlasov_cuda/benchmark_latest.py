"""Paired strong-scaling measurements after latest native-window checks pass."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--source-state", type=Path)
    parser.add_argument("--nx", nargs="+", type=int, default=[134400, 268800])
    parser.add_argument("--placements", nargs="+", default=["1:4", "2:8"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for nx in args.nx:
        reference = args.data / f"native-nx{nx}"
        checks = json.loads((args.data / f"comparison-nx{nx}-window/comparison.json").read_text())
        if not checks or not all(row["passed"] for row in checks):
            raise RuntimeError("native-state comparison must pass before timing")
        for placement in args.placements:
            nodes, ranks = map(int, placement.split(":"))
            for layout in ["transpose", "halos"]:
                if not any(
                    row["nodes"] == nodes and row["ranks"] == ranks and row["layout"] == layout and row["steps"] == 100
                    for row in checks
                ):
                    raise RuntimeError(f"missing 100-step native comparison: {nx}, {placement}, {layout}")
        metadata = json.loads((reference / "provenance.json").read_text())
        if metadata["nx"] != nx or metadata["nv"] != 4096 or metadata["dt"] != 0.02 or metadata["nu"] != 1e-12:
            raise RuntimeError("not the pinned latest turbulence target")
        for layout in ["transpose", "halos"]:
            command = [
                sys.executable,
                str(Path(__file__).with_name("measure.py")),
                str(args.binary),
                "--launcher",
                args.launcher,
                "--placements",
                *args.placements,
                "--nx",
                str(nx),
                "--nv",
                "4096",
                "--length",
                "12600",
                "--vmax",
                "15",
                "--dt",
                ".02",
                "--integrator",
                "sixth",
                "--velocity",
                "cubic",
                "--layout",
                layout,
                "--cells",
                "--nu",
                "1e-12",
                "--steps",
                str(args.steps),
                "--warmup",
                "3",
                "--repeats",
                str(args.repeats),
                "--expect-cuda",
                "--input",
                str(reference / "initial.bin"),
                "--v-grid",
                str(reference / "v.bin"),
                "--background",
                str(reference / "background.bin"),
                "--max-mass-change=-1",
                "--output",
                str(args.output / f"nx{nx}-{layout}.jsonl"),
            ]
            if args.source_state:
                command.extend(["--source-state", str(args.source_state)])
            print(json.dumps({"nx": nx, "layout": layout, "status": "measuring"}), flush=True)
            # Conservation diagnostics are recorded without changing the target
            # operator. Full-state/field/N parity is gated independently above.
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
