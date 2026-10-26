"""Compare CUDA layouts at full native resolution with a pinned ADEPT window."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np


def hole_counts(electric, k):
    # Same whole-box observer as the latest Nv4096 comparison: Gaussian
    # sigma_x=3, adjacent-minimum potential depths / vb^2, strict cuts.
    kernel = np.zeros(len(k), dtype=complex)
    kernel[1:] = 1j / k[1:] * np.exp(-0.5 * (3 * k[1:]) ** 2)
    phi = np.fft.irfft(np.fft.rfft(electric) * kernel, n=len(electric))
    peaks = np.flatnonzero((phi > np.roll(phi, 1)) & (phi > np.roll(phi, -1)))
    minima = np.flatnonzero((phi < np.roll(phi, 1)) & (phi < np.roll(phi, -1)))
    if not len(minima):
        return [0, 0]
    i = np.searchsorted(minima, peaks)
    depth = phi[peaks] - np.maximum(phi[minima[(i - 1) % len(minima)]], phi[minima[i % len(minima)]])
    return [int(np.count_nonzero(depth / 9 > cut)) for cut in [0.02, 0.5]]


def compare(candidate, reference, nx, nv, dv, length, background):
    actual = np.memmap(candidate, mode="r", dtype=np.float64, shape=(nv, nx))
    expected = np.memmap(reference, mode="r", dtype=np.float64, shape=(nv, nx))
    error2, reference2, maximum = 0.0, 0.0, 0.0
    density = np.zeros(nx)
    reference_density = np.zeros(nx)
    for start in range(0, nv, 64):
        a, b = actual[start : start + 64], expected[start : start + 64]
        difference = a - b
        if not np.isfinite(difference).all():
            raise RuntimeError("nonfinite checkpoint comparison")
        maximum = max(maximum, float(np.max(np.abs(difference))))
        error2 += float(np.sum(difference * difference))
        reference2 += float(np.sum(b * b))
        density += np.sum(a, axis=0) * dv
        reference_density += np.sum(b, axis=0) * dv
    k = 2 * np.pi * np.fft.rfftfreq(nx, d=length / nx)
    ef, rf = np.zeros(len(k), dtype=complex), np.zeros(len(k), dtype=complex)
    ef[1:] = np.fft.rfft(background - density)[1:] / (1j * k[1:])
    rf[1:] = np.fft.rfft(background - reference_density)[1:] / (1j * k[1:])
    ef[-1] = rf[-1] = 0
    electric, reference_electric = np.fft.irfft(ef, n=nx), np.fft.irfft(rf, n=nx)
    counts, reference_counts = hole_counts(electric, k), hole_counts(reference_electric, k)
    return dict(
        max_abs_f_error=maximum,
        relative_l2_f_error=float(np.sqrt(error2 / reference2)),
        relative_l2_e_error=float(np.linalg.norm(electric - reference_electric) / np.linalg.norm(reference_electric)),
        max_abs_e_error=float(np.max(np.abs(electric - reference_electric))),
        hole_counts=counts,
        reference_hole_counts=reference_counts,
        hole_counts_match=counts == reference_counts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--placements", nargs="+", default=["1:4", "2:8"])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads((args.reference / "provenance.json").read_text())
    nx, nv, dt, nu = (source[key] for key in ["nx", "nv", "dt", "nu"])
    # Thresholds are implementation-matching checks for a short mature window,
    # not criteria for long-time turbulent convergence or positivity.
    limits = dict(max_abs_f_error=1e-7, relative_l2_f_error=1e-8, relative_l2_e_error=1e-7)
    background = np.fromfile(args.reference / "background.bin", dtype=np.float64)
    records = []
    args.output.mkdir(parents=True, exist_ok=False)
    for placement in args.placements:
        nodes, ranks = map(int, placement.split(":"))
        for layout in ["transpose", "halos"]:
            for steps in args.steps:
                candidate = args.output / f"{nodes}n-{ranks}g-{layout}-{steps:04d}.bin"
                command = [
                    *shlex.split(args.launcher.format(nodes=nodes, ranks=ranks, ranks_per_node=ranks // nodes)),
                    str(args.binary.resolve()),
                    "--nx",
                    str(nx),
                    "--nv",
                    str(nv),
                    "--length",
                    "12600",
                    "--vmax",
                    "15",
                    "--dt",
                    str(dt),
                    "--nu",
                    str(nu),
                    "--cells",
                    "--velocity",
                    "cubic",
                    "--integrator",
                    "sixth",
                    "--layout",
                    layout,
                    "--steps",
                    str(steps),
                    "--warmup",
                    "0",
                    "--input",
                    str(args.reference / "initial.bin"),
                    "--v-grid",
                    str(args.reference / "v.bin"),
                    "--background",
                    str(args.reference / "background.bin"),
                    "--dump",
                    str(candidate),
                ]
                result = subprocess.run(command, text=True, capture_output=True, timeout=300)
                if result.returncode:
                    raise RuntimeError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
                metrics = compare(
                    candidate, args.reference / f"reference-{steps:04d}.bin", nx, nv, 30.0 / nv, 12600.0, background
                )
                passed = (
                    all(metrics[key] <= threshold for key, threshold in limits.items()) and metrics["hole_counts_match"]
                )
                row = dict(
                    nodes=nodes,
                    ranks=ranks,
                    layout=layout,
                    steps=steps,
                    **metrics,
                    passed=passed,
                    limits=limits,
                    benchmark=json.loads(result.stdout.strip().splitlines()[-1]),
                    command=command,
                    rank_log=result.stderr,
                )
                records.append(row)
                (args.output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
                print(
                    json.dumps({key: row[key] for key in ["nodes", "ranks", "layout", "steps", *metrics, "passed"]}),
                    flush=True,
                )
                if not passed:
                    raise RuntimeError("native checkpoint comparison did not meet predeclared matching limits")


if __name__ == "__main__":
    main()
