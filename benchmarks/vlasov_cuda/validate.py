"""Numerical checks against NumPy FFTs and SciPy's natural CubicSpline.

Use the CPU build locally or the CUDA build inside a GPU allocation. The launcher
is an argument template, e.g. 'mpiexec -n {ranks}' or 'srun -n {ranks} ...'.
"""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


def initial(nx, nv):
    x = np.arange(nx) * (4 * np.pi / nx)
    v = np.linspace(-8, 8, nv)
    m = np.exp(-(v**2) / 2)
    m /= np.trapezoid(m, v)
    return x, v, m[:, None] * (1 + 0.01 * np.cos(0.5 * x))[None, :]


def stream(f, v, dt):
    k = 0.5 * np.arange(f.shape[1] // 2 + 1)
    fk = np.fft.rfft(f, axis=1) * np.exp(-1j * v[:, None] * k[None, :] * dt)
    return np.fft.irfft(fk, n=f.shape[1], axis=1)


def field(f, v):
    rho = 1 - np.trapezoid(f, v, axis=0)
    rk = np.fft.rfft(rho)
    ek = np.zeros_like(rk)
    k = 0.5 * np.arange(len(rk))
    ek[1:] = rk[1:] / (1j * k[1:])
    if f.shape[1] % 2 == 0:
        ek[-1] = 0
    return np.fft.irfft(ek, n=f.shape[1])


def accelerate(f, v, a, dt):
    q = v[:, None] - np.broadcast_to(a, (f.shape[1],))[None, :] * dt
    # Evaluate one spatial column per query column (SciPy otherwise broadcasts
    # the whole output axis). Boundary values outside the velocity box are zero.
    result = np.empty_like(f)
    for x in range(f.shape[1]):
        spline = CubicSpline(v, f[:, x], bc_type="natural", extrapolate=False)
        result[:, x] = np.nan_to_num(spline(q[:, x]), nan=0.0)
    return result


def reference(mode, nx, nv, steps, dt):
    _, v, f = initial(nx, nv)
    for _ in range(steps):
        if mode == "stream":
            f = stream(f, v, dt)
        elif mode == "accelerate":
            f = accelerate(f, v, 0.2, dt)
        elif mode == "landau":
            f = stream(f, v, dt / 2)
            e = field(f, v)
            f = accelerate(f, v, -e, dt)
            f = stream(f, v, dt / 2)
    return f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--launcher", default="mpiexec -n {ranks}")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--host-mpi", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    nx, nv, dt, steps = 128, 128, 0.1, 4
    with tempfile.TemporaryDirectory(prefix="vlasov-validation-") as directory:
        for mode in ["transpose", "stream", "accelerate", "landau"]:
            expected = reference(mode, nx, nv, steps, dt)
            first = None
            for ranks in args.ranks:
                # MPI ranks on different nodes need a shared destination.
                # Only rank 0 writes; MPI_Gather collects all velocity slabs.
                dump = Path(directory) / f"{mode}-{ranks}.bin"
                command = [
                    *shlex.split(args.launcher.format(ranks=ranks)),
                    str(args.binary.resolve()),
                    "--nx",
                    str(nx),
                    "--nv",
                    str(nv),
                    "--steps",
                    str(steps),
                    "--dt",
                    str(dt),
                    "--warmup",
                    "1",
                    "--mode",
                    mode,
                    "--dump",
                    str(dump),
                ]
                if args.host_mpi:
                    command.append("--host-mpi")
                completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
                if completed.returncode:
                    raise RuntimeError(f"{' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
                report = json.loads(completed.stdout.strip().splitlines()[-1])
                actual = np.fromfile(dump, dtype=np.float64).reshape(nv, nx)
                error = float(np.max(np.abs(actual - expected)))
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=3e-13)
                if first is None:
                    first = actual
                else:
                    np.testing.assert_allclose(actual, first, rtol=2e-11, atol=3e-13)
                if mode in ("transpose", "stream"):
                    assert abs(report["mass_relative_change"]) < 1e-12, report
                if mode == "stream":
                    x, v, f0 = initial(nx, nv)
                    m = f0[:, 0] / 1.01
                    analytic = m[:, None] * (1 + 0.01 * np.cos(0.5 * (x[None, :] - v[:, None] * steps * dt)))
                    np.testing.assert_allclose(actual, analytic, rtol=2e-12, atol=3e-14)
                if mode == "landau":
                    assert report["final_field_energy"] < report["initial_field_energy"], report
                    assert abs(report["mass_relative_change"]) < 1e-9, report
                    initial_energy = report["initial_kinetic_energy"] + report["initial_field_energy"]
                    final_energy = report["final_kinetic_energy"] + report["final_field_energy"]
                    assert abs(final_energy / initial_energy - 1) < 1e-6, report
                row = {"mode": mode, "ranks": ranks, "max_abs_error": error, "passed": True}
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
