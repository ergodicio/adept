"""Compare C1 cubic halos and sixth-order splitting with interpax/NumPy.

This is an operator test. Native-checkpoint and collision comparisons are
separate requirements before claiming turbulence-run reproduction.
"""

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from interpax import interp1d


def cubic(f, v, acceleration, dt):
    a = np.broadcast_to(acceleration, f.shape[1])
    q = v[:, None] - a[None, :] * dt
    return np.asarray(
        jax.vmap(lambda query, values: interp1d(query, jnp.asarray(v), values, extrap=1e-30))(
            jnp.asarray(q.T), jnp.asarray(f.T)
        )
    ).T


def stream(f, v, dt, length):
    k = 2 * np.pi * np.fft.rfftfreq(f.shape[1], d=length / f.shape[1])
    return np.fft.irfft(np.fft.rfft(f, axis=1) * np.exp(-1j * v[:, None] * k * dt), n=f.shape[1], axis=1)


def field(f, dv, length, background):
    rho = background - np.sum(f, axis=0) * dv
    k = 2 * np.pi * np.fft.rfftfreq(f.shape[1], d=length / f.shape[1])
    ek = np.zeros(len(k), dtype=complex)
    ek[1:] = np.fft.rfft(rho)[1:] / (1j * k[1:])
    if f.shape[1] % 2 == 0:
        ek[-1] = 0
    return np.fft.irfft(ek, n=f.shape[1])


def sixth(f, v, dt, length, background):
    a = [
        0.168735950563437422448196,
        0.377851589220928303880766,
        -0.093175079568731452657924,
        0.377851589220928303880766,
        0.168735950563437422448196,
    ]
    d1 = 0.049086460976116245491441 - 2 * 0.000069728715055305084099 * dt**2
    d2 = 0.264177609888976700200146 - 2 * 0.000625704827430047189169 * dt**2 - 4 * 2.916600457689847816445691e-6 * dt**4
    d3 = (
        0.186735929134907054308413
        - 2 * 0.002213085124045325561636 * dt**2
        + 4 * 3.048480261700038788680723e-5 * dt**4
        - 8 * 4.985549387875068121593988e-7 * dt**6
    )
    for i, d in enumerate([d1, d2, d3, d3, d2, d1]):
        e = field(f, v[1] - v[0], length, background)
        f = cubic(f, v, -e, d * dt)
        if i < 5:
            f = stream(f, v, a[i] * dt, length)
    return f


def collide(f, v, dt, nu):
    from adept._vlasov1d.solvers.pushers.fokker_planck import Collisions

    cfg = {
        "grid": {
            "parallel": False,
            "species_grids": {"electron": {"v": jnp.asarray(v), "dv": v[1] - v[0]}},
            "species_params": {"electron": {"mass": 1.0}},
        },
        "terms": {
            "species": [{"name": "electron"}],
            "fokker_planck": {"is_on": True, "type": "chang_cooper_dougherty"},
            "krook": {"is_on": False},
        },
    }
    result = Collisions(cfg)(jnp.full(f.shape[1], nu), jnp.zeros(f.shape[1]), jnp.asarray(f.T), dt)
    return np.asarray(result).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--launcher", default="mpiexec -n {ranks}")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    nx, nv, vmax, length = 128, 128, 8.0, 4 * np.pi
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
    x = (np.arange(nx) + 0.5) * length / nx
    smooth = np.exp(-(v[:, None] ** 2) / 2) / np.sqrt(2 * np.pi) * (1 + 0.03 * np.cos(0.5 * x))
    signed = np.random.default_rng(4).normal(size=(nv, nx))
    # A shift longer than a rank's velocity block tests multi-neighbour halos.
    tests = [
        ("signed-positive", signed, 1.2, 0.1, 0.0),
        ("signed-negative", signed, -1.2, 0.1, 0.0),
        ("signed-wide", signed, 45.0, 0.1, 0.0),
        ("sixth", smooth, 0.0, 0.02, 0.0),
        ("collide", smooth, 0.0, 0.1, 1e-3),
        ("sixth-collide", smooth, 0.0, 0.02, 1e-12),
    ]
    records = []
    with tempfile.TemporaryDirectory(prefix="vlasov-local-") as directory:
        tmp = Path(directory)
        v.tofile(tmp / "v.bin")
        background = np.ones(nx)
        background.tofile(tmp / "bg.bin")
        for name, initial, acceleration, dt, nu in tests:
            initial.tofile(tmp / "initial.bin")
            if name.startswith("sixth"):
                expected = sixth(initial, v, dt, length, background)
            elif name == "collide":
                expected = initial
            else:
                expected = cubic(initial, v, acceleration, dt)
            if nu:
                expected = collide(expected, v, dt, nu)
            for layout in ["transpose", "halos"]:
                for ranks in args.ranks:
                    output = tmp / "out.bin"
                    cmd = [
                        *shlex.split(args.launcher.format(ranks=ranks)),
                        str(args.binary.resolve()),
                        "--nx",
                        str(nx),
                        "--nv",
                        str(nv),
                        "--vmax",
                        str(vmax),
                        "--length",
                        str(length),
                        "--dt",
                        str(dt),
                        "--steps",
                        "1",
                        "--warmup",
                        "0",
                        "--cells",
                        "--velocity",
                        "cubic",
                        "--layout",
                        layout,
                        "--halo-capacity",
                        "64",
                        "--integrator",
                        "sixth",
                        "--mode",
                        "landau" if name.startswith("sixth") else ("collide" if name == "collide" else "accelerate"),
                        "--nu",
                        str(nu),
                        "--acceleration",
                        str(acceleration),
                        "--input",
                        str(tmp / "initial.bin"),
                        "--v-grid",
                        str(tmp / "v.bin"),
                        "--background",
                        str(tmp / "bg.bin"),
                        "--dump",
                        str(output),
                    ]
                    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                    if completed.returncode:
                        raise RuntimeError(f"{' '.join(cmd)}\n{completed.stdout}\n{completed.stderr}")
                    actual = np.fromfile(output, dtype=np.float64).reshape(nv, nx)
                    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)
                    row = dict(
                        test=name,
                        layout=layout,
                        ranks=ranks,
                        max_abs_error=float(np.max(np.abs(actual - expected))),
                        passed=True,
                    )
                    records.append(row)
                    print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
