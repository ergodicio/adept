"""Read a committed native checkpoint and advance its pinned ADEPT operators.

Run with the production vp-turbulence environment on an allocated GPU node.
Only --output is written; the source campaign and checkpoint are read-only.
This deliberately tests numerical operators directly, without production
tracking, save buffers, or changes to either existing MLflow parent.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_vx(path, array):
    """Write x,v data as a contiguous v,x file with bounded copy temporaries."""
    import numpy as np

    out = np.memmap(path, mode="w+", dtype=np.float64, shape=array.shape[::-1])
    for v0 in range(0, array.shape[1], 64):
        out[v0 : v0 + 64] = array[:, v0 : v0 + 64].T
    out.flush()
    del out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="committed native input; omit to export the original seeded t0 initializer"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vp-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--expected-manifest")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.checkpoint and (
        not args.expected_manifest or sha(args.checkpoint / "manifest.json") != args.expected_manifest
    ):
        raise RuntimeError("native checkpoint manifest checksum mismatch")
    distribution = importlib.metadata.distribution("adept")
    direct = json.loads(distribution.read_text("direct_url.json"))
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit != "ddd70b957dc3df85f0b5ec84af927c25d09d24e0":
        raise RuntimeError(f"wrong ADEPT environment: {commit}")
    sys.path.insert(0, str(args.vp_source))
    import adept

    if Path(adept.__file__).resolve().parent != Path(distribution.locate_file("adept")).resolve():
        raise RuntimeError("PYTHONPATH shadows the pinned ADEPT installation")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import yaml
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from vp_turbulence.adeptmodules import VPTwoStream1D
    from vp_turbulence.multinode import initialize_module_sharded

    jax.config.update("jax_enable_x64", True)
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["terms"]["time"] != "sixth" or cfg["terms"]["edfdv"] != "cubic-spline":
        raise RuntimeError("expected the latest sixth-order/cubic target")
    if cfg["drivers"]["ex"] or cfg["drivers"]["ey"]:
        raise RuntimeError("external forcing is outside this comparison")
    manifest = (
        json.loads((args.checkpoint / "manifest.json").read_text())
        if args.checkpoint
        else {"actual_time": 0.0, "ranks": []}
    )
    nx, nv = int(cfg["grid"]["nx"]), int(cfg["grid"]["nv"])
    state = np.empty((nx, nv), dtype=np.float64) if args.checkpoint else None
    covered = np.zeros(nx, dtype=bool)
    for rank in manifest["ranks"]:
        leaf = rank["leaves"]["electron"]
        path = args.checkpoint / leaf["file"]
        if sha(path) != leaf["sha256"]:
            raise RuntimeError(f"native shard checksum mismatch: {path}")
        (start, stop, stride), vrange = leaf["index"]
        if stride != 1 or vrange != [0, nv, 1] or np.any(covered[start:stop]):
            raise RuntimeError("unexpected native shard coverage")
        state[start:stop] = np.load(path, mmap_mode="r", allow_pickle=False)
        covered[start:stop] = True
    if args.checkpoint and (not covered.all() or not np.isfinite(state).all()):
        raise RuntimeError("incomplete or nonfinite native input")
    v = (
        np.load(args.checkpoint / manifest["coordinates"]["v_electron"]["file"], allow_pickle=False)
        if args.checkpoint
        else None
    )
    # Rebuild the original ion background/geometry from the frozen initializer;
    # the native perturbed background is not replaced with uniform density.
    module = VPTwoStream1D(copy.deepcopy(cfg))
    module.write_units()
    module.get_derived_quantities()
    initialize_module_sharded(module)
    native_v = np.asarray(module.cfg["grid"]["species_grids"]["electron"]["v"])
    if args.checkpoint is None:
        state = np.asarray(module.state["electron"])
        v = native_v
    if not np.array_equal(v, native_v):
        raise RuntimeError("native velocity coordinates differ from pinned setup")
    v.tofile(args.output / "v.bin")
    np.asarray(module.cfg["grid"]["ion_charge"]).tofile(args.output / "background.bin")
    save_vx(args.output / "initial.bin", state)
    module.init_diffeqsolve()
    operator = module.diffeqsolve_quants["terms"].vector_field.vpfp
    if type(operator.vlasov_poisson).__name__ != "SixthOrderHamIntegrator":
        raise RuntimeError("incorrect time integrator")
    if operator.fp._sc_max_steps != 0:
        raise RuntimeError("this prototype expects production self-consistent beta disabled")
    sharding = NamedSharding(Mesh(np.array(jax.devices()), ("device",)), P("device", None))
    f = jax.make_array_from_callback(state.shape, sharding, lambda index: state[index])
    module.state = {"electron": f}
    nu = float(cfg["terms"]["fokker_planck"]["time"]["baseline"])
    a, prev, external = jnp.zeros(nx + 2), jnp.zeros(nx), jnp.zeros((6, nx))
    nu_fp, nu_k = jnp.full(nx, nu), jnp.zeros(nx)

    @jax.jit
    def advance(values, count):
        def body(_, current):
            _, result, _ = operator({"electron": current}, a, prev, external, nu_fp, nu_k)
            return result["electron"]

        return jax.lax.fori_loop(0, count, body, values)

    compiled = advance.lower(f, jnp.asarray(1)).compile()
    record = dict(
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        manifest_sha256=args.expected_manifest,
        actual_time=manifest["actual_time"],
        config_sha256=sha(args.config),
        adept_commit=commit,
        jax=jax.__version__,
        nx=nx,
        nv=nv,
        dt=cfg["grid"]["dt"],
        nu=nu,
        devices=[str(d) for d in jax.devices()],
        milestones=[],
    )
    (args.output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"prepared": str(args.output), "nx": nx, "nv": nv, "time": manifest["actual_time"]}), flush=True)
    previous = 0
    for stop in sorted(set(args.steps)):
        if stop < 1:
            raise ValueError("milestones must be positive")
        start = time.perf_counter()
        f = compiled(f, jnp.asarray(stop - previous))
        f.block_until_ready()
        elapsed = time.perf_counter() - start
        result = np.asarray(f)
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite reference result")
        save_vx(args.output / f"reference-{stop:04d}.bin", result)
        row = dict(steps=stop, advanced_steps=stop - previous, seconds=elapsed)
        record["milestones"].append(row)
        (args.output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        previous = stop


if __name__ == "__main__":
    main()
