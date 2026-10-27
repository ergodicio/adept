"""Time pinned ADEPT operators on prepared latest-run checkpoints, without saves.

Launch with one Python process per GPU using the pinned vp-turbulence venv.
Compilation, initial input, warmup and correctness reductions are outside timing.
"""

import argparse
import copy
import importlib.metadata
import json
import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vp-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vp_source))
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.distributed.initialize(local_device_ids=[int(os.environ["SLURM_LOCALID"])])
    import numpy as np
    import yaml
    from checkpoint_reference import sha
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from vp_turbulence.adeptmodules import VPTwoStream1D
    from vp_turbulence.multinode import initialize_module_sharded

    rank, ranks = jax.process_index(), jax.process_count()
    dist = importlib.metadata.distribution("adept")
    commit = json.loads(dist.read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
    if commit != "ddd70b957dc3df85f0b5ec84af927c25d09d24e0" or jax.__version__ != "0.10.2":
        raise RuntimeError("incorrect pinned numerical environment")
    metadata = json.loads((args.prepared / "provenance.json").read_text())
    if sha(args.config) != metadata["config_sha256"]:
        raise RuntimeError("config differs from verified checkpoint")
    nx, nv = metadata["nx"], metadata["nv"]
    cfg = yaml.safe_load(args.config.read_text())
    module = VPTwoStream1D(copy.deepcopy(cfg))
    module.write_units()
    module.get_derived_quantities()
    initialize_module_sharded(module)
    np.testing.assert_array_equal(
        np.fromfile(args.prepared / "v.bin", dtype=np.float64),
        np.asarray(module.cfg["grid"]["species_grids"]["electron"]["v"]),
    )
    np.testing.assert_array_equal(
        np.fromfile(args.prepared / "background.bin", dtype=np.float64), np.asarray(module.cfg["grid"]["ion_charge"])
    )
    source = np.memmap(args.prepared / "initial.bin", mode="r", dtype=np.float64, shape=(nv, nx))
    mesh = Mesh(np.array(jax.devices()), ("device",))
    sharding = NamedSharding(mesh, P("device", None))
    initial = jax.make_array_from_callback((nx, nv), sharding, lambda index: source.T[index])
    module.state = {"electron": initial}
    module.init_diffeqsolve()
    operator = module.diffeqsolve_quants["terms"].vector_field.vpfp
    if operator.fp._sc_max_steps != 0:
        raise RuntimeError("unexpected self-consistent beta")
    replicated = NamedSharding(mesh, P())

    def replicate(value):
        host = np.asarray(value)
        return jax.make_array_from_callback(host.shape, replicated, lambda index: host[index])

    a, prev = replicate(np.zeros(nx + 2)), replicate(np.zeros(nx))
    external = replicate(np.zeros((6, nx)))
    nu_fp, nu_k = replicate(np.full(nx, metadata["nu"])), replicate(np.zeros(nx))

    @jax.jit
    def advance(values, iterations):
        def body(_, current):
            _, result, _ = operator({"electron": current}, a, prev, external, nu_fp, nu_k)
            return result["electron"]

        return jax.lax.fori_loop(0, iterations, body, values)

    iterations = replicate(np.asarray(args.steps, dtype=np.int32))
    start = time.perf_counter()
    compiled = advance.lower(initial, iterations).compile()
    compile_seconds = time.perf_counter() - start
    warm = compiled(initial, replicate(np.asarray(args.warmup, dtype=np.int32)))
    warm.block_until_ready()
    del warm
    rows = []
    for repeat in range(args.repeats):
        # Reset to the same checkpoint each repeat; execution uses immutable input.
        mh.sync_global_devices(f"before-{repeat}")
        start = time.perf_counter()
        result = compiled(initial, iterations)
        result.block_until_ready()
        seconds = time.perf_counter() - start
        all_seconds = np.asarray(mh.process_allgather(np.asarray(seconds)))
        local = np.asarray(result.addressable_shards[0].data)
        if not np.isfinite(local).all():
            raise RuntimeError("nonfinite ADEPT output")
        local_sums = np.asarray([local.sum(), np.square(local).sum(), local.min(), local.max()])
        sums = np.asarray(mh.process_allgather(local_sums)).reshape(ranks, 4)
        if rank == 0:
            row = dict(
                backend="adept-jax",
                nx=nx,
                nv=nv,
                ranks=ranks,
                repeat=repeat,
                steps=args.steps,
                warmup=args.warmup,
                step_ms=1000 * float(all_seconds.max()) / args.steps,
                rank_seconds=all_seconds.tolist(),
                sum=float(sums[:, 0].sum()),
                sum_squares=float(sums[:, 1].sum()),
                minimum=float(sums[:, 2].min()),
                maximum=float(sums[:, 3].max()),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        del result, local
    hosts = np.asarray(
        mh.process_allgather(np.asarray(list(socket.gethostname().encode().ljust(64, b"\0")), dtype=np.uint8))
    )
    if rank == 0:
        report = dict(
            rows=rows,
            checkpoint=metadata,
            adept_commit=commit,
            libraries={name: importlib.metadata.version(name) for name in ["jax", "jaxlib", "interpax", "diffrax"]},
            hosts=[bytes(host).rstrip(b"\0").decode() for host in hosts.reshape(ranks, 64)],
            compile_seconds=compile_seconds,
            devices=[str(d) for d in jax.devices()],
            scope="Warmed operator loop; no SaveAt, diagnostics, I/O, initialization or compilation in timings",
        )
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
