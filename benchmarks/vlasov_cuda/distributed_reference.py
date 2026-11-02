"""Advance a verified prepared native state with pinned ADEPT over GPU ranks.

Used when the production collision workspace does not fit four 40GB GPUs.
All writes go under a new --output; the prepared input is read-only.
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
os.environ.setdefault("JAX_ENABLE_X64", "true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vp-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 100])
    args = parser.parse_args()
    sys.path.insert(0, str(args.vp_source))
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.distributed.initialize(local_device_ids=[int(os.environ["SLURM_LOCALID"])])
    import jax.numpy as jnp
    import numpy as np
    import yaml
    from checkpoint_reference import save_vx, sha
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from vp_turbulence.adeptmodules import VPTwoStream1D
    from vp_turbulence.multinode import initialize_module_sharded

    rank, count = jax.process_index(), jax.process_count()
    if count != 8 or len(jax.local_devices()) != 1:
        raise RuntimeError("expected eight ranks with one GPU each")
    d = importlib.metadata.distribution("adept")
    commit = json.loads(d.read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
    if commit != "ddd70b957dc3df85f0b5ec84af927c25d09d24e0" or jax.__version__ != "0.10.2":
        raise RuntimeError("incorrect pinned numerical environment")
    metadata = json.loads((args.prepared / "provenance.json").read_text())
    if sha(args.config) != metadata["config_sha256"]:
        raise RuntimeError("configuration differs from verified prepared input")
    nx, nv = metadata["nx"], metadata["nv"]
    if (args.prepared / "initial.bin").stat().st_size != nx * nv * 8:
        raise RuntimeError("incorrect prepared state size")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        for name in ["initial.bin", "v.bin", "background.bin"]:
            os.link(args.prepared / name, args.output / name)
        metadata.update(
            process_count=count,
            devices=[str(d) for d in jax.devices()],
            distributed=True,
            milestones=[],
            prepared_input=str(args.prepared),
        )
        (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    multihost_utils.sync_global_devices("output-created")
    (args.output / f"rank-{rank:02d}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                host=socket.gethostname(),
                pid=os.getpid(),
                local_devices=[str(d) for d in jax.local_devices()],
            )
        )
        + "\n"
    )
    cfg = yaml.safe_load(args.config.read_text())
    module = VPTwoStream1D(copy.deepcopy(cfg))
    module.write_units()
    module.get_derived_quantities()
    initialize_module_sharded(module)
    v = np.fromfile(args.prepared / "v.bin", dtype=np.float64)
    np.testing.assert_array_equal(v, np.asarray(module.cfg["grid"]["species_grids"]["electron"]["v"]))
    np.testing.assert_array_equal(
        np.fromfile(args.prepared / "background.bin", dtype=np.float64), np.asarray(module.cfg["grid"]["ion_charge"])
    )
    source = np.memmap(args.prepared / "initial.bin", mode="r", dtype=np.float64, shape=(nv, nx))
    mesh = Mesh(np.array(jax.devices()), ("device",))
    sharding = NamedSharding(mesh, P("device", None))
    f = jax.make_array_from_callback((nx, nv), sharding, lambda index: source.T[index])
    module.state = {"electron": f}
    module.init_diffeqsolve()
    operator = module.diffeqsolve_quants["terms"].vector_field.vpfp
    if operator.fp._sc_max_steps != 0:
        raise RuntimeError("unexpected self-consistent-beta iterations")
    replicated = NamedSharding(mesh, P())

    def replicate(value):
        host = np.asarray(value)
        return jax.make_array_from_callback(host.shape, replicated, lambda index: host[index])

    a = replicate(np.zeros(nx + 2))
    prev = replicate(np.zeros(nx))
    external = replicate(np.zeros((6, nx)))
    nu_fp = replicate(np.full(nx, metadata["nu"]))
    nu_k = replicate(np.zeros(nx))

    @jax.jit
    def advance(values, iterations):
        def body(_, current):
            _, result, _ = operator({"electron": current}, a, prev, external, nu_fp, nu_k)
            return result["electron"]

        return jax.lax.fori_loop(0, iterations, body, values)

    compiled = advance.lower(f, replicate(np.asarray(1, dtype=np.int32))).compile()
    if rank == 0:
        print(json.dumps(dict(status="compiled", nx=nx, nv=nv, ranks=count)), flush=True)
    previous = 0
    for stop in sorted(set(args.steps)):
        start = time.perf_counter()
        f = compiled(f, replicate(np.asarray(stop - previous, dtype=np.int32)))
        f.block_until_ready()
        elapsed = time.perf_counter() - start
        shards = f.addressable_shards
        if len(shards) != 1:
            raise RuntimeError("unexpected endpoint layout")
        shard = shards[0]
        values = np.asarray(shard.data)
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite reference result")
        np.save(args.output / f"step-{stop:04d}-rank-{rank:02d}.npy", values, allow_pickle=False)
        (args.output / f"step-{stop:04d}-rank-{rank:02d}.json").write_text(
            json.dumps(dict(start=shard.index[0].start or 0, stop=shard.index[0].stop or nx, seconds=elapsed)) + "\n"
        )
        multihost_utils.sync_global_devices(f"shards-saved-{stop}")
        if rank == 0:
            whole = np.empty((nx, nv), dtype=np.float64)
            for r in range(count):
                part = json.loads((args.output / f"step-{stop:04d}-rank-{r:02d}.json").read_text())
                whole[part["start"] : part["stop"]] = np.load(
                    args.output / f"step-{stop:04d}-rank-{r:02d}.npy", mmap_mode="r"
                )
            save_vx(args.output / f"reference-{stop:04d}.bin", whole)
            del whole
            row = dict(steps=stop, advanced_steps=stop - previous, rank0_seconds=elapsed)
            metadata["milestones"].append(row)
            (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        multihost_utils.sync_global_devices(f"reference-saved-{stop}")
        previous = stop
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
