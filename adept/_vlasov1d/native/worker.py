"""Internal MPI worker. Run by filename so JAX initializes before importing ADEPT."""

import argparse
import gc
import importlib.metadata
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

from bridge import NativeCore


def run(request):
    import numpy as np

    root = request.parent
    spec = json.loads(request.read_text())
    os.environ["JAX_PLATFORMS"] = "cpu" if spec["device"] == "cpu" else "cuda,cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    core = NativeCore(spec["library"], spec["options"])
    try:
        for species in spec["species"][1:]:
            core.add_species(species["options"])
        if core.size != spec["ranks"]:
            raise RuntimeError("MPI launcher rank count differs from native.ranks")
        if core.device_type != (2 if spec["device"] == "cuda" else 1):
            raise RuntimeError("native library was built for a different device")
        core.set_field(root / "field.bin")
        import jax

        jax.config.update("jax_enable_x64", True)
        if core.size > 1:
            coordinator = ""
            if core.rank == 0:
                with socket.socket() as connection:
                    connection.bind(("", 0))
                    coordinator = f"{socket.gethostname()}:{connection.getsockname()[1]}"
            coordinator = core.broadcast(coordinator)
            jax.distributed.initialize(
                coordinator_address=coordinator,
                num_processes=core.size,
                process_id=core.rank,
                local_device_ids=[core.local_rank] if core.device_type == 2 else [0],
            )
        elif core.device_type == 2:
            jax.config.update("jax_cuda_visible_devices", str(core.local_rank))
        sys.path[:0] = spec.get("pythonpath", [])
        import cloudpickle
        import jax.numpy as jnp
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        if len(jax.devices()) != core.size or len(jax.local_devices()) != 1:
            raise RuntimeError("native workers require exactly one JAX device per MPI rank")
        mesh = Mesh(np.array(jax.devices()), ("device",))
        slab = NamedSharding(mesh, P(None, "device"))
        replicated = NamedSharding(mesh, P())
        with (root / "callbacks.pkl").open("rb") as stream:
            payload = cloudpickle.load(stream)

        def replicate(value):
            host = np.asarray(value)
            return jax.make_array_from_callback(host.shape, replicated, lambda index: host[index])

        static = jax.tree.map(replicate, payload["state"])
        callbacks = {
            key: jax.jit(lambda t, y, fn=fn: fn(t, y, payload["args"])) for key, fn in payload["callbacks"].items()
        }

        def snapshot():
            # FFTW may align host allocations to 32 bytes; JAX CPU requires 64.
            # CUDA allocations satisfy JAX's alignment, so GPU views stay strict.
            copy = False if core.device_type == 2 else None
            e = jax.dlpack.from_dlpack(core.view(field=True), copy=copy)
            state = dict(static, e=jax.make_array_from_single_device_arrays((core.nx,), replicated, [e]))
            for index, species in enumerate(spec["species"]):
                f = jax.dlpack.from_dlpack(core.view(species=index), copy=copy)
                state[species["name"]] = jax.make_array_from_single_device_arrays((core.nx, species["nv"]), slab, [f])
            return state

        directory = root / f"rank-{core.rank:05d}"
        directory.mkdir()
        writers, metadata, layout = {}, {}, {}

        def save(key, frame, t, state):
            value = callbacks[key](replicate(np.asarray(t)), state)
            jax.block_until_ready(value)
            leaves, tree = jax.tree.flatten(value)
            if key not in layout:
                layout[key] = {"tree": tree, "leaves": []}
                metadata[key] = []
                for i, leaf in enumerate(leaves):
                    layout[key]["leaves"].append(dict(shape=leaf.shape, dtype=str(leaf.dtype)))
                    if leaf.is_fully_replicated and core.rank != 0:
                        metadata[key].append(None)
                        continue
                    shards = leaf.addressable_shards
                    if len(shards) != 1:
                        raise RuntimeError("unexpected native save sharding")
                    shard = shards[0]
                    index = [
                        [part.start or 0, part.stop if part.stop is not None else leaf.shape[d], part.step or 1]
                        for d, part in enumerate(shard.index)
                    ]
                    filename = f"{len(layout) - 1:04d}-{i:03d}.npy"
                    shape = (len(spec["times"][key]), *shard.data.shape)
                    writers[key, i] = np.lib.format.open_memmap(
                        directory / filename, mode="w+", dtype=np.dtype(leaf.dtype), shape=shape
                    )
                    metadata[key].append(dict(file=str((directory / filename).relative_to(root)), index=index))
            for i, leaf in enumerate(leaves):
                if (key, i) in writers:
                    (shard,) = leaf.addressable_shards
                    expected = metadata[key][i]["index"]
                    index = [
                        [part.start or 0, part.stop if part.stop is not None else leaf.shape[d], part.step or 1]
                        for d, part in enumerate(shard.index)
                    ]
                    if index != expected:
                        raise RuntimeError("save output sharding changed between frames")
                    writers[key, i][frame] = np.asarray(shard.data)
            # No device save output survives a native mutation, including identity saves.
            del leaves, value

        requests = {}
        for key, times in spec["times"].items():
            for frame, t in enumerate(times):
                position = (t - spec["t0"]) / spec["dt"]
                nearest = round(position)
                if abs(position - nearest) <= 1e-9:
                    position = float(nearest)
                upper = int(np.ceil(position))
                requests.setdefault(upper, []).append((key, frame, t, position - (upper - 1)))
        current = 0
        started = time.perf_counter()
        for upper, saves in sorted(requests.items()):
            off_grid = any(0 < alpha < 1 for _, _, _, alpha in saves)
            previous = None
            if off_grid:
                core.step(upper - 1 - current)
                source = snapshot()
                previous = jax.tree.map(lambda value: jnp.array(value, copy=True), source)
                jax.block_until_ready(previous)
                del source
                core.step(1)
            else:
                core.step(upper - current)
            current = upper
            state = snapshot()
            for key, frame, t, alpha in saves:
                if 0 < alpha < 1:
                    sample = jax.tree.map(lambda a, b, alpha=alpha: a + (b - a) * alpha, previous, state)
                    save(key, frame, t, sample)
                    del sample
                else:
                    save(key, frame, t, state)
            del state, previous
            core.barrier()
        core.step(spec["steps"] - current)
        for writer in writers.values():
            writer.flush()
        (directory / "index.json").write_text(json.dumps(metadata))
        if core.rank == 0:
            with (root / "layout.pkl").open("wb") as stream:
                cloudpickle.dump(layout, stream)
        stats = core.stats()
        if core.rank == 0:
            (root / "stats.json").write_text(
                json.dumps(
                    dict(
                        native_seconds=stats,
                        solve_and_save_seconds=time.perf_counter() - started,
                        steps=spec["steps"],
                        ranks=core.size,
                        device=spec["device"],
                        species=[
                            {key: value for key, value in species.items() if key != "options"}
                            for species in spec["species"]
                        ],
                        libraries={
                            name: importlib.metadata.version(name)
                            for name in ["jax", "jaxlib", "cloudpickle", "diffrax"]
                        },
                    ),
                    indent=2,
                )
            )
        core.barrier()
        gc.collect()
        core.close()
        if core.size > 1:
            jax.distributed.shutdown()
    except BaseException:
        traceback.print_exc()
        core.lib.av_abort(1)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    run(parser.parse_args().request)
