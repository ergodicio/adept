"""JAX-aware preparation utilities loaded only by concrete solver builders."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import jax

from .contracts import SimulationSpec


def normalize_key(key: int | jax.Array) -> tuple[jax.Array, int, str]:
    """Normalize an integer or JAX key and return reproducibility metadata."""

    normalized = jax.random.key(key) if isinstance(key, int) else key
    key_words = jax.device_get(jax.random.key_data(normalized)).reshape(-1).tolist()
    seed = key if isinstance(key, int) else 0
    if not isinstance(key, int):
        for word in key_words:
            seed = (seed << 32) | int(word)
    provenance = "jax-key:" + ":".join(f"{int(word):08x}" for word in key_words)
    return normalized, seed, provenance


def structural_fingerprint(spec: SimulationSpec, *pytrees: Any) -> str:
    """Fingerprint configuration plus PyTree schemas without materializing values."""

    leaves, structure = jax.tree.flatten(pytrees)
    leaf_schemas = []
    for leaf in leaves:
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        leaf_schemas.append(
            {
                "type": f"{type(leaf).__module__}.{type(leaf).__qualname__}",
                "shape": list(shape) if shape is not None else None,
                "dtype": str(dtype) if dtype is not None else None,
            }
        )
    payload = {
        "spec": spec.to_dict(),
        "tree": str(structure),
        "leaves": leaf_schemas,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["normalize_key", "structural_fingerprint"]
