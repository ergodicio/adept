"""Tracked direct/treecode field audit of saved AMR states; no time evolution.

Run on a compute node for large saved states. Both evaluators see the identical
particle positions and charge weights, independently of the original force path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def field_errors(reference, approximation, charges):
    """Absolute errors remain meaningful when the reference field vanishes."""
    reference, approximation, charges = map(np.asarray, (reference, approximation, charges))
    if reference.shape != approximation.shape or not reference.size:
        raise ValueError("Field arrays must have the same nonempty shape")
    if any(not np.isfinite(value).all() for value in (reference, approximation, charges)):
        raise ArithmeticError("Field audit encountered nonfinite values")
    difference = approximation - reference
    norm = float(np.linalg.norm(reference.ravel()))
    absolute = float(np.linalg.norm(difference.ravel()))
    maximum = float(np.max(np.abs(difference)))
    charge_l1 = float(np.sum(np.abs(charges)))
    # Below this scale a relative field error magnifies quadrature roundoff.
    floor = 64 * np.finfo(float).eps * max(charge_l1, np.finfo(float).tiny) * np.sqrt(reference.size)
    return {
        "reference_l2_norm": norm,
        "absolute_l2_error": absolute,
        "max_absolute_error": maximum,
        "charge_l1": charge_l1,
        "max_absolute_error_per_charge": maximum / charge_l1 if charge_l1 > 0 else None,
        "relative_l2_error": absolute / norm if norm > floor else None,
        "relative_l2_reference_floor": float(floor),
    }


def _state_dataset(state, t):
    import xarray as xr

    if "valid" in state and not bool(np.asarray(state["valid"])):
        raise ValueError("Saved final state is explicitly invalid")
    names = ("x", "v", "f", "weights", "active", "panel_id", "level")
    if any(name not in state for name in names):
        raise ValueError("Saved state must contain AMR coordinates, values, weights, activity and panel identifiers")
    return xr.Dataset(
        {
            name: (("t", "panel", "node") if name in names[:4] else ("t", "panel"), np.asarray(state[name])[None])
            for name in names
        },
        coords={"t": [t]},
    )


def load_saved_states(source, config):
    """Load distributions and verify/append the final state, without rebuilding f."""
    import xarray as xr

    source = Path(source)
    distribution, final = source / "distribution.nc", source / "final_state.npz"
    dataset, used = None, []
    if distribution.is_file():
        with xr.open_dataset(distribution, engine="h5netcdf") as saved:
            dataset = saved.load()
        used.append(distribution)
    if final.is_file():
        with np.load(final, allow_pickle=False) as saved:
            final_dataset = _state_dataset(dict(saved), config["time"]["tmax"])
        used.append(final)
        if dataset is None:
            dataset = final_dataset
        elif dataset.sizes.get("t", 0) and np.isclose(dataset.t.values[-1], config["time"]["tmax"], rtol=0, atol=1e-10):
            active = np.asarray(final_dataset.active[0], dtype=bool)
            for name in ("x", "v", "f", "weights", "active", "panel_id", "level"):
                if name not in dataset or dataset[name].shape[1:] != final_dataset[name].shape[1:]:
                    raise ValueError("Distribution and final_state shapes disagree")
                left, right = np.asarray(dataset[name][-1]), np.asarray(final_dataset[name][0])
                mask = slice(None) if name == "active" else active
                if not np.array_equal(left[mask], right[mask]):
                    raise ValueError(f"Distribution and final_state disagree in {name}")
        else:
            dataset = xr.concat([dataset, final_dataset], dim="t", join="exact")
    if dataset is None:
        raise FileNotFoundError("A completed AMR source needs distribution.nc and/or final_state.npz")
    return dataset, used


def _validate_frames(dataset, config):
    from .movies import reconstruct_amr_panels

    times = np.asarray(dataset.t)
    if times.ndim != 1 or not times.size or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Saved times must be finite, nonempty and strictly increasing")
    if times[0] < config["time"].get("tmin", 0.0) - 1e-10 or times[-1] > config["time"]["tmax"] + 1e-10:
        raise ValueError("Saved times lie outside the source time interval")
    if "weights" not in dataset or dataset.weights.dims != ("t", "panel", "node"):
        raise ValueError("AMR weights must have dimensions (t, panel, node)")
    reconstruct_amr_panels(
        dataset, np.empty(0), np.empty(0), config["grid"], max_level=config["amr"].get("max_level", 1)
    )
    active, weights = np.asarray(dataset.active, dtype=bool), np.asarray(dataset.weights)
    if not np.isfinite(weights).all() or np.any(weights[~active] != 0):
        raise ValueError("Inactive AMR weights must be zero and all weights finite")
    axis = np.array(
        [1.0, 2.0, 1.0] if config["numerical"].get("quadrature", "trapezoid") == "trapezoid" else [1.0, 4.0, 1.0]
    )
    unit = np.outer(axis, axis).ravel() / axis.sum() ** 2
    area = np.ptp(np.asarray(dataset.x)[active], axis=1) * np.ptp(np.asarray(dataset.v)[active], axis=1)
    if not np.allclose(weights[active], area[:, None] * unit, rtol=2e-12, atol=0):
        raise ValueError("Active AMR weights disagree with configured panel quadrature")


def evaluate_saved_fields(dataset, config, *, nx=128, probe_count=257, chunk_size=None):
    """Recompute fields on a shared periodic grid and off-grid endpoint probes.

    No distribution interpolation, clipping, normalization or integration in
    time is performed. Returned errors isolate tree summation on the same state.
    """
    import jax
    import jax.numpy as jnp
    import xarray as xr

    from adept.farsight1d.config import Farsight1DConfig
    from adept.farsight1d.numerics import electric_field
    from adept.farsight1d.treecode import electric_field_treecode_with_info

    if not jax.config.jax_enable_x64:
        raise RuntimeError("Field audits require jax_enable_x64")
    resolved = Farsight1DConfig.model_validate({key: value for key, value in config.items() if key != "solver"})
    if not resolved.amr.enabled:
        raise ValueError("This field audit requires an AMR FARSIGHT source")
    for name, value in (("nx", nx), ("probe_count", probe_count)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    chunk_size = resolved.numerical.chunk_size if chunk_size is None else chunk_size
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    _validate_frames(dataset, resolved.model_dump())
    grid, numerical = resolved.grid, resolved.numerical
    length = grid.xmax - grid.xmin
    x = grid.xmin + np.arange(nx) * length / nx
    probe_x = grid.xmin + (np.arange(probe_count) + 0.317) * length / probe_count
    tree_settings = numerical.treecode.model_dump()
    direct = jax.jit(lambda t, s, q: electric_field(t, s, q, length, numerical.epsilon, chunk_size))
    tree = jax.jit(
        lambda t, s, q: electric_field_treecode_with_info(
            t, s, q, length, numerical.epsilon, chunk_size=chunk_size, **tree_settings
        )
    )
    indices = sorted({0, dataset.sizes["t"] - 1})
    common_direct, common_tree, probe_direct, probe_tree = [], [], [], []
    common_metrics, probe_metrics = [], []
    for frame, t in enumerate(np.asarray(dataset.t)):
        active = np.asarray(dataset.active[frame], dtype=bool)
        sources = np.where(active[:, None], np.asarray(dataset.x[frame]), 0.0).ravel()
        charges = np.zeros_like(np.asarray(dataset.f[frame]))
        charges[active] = -np.asarray(dataset.weights[frame])[active] * np.asarray(dataset.f[frame])[active]
        charges = charges.ravel()

        def compare(targets, sources=sources, charges=charges, t=t, active_panels=int(active.sum())):
            expected = np.asarray(direct(jnp.asarray(targets), jnp.asarray(sources), jnp.asarray(charges)))
            actual, work = tree(jnp.asarray(targets), jnp.asarray(sources), jnp.asarray(charges))
            actual = np.asarray(actual)
            metrics = field_errors(expected, actual, charges)
            metrics.update(
                t=float(t), active_panels=active_panels, tree_work={key: int(value) for key, value in work.items()}
            )
            return expected, actual, metrics

        expected, actual, metrics = compare(x)
        common_direct.append(expected)
        common_tree.append(actual)
        common_metrics.append(metrics)
        if frame in indices:
            expected, actual, metrics = compare(probe_x)
            probe_direct.append(expected)
            probe_tree.append(actual)
            metrics["frame_role"] = (
                "initial"
                if np.isclose(t, resolved.time.tmin, rtol=0, atol=1e-10)
                else ("final" if np.isclose(t, resolved.time.tmax, rtol=0, atol=1e-10) else "saved_endpoint")
            )
            probe_metrics.append(metrics)
    common_direct, common_tree = map(np.asarray, (common_direct, common_tree))
    fields = xr.Dataset(
        {
            "electric_field_direct": (("t", "x"), common_direct),
            "electric_field_treecode": (("t", "x"), common_tree),
            "electric_energy_direct": ("t", 0.5 * length * np.mean(common_direct**2, axis=1)),
            "electric_energy_treecode": ("t", 0.5 * length * np.mean(common_tree**2, axis=1)),
            "probe_field_direct": (("probe_t", "probe_x"), np.asarray(probe_direct)),
            "probe_field_treecode": (("probe_t", "probe_x"), np.asarray(probe_tree)),
        },
        coords={"t": np.asarray(dataset.t), "x": x, "probe_t": np.asarray(dataset.t)[indices], "probe_x": probe_x},
        attrs={
            "description": "Direct and treecode fields on identical saved AMR states; no time evolution",
            "force_model": "FARSIGHT softened periodic kernel",
            "epsilon": numerical.epsilon,
            "source_observation_nx": grid.nx,
            "common_observation_nx": nx,
            "energy_note": "Physical E^2/2 quadrature, not the softened interaction Hamiltonian",
        },
    )
    audit = {
        "scope": "Same-state field evaluation only; no nonlinear evolution or trajectory comparison",
        "original_field_solver": numerical.field_solver,
        "treecode_settings": tree_settings,
        "epsilon": numerical.epsilon,
        "chunk_size": int(chunk_size),
        "source_observation_nx": grid.nx,
        "common_observation_nx": int(nx),
        "probe_count": int(probe_count),
        "probe_offset": 0.317,
        "probe_formula": "xmin + (i + 0.317) * L / probe_count",
        "relative_error_note": "Null when the reference norm is at or below the recorded charge-scaled roundoff floor",
        "work_note": "Logical tree interactions include padded near-leaf slots; not elapsed time or GPU instructions",
        "common_grid": common_metrics,
        "offgrid_endpoints": probe_metrics,
    }
    return fields, audit


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(
    source, output, *, nx=128, probe_count=257, chunk_size=None, experiment="farsight-comparison", tracking_uri=None
):
    """Create a verified MLflow artifact run linked to one completed AMR run."""
    from adept import Artifact, MetricEvent, RunRequest, RunStatus
    from adept.farsight1d import amr, numerics, treecode

    from .movies import farsight_method
    from .run import _provenance, _services, _write_json
    from .scan import require_compute_node

    require_compute_node()
    source, output = Path(source).resolve(), Path(output).resolve()
    config = json.loads((source / "config.json").read_text())
    original = json.loads((source / "run.json").read_text())
    if (
        original.get("status") != "FINISHED"
        or not isinstance(original.get("run_id"), str)
        or not original["run_id"].strip()
    ):
        raise ValueError("Field audits require a FINISHED source with an MLflow run_id")
    if (
        original.get("solver") != "farsight"
        or config.get("solver") != "farsight-1d"
        or not config.get("amr", {}).get("enabled", False)
    ):
        raise ValueError("Field audits require an AMR FARSIGHT source")
    if original.get("field_model") != "farsight-softened":
        raise ValueError("Source must identify the FARSIGHT softened field model")
    method = farsight_method(config)
    original_field_solver = config["numerical"].get("field_solver", "direct")
    if (
        original.get("method", method) != method
        or original.get("field_solver", original_field_solver) != original_field_solver
    ):
        raise ValueError("Source method metadata disagrees with its configuration")
    dataset, input_paths = load_saved_states(source, config)
    source_info = {
        "directory": str(source),
        "run_id": original["run_id"],
        "method": method,
        "sha256": {path.name: _sha256(path) for path in [source / "config.json", source / "run.json", *input_paths]},
    }
    if (source / "provenance.json").is_file():
        source_info["provenance"] = json.loads((source / "provenance.json").read_text())
        source_info["sha256"]["provenance.json"] = _sha256(source / "provenance.json")
    output.mkdir(parents=True, exist_ok=False)
    tracker, sink = _services(tracking_uri)
    request = RunRequest(
        experiment=experiment,
        name=f"{original.get('case', 'farsight')}-field-audit-{method}-{nx}",
        tags={
            "comparison.phase": "field-audit",
            "comparison.farsight_run_id": original["run_id"],
            "comparison.method": method,
        },
    )
    tracker.preflight(request)
    sink.preflight()
    handle = tracker.start(request)
    summary = {"run_id": handle.run_id, "status": "RUNNING", "source": source_info, "output": str(output)}
    print(json.dumps(summary), flush=True)
    try:
        _write_json(output / "run.json", summary)
        sink.validate(handle)
        provenance = _provenance(output)
        _write_json(output / "source_config.json", config)
        _write_json(output / "source_run.json", original)
        started = time.perf_counter()
        fields, audit = evaluate_saved_fields(dataset, config, nx=nx, probe_count=probe_count, chunk_size=chunk_size)
        audit["evaluation_seconds_including_compilation"] = time.perf_counter() - started
        code_paths = [Path(__file__), Path(amr.__file__), Path(numerics.__file__), Path(treecode.__file__)]
        audit.update(
            source=source_info,
            source_code_sha256={str(path): _sha256(path) for path in code_paths},
            audit_source_archive_sha256=provenance["source_archive_sha256"],
        )
        fields.attrs["source_run_id"] = original["run_id"]
        fields.to_netcdf(output / "fields.nc", engine="h5netcdf")
        _write_json(output / "audit.json", audit)
        metrics = {
            f"final_common_{key}": value
            for key, value in audit["common_grid"][-1].items()
            if isinstance(value, (float, int))
        }
        tracker.log_metrics(handle, [MetricEvent(metrics)])
        summary.update(
            status="FINISHED", artifacts={"fields": str(output / "fields.nc"), "audit": str(output / "audit.json")}
        )
        _write_json(output / "run.json", summary)
        for path in sorted(output.iterdir()):
            if path.is_file():
                receipt = sink.put(handle, Artifact(path, artifact_path="field-audit"))
                sink.verify(handle, receipt)
        for path in code_paths:
            receipt = sink.put(handle, Artifact(path, artifact_path="field-audit/source"))
            sink.verify(handle, receipt)
        tracker.finish(handle, RunStatus.FINISHED)
    except Exception as error:  # Record any failure, then preserve the original exception.
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        try:
            _write_json(output / "run.json", summary)
        except Exception as failure:  # noqa: BLE001 — a reporting failure must not replace the original error.
            error.add_note(f"Failure summary persistence also failed: {type(failure).__name__}: {failure}")
        try:
            receipt = sink.put(handle, Artifact(output / "run.json", artifact_path="field-audit"))
            sink.verify(handle, receipt)
        except Exception as failure:  # noqa: BLE001 — a reporting failure must not replace the original error.
            error.add_note(f"Failure artifact upload also failed: {type(failure).__name__}: {failure}")
        try:
            tracker.finish(handle, RunStatus.FAILED, error=summary["error"])
        except Exception as failure:  # noqa: BLE001 — a reporting failure must not replace the original error.
            error.add_note(f"Failure status update also failed: {type(failure).__name__}: {failure}")
        raise
    print(json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed AMR comparison run directory")
    parser.add_argument("--output", type=Path, required=True, help="New audit directory, never overwritten")
    parser.add_argument("--nx", type=int, default=128, help="Common unique periodic field targets")
    parser.add_argument("--probe-count", type=int, default=257)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--experiment", default="farsight-comparison")
    parser.add_argument("--tracking-uri", help="Defaults to MLFLOW_TRACKING_URI")
    args = vars(parser.parse_args(argv))
    import jax

    jax.config.update("jax_enable_x64", True)
    audit_run(**args)


if __name__ == "__main__":
    main()
