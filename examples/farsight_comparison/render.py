"""Render and log a comparison from two completed local run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _validate_pair(runs, e_config, f_config, e_case, f_case):
    """Reject misleading pairings before creating a tracking run or artifacts."""
    from .cases import ComparisonCase
    from .movies import farsight_method

    for solver, run in runs.items():
        if not isinstance(run.get("run_id"), str) or not run["run_id"].strip():
            raise ValueError(f"The {solver} source must identify its MLflow run")
        if run.get("status") != "FINISHED":
            raise ValueError(f"Cannot render unfinished {solver} run {run['run_id']}")
    field_model = runs["eulerian"].get("field_model")
    if field_model not in {"poisson", "farsight-softened"}:
        raise ValueError("Unknown Eulerian field_model; cannot label a matched comparison")
    expected_solver = "eulerian-poisson" if field_model == "poisson" else "eulerian-softened"
    if runs["eulerian"].get("solver") != expected_solver or runs["farsight"].get("solver") != "farsight":
        raise ValueError("Source solver identities do not match the Eulerian/FARSIGHT roles and field model")
    if runs["farsight"].get("field_model") != "farsight-softened":
        raise ValueError("FARSIGHT source must explicitly identify its softened field model")
    if runs["eulerian"]["run_id"] == runs["farsight"]["run_id"]:
        raise ValueError("Eulerian and FARSIGHT sources must identify distinct runs")
    benchmark = e_config.get("benchmark", {})
    if benchmark.get("field_model") != field_model:
        raise ValueError("Eulerian config and run field_model metadata disagree")
    physical = lambda case: {key: value for key, value in case.items() if key not in {"nx", "nv"}}
    if physical(e_case) != physical(f_case) or benchmark.get("case") != e_case:
        raise ValueError("Case specifications differ; explicitly review matching before comparison")
    case = ComparisonCase(**f_case)
    eulerian_case = ComparisonCase(**e_case)
    if any(run.get("case") != case.name for run in runs.values()):
        raise ValueError("Source run case labels disagree with their case specifications")
    if e_config.get("solver") != "vlasov-1d" or f_config.get("solver") != "farsight-1d":
        raise ValueError("Source configurations do not identify the expected solvers")
    expected = case.to_farsight_config()
    if any(f_config.get(name) != expected[name] for name in ("grid", "initial", "time")):
        raise ValueError("FARSIGHT grid, initial condition, or time config disagrees with the shared case")
    if f_config.get("numerical", {}).get("epsilon") != case.epsilon:
        raise ValueError("FARSIGHT epsilon disagrees with the shared case")
    e_grid = e_config.get("grid", {})
    expected_e_grid = {
        **eulerian_case.to_farsight_config()["grid"],
        "tmin": 0.0,
        "tmax": eulerian_case.tmax,
        "dt": eulerian_case.dt,
    }
    if any(e_grid.get(name) != value for name, value in expected_e_grid.items()):
        raise ValueError("Eulerian grid or time config disagrees with the shared case")
    terms = e_config.get("terms", {})
    if terms.get("vdfdx") != "exponential" or terms.get("edfdv") != "cubic-spline":
        raise ValueError("Eulerian source must use the requested spectral-x / cubic-spline-v solver")
    method = farsight_method(f_config)
    saved_method = runs["farsight"].get("method")
    if saved_method is not None and saved_method != method:
        raise ValueError("FARSIGHT method metadata disagrees with its AMR/field-solver configuration")
    for key, actual in (
        ("amr_enabled", f_config.get("amr", {}).get("enabled", False)),
        ("field_solver", f_config.get("numerical", {}).get("field_solver", "direct")),
    ):
        if key in runs["farsight"] and runs["farsight"][key] != actual:
            raise ValueError(f"FARSIGHT {key} metadata disagrees with its configuration")
    return field_model


def render_pair(eulerian: Path, farsight: Path, output: Path, *, experiment="farsight-comparison"):
    import xarray as xr

    from adept import Artifact, MetricEvent, RunRequest, RunStatus

    from .movies import farsight_method, render_comparison
    from .run import _services

    eulerian, farsight, output = (Path(p).resolve() for p in (eulerian, farsight, output))
    runs = {
        "eulerian": json.loads((eulerian / "run.json").read_text()),
        "farsight": json.loads((farsight / "run.json").read_text()),
    }
    f_config = json.loads((farsight / "config.json").read_text())
    e_config = json.loads((eulerian / "config.json").read_text())
    field_model = _validate_pair(
        runs,
        e_config,
        f_config,
        json.loads((eulerian / "case.json").read_text()),
        json.loads((farsight / "case.json").read_text()),
    )
    label = (
        "ordinary Poisson" if field_model == "poisson" else f"matched softened ε = {f_config['numerical']['epsilon']:g}"
    )
    method = farsight_method(f_config)
    name = f"{runs['farsight']['case']}-movie-{method}-{field_model}-{f_config['grid']['nx']}x{f_config['grid']['nv']}"
    output.mkdir(parents=True, exist_ok=False)
    tracker, sink = _services()
    request = RunRequest(
        experiment=experiment,
        name=name,
        tags={
            "comparison.phase": "render",
            "comparison.case": runs["farsight"]["case"],
            "comparison.eulerian_run_id": runs["eulerian"]["run_id"],
            "comparison.farsight_run_id": runs["farsight"]["run_id"],
            "comparison.field_model": field_model,
            "comparison.farsight_method": method,
        },
    )
    tracker.preflight(request)
    sink.preflight()
    handle = tracker.start(request)
    summary = {
        "run_id": handle.run_id,
        "name": name,
        "status": "RUNNING",
        "source_runs": runs,
        "farsight_method": method,
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"run_id": handle.run_id, "name": name}), flush=True)
    try:
        sink.validate(handle)

        # Load into memory and close files before movie encoding.
        def dataset(root, name):
            with xr.open_dataset(root / f"{name}.nc", engine="h5netcdf") as source:
                return source.load()

        paths = render_comparison(
            dataset(eulerian, "distribution"),
            dataset(farsight, "distribution"),
            dataset(eulerian, "scalars"),
            dataset(farsight, "scalars"),
            {"farsight": f_config, "eulerian": {"field_model": label}},
            output,
            title=f"{runs['farsight']['case']} — {method} / spectral-x cubic-v",
        )
        diagnostic = json.loads(paths["diagnostics"].read_text())
        tracker.log_metrics(
            handle,
            [
                MetricEvent(
                    {
                        "final_relative_l2_distribution_difference": diagnostic["relative_l2_distribution_difference"][
                            -1
                        ],
                        **{
                            f"{solver}_relative_{quantity}_change": diagnostic["native_scalars"][solver][
                                f"relative_{quantity}"
                            ][-1]
                            for solver in ("eulerian", "farsight")
                            for quantity in ("mass", "c2")
                        },
                        **{
                            f"farsight_representation_relative_{quantity}_change": diagnostic[
                                "farsight_representation"
                            ][f"relative_{quantity}"][-1]
                            for quantity in ("mass", "c2")
                        },
                    }
                )
            ],
        )
        summary.update(status="FINISHED", artifacts={key: str(path) for key, path in paths.items()})
        (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
        for path in [
            *paths.values(),
            output / "run.json",
            Path(__file__),
            Path(__file__).with_name("movies.py"),
            Path(__file__).with_name("representation.py"),
        ]:
            receipt = sink.put(handle, Artifact(path, artifact_path="comparison"))
            sink.verify(handle, receipt)
        tracker.finish(handle, RunStatus.FINISHED)
    except Exception as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
        try:
            receipt = sink.put(handle, Artifact(output / "run.json", artifact_path="comparison"))
            sink.verify(handle, receipt)
        except Exception as tracking_error:  # noqa: BLE001 — preserve the original render failure
            error.add_note(f"Failure artifact upload also failed: {type(tracking_error).__name__}: {tracking_error}")
        try:
            tracker.finish(handle, RunStatus.FAILED, error=summary["error"])
        except Exception as tracking_error:  # noqa: BLE001 — status failure must not replace the render failure
            error.add_note(f"Failure status update also failed: {type(tracking_error).__name__}: {tracking_error}")
        raise
    print(json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eulerian", type=Path, required=True)
    parser.add_argument("--farsight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="farsight-comparison")
    args = parser.parse_args(argv)
    render_pair(args.eulerian, args.farsight, args.output, experiment=args.experiment)


if __name__ == "__main__":
    main()
