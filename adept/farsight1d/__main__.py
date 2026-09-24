"""Run a FARSIGHT config through the explicit host runtime, with local artifacts."""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FileAnalyzer:
    """Host-only adapter: persist the pure analyzer's datasets and full final state."""

    analyzer: Any
    directory: Path

    def analyze(self, result, manifest):
        import numpy as np

        from adept import Artifact

        report = self.analyzer.analyze(result, manifest)
        for name, dataset in report.result.items():
            # NetCDF does not have a boolean type. Preserve the validity flag as
            # a byte, while the NPZ retains the exact final-state dtypes.
            serializable = dataset.copy()
            for key in serializable:
                if serializable[key].dtype == bool:
                    serializable[key] = serializable[key].astype("int8")
            serializable.to_netcdf(self.directory / f"{name}.nc", engine="h5netcdf")
        np.savez(self.directory / "final_state.npz", **result.final_state)
        (self.directory / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
        (self.directory / "metrics.json").write_text(
            json.dumps({key: value for event in report.metrics for key, value in event.values.items()}, indent=2) + "\n"
        )
        artifacts = tuple(Artifact(path, artifact_path="farsight") for path in sorted(self.directory.iterdir()))
        return replace(report, artifacts=artifacts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the experimental fixed-panel FARSIGHT solver")
    parser.add_argument("--cfg", required=True, type=Path, help="YAML config, including its .yaml extension")
    parser.add_argument("--output", required=True, type=Path, help="New directory for local datasets and manifest")
    parser.add_argument("--seed", type=int, default=42, help="Recorded PRNG seed (the current solver is deterministic)")
    parser.add_argument("--tracking-uri", help="Optional MLflow URI; uploads the same local artifacts and metrics")
    parser.add_argument("--experiment", default="farsight-1d")
    parser.add_argument("--name", help="Optional tracked run name")
    args = parser.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    from adept import RunRequest, SimulationSpec, run_prepared, solver_registry

    config = yaml.safe_load(args.cfg.read_text())
    prepared = solver_registry.prepare(SimulationSpec.from_legacy_config(config), key=args.seed)
    # Refuse to overwrite a previous experiment, including an empty directory.
    directory = args.output.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    prepared = replace(prepared, analyzer=FileAnalyzer(prepared.analyzer, directory))
    services = {}
    if args.tracking_uri:
        from adept import MLflowArtifactSink, MLflowTracker

        services = {
            "tracker": MLflowTracker(tracking_uri=args.tracking_uri),
            "artifact_sink": MLflowArtifactSink(tracking_uri=args.tracking_uri),
        }
    completed = run_prepared(
        prepared,
        key=jax.random.key(args.seed),
        request=RunRequest(experiment=args.experiment, name=args.name or args.cfg.stem),
        **services,
    )
    summary = {
        "run_id": completed.handle.run_id,
        "tracking_backend": completed.handle.backend,
        "output": str(directory),
        "run_time_seconds": completed.run_time_seconds,
        "metrics": {key: value for event in completed.report.metrics for key, value in event.values.items()},
    }
    (directory / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
