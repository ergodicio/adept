"""Exercise the public CLI and its local and tracked artifact lifecycle."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from tests.test_farsight1d.test_builder import small_config


def invoke_cli(tmp_path, output, *extra):
    config = small_config()
    config["solver"] = "farsight-1d"
    config["grid"].update(nx=8, nv=8)
    config["time"].update(dt=0.02, tmax=0.02)
    config["numerical"]["epsilon"] = 1.0
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    root = Path(__file__).parents[2]
    environment = {**os.environ, "PYTHONPATH": str(root)}
    # A local SQLite test must never consult an ambient registry server.
    environment.pop("MLFLOW_TRACKING_URI", None)
    environment.pop("MLFLOW_REGISTRY_URI", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "adept.farsight1d",
            "--cfg",
            str(config_path),
            "--output",
            str(output),
            "--seed",
            "17",
            *extra,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )


def test_cli_writes_readable_datasets_final_state_and_manifest(tmp_path):
    output = tmp_path / "outputs"
    process = invoke_cli(tmp_path, output)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["tracking_backend"] == "null"
    assert summary["output"] == str(output.resolve())
    assert summary["metrics"]["final_valid"] == 1.0
    assert summary == json.loads((output / "run.json").read_text())

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["seed"] == 17
    assert manifest["structural_fingerprint"].startswith("sha256:")
    assert manifest["resolved_config"]["time"]["num_steps"] == 1
    assert manifest["raw_config"]["numerical"]["epsilon"] == 1.0
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics == summary["metrics"]

    with xr.open_dataset(output / "scalars.nc", engine="h5netcdf") as scalars:
        np.testing.assert_allclose(scalars.t, [0.0, 0.02])
        assert scalars.valid.dtype == np.dtype("int8")
        assert bool(scalars.valid.all())
        np.testing.assert_allclose(float(scalars.c2[-1]), metrics["final_c2"])
    with xr.open_dataset(output / "fields.nc", engine="h5netcdf") as fields:
        assert fields.electric_field.shape == (2, 8)
        np.testing.assert_allclose(fields.x, np.arange(8) * 1.5)
    with np.load(output / "final_state.npz", allow_pickle=False) as final_state:
        assert final_state["valid"].dtype == np.dtype("bool")
        assert bool(final_state["valid"])
        with xr.open_dataset(output / "distribution.nc", engine="h5netcdf") as distribution:
            assert distribution.f.shape == (2, 9, 9)
            for name in ("x", "v", "f"):
                np.testing.assert_array_equal(distribution[name][-1], final_state[name])


@pytest.mark.parametrize("empty", [False, True])
def test_cli_refuses_to_overwrite_existing_output_directory(tmp_path, empty):
    output = tmp_path / "outputs"
    output.mkdir()
    if not empty:
        (output / "existing.txt").write_text("preserve previous results\n")
    previous = {path.name: path.read_bytes() for path in output.iterdir()}
    process = invoke_cli(tmp_path, output)
    assert process.returncode != 0
    assert "FileExistsError" in process.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous


def test_cli_tracks_metrics_uploads_artifacts_and_finishes_a_local_run(tmp_path):
    from mlflow.tracking import MlflowClient

    output = tmp_path / "outputs"
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    process = invoke_cli(
        tmp_path,
        output,
        "--tracking-uri",
        tracking_uri,
        "--experiment",
        "farsight-cli-test",
        "--name",
        "one-step-smoke",
    )
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["tracking_backend"] == "mlflow"
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    run = client.get_run(summary["run_id"])
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "one-step-smoke"
    assert run.data.metrics["final_valid"] == 1.0
    assert run.data.metrics["final_remesh_count"] == 1.0
    assert run.data.tags["adept.structural_fingerprint"].startswith("sha256:")
    expected = {"scalars.nc", "fields.nc", "distribution.nc", "final_state.npz", "manifest.json", "metrics.json"}
    uploaded = client.list_artifacts(run.info.run_id, "farsight")
    assert {Path(artifact.path).name for artifact in uploaded} == expected
    download_directory = tmp_path / "downloaded"
    download_directory.mkdir()
    for name in expected:
        downloaded = Path(client.download_artifacts(run.info.run_id, f"farsight/{name}", str(download_directory)))
        assert downloaded.read_bytes() == (output / name).read_bytes()
