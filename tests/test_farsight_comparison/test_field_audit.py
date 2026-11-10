"""Small offline saved-state audits with fake tracking; no simulation or network."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from adept import RunHandle
from examples.farsight_comparison import field_audit, run


@pytest.fixture
def saved():
    config = {
        "solver": "farsight-1d",
        "grid": {"nx": 4, "nv": 4, "xmin": 0.0, "xmax": 2.0, "vmin": -1.0, "vmax": 1.0},
        "time": {"tmin": 0.0, "tmax": 0.1, "dt": 0.05},
        "numerical": {
            "epsilon": 0.1,
            "quadrature": "trapezoid",
            "field_solver": "direct",
            "treecode": {"degree": 8, "theta": 0.5, "leaf_size": 2},
        },
        "amr": {"enabled": True, "max_level": 1, "max_panels": 6},
    }
    px, pv = [], []
    for xlo in (0.0, 1.0):
        for vlo in (-1.0, 0.0):
            x, v = np.meshgrid(np.linspace(xlo, xlo + 1, 3), np.linspace(vlo, vlo + 1, 3), indexing="ij")
            px.append(x.ravel())
            pv.append(v.ravel())
    # Invalid inactive coordinates must not contaminate zero source weights.
    x = np.concatenate((px, np.full((2, 9), np.nan)))
    v = np.concatenate((pv, np.full((2, 9), np.nan)))
    weights = np.zeros_like(x)
    weights[:4] = np.outer([1, 2, 1], [1, 2, 1]).ravel() / 16
    times = np.array([0.0, 0.1])
    values = [(1.0 + (0.1 + t) * np.cos(np.pi * x)) * np.exp(-(v**2)) for t in times]
    dataset = xr.Dataset(
        {
            "x": (("t", "panel", "node"), np.broadcast_to(x, (2, 6, 9))),
            "v": (("t", "panel", "node"), np.broadcast_to(v, (2, 6, 9))),
            "f": (("t", "panel", "node"), np.asarray(values)),
            "weights": (("t", "panel", "node"), np.broadcast_to(weights, (2, 6, 9))),
            "active": (("t", "panel"), np.tile([1, 1, 1, 1, 0, 0], (2, 1))),
            "panel_id": (("t", "panel"), np.tile([0, 1, 2, 3, -1, -1], (2, 1))),
            "level": (("t", "panel"), np.tile([0, 0, 0, 0, -1, -1], (2, 1))),
        },
        coords={"t": times},
    )
    return dataset, config


def _reference_field(targets, dataset, frame, epsilon=0.1):
    active = dataset.active.values[frame].astype(bool)
    sources = dataset.x.values[frame, active].ravel()
    charges = -(dataset.weights.values[frame, active] * dataset.f.values[frame, active]).ravel()
    separation = (targets[:, None] - sources[None, :]) / 2.0
    separation -= np.floor(separation + 0.5)
    alpha = epsilon / 2.0
    kernel = 0.5 * separation * np.sqrt(1 + 4 * alpha**2) / np.sqrt(separation**2 + alpha**2) - separation
    return kernel @ charges


def test_common_grid_and_offgrid_fields_use_identical_saved_sources(saved):
    dataset, config = saved
    fields, audit = field_audit.evaluate_saved_fields(dataset, config, nx=32, probe_count=37, chunk_size=8)
    np.testing.assert_array_equal(fields.x, np.arange(32) * 2.0 / 32)
    np.testing.assert_allclose(fields.probe_x, (np.arange(37) + 0.317) * 2.0 / 37)
    assert fields.x.values[-1] < 2.0 and fields.sizes["t"] == 2
    for frame in range(2):
        np.testing.assert_allclose(
            fields.electric_field_direct[frame], _reference_field(fields.x.values, dataset, frame), atol=2e-15
        )
        np.testing.assert_allclose(
            fields.probe_field_direct[frame], _reference_field(fields.probe_x.values, dataset, frame), atol=2e-15
        )
    np.testing.assert_allclose(fields.electric_field_treecode, fields.electric_field_direct, atol=1e-8)
    assert [entry["frame_role"] for entry in audit["offgrid_endpoints"]] == ["initial", "final"]
    assert audit["source_observation_nx"] == 4 and audit["common_observation_nx"] == 32
    assert audit["treecode_settings"] == {"degree": 8, "theta": 0.5, "leaf_size": 2}
    assert all(entry["active_panels"] == 4 for entry in audit["common_grid"])


def test_zero_field_and_charge_have_explicitly_undefined_relative_norms():
    result = field_audit.field_errors(np.zeros(5), np.zeros(5), np.zeros(7))
    assert result["relative_l2_error"] is None
    assert result["max_absolute_error_per_charge"] is None
    assert result["max_absolute_error"] == result["absolute_l2_error"] == 0
    result = field_audit.field_errors(np.full(5, 1e-17), np.full(5, 2e-17), np.ones(7))
    assert result["relative_l2_error"] is None
    assert result["absolute_l2_error"] > 0
    json.dumps(result, allow_nan=False)


def test_validation_uses_resolved_configuration_defaults(saved):
    dataset, config = saved
    del config["grid"]["xmin"]
    del config["time"]["tmin"]
    del config["numerical"]["quadrature"]
    fields, _ = field_audit.evaluate_saved_fields(dataset, config, nx=8, probe_count=9, chunk_size=8)
    np.testing.assert_allclose(
        fields.electric_field_direct[0], _reference_field(fields.x.values, dataset, 0), atol=2e-15
    )


@pytest.mark.parametrize("kind", ["inactive-weight", "quadrature", "geometry", "times"])
def test_invalid_saved_state_is_not_silently_repaired(saved, kind):
    dataset, config = saved
    dataset = dataset.copy(deep=True)
    if kind == "inactive-weight":
        dataset.weights.values[:, -1] = 1.0
    elif kind == "quadrature":
        dataset.weights.values[:, 0, 0] *= 2
    elif kind == "geometry":
        dataset.x.values[:, 0, 0] += 0.01
    else:
        dataset = dataset.assign_coords(t=[0.1, 0.0])
    with pytest.raises(ValueError):
        field_audit.evaluate_saved_fields(dataset, config, nx=8, probe_count=9)


class FakeTracker:
    def __init__(self):
        self.requests, self.finished, self.metrics = [], [], []
        self.fail_finish = False

    def preflight(self, request):
        pass

    def start(self, request):
        self.requests.append(request)
        return RunHandle("field-audit-run", "mlflow")

    def log_metrics(self, handle, events):
        self.metrics.extend(events)

    def finish(self, handle, status, **kwargs):
        self.finished.append(status.value)
        if self.fail_finish:
            raise OSError("status service unavailable")


class FakeSink:
    def __init__(self):
        self.uploads, self.verified = [], []
        self.fail_put = False

    def preflight(self):
        pass

    def validate(self, handle):
        pass

    def put(self, handle, artifact):
        self.uploads.append(artifact)
        if self.fail_put:
            raise OSError("artifact service unavailable")
        return artifact

    def verify(self, handle, receipt):
        self.verified.append(receipt)


@pytest.fixture
def inputs(saved, tmp_path, monkeypatch):
    dataset, config = saved
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    (source / "run.json").write_text(
        json.dumps(
            {
                "status": "FINISHED",
                "run_id": "source-run",
                "solver": "farsight",
                "case": "two-stream",
                "field_model": "farsight-softened",
                "method": "farsight-amr-direct",
                "field_solver": "direct",
            }
        )
    )
    dataset.to_netcdf(source / "distribution.nc", engine="h5netcdf")
    final = {name: np.asarray(dataset[name][-1]) for name in dataset.data_vars}
    np.savez(source / "final_state.npz", valid=np.asarray(True), **final)
    tracker, sink = FakeTracker(), FakeSink()
    monkeypatch.setattr(run, "_services", lambda tracking_uri=None: (tracker, sink))

    def provenance(output):
        (output / "source.zip").write_bytes(b"test provenance placeholder")
        result = {"source_archive_sha256": hashlib.sha256((output / "source.zip").read_bytes()).hexdigest()}
        (output / "provenance.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(run, "_provenance", provenance)
    return SimpleNamespace(
        source=source, output=tmp_path / "audit", tracker=tracker, sink=sink, dataset=dataset, config=config
    )


def test_loading_saved_final_only_is_labeled_final(inputs):
    (inputs.source / "distribution.nc").unlink()
    dataset, paths = field_audit.load_saved_states(inputs.source, inputs.config)
    assert dataset.sizes["t"] == 1 and len(paths) == 1
    _, audit = field_audit.evaluate_saved_fields(dataset, inputs.config, nx=8, probe_count=9)
    assert [entry["frame_role"] for entry in audit["offgrid_endpoints"]] == ["final"]


def test_appends_final_snapshot_when_distribution_ends_earlier(inputs):
    inputs.dataset.isel(t=slice(0, 1)).to_netcdf(inputs.source / "distribution.nc", engine="h5netcdf")
    dataset, paths = field_audit.load_saved_states(inputs.source, inputs.config)
    np.testing.assert_array_equal(dataset.t, [0.0, 0.1])
    assert len(paths) == 2


def test_mismatched_final_snapshot_is_rejected(inputs):
    dataset = inputs.dataset.copy(deep=True)
    dataset.f.values[-1, 0, 0] += 0.1
    dataset.to_netcdf(inputs.source / "distribution.nc", engine="h5netcdf")
    with pytest.raises(ValueError, match="disagree in f"):
        field_audit.load_saved_states(inputs.source, inputs.config)


def test_success_links_source_hashes_and_verifies_all_artifacts(inputs):
    result = field_audit.audit_run(inputs.source, inputs.output, nx=16, probe_count=17)
    assert result["status"] == "FINISHED" and result["run_id"] == "field-audit-run"
    audit = json.loads((inputs.output / "audit.json").read_text())
    assert audit["source"]["run_id"] == "source-run"
    expected_hash = hashlib.sha256((inputs.source / "distribution.nc").read_bytes()).hexdigest()
    assert audit["source"]["sha256"]["distribution.nc"] == expected_hash
    assert {Path(path).name for path in audit["source_code_sha256"]} == {
        "field_audit.py",
        "amr.py",
        "numerics.py",
        "treecode.py",
    }
    assert audit["evaluation_seconds_including_compilation"] > 0
    with xr.open_dataset(inputs.output / "fields.nc", engine="h5netcdf") as fields:
        assert fields.sizes["x"] == 16 and fields.attrs["source_run_id"] == "source-run"
    assert inputs.tracker.finished == ["FINISHED"]
    assert inputs.tracker.requests[0].tags["comparison.farsight_run_id"] == "source-run"
    assert inputs.sink.verified == inputs.sink.uploads
    assert {Path(artifact.source).name for artifact in inputs.sink.uploads} >= {
        "run.json",
        "fields.nc",
        "audit.json",
        "source_config.json",
        "source_run.json",
        "provenance.json",
        "source.zip",
    }


def test_source_failure_is_rejected_before_tracking(inputs):
    source_run = json.loads((inputs.source / "run.json").read_text())
    source_run["status"] = "FAILED"
    (inputs.source / "run.json").write_text(json.dumps(source_run))
    with pytest.raises(ValueError, match="FINISHED"):
        field_audit.audit_run(inputs.source, inputs.output)
    assert not inputs.output.exists() and not inputs.tracker.requests


def test_existing_output_directory_is_never_overwritten(inputs):
    inputs.output.mkdir()
    with pytest.raises(FileExistsError):
        field_audit.audit_run(inputs.source, inputs.output)
    assert not inputs.tracker.requests


def test_nersc_login_node_is_rejected_before_field_work(inputs, monkeypatch):
    from examples.farsight_comparison import scan

    monkeypatch.setenv("NERSC_HOST", "perlmutter")
    monkeypatch.setattr(scan.socket, "gethostname", lambda: "perlmutter-login01")
    with pytest.raises(RuntimeError, match="non-compute"):
        field_audit.audit_run(inputs.source, inputs.output)
    assert not inputs.output.exists() and not inputs.tracker.requests


def test_audit_failure_is_preserved_when_failure_tracking_also_fails(inputs, monkeypatch):
    def fail(*args, **kwargs):
        raise ArithmeticError("field evaluator failed")

    monkeypatch.setattr(field_audit, "evaluate_saved_fields", fail)
    inputs.tracker.fail_finish = inputs.sink.fail_put = True
    with pytest.raises(ArithmeticError, match="field evaluator failed") as caught:
        field_audit.audit_run(inputs.source, inputs.output)
    summary = json.loads((inputs.output / "run.json").read_text())
    assert summary["status"] == "FAILED" and summary["source"]["run_id"] == "source-run"
    assert inputs.tracker.finished == ["FAILED"]
    assert len(caught.value.__notes__) == 2


def test_cli_routes_explicit_settings_without_launching_a_simulation(monkeypatch):
    calls = []
    monkeypatch.setattr(field_audit, "audit_run", lambda **kwargs: calls.append(kwargs))
    field_audit.main(["--source", "/tmp/source", "--output", "/tmp/audit", "--nx", "256", "--probe-count", "509"])
    assert calls[0]["nx"] == 256 and calls[0]["probe_count"] == 509
    assert calls[0]["source"] == Path("/tmp/source")
