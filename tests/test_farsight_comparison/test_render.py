"""Offline comparison-wrapper validation and tracking; no simulations or network."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import xarray as xr

from adept import RunHandle
from examples.farsight_comparison import movies, render, run
from examples.farsight_comparison.cases import get_case
from examples.farsight_comparison.eulerian import build_eulerian_config


class FakeTracker:
    def __init__(self):
        self.requests = []
        self.finished = []
        self.metrics = []
        self.fail_finish = False

    def preflight(self, request):
        pass

    def start(self, request):
        self.requests.append(request)
        return RunHandle("render-run", "mlflow")

    def log_metrics(self, handle, events):
        self.metrics.extend(events)

    def finish(self, handle, status, **kwargs):
        self.finished.append(status.value)
        if self.fail_finish:
            raise OSError("status service unavailable")


class FakeSink:
    def __init__(self):
        self.uploads = []
        self.verified = []
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
def inputs(tmp_path, monkeypatch):
    case = get_case("two-stream", nx=16, nv=32, tmax=1.0, dt=0.05, frame_dt=0.5)
    eulerian, farsight = tmp_path / "eulerian", tmp_path / "farsight"
    for directory, solver, config, run_id in (
        (
            eulerian,
            "eulerian-softened",
            build_eulerian_config(case, field_model="farsight-softened"),
            "eulerian-source",
        ),
        (farsight, "farsight", case.to_farsight_config(), "farsight-source"),
    ):
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "case.json").write_text(json.dumps(case.to_dict()))
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "status": "FINISHED",
                    "run_id": run_id,
                    "solver": solver,
                    "case": case.name,
                    "field_model": "farsight-softened",
                }
            )
        )
        # Tiny input arrays exercise the real load-and-close path. Rendering is
        # mocked: numerical reconstruction is covered separately in test_movies.
        for name in ("distribution", "scalars"):
            xr.Dataset({"sentinel": ("t", [1.0, 2.0])}, coords={"t": [0.0, 1.0]}).to_netcdf(
                directory / f"{name}.nc", engine="h5netcdf"
            )
    tracker, sink = FakeTracker(), FakeSink()
    monkeypatch.setattr(run, "_services", lambda: (tracker, sink))
    calls = []

    def fake_render(e_dist, f_dist, e_scalars, f_scalars, config, output, **kwargs):
        calls.append({"datasets": (e_dist, f_dist, e_scalars, f_scalars), "config": config, **kwargs})
        paths = {"movie": output / "comparison.mp4", "diagnostics": output / "diagnostics.json"}
        paths["movie"].write_bytes(b"synthetic test artifact; not a movie")
        paths["diagnostics"].write_text(
            json.dumps(
                {
                    "relative_l2_distribution_difference": [0.01, 0.02],
                    "native_scalars": {
                        "eulerian": {"relative_mass": [0, 1e-8], "relative_c2": [0, -1e-4]},
                        "farsight": {"relative_mass": [0, -1e-7], "relative_c2": [0, -2e-4]},
                    },
                    "farsight_representation": {"relative_mass": [0, -3e-7], "relative_c2": [0, -4e-4]},
                }
            )
        )
        return paths

    monkeypatch.setattr(movies, "render_comparison", fake_render)
    return SimpleNamespace(
        eulerian=eulerian,
        farsight=farsight,
        output=tmp_path / "render",
        tracker=tracker,
        sink=sink,
        calls=calls,
    )


def update_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_success_links_sources_labels_and_native_metrics(inputs):
    result = render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert result["status"] == "FINISHED"
    assert result["run_id"] == "render-run"
    assert result["source_runs"]["eulerian"]["run_id"] == "eulerian-source"
    assert result["source_runs"]["farsight"]["run_id"] == "farsight-source"
    tags = inputs.tracker.requests[0].tags
    assert tags["comparison.eulerian_run_id"] == "eulerian-source"
    assert tags["comparison.farsight_run_id"] == "farsight-source"
    assert tags["comparison.field_model"] == "farsight-softened"
    assert inputs.tracker.finished == ["FINISHED"]
    assert len(inputs.calls) == 1
    assert inputs.calls[0]["config"]["eulerian"]["field_model"] == "matched softened ε = 1.5"
    assert all(dataset.sentinel.values.tolist() == [1, 2] for dataset in inputs.calls[0]["datasets"])
    assert inputs.tracker.metrics[0].values["final_relative_l2_distribution_difference"] == 0.02
    assert inputs.tracker.metrics[0].values["farsight_relative_c2_change"] == -2e-4
    assert inputs.tracker.metrics[0].values["farsight_representation_relative_c2_change"] == -4e-4
    assert inputs.sink.verified == inputs.sink.uploads
    assert {Path(artifact.source).name for artifact in inputs.sink.uploads} == {
        "comparison.mp4",
        "diagnostics.json",
        "run.json",
        "render.py",
        "movies.py",
        "representation.py",
    }


@pytest.mark.parametrize("source", ["eulerian", "farsight"])
def test_rejects_failed_source_before_tracking_or_rendering(inputs, source):
    update_json(getattr(inputs, source) / "run.json", lambda value: value.update(status="FAILED"))
    with pytest.raises(ValueError, match="unfinished"):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert not inputs.output.exists()
    assert not inputs.tracker.requests and not inputs.calls


@pytest.mark.parametrize("source", ["eulerian", "farsight"])
def test_rejects_mismatched_case_before_tracking(inputs, source):
    update_json(getattr(inputs, source) / "case.json", lambda value: value.update(amplitude=0.2))
    with pytest.raises(ValueError, match="Case specifications differ"):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert not inputs.tracker.requests


@pytest.mark.parametrize("field_model", ["unknown", "poisson"])
def test_rejects_unknown_or_inconsistent_force_metadata(inputs, field_model):
    update_json(inputs.eulerian / "run.json", lambda value: value.update(field_model=field_model))
    with pytest.raises(ValueError, match=r"field_model|field model"):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert not inputs.tracker.requests


def test_rejects_config_force_model_mismatch(inputs):
    update_json(inputs.eulerian / "config.json", lambda value: value["benchmark"].update(field_model="poisson"))
    with pytest.raises(ValueError, match="field_model metadata disagree"):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert not inputs.tracker.requests


def test_poisson_control_is_not_labeled_matched_softened(inputs):
    update_json(
        inputs.eulerian / "run.json", lambda value: value.update(solver="eulerian-poisson", field_model="poisson")
    )
    update_json(inputs.eulerian / "config.json", lambda value: value["benchmark"].update(field_model="poisson"))
    render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert inputs.calls[0]["config"]["eulerian"]["field_model"] == "ordinary Poisson"


@pytest.mark.parametrize(
    "source,section,key,value,match",
    [
        ("farsight", "initial", "amplitude", 0.7, "initial condition"),
        ("farsight", "numerical", "epsilon", 0.1, "epsilon"),
        ("eulerian", "grid", "dt", 0.1, "grid or time"),
        ("eulerian", "terms", "edfdv", "exponential", "cubic-spline"),
    ],
)
def test_rejects_config_case_or_solver_mismatch(inputs, source, section, key, value, match):
    update_json(getattr(inputs, source) / "config.json", lambda config: config[section].update({key: value}))
    with pytest.raises(ValueError, match=match):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert not inputs.tracker.requests


def test_requires_distinct_source_run_ids(inputs):
    update_json(inputs.eulerian / "run.json", lambda value: value.update(run_id="farsight-source"))
    with pytest.raises(ValueError, match="distinct runs"):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)


def test_render_failure_marks_failed_uploads_summary_and_preserves_original_error(inputs, monkeypatch):
    def fail_render(*args, **kwargs):
        raise ArithmeticError("unremeshed geometry")

    monkeypatch.setattr(movies, "render_comparison", fail_render)
    inputs.tracker.fail_finish = True
    inputs.sink.fail_put = True
    with pytest.raises(ArithmeticError, match="unremeshed geometry") as error:
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    result = json.loads((inputs.output / "run.json").read_text())
    assert result["status"] == "FAILED" and result["run_id"] == "render-run"
    assert result["source_runs"]["farsight"]["run_id"] == "farsight-source"
    assert inputs.tracker.finished == ["FAILED"]
    assert Path(inputs.sink.uploads[0].source) == inputs.output / "run.json"
    assert len(error.value.__notes__) == 2


def test_refuses_existing_output_directory(inputs):
    inputs.output.mkdir()
    with pytest.raises(FileExistsError):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert not inputs.tracker.requests


def test_amr_treecode_method_label_and_finer_eulerian_grid_are_preserved(inputs):
    update_json(inputs.farsight / "config.json", lambda value: value.update(amr={"enabled": True, "max_level": 2}))
    update_json(inputs.farsight / "config.json", lambda value: value["numerical"].update(field_solver="treecode"))
    update_json(
        inputs.farsight / "run.json",
        lambda value: value.update(method="farsight-amr-treecode", field_solver="treecode", amr_enabled=True),
    )
    update_json(inputs.eulerian / "case.json", lambda value: value.update(nx=64, nv=128))
    update_json(inputs.eulerian / "config.json", lambda value: value["benchmark"]["case"].update(nx=64, nv=128))
    update_json(inputs.eulerian / "config.json", lambda value: value["grid"].update(nx=64, nv=128))
    result = render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert result["farsight_method"] == "farsight-amr-treecode"
    assert inputs.tracker.requests[0].tags["comparison.farsight_method"] == "farsight-amr-treecode"
    assert "farsight-amr-treecode" in inputs.calls[0]["title"]
    assert inputs.calls[0]["config"]["farsight"]["amr"]["enabled"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"method": "farsight-amr-treecode"},
        {"amr_enabled": True},
        {"field_solver": "treecode"},
    ],
)
def test_rejects_misleading_farsight_method_metadata(inputs, metadata):
    update_json(inputs.farsight / "run.json", lambda value: value.update(metadata))
    with pytest.raises(ValueError, match="metadata disagrees"):
        render.render_pair(inputs.eulerian, inputs.farsight, inputs.output)
    assert not inputs.tracker.requests
