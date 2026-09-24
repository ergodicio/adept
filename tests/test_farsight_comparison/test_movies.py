"""Tests for representation-faithful, non-misleading comparison rendering."""

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from examples.farsight_comparison import movies
from examples.farsight_comparison.movies import reconstruct_fixed_panels, render_comparison


@pytest.fixture
def data():
    grid = {"nx": 8, "nv": 6, "xmin": -1.0, "xmax": 3.0, "vmin": -2.0, "vmax": 2.0}
    x, v = np.meshgrid(np.linspace(-1, 3, 9), np.linspace(-2, 2, 7), indexing="ij")

    def polynomial(x, v):
        # A genuinely biquadratic polynomial with matching periodic endpoints.
        return 1.0 + (x + 1) * (3 - x) * (1 + 0.2 * v + 0.1 * v**2)

    t = np.array([0.0, 0.5, 1.0])
    f = np.stack([polynomial(x, v) * (1 + 0.01 * time) for time in t])
    farsight = xr.Dataset(
        {
            name: (("t", "x_node", "v_node"), values)
            for name, values in (("x", np.broadcast_to(x, f.shape)), ("v", np.broadcast_to(v, f.shape)), ("f", f))
        },
        coords={"t": t},
    )
    xe = np.linspace(-1, 3, 12, endpoint=False) + 4 / 24
    ve = np.linspace(-2, 2, 16, endpoint=False) + 4 / 32
    eulerian = xr.Dataset(
        {"f": (("t", "x", "v"), np.stack([polynomial(xe[:, None], ve[None, :]) * (1 + 0.01 * time) for time in t]))},
        coords={"t": t, "x": xe, "v": ve},
    )
    scalars = xr.Dataset({"mass": ("t", [1.0, 1.01, 1.02]), "c2": ("t", [2.0, 2.02, 2.04])}, coords={"t": t})
    config = {
        "farsight": {"grid": grid, "numerical": {"epsilon": 0.5}},
        "eulerian": {"field_model": "unsoftened periodic Poisson"},
    }
    return farsight, eulerian, scalars, config, polynomial


def test_exact_biquadratic_and_panel_seams(data):
    farsight, _, _, config, polynomial = data
    # Include both sides and the exact values of internal panel seams.
    x = np.array([-1, -0.01, 0, 1e-10, 0.7, 1, 2, 2.9, 3])
    v = np.array([-2, -2 / 3, 0.01, 2 / 3, 2])
    result = reconstruct_fixed_panels(farsight, x, v, config["farsight"]["grid"])
    expected = polynomial(x[:, None], v[None, :])
    np.testing.assert_allclose(result[0], expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(result[-1], 1.01 * expected, rtol=2e-15, atol=2e-15)


def test_periodic_x_and_zero_velocity_exterior(data):
    farsight, _, _, config, _ = data
    x = np.array([-5.0, -1.0, 3.0, 7.0, -0.8, 3.2])
    v = np.array([-2.001, -2.0, 0.0, 2.0, 2.001])
    values = reconstruct_fixed_panels(farsight, x, v, config["farsight"]["grid"])
    np.testing.assert_array_equal(values[:, :, [0, -1]], 0)
    for index in (1, 2, 3):
        np.testing.assert_allclose(values[:, index], values[:, 0])
    np.testing.assert_allclose(values[:, 4], values[:, 5])
    assert np.all(values[:, :, 1:-1] > 0)


def test_reproduces_all_nodes_for_arbitrary_piecewise_panels(data):
    farsight, _, _, config, _ = data
    rng = np.random.default_rng(2026)
    values = rng.normal(size=farsight.f.shape)
    values[:, -1] = values[:, 0]
    farsight = farsight.copy(deep=True)
    farsight["f"] = (farsight.f.dims, values)
    result = reconstruct_fixed_panels(
        farsight, farsight.x.values[0, :, 0], farsight.v.values[0, 0, :], config["farsight"]["grid"]
    )
    np.testing.assert_allclose(result, values, atol=2e-14)


@pytest.mark.parametrize("coordinate", ["x", "v"])
def test_rejects_deformed_or_unremeshed_geometry(data, coordinate):
    farsight, _, _, config, _ = data
    farsight = farsight.copy(deep=True)
    farsight[coordinate].values[1, 2, 3] += 0.01
    with pytest.raises(ValueError, match="moving/deformed"):
        reconstruct_fixed_panels(farsight, [0], [0], config["farsight"]["grid"])


def test_rejects_amr_geometry(data):
    farsight, _, _, config, _ = data
    farsight["active"] = ("panel", [True])
    with pytest.raises(ValueError, match="AMR"):
        reconstruct_fixed_panels(farsight, [0], [0], config["farsight"]["grid"])


def test_rejects_inconsistent_periodic_seam(data):
    farsight, _, _, config, _ = data
    farsight = farsight.copy(deep=True)
    farsight.f.values[0, -1, 2] += 0.1
    with pytest.raises(ValueError, match="periodic endpoints"):
        reconstruct_fixed_panels(farsight, [0], [0], config["farsight"]["grid"])


def test_static_artifacts_and_separate_native_diagnostics(data, tmp_path):
    farsight, eulerian, scalars, config, _ = data
    other_scalars = scalars.copy(deep=True)
    other_scalars["c2"] = ("t", [3.0, 3.06, 3.12])
    paths = render_comparison(eulerian, farsight, scalars, other_scalars, config, tmp_path, make_movie=False)
    assert paths.keys() == {"contact_sheet", "conservation", "diagnostics"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    diagnostics = json.loads(paths["diagnostics"].read_text())
    np.testing.assert_allclose(diagnostics["relative_l2_distribution_difference"], 0, atol=1e-15)
    assert diagnostics["native_scalars"]["eulerian"]["relative_c2"][-1] == pytest.approx(0.02)
    assert diagnostics["native_scalars"]["farsight"]["relative_c2"][-1] == pytest.approx(0.04)
    assert diagnostics["eulerian_cells"] == [12, 16]
    assert diagnostics["farsight_intervals"] == [8, 6]
    assert diagnostics["farsight_field_model"]["epsilon"] == 0.5
    assert "unsoftened" in diagnostics["eulerian_field_model"]
    assert "c2_negative" not in diagnostics["native_scalars"]["farsight"]
    assert diagnostics["native_scalar_notes"]["initial_limiter"] is None
    with pytest.raises(FileExistsError, match="already exist"):
        render_comparison(eulerian, farsight, scalars, scalars, config, tmp_path, make_movie=False)


def test_optional_native_sign_and_limiter_scalars_are_not_dropped(data, tmp_path):
    farsight, eulerian, scalars, config, _ = data
    diagnostic_names = (
        "c2_positive",
        "c2_negative",
        "positive_mass",
        "negative_mass",
        "negative_node_count",
        "min_f",
        "positivity_failed_panels",
        "min_bernstein_coefficient",
        "initial_positivity_c2_change",
        "initial_positivity_polynomial_c2_change",
        "initial_positivity_mass_change",
        "initial_positivity_limited_panels",
        "initial_positivity_failed_panels",
        "initial_positivity_min_theta",
        "interpolation_c2_change",
        "source_limiter_c2_change",
        "source_limiter_mass_change",
        "source_limiter_panels",
        "source_limiter_min_theta",
        "destination_limiter_c2_change",
        "destination_limiter_mass_change",
        "destination_limiter_polynomial_c2_change",
        "destination_limiter_panels",
        "destination_limiter_min_theta",
        "regrid_c2_change",
        "remap_c2_change",
        "remap_c2_abs_change",
    )
    enriched = scalars.copy(deep=True)
    for index, name in enumerate(diagnostic_names):
        enriched[name] = ("t", np.array([-0.25, 0.0, 0.5]) + index)
    paths = render_comparison(eulerian, farsight, scalars, enriched, config, tmp_path, make_movie=False)
    diagnostic = json.loads(paths["diagnostics"].read_text())
    native = diagnostic["native_scalars"]
    for name in diagnostic_names:
        np.testing.assert_array_equal(native["farsight"][name], enriched[name])
        assert name not in native["eulerian"]
    np.testing.assert_array_equal(native["farsight"]["relative_c2"], (scalars.c2 - scalars.c2[0]) / scalars.c2[0])
    assert "either sign" in diagnostic["native_scalar_notes"]["stage_budgets"]
    assert "not inferred to be zero" in diagnostic["native_scalar_notes"]["availability"]


@pytest.mark.parametrize(
    "name", ["c2_negative", "negative_mass", "initial_positivity_c2_change", "source_limiter_panels"]
)
def test_optional_native_scalars_require_finite_time_series(data, name):
    _, _, scalars, _, _ = data
    invalid = scalars.copy(deep=True)
    invalid[name] = ("t", [0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match=rf"{name}.*finite"):
        movies._scalar_data(invalid, "FARSIGHT", np.asarray(scalars.t))
    invalid[name] = (("t", "extra"), np.zeros((3, 1)))
    with pytest.raises(ValueError, match=rf"{name} must have dimension"):
        movies._scalar_data(invalid, "FARSIGHT", np.asarray(scalars.t))


def test_rejects_unequal_saved_times(data, tmp_path):
    farsight, eulerian, scalars, config, _ = data
    farsight = farsight.assign_coords(t=[0, 0.51, 1])
    with pytest.raises(ValueError, match="times must match"):
        render_comparison(eulerian, farsight, scalars, scalars, config, tmp_path, make_movie=False)


def test_rejects_mismatched_domains(data, tmp_path):
    farsight, eulerian, scalars, config, _ = data
    eulerian = eulerian.assign_coords(x=eulerian.x + 0.1)
    with pytest.raises(ValueError, match="domain must match"):
        render_comparison(eulerian, farsight, scalars, scalars, config, tmp_path, make_movie=False)


def test_requires_explicit_field_model(data, tmp_path):
    farsight, eulerian, scalars, config, _ = data
    del config["eulerian"]["field_model"]
    with pytest.raises(ValueError, match="field_model"):
        render_comparison(eulerian, farsight, scalars, scalars, config, tmp_path, make_movie=False)


def test_rejects_nonfinite_distribution(data):
    farsight, _, _, config, _ = data
    farsight = farsight.copy(deep=True)
    farsight.f.values[1, 3, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reconstruct_fixed_panels(farsight, [0], [0], config["farsight"]["grid"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="No existing ffmpeg encoder")
def test_mp4_encoder_smoke(data, tmp_path):
    farsight, eulerian, scalars, config, _ = data
    paths = render_comparison(eulerian, farsight, scalars, scalars, config, tmp_path, fps=6)
    assert paths["movie"].stat().st_size > 1000
    assert paths["movie"].read_bytes()[4:8] == b"ftyp"


def test_ffmpeg_prefers_system_with_h264(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/system/ffmpeg")
    calls = []

    def encoders(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout=" V....D libx264 H.264 encoder\n", stderr="")

    monkeypatch.setattr(subprocess, "run", encoders)
    assert movies._ffmpeg_path() == "/system/ffmpeg"
    assert calls == [["/system/ffmpeg", "-hide_banner", "-encoders"]]


def test_ffmpeg_uses_bundled_encoder_when_system_lacks_h264(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/system/ffmpeg")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "/bundled/ffmpeg"))
    calls = []

    def encoders(command, **kwargs):
        calls.append(command[0])
        output = " V....D libx264 H.264 encoder" if command[0] == "/bundled/ffmpeg" else " V..... mpeg4 MPEG-4 encoder"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", encoders)
    assert movies._ffmpeg_path() == "/bundled/ffmpeg"
    assert calls == ["/system/ffmpeg", "/bundled/ffmpeg"]


@pytest.mark.parametrize("bundled_available", [True, False])
def test_ffmpeg_without_usable_h264_fails_clearly(monkeypatch, bundled_available):
    monkeypatch.setattr(shutil, "which", lambda _: "/system/ffmpeg")
    bundled = SimpleNamespace(get_ffmpeg_exe=lambda: "/bundled/ffmpeg") if bundled_available else None
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", bundled)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" V..... mpeg4 MPEG-4 encoder", stderr=""),
    )
    with pytest.raises(RuntimeError, match="libx264 encoder"):
        movies._ffmpeg_path()


def test_ffmpeg_probe_timeout_is_not_usable(monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    assert not movies._has_h264_encoder("/broken/ffmpeg")


@pytest.fixture
def amr_data():
    """Independent nx=nv=4 hierarchy: refine root 0, then all four roots."""
    grid = {"nx": 4, "nv": 4, "xmin": 0.0, "xmax": 4.0, "vmin": -2.0, "vmax": 2.0}
    capacity, times = 18, [0.0, 0.5]
    arrays = {name: np.full((2, capacity, 9), np.nan) for name in ("x", "v", "f")}
    active = np.zeros((2, capacity), dtype=np.int8)
    levels = np.full((2, capacity), -1, dtype=int)
    panel_ids = np.full((2, capacity), -1, dtype=int)

    def polynomial(x, v):
        return 1 + x * (4 - x) * (1 + 0.2 * v + 0.1 * v**2)

    for frame, leaves in enumerate(([1, 2, 3, 4, 5, 8, 9], list(range(4, 20)))):
        for slot, panel_id in enumerate(leaves):
            level = int(panel_id >= 4)
            offset, side = (4, 4) if level else (0, 2)
            ix, iv = divmod(panel_id - offset, side)
            x = (2 * ix + np.arange(3)) / 2**level
            v = -2 + (2 * iv + np.arange(3)) / 2**level
            px, pv = np.meshgrid(x, v, indexing="ij")
            arrays["x"][frame, slot] = px.ravel()
            arrays["v"][frame, slot] = pv.ravel()
            arrays["f"][frame, slot] = (1 + times[frame]) * polynomial(px, pv).ravel()
            active[frame, slot], levels[frame, slot], panel_ids[frame, slot] = 1, level, panel_id
    dataset = xr.Dataset(
        {
            **{name: (("t", "panel", "node"), values) for name, values in arrays.items()},
            "active": (("t", "panel"), active),
            "level": (("t", "panel"), levels),
            "panel_id": (("t", "panel"), panel_ids),
        },
        coords={"t": times},
    )
    return dataset, grid, polynomial


def test_amr_exact_biquadratic_across_coarse_fine_boundaries_and_regridding(amr_data):
    dataset, grid, polynomial = amr_data
    x = np.array([0, 0.2, 1, 2, 2.00001, 3, 3.9, 4])
    v = np.array([-2, -1, -0.3, 0, 1, 2])
    values = movies.reconstruct_amr_panels(dataset, x, v, grid, max_level=1)
    expected = polynomial(x[:, None], v[None, :])
    np.testing.assert_allclose(values[0], expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(values[1], 1.5 * expected, rtol=2e-15, atol=2e-15)
    frames = list(movies.iter_validated_amr_frames(dataset, grid, {"max_level": 1}))
    assert [frame[0].shape for frame in frames] == [(7, 9), (16, 9)]
    assert all(np.isfinite(frame[2]).all() for frame in frames)


def test_amr_finest_boundary_owner_and_deterministic_same_level_ties(amr_data):
    dataset, grid, _ = amr_data
    dataset = dataset.isel(t=[0]).copy(deep=True)
    for slot in np.flatnonzero(dataset.active.values[0]):
        dataset.f.values[0, slot] = dataset.panel_id.values[0, slot] + 1
    x, v = np.array([0, 1, 2 - 1e-5, 2, 2 + 1e-5, 4, 8]), np.array([-1.5, -1, 0, 1])
    result = movies.reconstruct_amr_panels(dataset, x, v, grid, max_level=1)[0]
    assert result[0, 0] == 5  # fine ID 4 beats coarse periodic neighbor ID 2
    assert result[1, 1] == 5  # four equal-level leaves: smallest ID 4 wins
    assert result[2, 0] == pytest.approx(9)
    assert result[3, 0] == 9  # fine ID 8 beats coarse ID 2 on x=2
    assert result[4, 0] == pytest.approx(3)
    assert result[0, 2] == 6  # fine ID 5 beats coarse ID 1 on v=0
    assert result[0, 3] == 2  # periodic equal-level tie: ID 1 beats ID 3
    np.testing.assert_array_equal(result[0], result[5])
    np.testing.assert_array_equal(result[0], result[6])
    # Packed slots may change after remeshing; reference IDs determine ties.
    reversed_slots = dataset.isel(panel=slice(None, None, -1))
    np.testing.assert_array_equal(movies.reconstruct_amr_panels(reversed_slots, x, v, grid, max_level=1)[0], result)


def test_amr_velocity_edges_and_zero_exterior(amr_data):
    dataset, grid, polynomial = amr_data
    values = movies.reconstruct_amr_panels(dataset, [0.5], [-2.1, -2, 2, 2.1], grid, max_level=1)
    np.testing.assert_array_equal(values[:, :, [0, -1]], 0)
    np.testing.assert_allclose(values[0, 0, 1:3], polynomial(0.5, np.array([-2, 2])))


def test_amr_rejects_advected_coordinates(amr_data):
    dataset, grid, _ = amr_data
    dataset.x.values[0, 0, 4] += 0.001
    with pytest.raises(ValueError, match="moving/deformed"):
        movies.reconstruct_amr_panels(dataset, [0], [0], grid, max_level=1)


def test_amr_rejects_uncovered_leaf_even_outside_query_region(amr_data):
    dataset, grid, _ = amr_data
    dataset.active.values[0, 0] = 0
    with pytest.raises(ValueError, match="uncovered reference cells"):
        movies.reconstruct_amr_panels(dataset, [3], [-1], grid, max_level=1)


def test_amr_rejects_overlapping_parent_and_child(amr_data):
    dataset, grid, _ = amr_data
    dataset.active.values[0, 7] = 1
    dataset.level.values[0, 7], dataset.panel_id.values[0, 7] = 0, 0
    x, v = np.meshgrid([0, 1, 2], [-2, -1, 0], indexing="ij")
    dataset.x.values[0, 7], dataset.v.values[0, 7], dataset.f.values[0, 7] = x.ravel(), v.ravel(), np.ones(9)
    with pytest.raises(ValueError, match="overlapping active interiors"):
        movies.reconstruct_amr_panels(dataset, [3], [-1], grid, max_level=1)


def test_amr_rejects_mismatched_level_and_panel_id(amr_data):
    dataset, grid, _ = amr_data
    dataset.level.values[0, 0] = 1
    with pytest.raises(ValueError, match="IDs disagree"):
        movies.reconstruct_amr_panels(dataset, [0], [0], grid, max_level=1)


@pytest.mark.parametrize("field_solver", ["direct", "treecode"])
@pytest.mark.parametrize("limiter", ["none", "bernstein"])
def test_amr_render_reports_actual_method_and_active_panels(amr_data, tmp_path, field_solver, limiter, monkeypatch):
    from matplotlib.figure import Figure

    dataset, grid, polynomial = amr_data
    x, v = (np.arange(12) + 0.5) / 3, -2 + (np.arange(16) + 0.5) / 4
    f = np.stack([(1 + time) * polynomial(x[:, None], v[None, :]) for time in dataset.t.values])
    eulerian = xr.Dataset({"f": (("t", "x", "v"), f)}, coords={"t": dataset.t, "x": x, "v": v})
    scalars = xr.Dataset({"mass": ("t", [1, 1]), "c2": ("t", [2, 2])}, coords={"t": dataset.t})
    if limiter == "bernstein":
        scalars["initial_positivity_c2_change"] = ("t", [-0.012, -0.012])
        scalars["initial_positivity_mass_change"] = ("t", [0.0, 0.0])
        scalars["initial_positivity_polynomial_c2_change"] = ("t", [-0.01, -0.01])
    config = {
        "farsight": {
            "grid": grid,
            "amr": {"enabled": True, "max_level": 1},
            "numerical": {
                "epsilon": 0.3,
                "field_solver": field_solver,
                "positivity_limiter": limiter,
                "quadrature": "simpson",
            },
        },
        "eulerian": {"field_model": "matched softened ε = 0.3"},
    }
    footers = []
    original_supxlabel = Figure.supxlabel

    def capture_footer(figure, text, *args, **kwargs):
        footers.append(text)
        return original_supxlabel(figure, text, *args, **kwargs)

    monkeypatch.setattr(Figure, "supxlabel", capture_footer)
    make_movie = field_solver == "treecode" and limiter == "none" and shutil.which("ffmpeg") is not None
    paths = render_comparison(eulerian, dataset, scalars, scalars, config, tmp_path, make_movie=make_movie)
    if make_movie:
        assert paths["movie"].read_bytes()[4:8] == b"ftyp"
    diagnostics = json.loads(paths["diagnostics"].read_text())
    assert diagnostics["farsight_method"] == f"farsight-amr-{field_solver}" + (
        "-bernstein" if limiter == "bernstein" else ""
    )
    assert diagnostics["farsight_amr"]["active_panels"] == [7, 16]
    assert diagnostics["farsight_amr"]["capacity"] == 18
    assert "AMR reference leaves" in diagnostics["reconstruction"]
    assert diagnostics["farsight_representation"]["relative_c2"][-1] == pytest.approx(1.25)
    np.testing.assert_allclose(diagnostics["relative_l2_distribution_difference"], 0, atol=1e-15)
    if limiter == "bernstein":
        assert "excluded from relative drift" in diagnostics["native_scalar_notes"]["initial_limiter"]
        assert any("initial native ΔC2 = -1.200e-02" in text for text in footers)
        assert any("initial polynomial ΔC2 = -1.000e-02" in text for text in footers)
    else:
        assert diagnostics["native_scalar_notes"]["initial_limiter"] is None
        assert all("Initial limiting" not in text for text in footers)


def test_amr_exact_polynomial_integrals_are_independent_of_leaf_partition(amr_data):
    from examples.farsight_comparison.representation import representation_integrals

    dataset, grid, _ = amr_data
    result = representation_integrals(dataset, {"grid": grid, "amr": {"enabled": True, "max_level": 1}})
    # Analytic integrals of 1+x(4-x)(1+v/5+v²/10), x∈[0,4], v∈[-2,2].
    mass, c2 = 2896 / 45, 1672336 / 5625
    np.testing.assert_allclose(result["mass"], [mass, 1.5 * mass], rtol=2e-15)
    np.testing.assert_allclose(result["c2"], [c2, 2.25 * c2], rtol=2e-15)
    np.testing.assert_allclose(result["relative_mass"], [0, 0.5], atol=1e-15)
    np.testing.assert_allclose(result["relative_c2"], [0, 1.25], atol=1e-15)
