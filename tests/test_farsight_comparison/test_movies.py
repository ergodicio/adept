"""Tests for representation-faithful, non-misleading comparison rendering."""

import json
import shutil

import numpy as np
import pytest
import xarray as xr

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
    with pytest.raises(FileExistsError, match="already exist"):
        render_comparison(eulerian, farsight, scalars, scalars, config, tmp_path, make_movie=False)


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
