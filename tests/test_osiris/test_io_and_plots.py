"""Tests for the io.py loaders and plots.py canned views."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from adept.osiris import io as oio
from adept.osiris import plots as oplt

# Point this at a completed two-stream OSIRIS run to exercise the io/plots
# loaders against real data; the tests skip cleanly when it is unset.
EXISTING_RUN_ENV = "OSIRIS_TWOSTREAM_RUN"


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(EXISTING_RUN_ENV)
    if not raw or not Path(raw).is_dir():
        pytest.skip(f"set {EXISTING_RUN_ENV} to an existing two-stream run dir")
    return Path(raw)


def test_load_grid_h5_shape_and_coords(run_dir: Path) -> None:
    da = oio.load_grid_h5(run_dir / "MS" / "FLD" / "e1" / "e1-000600.h5")
    assert da.dims == ("x1",)
    assert da.shape == (64,)
    assert "time" in da.attrs and "iter" in da.attrs
    assert da.attrs["iter"] == 600
    assert da.coords["x1"].size == 64


def test_load_phasespace_shape(run_dir: Path) -> None:
    p = next((run_dir / "MS/PHA/x1p1/beam_pos").glob("*.h5"))
    da = oio.load_phasespace_h5(p)
    assert da.ndim == 2
    assert set(da.dims) <= {"x1", "p1"}


def test_load_series_stacks_time(run_dir: Path) -> None:
    da = oio.load_series(run_dir / "MS" / "FLD" / "e1")
    assert da.dims[0] == "t"
    # Two-stream baseline ran 30/0.05 = 600 steps, 601 snapshots saved.
    assert da.shape[0] == 601
    assert da.shape[1] == 64
    t = da.coords["t"].values
    assert t[0] == 0.0
    assert np.isclose(t[-1], 30.0, atol=0.05)


def test_list_diagnostics(run_dir: Path) -> None:
    diags = oio.list_diagnostics(run_dir)
    assert "FLD/e1" in diags
    assert any(k.startswith("PHA/x1p1/") for k in diags)


def test_field_energy_series_nontrivial(run_dir: Path) -> None:
    series = oplt.field_energy_series(run_dir)
    assert series.dims == ("t",)
    assert series.size > 0
    assert np.all(series.values >= 0)


def test_omega_k_returns_axes(run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ser = oio.load_series(run_dir / "MS/FLD/e1")
    fig, ax = plt.subplots()
    out = oplt.plot_omega_k(ser, ax=ax, show_em=True, show_langmuir=False)
    assert out is ax
    # Y limits should bracket some non-zero omega.
    ylo, yhi = ax.get_ylim()
    assert ylo < 0 < yhi
    plt.close(fig)


def test_save_canned_plots_writes_all_expected(tmp_path: Path, run_dir: Path) -> None:
    written = oplt.save_canned_plots(run_dir, tmp_path, v_th=0.0707)
    # At minimum: one spacetime + omega_k for e1, energy_vs_time, both species PS.
    assert "spacetime/e1" in written
    assert "omega_k/e1" in written
    assert "energy_vs_time" in written
    assert "phasespace/beam_pos/x1p1" in written
    assert "phasespace/beam_neg/x1p1" in written
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0
