"""The parity harness (adept/_lpse2d/parity): reference-run location and the comparison
metrics that reproduce the 2026-09-14 ``log_lpse_reference.py`` baseline on MLflow."""

import os
from pathlib import Path

import numpy as np
import pytest

from adept._lpse2d.parity import (
    DEFAULT_WINDOWS,
    compare,
    deck_path,
    energy_floor_crossings,
    growth_rate,
    lpse_root,
    parse_windows,
    read_flux,
    reference_run_dir,
)


def test_parse_windows_and_growth_rate():
    assert parse_windows("0.3-0.9,0.9-1.5") == ((0.3, 0.9), (0.9, 1.5))
    t = np.linspace(0.0, 2.0, 201)
    e = 1e-3 * np.exp(4.0 * t)
    assert growth_rate(t, e, 0.5, 1.5) == pytest.approx(4.0, rel=1e-10)
    # fewer than four positive samples in the window -> nan, as the reference script
    assert np.isnan(growth_rate(t, e, 0.5, 0.52))
    assert np.isnan(growth_rate(t, np.zeros_like(t), 0.5, 1.5))


def _synthetic(gamma_adept=3.0, gamma_lpse=2.5, gamma_raman=4.0):
    t = np.linspace(0.0, 2.0, 401)
    tl = np.linspace(0.0, 2.0, 201)
    series = {
        "t (ps)": t,
        "epw_energy": 1e-4 * np.exp(gamma_adept * t),
        "e1_sq": 1e-6 * np.exp(gamma_raman * t),
        "iaw_density_abs_max": 1e-3 * (1.0 + t),
    }
    lpse = {
        "time": tl,
        "EPW_energy": 1e-4 * np.exp(gamma_lpse * tl),
        "E1_energy": 1e-6 * np.exp(gamma_raman * tl),
        "Nelf_max": 2e-3 * (1.0 + tl),
    }
    return series, lpse


def test_compare_keys_and_values_match_the_reference_script():
    series, lpse = _synthetic()
    m = compare(series, lpse, "0.5-1.0,1.0-1.5")
    assert set(m) == {
        "epw_energy_growth_adept_0.5_1ps",
        "epw_energy_growth_lpse_0.5_1ps",
        "epw_energy_growth_ratio_0.5_1ps",
        "raman_energy_growth_adept_0.5_1ps",
        "raman_energy_growth_lpse_0.5_1ps",
        "raman_energy_growth_ratio_0.5_1ps",
        "epw_energy_growth_adept_1_1.5ps",
        "epw_energy_growth_lpse_1_1.5ps",
        "epw_energy_growth_ratio_1_1.5ps",
        "raman_energy_growth_adept_1_1.5ps",
        "raman_energy_growth_lpse_1_1.5ps",
        "raman_energy_growth_ratio_1_1.5ps",
        "epw_half_max_time_adept_ps",
        "epw_half_max_time_lpse_ps",
        "iaw_max_adept",
        "iaw_max_lpse",
    }
    assert m["epw_energy_growth_ratio_0.5_1ps"] == pytest.approx(3.0 / 2.5, rel=1e-9)
    assert m["raman_energy_growth_ratio_1_1.5ps"] == pytest.approx(1.0, rel=1e-9)
    # half max of exp(g t) on [0, 2]: t = 2 - ln 2 / g
    assert m["epw_half_max_time_adept_ps"] == pytest.approx(2.0 - np.log(2.0) / 3.0, abs=0.01)
    assert m["epw_half_max_time_lpse_ps"] == pytest.approx(2.0 - np.log(2.0) / 2.5, abs=0.01)
    assert m["iaw_max_adept"] == pytest.approx(3e-3) and m["iaw_max_lpse"] == pytest.approx(6e-3)
    # windows as tuples give the same result
    assert compare(series, lpse, ((0.5, 1.0), (1.0, 1.5))) == m


def test_compare_without_raman_or_iaw_columns():
    series, lpse = _synthetic()
    del series["e1_sq"], lpse["Nelf_max"]
    m = compare(series, lpse, ((0.5, 1.0),))
    assert not any(k.startswith(("raman", "iaw")) for k in m)
    crossings = energy_floor_crossings(series, lpse, floor_window=(0.0, 0.2), factors=(10,))
    # floor = median of exp(g t) on (0, 0.2) ~ exp(0.1 g); 10x floor at t = 0.1 + ln 10 / g
    assert crossings["epw_10x_floor_time_adept_ps"] == pytest.approx(0.1 + np.log(10.0) / 3.0, abs=0.02)


def test_default_windows_cover_the_baseline_decks():
    assert set(DEFAULT_WINDOWS) == {
        "hom_srs_020", "hom_tpd_023", "test_006", "test_025", "test_029", "test_010",
        "test_032", "test_013", "test_022", "test_036", "test_037",
    }  # fmt: skip
    for windows in DEFAULT_WINDOWS.values():
        assert all(lo < hi for lo, hi in windows)


def test_lpse_root_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LPSE_ROOT", str(tmp_path))
    assert lpse_root() == tmp_path
    monkeypatch.setenv("LPSE_ROOT", str(tmp_path / "missing"))
    assert lpse_root() is None
    assert deck_path("test_006") is None
    monkeypatch.delenv("LPSE_ROOT")
    root = lpse_root()
    assert root is None or (root / "examples").is_dir() or (root / "runs").is_dir()


def test_reference_run_dir_without_download(monkeypatch, tmp_path):
    monkeypatch.setenv("LPSE_ROOT", str(tmp_path))
    monkeypatch.setenv("LPSE_REFERENCE_CACHE", str(tmp_path / "cache"))
    assert reference_run_dir("test_025", download=False) is None
    run = tmp_path / "runs" / "test_025" / "data"
    run.mkdir(parents=True)
    (run / "lpse.metrics").write_text("# (1) time\n0.0\n")
    assert reference_run_dir("test_025", download=False) == tmp_path / "runs" / "test_025"


RUN_036 = reference_run_dir("test_036", download=False)


@pytest.mark.skipif(
    RUN_036 is None or not (RUN_036 / "data" / "lpse.flux").is_file(), reason="no test_036 reference run"
)
def test_read_flux_test_036():
    t, flux = read_flux(RUN_036 / "data" / "lpse.flux")
    assert flux.shape[1:] == (7, 6) and t[0] == pytest.approx(0.12)
    assert np.all(flux[:, :, 4:] == 0.0)  # no z faces in 2-D
    assert np.all(flux[:, :4, :4] > 0.0)


RUN_006 = reference_run_dir("test_006", download=False)


@pytest.mark.skipif(RUN_006 is None, reason="no test_006 reference run")
def test_compare_on_the_test_006_reference_against_itself():
    """Feeding LPSE's own energies back as the adept series gives ratios of exactly 1."""
    from adept._lpse2d.lpse_deck import read_metrics

    d = read_metrics(RUN_006 / "data" / "lpse.metrics")
    series = {"t (ps)": d["time"], "epw_energy": d["EPW_energy"], "e1_sq": d["E1_energy"]}
    m = compare(series, d, DEFAULT_WINDOWS["test_006"])
    for k, v in m.items():
        if "ratio" in k:
            assert v == pytest.approx(1.0, rel=1e-12), k
    assert m["epw_half_max_time_adept_ps"] == m["epw_half_max_time_lpse_ps"]
    # the stored baseline value for this deck (MLflow lpse-parity bf56ee6c, 2026-09-14)
    assert m["epw_energy_growth_lpse_2.5_3.5ps"] == pytest.approx(5.707053668690972, rel=1e-9)


@pytest.mark.skipif(not os.environ.get("MLFLOW_TRACKING_URI"), reason="no MLflow tracking environment")
def test_verify_reproduces_the_stored_baseline_metrics(tmp_path):
    """``verify`` on the test_006 cross-check run (bf56ee6c…): the ratio metrics logged on
    2026-09-14 by ``log_lpse_reference.py`` are reproduced from the run's own artifacts."""
    from adept._lpse2d.parity.__main__ import main

    rc = main(["verify", "bf56ee6c3072421d80bdcff0080a5c84", "--dest", str(tmp_path), "--rtol", "1e-6"])
    assert rc == 0
    assert (Path(tmp_path) / "binary" / "series.xr").is_file()
