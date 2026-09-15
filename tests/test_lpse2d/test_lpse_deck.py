"""Translation of original-LPSE decks and reading of LPSE outputs (lpse_deck.py).

The ZAK-unit port is checked against the derived quantities LPSE prints in its own
parameter summary for the test_025 deck; the deck parser and translator are checked on
the shipped example decks; the output readers on a reference run when one is present.
"""

from pathlib import Path

import numpy as np
import pytest

LPSE_ROOT = Path("/home/phil/Desktop/Ergodic-projects/original-lpse")
DECKS = LPSE_ROOT / "examples" / "testRuns"


def _finish(cfg):
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def test_zak_units_match_lpse_printed_summary():
    """test_025: Z = 1, Te = 2 keV, Ti = 1 keV, Mi/Me = 1836, n0/nc = 0.2, 0.351 um. LPSE prints
    critical density 9.0491e+21 1/cm^3, envelope density 1.8098e+21, plasma frequency
    2.4000e+15 rad/s, Debye length 7.8148e-07 cm (runs/test_025/lpse.out)."""
    from adept._lpse2d.lpse_deck import ZakUnits

    zak = ZakUnits(2.0, 1.0, 1.0, 1836.0, 0.2, 0.351)
    assert zak.nc == pytest.approx(9.0491e21, rel=2e-4)
    assert zak.n0 == pytest.approx(1.8098e21, rel=2e-4)
    assert zak.plasma_frequency == pytest.approx(2.4000e15, rel=2e-4)
    assert zak.debye_length_cm == pytest.approx(7.8148e-7, rel=2e-4)
    # the dispersion coefficient 3 ve^2 / (2 wpe) is exactly 1 in ZAK units
    assert 3.0 * zak.ve**2 / (2.0 * zak.wpe) == pytest.approx(1.0)
    # ZAK k unit vs the Debye wavenumber: kde = wpe / ve
    assert zak.kde == pytest.approx(zak.wpe / zak.ve)
    assert zak.zak_per_ps > 0 and zak.zak_per_um > 0 and np.isfinite(zak.potential_zak_to_adept)


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
def test_parse_deck_with_include():
    from adept._lpse2d.lpse_deck import parse_parms

    parms = parse_parms(DECKS / "test_006" / "lpse.parms")
    assert parms["lw.solver"] == "spectral" and parms["raman.solver"] == "spectral"
    assert parms["laser.1.intensity"] == "1e+15"  # from laser_include.txt
    assert parms["grid.nodes"] == "360 360" and parms["densityProfile.NminLocation"] == "-10 0"


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
def test_translate_test_006_builds_a_runnable_config():
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    parms = parse_parms(DECKS / "test_006" / "lpse.parms")
    cfg, report = translate_parms(parms, run="test_006")
    assert cfg["terms"]["epw"]["source"] == {
        "noise": True,
        "noise_model": "thermal",
        "noise_debye_factor": False,
        "noise_calibrate": False,
        "noise_amplitude": pytest.approx(cfg["terms"]["epw"]["source"]["noise_amplitude"]),
        "noise_seed": 1,
        "tpd": False,
        "tpd_form": "lpse",
        "srs": True,
    }
    assert cfg["terms"]["light"]["solver"] == "spectral" and cfg["terms"]["light"]["pump_depletion"] is False
    assert cfg["terms"]["light"]["max_wavenumber"] == 1.0
    assert cfg["terms"]["epw"]["damping"]["landau_form"] == "relativistic"
    assert cfg["terms"]["epw"]["damping"]["collisions"] == 0.1
    # EPW dt = min(lw.spectral.dt, maxLightStepsPerStep * raman.dt) = min(0.002, 4 * 0.0001)
    assert cfg["grid"]["dt"] == "0.0004ps" and cfg["grid"]["light_substeps"] == 4
    assert cfg["grid"]["dealias"] == "rectangular" and cfg["grid"]["low_pass_filter"] == pytest.approx(0.666)
    assert cfg["units"]["laser intensity"] == "1e+15W/cm^2"
    assert not report["unsupported"]

    cfg = _finish(cfg)
    grid = cfg["grid"]
    assert grid["nx"] == 360 and grid["ny"] == 360
    assert grid["dx"] == pytest.approx(20.0 / 359.0)
    n = np.asarray(grid["background_density"])[:, 0]
    x = np.asarray(grid["x"])
    # linear from 0.2 at the N_min location (x = 0) to 0.28 at N_max (x = 20), clipped outside
    inside = (x > 0.0) & (x < 20.0)
    np.testing.assert_allclose(n[inside], 0.2 + 0.08 * x[inside] / 20.0, rtol=1e-12)
    assert n.min() >= 0.2 - 1e-12 and n.max() <= 0.28 + 1e-12
    # rectangular anti-aliasing: outer 33.4 % of each axis zeroed
    band = np.asarray(grid["low_pass_filter_grid"])
    kx = np.abs(np.asarray(grid["kx"]))[:, None]
    ky = np.abs(np.asarray(grid["ky"]))[None, :]
    expected = (kx < 0.666 * kx.max()) & (ky < 0.666 * ky.max())
    np.testing.assert_array_equal(band > 0, expected)


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
def test_translate_combined_iaw_deck():
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    cfg, report = translate_parms(parse_parms(DECKS / "test_032" / "lpse.parms"), run="test_032")
    assert cfg["terms"]["epw"]["solver"] == "combined"
    assert cfg["terms"]["epw"]["source"]["tpd"] and cfg["terms"]["epw"]["source"]["srs"]
    assert cfg["terms"]["iaw"]["active"] and cfg["terms"]["iaw"]["solver"] == "spectral"
    assert cfg["terms"]["light"]["solver"] == "spectral"
    cfg = _finish(cfg)
    assert cfg["terms"]["iaw"]["damping"]["landau"] > 0.0


def test_lpse_exp_profile_and_thermal_noise_without_debye_factor():
    import yaml

    from adept._lpse2d.core.epw import noise_kick_spectrum

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["density"] = {"basis": "lpse-exp", "min": 0.2, "max": 0.28, "min_location": "2um", "max_location": "8um"}
    cfg["grid"].update({"xmax": "10um", "dx": "0.1um", "ymax": "0.2um", "ymin": "-0.2um", "tmax": "10fs", "dt": "1fs"})
    cfg["terms"]["epw"]["source"].update(
        {"noise": True, "noise_model": "thermal", "noise_debye_factor": False, "tpd": False}
    )
    cfg["terms"]["epw"]["damping"]["collisions"] = 1.0
    cfg = _finish(cfg)
    n = np.asarray(cfg["grid"]["background_density"])[:, 0]
    x = np.asarray(cfg["grid"]["x"])
    ln = 6.0 / np.log(0.28 / 0.2)
    inside = (x > 2.0) & (x < 8.0)
    np.testing.assert_allclose(n[inside], 0.28 * np.exp(-(8.0 - x[inside]) / ln), rtol=1e-12)
    assert n[x < 2.0].max() == pytest.approx(0.2) and n[x > 8.0].min() == pytest.approx(0.28)

    kick = np.asarray(noise_kick_spectrum(cfg))
    cfg["terms"]["epw"]["source"]["noise_debye_factor"] = True
    kick_debye = np.asarray(noise_kick_spectrum(cfg))
    derived = cfg["units"]["derived"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    lam_sq = derived["vte_sq"] / derived["wp0"] ** 2
    band = kick > 0
    np.testing.assert_allclose(kick[band] / kick_debye[band], np.sqrt(1.0 + (kx**2 + ky**2) * lam_sq)[band], rtol=1e-12)


RUN_025 = LPSE_ROOT / "runs" / "test_025" / "data"


@pytest.mark.skipif(not RUN_025.exists(), reason="no LPSE reference run present")
def test_read_lpse_metrics_and_frames():
    from adept._lpse2d.lpse_deck import read_frames, read_metrics

    metrics = read_metrics(RUN_025 / "lpse.metrics")
    assert metrics["time"][0] == pytest.approx(0.01) and metrics["time"][-1] >= 1.0  # one sample past the end
    assert "EPW_energy" in metrics and np.all(metrics["EPW_energy"] > 0)
    frames = read_frames(RUN_025 / "lpse.pots")
    assert len(frames) == 2
    header, pots = frames[0]
    assert header["Nx"] == 720 and header["Ny"] == 90 and header["time"] == pytest.approx(0.5)
    assert pots.shape == (720, 90) and np.iscomplexobj(pots) and np.all(np.isfinite(pots))
    assert frames[1][0]["time"] == pytest.approx(1.0)
    # the x-space field is smooth along x on the scale of one cell but not constant
    assert np.abs(pots).std() > 0
