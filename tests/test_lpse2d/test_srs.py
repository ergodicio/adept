"""Tests for the SRS (stimulated Raman scattering) path of the envelope-2d solver.

These are ports of the physics exercised by the `srs_1D` case of lpse-matlab
(m201805_matlabLpse_v11.m), run as small quasi-1D 2D boxes.
"""

import numpy as np
import pytest
import yaml


def _load_cfg():
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)
    return exo, sol


def _predicted_gamma0(cfg_units, cfg_grid):
    """
    Homogeneous backward-SRS growth rate gamma0 = k vos/4 * wpe / sqrt(w_ek * w_s)
    evaluated at the phase-matched EPW wavenumber k = k0 + k1.
    """
    c = cfg_units["derived"]["c"]
    w0 = cfg_units["derived"]["w0"]
    wp0 = cfg_units["derived"]["wp0"]
    vte_sq = cfg_units["derived"]["vte_sq"]
    e = cfg_units["derived"]["e"]
    me = cfg_units["derived"]["me"]
    E0_source = cfg_units["derived"]["E0_source"]
    n = cfg_units["envelope density"]  # uniform box at the envelope density

    # pump wavenumber as launched by the solver (snapped to the FFT grid)
    dk = 2.0 * np.pi / (cfg_grid["nx"] * cfg_grid["dx"])
    k0 = np.sqrt(w0**2 - wp0**2) / c
    k0 = np.round(k0 / dk) * dk

    # fixed point for the phase-matched EPW wavenumber
    k = k0
    for _ in range(20):
        w_ek = wp0 + 1.5 * k**2 * vte_sq / wp0
        w_s = w0 - w_ek
        k1 = np.sqrt(w_s**2 - wp0**2) / c
        k = k0 + k1

    # local pump field including the density swelling factor
    E0_local = E0_source * (1.0 - n) ** -0.25
    vos = e * E0_local / (me * w0)

    gamma0 = k * vos / 4.0 * wp0 / np.sqrt(w_ek * w_s)
    return gamma0, k1


@pytest.mark.parametrize("ny", ["2d", "1d"])
def test_srs_growth_rate(ny):
    """Noise-seeded homogeneous SRS: the EPW energy should grow at ~2*gamma0.

    The "1d" variant shrinks the y extent below one cell so the box collapses to ny=1,
    which is the cheap true-1D configuration (the MATLAB srs_1D case).
    """
    cfg = _load_cfg()
    if ny == "1d":
        cfg["grid"]["ymax"] = "0.02um"
        cfg["grid"]["ymin"] = "-0.02um"
        cfg["mlflow"]["run"] = "srs-test-1d"
    exo, sol = _run(cfg)

    result = sol["solver result"]
    t = np.array(result.ts["default"])
    e_sq = np.array(result.ys["default"]["e_sq"])

    gamma0, _ = _predicted_gamma0(exo.adept_module.cfg["units"], exo.adept_module.cfg["grid"])

    # fit the log-slope over the last stretch of the run, where exponential growth
    # dominates the noise floor
    fit_window = (t > t[-1] - 0.5) & (t < t[-1] - 0.02)
    slope = np.polyfit(t[fit_window], np.log(e_sq[fit_window]), 1)[0]
    measured_gamma = slope / 2.0  # e_sq ~ |E|^2 grows at 2*gamma

    assert e_sq[-1] > 1e4 * np.min(e_sq[t > 0.1]), "no SRS growth occurred"
    np.testing.assert_allclose(measured_gamma, gamma0, rtol=0.35)


def test_srs_seed_propagation():
    """The Raman seed injector should launch a backward (-x) wave at the local k1."""
    cfg = _load_cfg()

    # seed-only setup: no pump, no noise -- E1 just propagates
    del cfg["drivers"]["E0"]
    cfg["drivers"]["E1"] = {
        "intensity": "1.0e+12W/cm^2",
        "delta_omega": 0.0,
        "turn_on_time": "10fs",
    }
    cfg["grid"]["xmax"] = "20um"
    cfg["grid"]["tmax"] = "0.2ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.2ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.02ps"
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["mlflow"]["run"] = "srs-seed-test"

    exo, sol = _run(cfg)

    result = sol["solver result"]
    e1_raw = np.array(result.ys["fields"]["E1"])
    e1 = e1_raw.view(np.complex64 if e1_raw.dtype == np.float32 else np.complex128)
    e1y_final = e1[-1, :, 0, 1]  # final time, y=first row, y-polarization

    assert np.max(np.abs(e1y_final)) > 0, "seed was not injected"

    dcfg = exo.adept_module.cfg
    x = np.array(dcfg["grid"]["x"])
    derived = dcfg["units"]["derived"]
    wp0, w1, c = derived["wp0"], derived["w1"], derived["c"]
    n = dcfg["units"]["envelope density"]
    k1 = np.sqrt(w1**2 - wp0**2) / c  # seed injected at delta_omega = 0

    # in the bulk (away from the injector near-field and the absorbers) the seed should be
    # a leftward traveling wave exp(-i k1 x)
    bulk = slice(np.argmin(np.abs(x - 6.0)), np.argmin(np.abs(x - 12.0)))
    phase = np.unwrap(np.angle(e1y_final[bulk]))
    k_measured = np.polyfit(x[bulk], phase, 1)[0]
    assert k_measured < 0, f"seed propagates the wrong way: k = {k_measured:.2f} 1/um"
    np.testing.assert_allclose(np.abs(k_measured), k1, rtol=0.05)

    # amplitude calibration of the injector: |E1| = E1_source * sinc(k1 dx) / eps1^(1/4)
    dx = dcfg["grid"]["dx"]
    eps1 = 1.0 - n * (derived["w0"] / w1) ** 2
    expected_amp = dcfg["drivers"]["E1"]["derived"]["amplitude"] * np.sin(k1 * dx) / (k1 * dx) / eps1**0.25
    np.testing.assert_allclose(np.mean(np.abs(e1y_final[bulk])), expected_amp, rtol=0.3)


if __name__ == "__main__":
    test_srs_seed_propagation()
    test_srs_growth_rate()
