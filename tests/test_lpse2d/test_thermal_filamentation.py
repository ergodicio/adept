"""Thermal-filamentation source on the IAW velocity divergence (LPSE thermalFil.*)."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _cfg(tf, absorption=True, collisions=1.0):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": 0.2}
    cfg["grid"].update({"dx": "0.1um", "xmax": "12.8um", "ymax": "3.2um", "ymin": "-3.2um", "dt": "1fs", "tmax": "2fs"})
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": False})
    cfg["terms"]["epw"]["damping"]["collisions"] = collisions
    cfg["terms"]["light"] = {"absorption": absorption}
    cfg["terms"]["iaw"] = {
        "active": True,
        "solver": "spectral",
        "damping": {"landau": 0.1, "collisions": 0.0},
        "thermal_filamentation": tf,
    }
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _fields(cfg, modulation_k=None):
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    x = np.asarray(cfg["grid"]["x"])
    e0 = np.zeros((nx, ny, 2), dtype=np.complex128)
    amp = 1.0e-3 if modulation_k is None else 1.0e-3 * np.sqrt(1.0 + 0.5 * np.cos(modulation_k * x))[:, None]
    e0[..., 1] = amp
    return jnp.zeros((nx, ny), dtype=jnp.complex128), jnp.asarray(e0), jnp.zeros((nx, ny, 2), dtype=jnp.complex128)


def test_uniform_intensity_gives_no_source_and_modulation_the_derived_coefficient():
    from adept._lpse2d.core.iaw import IonAcousticWave

    cfg = _cfg({"laser": True})
    iaw = IonAcousticWave(cfg)
    phi_k, e0, e1 = _fields(cfg)
    uniform_max = float(jnp.max(jnp.abs(iaw.thermal_filamentation_source(phi_k, e0, e1))))
    kx = np.asarray(cfg["grid"]["kx"])
    km = kx[4]
    phi_k, e0, e1 = _fields(cfg, modulation_k=km)
    source = np.asarray(iaw.thermal_filamentation_source(phi_k, e0, e1))
    assert uniform_max < 1e-9 * np.max(np.abs(source))  # only the box-mean subtraction round-off
    e_sq = np.sum(np.abs(np.asarray(e0)) ** 2, axis=-1)
    k_laser, nc_laser, power = iaw.thermal_terms["laser"]
    n = float(cfg["density"]["val"])
    expected = k_laser * (n / nc_laser) ** power * (e_sq - e_sq.mean())
    np.testing.assert_allclose(source, expected, rtol=1e-12, atol=1e-30)
    # heating where the intensity is high drives outflow: positive divergence there
    assert source[np.argmax(e_sq[:, 0]), 0] > 0.0
    # the coefficient: Z * 2 nu_abs (n/nc)^2 * fieldScale^2 / (8 pi m_i kappa') in 1/ps^2 per code |E|^2
    assert k_laser > 0.0 and np.isfinite(k_laser)


def test_nonlocal_correction_scales_as_k_to_the_four_thirds():
    from adept._lpse2d.core.iaw import IonAcousticWave

    local = IonAcousticWave(_cfg({"laser": True}))
    nonlocal_ = IonAcousticWave(_cfg({"laser": True, "nonlocal": True}))
    cfg = _cfg({"laser": True})
    kx = np.asarray(cfg["grid"]["kx"])
    ratios = []
    for i in (2, 4):
        phi_k, e0, e1 = _fields(cfg, modulation_k=kx[i])
        s_loc = np.fft.fft2(np.asarray(local.thermal_filamentation_source(phi_k, e0, e1)))
        s_nl = np.fft.fft2(np.asarray(nonlocal_.thermal_filamentation_source(phi_k, e0, e1)))
        ratios.append(np.abs(s_nl[i, 0]) / np.abs(s_loc[i, 0]) - 1.0)
    # the excess over the local source grows as k^(4/3): ratio of excesses = 2^(4/3)
    np.testing.assert_allclose(ratios[1] / ratios[0], 2.0 ** (4.0 / 3.0), rtol=1e-6)
    assert ratios[0] > 0.0


def test_requirements_and_translator():
    import pytest

    from adept._lpse2d.core.iaw import IonAcousticWave
    from adept._lpse2d.lpse_deck import translate_parms

    with pytest.raises(ValueError, match="absorption"):
        IonAcousticWave(_cfg({"laser": True}, absorption=False))
    with pytest.raises(ValueError, match="collisions"):
        IonAcousticWave(_cfg({"lw": True}, collisions=0.0))
    parms = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "lw.enable": "true",
        "iaw.enable": "true",
        "lw.spectral.dt": "0.005",
        "simulation.time.end": "1",
        "laser.1.intensity": "1e15",
        "thermalFil.laser.enable": "true",
        "thermalFil.isNonlocal": "true",
        "thermalFil.conductivityMultiplier": "2",
    }
    cfg, report = translate_parms(parms, experiment="x", run="y")
    tf = cfg["terms"]["iaw"]["thermal_filamentation"]
    assert tf["laser"] and not tf["raman"] and tf["nonlocal"] and tf["conductivity_multiplier"] == 2.0


def test_lpse_coulomb_logs_total_density_and_window_weighting():
    """Inventory A16 (ZakharovSolver::getThermalFilamentationSource; ParameterManager.cpp:1466-1481):
    Spitzer conductivity with LPSE's e-i Coulomb log at the envelope density, lambda_nl with its e-e
    one; the local term scales with the total density n_b (1 + Nelf) squared; the intensity average
    is weighted by the IAW window. Exact arithmetic (1e-12)."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    cfg = _cfg({"laser": True})
    iaw = IonAcousticWave(cfg)
    from adept._lpse2d.core.iaw import _Q_kev

    te_ev = 1.0e3 * float(_Q_kev(cfg["units"]["reference electron temperature"]))
    z = float(cfg["units"]["ionization state"])
    lambda_um = 2.0 * np.pi * cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]
    no = float(cfg["units"]["envelope density"]) * 1.1148e21 / lambda_um**2
    log_ei = 24.0 - np.log(np.sqrt(no) / te_ev) if te_ev / 1.0e3 >= 0.01 * z**2 else None
    log_ee = 23.5 - np.log(np.sqrt(no) * te_ev**-1.25) - np.sqrt(1e-5 + (np.log(te_ev) - 2.0) ** 2 / 16.0)
    assert iaw.thermal_log_ei == pytest.approx(max(2.0, log_ei), rel=1e-12)
    assert iaw.thermal_log_ee == pytest.approx(max(2.0, log_ee), rel=1e-12)

    kx = np.asarray(cfg["grid"]["kx"])
    phi_k, e0, e1 = _fields(cfg, modulation_k=kx[4])
    base = np.asarray(iaw.thermal_filamentation_source(phi_k, e0, e1))
    nelf = jnp.full((cfg["grid"]["nx"], cfg["grid"]["ny"]), 0.1)
    with_nelf = np.asarray(iaw.thermal_filamentation_source(phi_k, e0, e1, nelf))
    np.testing.assert_allclose(with_nelf, 1.1**2 * base, rtol=1e-12, atol=1e-30)

    half = {"width": ["6.4um", "0um"], "center": ["-3.2um", "0um"], "edge_width": "0um"}
    windowed_cfg = _cfg({"laser": True})
    windowed_cfg["terms"]["iaw"]["source_window"] = half
    from adept._lpse2d.helpers import get_solver_quantities

    windowed_cfg["grid"] = get_solver_quantities(windowed_cfg)
    windowed_cfg["grid"]["background_density"] = cfg["grid"]["background_density"]
    windowed = IonAcousticWave(windowed_cfg)
    rr = np.asarray(windowed_cfg["grid"]["iaw_window"])
    e_sq = np.sum(np.abs(np.asarray(e0)) ** 2, axis=-1)
    k_laser, nc_laser, power = windowed.thermal_terms["laser"]
    n = float(cfg["density"]["val"])
    mask = np.asarray(windowed_cfg["grid"]["iaw_source_mask"])
    want = k_laser * (n / nc_laser) ** power * (e_sq - np.sum(e_sq * rr) / np.sum(rr)) * mask**2
    got = np.asarray(windowed.thermal_filamentation_source(phi_k, e0, e1))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
