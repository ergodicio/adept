"""Thermal-filamentation source on the IAW velocity divergence (LPSE thermalFil.*)."""

from copy import deepcopy

import numpy as np
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
    expected = np.asarray(iaw.thermal_coeff["laser"]) * (e_sq - e_sq.mean())
    np.testing.assert_allclose(source, expected, rtol=1e-12, atol=1e-30)
    # heating where the intensity is high drives outflow: positive divergence there
    assert source[np.argmax(e_sq[:, 0]), 0] > 0.0
    # the coefficient: Z * 2 nu_abs (n/nc)^2 * fieldScale^2 / (8 pi m_i kappa') in 1/ps^2 per code |E|^2
    coeff = float(np.asarray(iaw.thermal_coeff["laser"])[0, 0])
    assert coeff > 0.0 and np.isfinite(coeff)


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
