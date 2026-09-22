"""The original LPSE density profiles (density.basis: lpse-<shape>)."""

from copy import deepcopy

import numpy as np
import pytest
import yaml


def _cfg(density, ny=8):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"dx": "0.1um", "xmax": "20um", "ymax": "0.4um", "ymin": "-0.4um", "dt": "1fs", "tmax": "2fs"})
    cfg["density"] = density
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    return cfg, np.asarray(get_density_profile(cfg))


def _at(cfg, prof, x_um):
    return float(prof[int(np.argmin(np.abs(np.asarray(cfg["grid"]["x"]) - x_um))), 0])


BASE = {"min": 0.1, "max": 0.3, "min_location": "2um", "max_location": "18um"}


def test_linear_and_exp_hit_the_endpoints_and_clip():
    cfg, lin = _cfg({"basis": "lpse-linear", **BASE})
    assert abs(_at(cfg, lin, 2.0) - 0.1) < 2e-3 and abs(_at(cfg, lin, 18.0) - 0.3) < 2e-3
    assert abs(_at(cfg, lin, 10.0) - 0.2) < 2e-3
    assert lin.min() >= 0.1 and lin.max() <= 0.3  # clipped beyond the endpoints
    cfg, ex = _cfg({"basis": "lpse-exp", **BASE})
    assert abs(_at(cfg, ex, 2.0) - 0.1) < 2e-3 and abs(_at(cfg, ex, 18.0) - 0.3) < 2e-3
    assert abs(_at(cfg, ex, 10.0) - np.sqrt(0.1 * 0.3)) < 2e-3  # geometric mean half way


@pytest.mark.parametrize("p", [2.0, 4.0])
def test_gaussian_and_inverse_power_match_lpse_formulas(p):
    cfg, g = _cfg({"basis": "lpse-gaussian", "sg_order": p, **BASE})
    assert abs(_at(cfg, g, 18.0) - 0.3) < 2e-3 and abs(_at(cfg, g, 2.0) - 0.1) < 3e-3
    sd = 16.0 / np.log(3.0) ** (1.0 / p)
    assert abs(_at(cfg, g, 10.0) - 0.3 * np.exp(-((8.0 / sd) ** p))) < 3e-3
    cfg, ip = _cfg({"basis": "lpse-inverse-power", "sg_order": p, **BASE})
    assert abs(_at(cfg, ip, 2.0) - 0.1) < 3e-3
    rc = 16.0 * (0.1 / 0.3) ** (1.0 / p)
    assert abs(_at(cfg, ip, 18.0 - 0.5 * rc) - 0.3) < 1e-9  # flat top inside r_c
    assert abs(_at(cfg, ip, 10.0) - 0.1 * (16.0 / 8.0) ** p) < 3e-3 or _at(cfg, ip, 10.0) == 0.3


def test_quadratic_passes_through_its_three_points():
    d = {
        "basis": "lpse-quadratic",
        "min": 0.1,
        "max": 0.3,
        "min_location": "5um",
        "max_location": "15um",
        "central_density": 0.2,
        "origin": "10um",
    }
    cfg, q = _cfg(d)
    assert abs(_at(cfg, q, 10.0) - 0.2) < 2e-3
    assert abs(_at(cfg, q, 5.0) - 0.1) < 2e-3
    assert abs(_at(cfg, q, 15.0) - 0.3) < 2e-3


def test_dips_sit_on_the_linear_ramp():
    cfg, lin = _cfg({"basis": "lpse-linear", **BASE})
    dip = {"dip_depth": 0.05, "dip_width": "4um", "dip_offset": "0um", "origin": "10um"}
    cfg, qd = _cfg({"basis": "lpse-qd", **BASE, **dip})
    assert abs(_at(cfg, qd, 10.0) - (_at(cfg, lin, 10.0) - 0.05)) < 2e-3  # full depth at the centre
    assert abs(_at(cfg, qd, 12.0) - _at(cfg, lin, 12.0)) < 2e-3  # zero at the edge of the dip
    assert abs(_at(cfg, qd, 15.0) - _at(cfg, lin, 15.0)) < 1e-9  # untouched outside
    cfg, gd = _cfg({"basis": "lpse-gd", "sg_order": 2.0, **BASE, **dip})
    assert abs(_at(cfg, gd, 10.0) - (_at(cfg, lin, 10.0) - 0.05)) < 2e-3
    assert abs(_at(cfg, gd, 12.0) - (_at(cfg, lin, 12.0) - 0.05 * np.exp(-1.0))) < 2e-3


def test_spherical_geometry_and_clip():
    cfg, s = _cfg({"basis": "lpse-linear", "geometry": "spherical", **BASE})
    # radial distance from the max location: symmetric about x = 18 um, so 20 um reads like 16 um
    assert abs(_at(cfg, s, 20.0) - _at(cfg, s, 16.0)) < 3e-3
    cfg, c = _cfg({"basis": "lpse-linear", "min": 1.0, "max": 2.0, "min_location": "2um", "max_location": "18um"})
    assert c.max() == 1.25


def test_file_profile_roundtrip(tmp_path):
    cfg, lin = _cfg({"basis": "lpse-linear", **BASE})
    path = tmp_path / "n.npy"
    np.save(path, lin)
    cfg2, loaded = _cfg({"basis": "lpse-file", "file": str(path)})
    np.testing.assert_allclose(loaded, lin)
    np.save(path, lin[:, 0])
    cfg3, loaded1d = _cfg({"basis": "lpse-file", "file": str(path)})
    np.testing.assert_allclose(loaded1d, lin)
    with pytest.raises(ValueError, match="shape"):
        np.save(path, lin[:-1])
        _cfg({"basis": "lpse-file", "file": str(path)})


def test_translator_maps_the_shape_keys():
    from adept._lpse2d.lpse_deck import translate_parms

    base = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "lw.enable": "true",
        "densityProfile.NminOverNc": "0.2",
        "densityProfile.NmaxOverNc": "0.28",
        "densityProfile.NminLocation": "-10 0",
        "densityProfile.NmaxLocation": "10 0",
        "lw.spectral.dt": "0.005",
        "simulation.time.end": "1",
        "laser.1.intensity": "1e15",
    }
    dip = {
        "densityProfile.shape": "gd",
        "densityProfile.sgOrder": "4",
        "densityProfile.dip.depth": "0.02",
        "densityProfile.dip.width": "3",
        "densityProfile.dip.offset": "1",
    }
    parms = dict(base, **dip)
    cfg, report = translate_parms(parms, experiment="x", run="y")
    d = cfg["density"]
    assert d["basis"] == "lpse-gd" and d["sg_order"] == 4.0 and d["dip_depth"] == 0.02
    assert d["dip_width"] == "3.0um" and d["dip_offset"] == "1.0um" and d["origin"] == "10.0um"
    assert d["min_location"] == "0.0um" and d["max_location"] == "20.0um"
    cfg, report = translate_parms(dict(base, **{"densityProfile.shape": "inverseSquare"}), experiment="x", run="y")
    assert cfg["density"]["basis"] == "lpse-inverse-power" and cfg["density"]["sg_order"] == 2.0
