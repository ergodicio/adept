"""Absorbing layers per field and the injector planes, against LPSE (absorbingBoundaries.cpp,
LightSolver::setup, SchrodingerSolver3::setAbcParameters / completeBoundaryConditions).

The profiles are checked against literal transcriptions of the two C++ loops (tolerance 1e-12,
fixed in advance: the port is exact up to round-off).
"""

import math
from copy import deepcopy

import numpy as np
import pytest
import yaml


def _c_round(v):
    return math.floor(v + 0.5)


def _lpse_single(n, d, lmin, lmax, max_rate, lam):
    """absorbingBoundaries.cpp setupAbsorbingBoundaries, exp branch, one axis."""
    coeff = max_rate / (math.exp(lam) - 1)
    idx = [_c_round(lmin / d), n - 1 - _c_round(lmax / d)]
    labc = [lmin, lmax]
    out = []
    for i in range(n):
        rate = 0.0
        for side in (0, 1):
            if labc[side] > 0.0 and ((side == 0 and i <= idx[0]) or (side == 1 and i >= idx[1])):
                dist = abs(idx[side] - i) * d
                rate = coeff * (math.exp(lam * dist / labc[side]) - 1)
        out.append(rate)
    return np.array(out)


def _lpse_double(n, d, l1, l2, max1, lam1, max2, lam2):
    """absorbingBoundaries.cpp setupAbsorbingBoundaries_doubleExponential, exp branch, one axis."""
    c1 = max1 / (math.exp(lam1) - 1)
    c2 = max2 / (math.exp(lam2) - 1)
    d_eq = [l1[s] / lam1 * math.log(max2 / c1 + 1) for s in (0, 1)]
    total = [l1[s] + l2[s] - d_eq[s] for s in (0, 1)]
    start = [_c_round(total[0] / d), n - 1 - _c_round(total[1] / d)]
    trans = [_c_round((total[0] - l2[0]) / d), n - 1 - _c_round((total[1] - l2[1]) / d)]
    if l1[0] == 0.0:
        trans = [-1, n + 1]
    out = []
    for i in range(n):
        rate = 0.0
        for s in (0, 1):
            if total[s] > 0.0 and ((s == 0 and i <= start[0]) or (s == 1 and i >= start[1])):
                if i <= trans[0] or i >= trans[1]:
                    dist = abs(start[s] - i) * d - l2[s] + d_eq[s]
                    rate = c1 * (math.exp(lam1 * dist / l1[s]) - 1)
                else:
                    dist = abs(start[s] - i) * d
                    rate = c2 * (math.exp(lam2 * dist / l2[s]) - 1)
        out.append(rate)
    return np.array(out)


@pytest.mark.parametrize("n, d, lmin, lmax", [(360, 0.0696, 3.0, 3.0), (128, 0.1, 2.0, 2.0), (200, 0.0333, 1.3, 2.7)])
def test_single_exponential_layer_is_lpse(n, d, lmin, lmax):
    from adept._lpse2d.helpers import lpse_exp_rate

    np.testing.assert_allclose(
        lpse_exp_rate(n, d, lmin, lmax, 200.0, 7.0), _lpse_single(n, d, lmin, lmax, 200.0, 7.0), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "n, d, strong, weak", [(360, 0.0696, 2.0, 2.0), (360, 0.0696, 3.0, 2.0), (256, 0.05, 1.5, 3.0)]
)
def test_double_exponential_layer_is_lpse(n, d, strong, weak):
    from adept._lpse2d.helpers import lpse_double_exp_rate

    got = lpse_double_exp_rate(n, d, (strong, strong), (weak, weak), 5.0e3, 7.0, 200.0, 7.0)
    want = _lpse_double(n, d, (strong, strong), (weak, weak), 5.0e3, 7.0, 200.0, 7.0)
    # (LPSE's index rounding of the total width and of the joining distance leaves the edge rate
    # a few % off max1, 4.7e3-5.4e3 here; the port reproduces it)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def _finish(cfg):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _raw(solver="separate", **light):
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["density"] = {"basis": "uniform", "val": 0.22}
    cfg["grid"].update(
        {
            "boundary_width": "1.0um",
            "boundary_profile": "exp",
            "dt": "1fs",
            "dx": "0.1um",
            "xmax": "12.8um",
            "tmax": "10fs",
            "ymax": "1.6um",
            "ymin": "-1.6um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": solver == "combined", "srs": solver == "combined"})
    cfg["terms"]["epw"]["solver"] = solver
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": True, **light}
    return cfg


def test_each_light_field_gets_its_own_layer_and_the_combined_field_the_double_exponential():
    """LPSE: laser.evolution.Labc for the pump, raman.evolution.Labc for the Raman light, and in
    combined mode the EPW layer inside the Raman light's (LightSolver::setup)."""
    from adept._lpse2d.core.combined import CombinedSolver
    from adept._lpse2d.helpers import lpse_double_exp_rate, lpse_exp_rate

    cfg = _finish(_raw("combined", boundary_width="2.0um", raman_boundary_width="1.5um"))
    g = cfg["grid"]
    nx, dx, dt = g["nx"], g["dx"], g["dt"]
    np.testing.assert_allclose(
        np.asarray(g["light_absorbing_boundaries"])[:, 0],
        np.exp(-lpse_exp_rate(nx, dx, 2.0, 2.0, 5e3, 7.0) * dt),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(g["raman_absorbing_boundaries"])[:, 0],
        np.exp(-lpse_exp_rate(nx, dx, 1.5, 1.5, 5e3, 7.0) * dt),
        rtol=1e-12,
    )
    rate = lpse_double_exp_rate(nx, dx, (1.5, 1.5), (1.0, 1.0), 5e3, 7.0, 200.0, 7.0)
    np.testing.assert_allclose(np.asarray(g["combined_absorbing_boundaries"])[:, 0], np.exp(-rate * dt), rtol=1e-12)
    solver = CombinedSolver(cfg)
    np.testing.assert_allclose(np.asarray(solver.boundary)[:, 0], np.exp(-rate * solver.dt_l), rtol=1e-12)


def test_fd_light_damps_pump_and_raman_light_with_their_own_layers():
    from adept._lpse2d.core.light import CoupledLight

    raw = _raw(boundary_width="2.0um", raman_boundary_width="1.0um")
    raw["terms"]["light"]["solver"] = "fd"
    cfg = _finish(raw)
    light = CoupledLight(cfg)
    n = light.n_sub
    np.testing.assert_allclose(
        np.asarray(light.sub_boundary0), np.asarray(cfg["grid"]["light_absorbing_boundaries"]) ** (1 / n)
    )
    np.testing.assert_allclose(
        np.asarray(light.sub_boundary), np.asarray(cfg["grid"]["raman_absorbing_boundaries"]) ** (1 / n)
    )
    assert not np.allclose(np.asarray(light.sub_boundary0), np.asarray(light.sub_boundary))


def _translate(tmp_path, extra):
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    deck = tmp_path / "lpse.parms"
    deck.write_text(
        "grid.sizes = 20 10;\ngrid.nodes = 201 101;\nsimulation.time.end = 1;\nlw.enable = true;\n"
        "laser.enable = true;\nlaser.solver = spectral;\nlaser.nBeams = 1;\nlaser.1.intensity = 1e15;\n"
        "raman.enable = true;\nraman.solver = spectral;\nraman.nBeams = 1;\nraman.1.intensity = 1e12;\n" + extra
    )
    return translate_parms(parse_parms(deck), run="x")


def test_translator_layers_planes_and_seed_rise_follow_lpse(tmp_path):
    """h = 0.1 um: pump plane int(3.05/h) + int(1.02/h) = 30 + 10 = 40 nodes from x-min -> adept offset
    39.5 h; seed plane int(2.5/h) + int(0.5/h) = 25 + 5 = 30 from x-max -> 30.5 h; raman riseTime default
    30 fs; iaw.Labc absent -> no IAW layer; probes 4 cells inside the deepest plane."""
    cfg, report = _translate(
        tmp_path,
        "lw.Labc = 2;\nlaser.evolution.Labc = 3.05;\nlaser.evolution.Loff = 1.02;\n"
        "raman.evolution.Labc = 2.5;\nraman.evolution.Loff = 0.5;\niaw.enable = true;\n",
    )
    grid, light = cfg["grid"], cfg["terms"]["light"]
    assert grid["boundary_width"] == "2.0um" and light["boundary_width"] == "3.05um"
    assert light["raman_boundary_width"] == "2.5um"
    assert float(cfg["drivers"]["E0"]["offset"][:-2]) == pytest.approx(39.5 * 0.1)
    assert float(cfg["drivers"]["E1"]["offset"][:-2]) == pytest.approx(30.5 * 0.1)
    assert cfg["drivers"]["E1"]["turn_on_time"] == "30.0fs"
    assert cfg["terms"]["iaw"]["boundary_width"] == "0.0um"
    assert float(grid["probe_offset"][:-2]) == pytest.approx(44 * 0.1)
    assert not report["unsupported"]
    _, report = _translate(tmp_path, "lw.Labc.min.x = 2;\nlw.Labc.max.x = 3;\n")
    assert any("lw.Labc.min.x" in u for u in report["unsupported"])
