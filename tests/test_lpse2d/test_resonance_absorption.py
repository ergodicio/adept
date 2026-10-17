"""Resonance absorption of the pump at the critical surface (terms.light.resonance_absorption;
LPSE laser.evolution.resonanceAbsorption) -- plan 2 L.3."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept._lpse2d.parity import deck_path


def _cfg(angle, ramp, ra, tmax=0.4):
    """A p-polarised super-Gaussian beam (1 um, order 4) at `angle` on a linear ramp n = 0.1 at
    2 um -> 1.9 at 10 um (n_c at 6 um, L_n = 4.4 um, (k0 L_n)^(1/3) = 4.3), FD order 4 at 6 cells per wavelength."""
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {
            "ymax": "2um",
            "ymin": "-2um",
            "xmax": "10um",
            "dx": "0.0625um",
            "dt": "1fs",
            "tmax": f"{tmax}ps",
            "boundary_width": "1.5um",
        }
    )
    cfg["terms"]["light"] = {"pump_depletion": True, "solver": "fd", "fd_order": 4}
    if ra:
        cfg["terms"]["light"]["resonance_absorption"] = True
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"].update({"noise": False, "srs": False, "tpd": True})
    cfg["units"]["envelope density"] = 0.25
    noise = {"max": 1e-9, "min": 1e-10, "type": "uniform"}
    if ramp:
        cfg["density"] = {
            "basis": "lpse-linear",
            "min": 0.1,
            "max": 1.9,
            "min_location": "2um",
            "max_location": "10um",
            "noise": noise,
        }
    else:
        cfg["density"] = {"basis": "uniform", "val": 0.1, "noise": noise}
    cfg["drivers"]["E0"].update({"angle": angle, "beam_width": "1.0um", "beam_sg_order": 4, "delta_omega_max": 0.0})
    cfg["save"]["fields"]["t"].update({"tmax": f"{tmax}ps", "dt": "0.1ps"})
    cfg["mlflow"]["run"] = f"ra-{angle}-{ramp}-{ra}"
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, _, _ = exo(modules)
    result = sol["solver result"]
    dcfg = exo.adept_module.cfg
    t = np.asarray(result.ts["default"])
    net = np.asarray(result.ys["default"]["incident_flux"])
    x = np.asarray(dcfg["grid"]["x"])
    raw = np.asarray(result.ys["fields"]["E0"])
    e0 = raw.view(np.complex64 if raw.dtype == np.float32 else np.complex128)
    band = (x > 5.5) & (x < 6.5)  # about the critical surface
    energy_nc = np.sum(np.abs(e0[:, band]) ** 2, axis=(1, 2, 3))
    return t, net, np.asarray(result.ts["fields"]), energy_nc


def test_resonance_absorption_at_the_denisov_peak_and_beyond():
    tau = 4.3 * np.sin(np.deg2rad([11.0, 25.0]))
    assert 0.75 < tau[0] < 0.9 and tau[1] > 1.7
    # the launched flux: the same beam in a uniform sub-critical box
    _, net_free, _, _ = _run(_cfg(11.0, ramp=False, ra=False))
    f_inc = np.mean(net_free[-40:])
    assert f_inc > 0.3

    t, net, tf, energy = _run(_cfg(11.0, ramp=True, ra=True))
    steady = t > 0.25
    absorbed = np.mean(net[steady]) / f_inc
    # net flux into the ramp = the absorbed fraction; the Denisov curve peaks (0.5-0.7 across
    # the cold / warm models) near tau 0.8 -- the envelope model with LPSE's warm and Landau
    # terms gives ~0.42 here
    assert 0.3 < absorbed < 0.7
    # a true steady state: the flux and the energy at n_c stop changing
    np.testing.assert_allclose(np.mean(net[t > 0.35]), np.mean(net[(t > 0.25) & (t < 0.3)]), rtol=2e-2)
    assert energy[-1] < 1.05 * energy[np.argmin(np.abs(tf - 0.2))]

    # (without RA the cold-plasma resonance at n_c -- epsilon = 0 with nothing to limit the
    # longitudinal field -- has no steady state: on the 0.05 um grid the energy there grew
    # 6x over 0.1-0.4 ps; the growth is grid-dependent, so it is not asserted here)

    # far past the peak (tau ~ 1.8) the absorption is small; the launched x-flux of the 25 deg
    # beam is the 11 deg one times cos(25)/cos(11) (measured 0.922 vs 0.923)
    t25, net25, _, _ = _run(_cfg(25.0, ramp=True, ra=True))
    absorbed25 = np.mean(net25[t25 > 0.25]) / (f_inc * np.cos(np.deg2rad(25.0)) / np.cos(np.deg2rad(11.0)))
    assert absorbed25 < 0.15


def test_validation_and_translator():
    from adept._lpse2d.helpers import get_derived_quantities, write_units

    cfg = _cfg(11.0, ramp=True, ra=True)
    cfg["terms"]["light"]["solver"] = "spectral"
    write_units(cfg)
    with pytest.raises(ValueError, match="fd light solver"):
        get_derived_quantities(cfg)

    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    if deck_path("test_011") is None:
        pytest.skip("no LPSE decks present")
    cfg, report = translate_parms(parse_parms(deck_path("test_011")), run="test_011")
    ra = cfg["terms"]["light"]["resonance_absorption"]
    assert cfg["terms"]["light"]["solver"] == "fd" and ra["landau_update"] == 10 and ra["filter"] is True
    assert (
        cfg["drivers"]["E0"]["beam_width"] == f"{3.0 / np.sqrt(2.0)}um" and cfg["drivers"]["E0"]["beam_sg_order"] == 4.0
    )
    assert abs(cfg["drivers"]["E0"]["angle"] - np.degrees(np.arctan2(0.1676, 0.98586))) < 1e-6
    assert not any("resonanceAbsorption" in u for u in report["unsupported"])
