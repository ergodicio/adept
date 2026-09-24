"""Light spectrum probes (save.light_spectrum; LPSE spectrum.N.{laser|raman}) -- plan 2 L.6."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept._lpse2d.parity import deck_path


def _cfg():
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"ymax": "0.02um", "ymin": "-0.02um", "xmax": "20um", "tmax": "0.4ps"})
    cfg["terms"]["light"] = {"pump_depletion": True}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["save"]["fields"]["t"].update({"tmax": "0.4ps", "dt": "0.1ps"})
    cfg["save"]["light_spectrum"] = [
        {"field": "E0", "interval": "2fs", "tmin": "0.15ps", "x": ["-2um", "2um"], "poynting": True},
        {"field": "E0", "interval": "10fs", "x": ["4um", "4um"]},
    ]
    cfg["mlflow"]["run"] = "light-spectrum-probe"
    return cfg


def test_probe_records_the_pump_at_its_detuning():
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(_cfg())
    sol, ppo, _ = exo(modules)
    dcfg = exo.adept_module.cfg
    derived = dcfg["units"]["derived"]
    x = np.asarray(dcfg["grid"]["x"])
    dx = float(dcfg["grid"]["dx"])
    spectra = ppo["light_spectrum"]
    assert len(spectra) == 2

    ds = spectra[0]
    # the sub-box: nearest nodes to 8 and 12 um (box centre 10 um), guard cells cropped
    xs = ds["x (um)"].values
    assert abs(xs[0] - (x[np.argmin(np.abs(x - 8.0))])) < 1e-9 and abs(xs[-1] - x[np.argmin(np.abs(x - 12.0))]) < 1e-9
    assert xs.size == round(4.0 / dx) + 1
    t = ds["t (ps)"].values
    np.testing.assert_allclose(t[0], 0.15, atol=1e-9)
    np.testing.assert_allclose(np.diff(t), 0.002, atol=1e-9)
    assert t[-1] <= 0.4 + 1e-9
    assert set(ds.data_vars) >= {"e_x", "e_y", "e_z", "s_x", "s_y", "spectrum_y", "power"}

    # the single-colour pump sits at the driver's delta_omega: E0 ~ e^{-i w0 dw t}
    from adept._lpse2d.modules.driver import UniformDriver

    _, args = UniformDriver(dcfg)({}, {"drivers": {}})
    dw = float(np.asarray(args["drivers"]["E0"]["delta_omega"])[0])
    e_y = ds["e_y"].values[:, xs.size // 2, 0]
    steady = t > 0.25  # the pump has crossed the probe (6 um -> 10 um in ~15 fs) and settled
    phase_rate = np.mean(np.angle(e_y[steady][1:] / e_y[steady][:-1])) / 0.002  # rad / ps
    np.testing.assert_allclose(-phase_rate / derived["w0"], dw, atol=5e-4)  # E0 ~ e^{-i w0 dw t}
    # and the spectrum peaks in the bin nearest that offset
    omega = ds["delta omega (w_carrier)"].values
    peak = omega[np.argmax(ds["power"].values)]
    assert abs(peak - dw) <= 0.5 * (omega[1] - omega[0]) + 1e-12
    # the Poynting flux of the launched wave is rightward, v_g |E|^2 times the central
    # difference's sin(k dx)/(k dx) (as the poynting field diagnostic)
    n = dcfg["units"]["envelope density"]
    k_dx = derived["w0"] / derived["c"] * np.sqrt((1.0 + dw) ** 2 - n) * dx
    v_g = derived["c"] * np.sqrt((1.0 + dw) ** 2 - n) / (1.0 + dw)
    s_x = ds["s_x"].values[steady][:, xs.size // 2, 0]
    e_sq = np.abs(ds["e_y"].values[steady][:, xs.size // 2, 0]) ** 2
    np.testing.assert_allclose(np.mean(s_x / e_sq), v_g * np.sin(k_dx) / k_dx, rtol=0.02)

    # a single-node probe without guard cells: one column, whole y axis
    ds1 = spectra[1]
    assert ds1["x (um)"].size == 1 and ds1["y (um)"].size == dcfg["grid"]["ny"]
    assert "s_x" not in ds1
    np.testing.assert_allclose(np.diff(ds1["t (ps)"].values), 0.01, atol=1e-9)
    assert abs(ds1["t (ps)"].values[0]) < 1e-9


def test_probe_validation():
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_save_quantities,
        get_solver_quantities,
        write_units,
    )

    def finish(cfg):
        write_units(cfg)
        cfg = get_derived_quantities(cfg)
        cfg["grid"] = get_solver_quantities(cfg)
        cfg["grid"]["background_density"] = get_density_profile(cfg)
        return get_save_quantities(cfg)

    cfg = _cfg()
    cfg["save"]["light_spectrum"] = [{"field": "E1", "interval": "1fs"}]
    cfg["terms"]["epw"]["source"].update({"srs": False, "tpd": True})  # no Raman field
    with pytest.raises(ValueError, match="E1"):
        finish(cfg)
    cfg = _cfg()
    cfg["save"]["light_spectrum"] = [{"field": "E0", "interval": "1fs", "x": ["2um", "-2um"]}]
    with pytest.raises(ValueError, match="reversed"):
        finish(cfg)
    cfg = _cfg()
    cfg["save"]["light_spectrum"] = [{"field": "Ez", "interval": "1fs"}]
    with pytest.raises(ValueError, match="E0 or E1"):
        finish(cfg)


@pytest.mark.skipif(deck_path("test_030") is None, reason="no LPSE decks present")
def test_translator_maps_the_spectrum_instruments():
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    parms = parse_parms(deck_path("test_030"))
    cfg, report = translate_parms(parms, run="test_030")
    probes = cfg["save"]["light_spectrum"]
    # the deck declares nRamanSpectrum = 2 but defines only spectrum.1 (the second names no
    # output file, so LPSE disables it)
    assert len(probes) == 1
    for n, probe in enumerate(probes, start=1):
        assert probe["field"] == "E1"
        assert probe["interval"] == f"{float(parms[f'spectrum.{n}.raman.interval'])}ps"
        assert probe["tmin"] == f"{float(parms[f'spectrum.{n}.raman.startTime'])}ps"
        lo = parms[f"spectrum.{n}.raman.location.min"].split()[0]
        assert probe["x"][0] == f"{float(lo)}um"
        assert probe["poynting"] == any(k.startswith(f"spectrum.{n}.raman.file.S0.") for k in parms)
    # a probe on a field that is not evolved is skipped with a note
    parms["laserSpectrum.nLaserSpectrum"] = "1"
    parms.update({"spectrum.1.laser.enable": "true", "spectrum.1.laser.file.E0.z": "x", "laser.solver": "static"})
    cfg, report = translate_parms(parms, run="test_030")
    assert len(cfg["save"]["light_spectrum"]) == len(probes)
    assert any("spectrum.1.laser" in n for n in report["notes"])
