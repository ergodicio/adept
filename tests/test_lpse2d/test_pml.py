"""PML absorbing layers for the FD light fields (terms.light.absorber: pml; LPSE abc.type = pml)
-- plan 2 L.2."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept._lpse2d.parity import deck_path


def _cfg(absorber, order=2):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {"ymax": "0.02um", "ymin": "-0.02um", "xmax": "20um", "tmax": "0.4ps", "boundary_width": "1.5um"}
    )
    cfg["terms"]["light"] = {"pump_depletion": True, "solver": "fd", "fd_order": order, "absorber": absorber}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["drivers"]["E0"]["delta_omega_max"] = 0.0
    cfg["save"]["fields"]["t"].update({"tmax": "0.4ps", "dt": "0.1ps"})
    cfg["mlflow"]["run"] = f"pml-{absorber}"
    return cfg


def _reflection(cfg):
    """Reflected over incident power of the steady pump in the bulk (7-15 um), by the sign of
    kx in a Hann-windowed spectrum, plus the flux probes."""
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, _, _ = exo(modules)
    result = sol["solver result"]
    dcfg = exo.adept_module.cfg
    x = np.asarray(dcfg["grid"]["x"])
    raw = np.asarray(result.ys["fields"]["E0"])
    e = raw.view(np.complex64 if raw.dtype == np.float32 else np.complex128)[-1, :, 0, 1]
    m = (x > 7.0) & (x < 15.0)
    spectrum = np.fft.fft(e[m] * np.hanning(m.sum()))
    k = np.fft.fftfreq(m.sum(), d=dcfg["grid"]["dx"]) * 2 * np.pi
    r = np.sum(np.abs(spectrum[k < 0]) ** 2) / np.sum(np.abs(spectrum[k > 0]) ** 2)
    t = np.asarray(result.ts["default"])
    steady = t > 0.3
    inc = np.mean(np.asarray(result.ys["default"]["incident_flux"])[steady])
    tr = np.mean(np.asarray(result.ys["default"]["transmitted_flux"])[steady])
    edge = np.abs(e[-1])
    return r, inc, tr, edge


def test_pml_reflects_less_than_the_exp_layer_and_kills_the_wall():
    r_exp, inc_exp, tr_exp, _ = _reflection(_cfg("exp"))
    r_pml, inc_pml, tr_pml, edge = _reflection(_cfg("pml"))
    # the exp layer (5e3/ps peak, 1.5 um) reflects ~5e-5 of the power; the PML < 1e-5
    assert r_pml < 1e-5 and r_pml < 0.2 * r_exp
    assert edge == 0.0  # the wall cell is held at zero
    # the launched flux is the same and transmitted to the far probe with either layer
    np.testing.assert_allclose(inc_pml, inc_exp, rtol=1e-2)
    np.testing.assert_allclose(tr_pml, inc_pml, rtol=1e-2)


def test_validation_and_translator():
    from adept._lpse2d.helpers import get_derived_quantities, write_units

    cfg = _cfg("pml")
    cfg["terms"]["light"]["solver"] = "spectral"
    write_units(cfg)
    with pytest.raises(ValueError, match="pml"):
        get_derived_quantities(cfg)

    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    if deck_path("test_011") is None:
        pytest.skip("no LPSE decks present")
    parms = parse_parms(deck_path("test_011"))
    cfg, report = translate_parms(parms, run="test_011")
    assert cfg["terms"]["light"]["absorber"] == "pml"
    parms["laser.evolution.abc.SabcDenom"] = "4"
    cfg, report = translate_parms(parms, run="test_011")
    assert cfg["terms"]["light"]["pml_denominator"] == 4.0
    parms["laser.solver"] = "spectral"
    cfg, report = translate_parms(parms, run="test_011")
    assert "absorber" not in cfg["terms"]["light"] and any("pml" in n for n in report["notes"])
