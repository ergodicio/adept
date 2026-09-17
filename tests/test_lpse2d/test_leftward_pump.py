"""Leftward pump beams from the x-max face (drivers.E0.angle / beams[].angle with |angle| > 90;
LPSE laser.N.direction with a negative x) -- plan 2 L.4a."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept._lpse2d.parity import deck_path


def _cfg(solver, angle=None, beams=None):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"ymax": "0.02um", "ymin": "-0.02um", "xmax": "20um", "tmax": "0.3ps"})
    cfg["terms"]["light"] = {"pump_depletion": True, "solver": solver}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    if angle is not None:
        cfg["drivers"]["E0"]["angle"] = angle
    if beams is not None:
        cfg["drivers"]["E0"]["beams"] = beams
    cfg["save"]["fields"]["t"].update({"tmax": "0.3ps", "dt": "0.05ps"})
    cfg["mlflow"]["run"] = f"leftward-pump-{solver}"
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, _, _ = exo(modules)
    result = sol["solver result"]
    dcfg = exo.adept_module.cfg
    x = np.asarray(dcfg["grid"]["x"])
    raw = np.asarray(result.ys["fields"]["E0"])
    e0 = raw.view(np.complex64 if raw.dtype == np.float32 else np.complex128)[-1, :, 0, :]
    t = np.asarray(result.ts["default"])
    series = {k: np.asarray(v)[t > 0.2] for k, v in result.ys["default"].items() if k.endswith("_flux")}
    return dcfg, x, e0, series


def test_fd_leftward_pump_mirrors_the_rightward_one():
    """angle 180 with the FD injector: the two-point rows sit at xmax - offset and launch
    e^{-i k0 x} into x < x_inject; amplitude, wavenumber and fluxes mirror the +x run."""
    dcfg, x, e0, series = _run(_cfg("fd", angle=180.0))
    derived = dcfg["units"]["derived"]
    n = dcfg["units"]["envelope density"]
    nominal = derived["E0_source"] * (1.0 - n) ** -0.25
    from adept._lpse2d.core.stencils import launched_amplitude_ratio
    from adept._lpse2d.modules.driver import UniformDriver

    _, args = UniformDriver(dcfg)({}, {"drivers": {}})
    dw = float(np.asarray(args["drivers"]["E0"]["delta_omega"])[0])
    k0_dx = derived["w0"] / derived["c"] * np.sqrt((1.0 + dw) ** 2 - n) * dcfg["grid"]["dx"]
    e0y = e0[:, 1]
    bulk = slice(np.argmin(np.abs(x - 7.0)), np.argmin(np.abs(x - 12.0)))
    np.testing.assert_allclose(np.mean(np.abs(e0y[bulk])), nominal * launched_amplitude_ratio(k0_dx, 2), rtol=5e-3)
    theta = np.angle(e0y[bulk][1:] / e0y[bulk][:-1])
    assert np.mean(theta) < 0  # leftward phase advance
    np.testing.assert_allclose(-np.mean(theta), np.arccos(1.0 - k0_dx**2 / 2.0), rtol=1e-3)
    # nothing behind the plane (x > 14 um + two rows), and the wave is absorbed at x-min
    behind = x > 14.0 + 3 * dcfg["grid"]["dx"]
    assert np.max(np.abs(e0y[behind])) < 0.02 * nominal
    assert np.max(np.abs(e0y[x < 1.0])) < 0.01 * nominal
    # the flux probes read a leftward (negative) flux of the nominal size on both sides, the
    # right probe upstream of the x-max rows
    assert np.mean(series["incident_flux"]) < 0 and np.mean(series["transmitted_flux"]) < 0
    np.testing.assert_allclose(np.mean(series["incident_flux"]), np.mean(series["transmitted_flux"]), rtol=1e-2)
    np.testing.assert_allclose(
        abs(np.mean(series["transmitted_flux"])), launched_amplitude_ratio(k0_dx, 2) ** 2, rtol=2e-2
    )


def test_spectral_counter_propagating_beams_form_a_standing_wave():
    """Two beams of half intensity at 0 and 180 deg (the CBET / SBS-backscatter geometry): the
    spectral injectors launch one from each x face; the bulk holds +k0 and -k0 with equal
    power and the mean intensity is the nominal one."""
    beams = [{"intensity": 0.5, "angle": 0.0}, {"intensity": 0.5, "angle": 180.0}]
    dcfg, x, e0, series = _run(_cfg("spectral", beams=beams))
    derived = dcfg["units"]["derived"]
    n = dcfg["units"]["envelope density"]
    nominal_sq = (derived["E0_source"] * (1.0 - n) ** -0.25) ** 2
    e0y = e0[:, 1]
    bulk = slice(np.argmin(np.abs(x - 7.0)), np.argmin(np.abs(x - 13.0)))
    # mean intensity: |a e^{ikx} + b e^{-ikx}|^2 averages to |a|^2 + |b|^2 = nominal
    np.testing.assert_allclose(np.mean(np.abs(e0y[bulk]) ** 2), nominal_sq, rtol=3e-2)
    # and it is a standing wave: the contrast is close to full
    envelope = np.abs(e0y[bulk])
    assert envelope.max() > 1.8 * envelope.min() + 0.5 * np.sqrt(nominal_sq)
    spectrum = np.fft.fft(e0y[bulk] * np.hanning(e0y[bulk].size))
    k = np.fft.fftfreq(e0y[bulk].size, d=dcfg["grid"]["dx"]) * 2 * np.pi
    p_plus = np.sum(np.abs(spectrum[k > 0]) ** 2)
    p_minus = np.sum(np.abs(spectrum[k < 0]) ** 2)
    np.testing.assert_allclose(p_plus, p_minus, rtol=0.1)
    # the incident probe (x-min side, clear of both injectors) sees the net flux ~ 0
    assert abs(np.mean(series["incident_flux"])) < 0.1


def test_validation_and_translator():
    from adept._lpse2d.helpers import get_derived_quantities, write_units

    cfg = _cfg("fd", angle=90.0)
    write_units(cfg)
    with pytest.raises(ValueError, match="y face"):
        get_derived_quantities(cfg)
    # an oblique leftward beam is accepted by the FD injector too (the general commutator
    # injector of plan 2 L.4) and marked leftward
    cfg = _cfg("fd", angle=160.0)
    write_units(cfg)
    assert bool(get_derived_quantities(cfg)["drivers"]["E0"]["derived"]["beam_leftward"][0])
    cfg = _cfg("fd", angle=-180.0)
    write_units(cfg)
    assert bool(get_derived_quantities(cfg)["drivers"]["E0"]["derived"]["beam_leftward"][0])

    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    if deck_path("test_006") is None:
        pytest.skip("no LPSE decks present")
    parms = parse_parms(deck_path("test_006"))
    parms["laser.1.direction"] = "-1 0 0"
    parms["laser.solver"] = "spectral"
    cfg, report = translate_parms(parms, run="test_006")
    assert cfg["drivers"]["E0"]["angle"] == 180.0  # a single beam is written as drivers.E0.angle
    assert not any("direction" in u for u in report["unsupported"])
    parms["laser.1.direction"] = "0 1 0"
    cfg, report = translate_parms(parms, run="test_006")
    assert any("direction" in u for u in report["unsupported"])
