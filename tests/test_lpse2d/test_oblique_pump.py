"""Oblique incidence of the pump (drivers.E0.angle): static pump and spectral injector."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


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


def _cfg(angle, *, pump_depletion=False):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "6.4um", "ymin": "-6.4um", "dx": "0.1um"})
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": pump_depletion}
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing" if pump_depletion else "periodic", "y": "periodic"}
    cfg["drivers"]["E0"]["angle"] = angle
    return _finish(cfg)


def _dominant_mode(field, cfg):
    kx = np.asarray(cfg["grid"]["kx"])
    ky = np.asarray(cfg["grid"]["ky"])
    power = np.abs(np.fft.fft2(field[..., 0])) ** 2 + np.abs(np.fft.fft2(field[..., 1])) ** 2
    i, j = np.unravel_index(np.argmax(power), power.shape)
    ex = np.fft.fft2(field[..., 0])[i, j]
    ey = np.fft.fft2(field[..., 1])[i, j]
    return kx[i], ky[j], ex, ey, power[i, j] / power.sum()


@pytest.mark.parametrize("angle", [0.0, 20.0])
def test_static_pump_is_a_transverse_grid_mode_at_the_requested_angle(angle):
    from adept._lpse2d.core.laser import Light

    cfg = _cfg(angle)
    ny = cfg["grid"]["ny"]
    light_wave = {
        "delta_omega": jnp.array([0.0]),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    E0 = np.asarray(Light(cfg).laser_update(0.0, None, light_wave))
    d = cfg["units"]["derived"]
    k0 = d["w0"] / d["c"] * np.sqrt(1.0 - d["wp0"] ** 2 / d["w0"] ** 2)
    kx, ky, ex, ey, fraction = _dominant_mode(E0, cfg)
    assert fraction > 0.999  # one grid mode
    dk = 2 * np.pi / (cfg["grid"]["nx"] * cfg["grid"]["dx"])
    dky = 2 * np.pi / (ny * cfg["grid"]["dy"])
    assert abs(kx - k0 * np.cos(np.deg2rad(angle))) <= 0.5 * dk + 1e-9
    assert abs(ky - k0 * np.sin(np.deg2rad(angle))) <= 0.5 * dky + 1e-9
    # transverse: k . E = 0
    assert abs(kx * ex + ky * ey) < 1e-9 * np.hypot(kx, ky) * np.hypot(abs(ex), abs(ey))
    # the same amplitude as at normal incidence (swelling at the uniform box density)
    expected = d["E0_source"] * (1.0 - float(cfg["density"]["val"])) ** -0.25
    np.testing.assert_allclose(np.abs(E0[..., 0] + 1j * 0) ** 2 + np.abs(E0[..., 1]) ** 2, expected**2, rtol=1e-6)


def test_spectral_injector_launches_the_oblique_pump():
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight
    from adept._lpse2d.modules.driver import UniformDriver

    cfg = _cfg(20.0, pump_depletion=True)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    light = SpectralCoupledLight(cfg)
    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    pump_args = args["drivers"]["E0"]
    E0 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    E1 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128)
    t, dt = 0.0, cfg["grid"]["dt"]
    for _ in range(int(0.05 / dt)):  # 50 fs: the front crosses ~15 um at c
        E0, E1 = light(t, E0, E1, phi_k, pump_args, None)
        t += dt
    E0 = np.asarray(E0)
    d = cfg["units"]["derived"]
    k0 = d["w0"] / d["c"] * np.sqrt(1.0 - float(cfg["density"]["val"]))
    kx, ky, ex, ey, _ = _dominant_mode(E0, cfg)
    # a launched wave with a front and absorbers is broad in kx, but lives in one ky row
    power_ky = np.sum(np.abs(np.fft.fft2(E0[..., 0])) ** 2 + np.abs(np.fft.fft2(E0[..., 1])) ** 2, axis=0)
    assert power_ky[np.argmin(np.abs(np.asarray(cfg["grid"]["ky"]) - ky))] > 0.95 * power_ky.sum()
    dky = 2 * np.pi / (ny * cfg["grid"]["dy"])
    assert abs(ky - k0 * np.sin(np.deg2rad(20.0))) <= 0.5 * dky + 1e-9
    assert abs(kx - np.sqrt(k0**2 - ky**2)) < 0.1 * k0
    # the absorber (an x-space envelope applied after the projection) mixes a little
    assert abs(kx * ex + ky * ey) < 1e-2 * np.hypot(kx, ky) * np.hypot(abs(ex), abs(ey))
    assert np.max(np.abs(E0)) > 0.0


def test_fd_injector_refuses_an_oblique_pump():
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["terms"]["light"] = {"solver": "fd", "pump_depletion": True}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["drivers"]["E0"]["angle"] = 10.0
    with pytest.raises(ValueError, match="spectral"):
        _finish(cfg)


# the beam lines of LPSE's test_010 deck (examples/testRuns/test_010/laser_include.txt) and the
# main-deck keys the translator needs, checked in so the test runs without an LPSE checkout
TEST_010_DECK = """grid.sizes = 20 10;
grid.nodes = 360 180;
laser.enable = true;
laser.solver = spectral;
lw.enable = true;
lw.spectral.dt = 0.002;
simulation.time.end = 3;
#include "laser_include.txt"
"""
TEST_010_LASER_INCLUDE = """laser.nBeams= 1;
laser.wavelength= 0.351;
laser.1.intensity = 1.5e+15;
laser.1.phase = 300.26;
laser.1.polarization = 0;
laser.1.frequencyShift = 0;
laser.1.group = 0;
laser.1.direction = 0.95783 0.28735 0;
laser.1.evolution.source = min.x;
laser.1.evolution.offset = 0 0 0;
laser.1.evolution.width = 0;
laser.1.evolution.sgOrder = 0;
"""


def test_translator_maps_the_test_010_beam_direction(tmp_path):
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    (tmp_path / "laser_include.txt").write_text(TEST_010_LASER_INCLUDE)
    deck = tmp_path / "lpse.parms"
    deck.write_text(TEST_010_DECK)
    cfg, report = translate_parms(parse_parms(deck), experiment="x", run="test_010")
    assert abs(cfg["drivers"]["E0"]["angle"] - np.degrees(np.arctan2(0.28735, 0.95783))) < 1e-3
    assert not any("direction" in u for u in report["unsupported"])


def test_two_beams_launch_both_transverse_wavenumbers():
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight
    from adept._lpse2d.modules.driver import UniformDriver

    cfg = _cfg(0.0, pump_depletion=True)
    cfg = _finish(_beams_cfg(cfg, [{"intensity": 1.0, "angle": 20.0}, {"intensity": 1.0, "angle": -20.0}]))
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    light = SpectralCoupledLight(cfg)
    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    pa = args["drivers"]["E0"]
    E0 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    E1 = jnp.zeros_like(E0)
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128)
    t, dt = 0.0, cfg["grid"]["dt"]
    for _ in range(int(0.05 / dt)):
        E0, E1 = light(t, E0, E1, phi_k, pa, None)
        t += dt
    E0 = np.asarray(E0)
    power_ky = np.sum(np.abs(np.fft.fft2(E0[..., 0])) ** 2 + np.abs(np.fft.fft2(E0[..., 1])) ** 2, axis=0)
    ky = np.asarray(cfg["grid"]["ky"])
    k0 = cfg["units"]["derived"]["w0"] / cfg["units"]["derived"]["c"] * np.sqrt(1.0 - float(cfg["density"]["val"]))
    jp = int(np.argmin(np.abs(ky - k0 * np.sin(np.deg2rad(20.0)))))
    jm = int(np.argmin(np.abs(ky + k0 * np.sin(np.deg2rad(20.0)))))
    assert power_ky[jp] + power_ky[jm] > 0.95 * power_ky.sum()
    np.testing.assert_allclose(power_ky[jp], power_ky[jm], rtol=0.05)


def _beams_cfg(cfg_finished, beams, **extra):
    """Rebuild the raw config with beams/extras (the finished one already carries derived arrays)."""
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        raw = yaml.safe_load(fi)
    raw = deepcopy(raw)
    raw["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "6.4um", "ymin": "-6.4um", "dx": "0.1um"})
    raw["terms"]["light"] = {"solver": "spectral", "pump_depletion": True}
    raw["terms"]["epw"]["source"]["noise"] = False
    raw["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    raw["drivers"]["E0"]["beams"] = beams
    raw["drivers"]["E0"].update(extra)
    return raw


def test_transverse_profile_kap_and_pulse_file(tmp_path):
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight
    from adept._lpse2d.modules.driver import UniformDriver

    pulse = tmp_path / "pulse.txt"
    np.savetxt(pulse, np.array([[0.0, 0.25], [1.0, 0.25]]))
    raw = _beams_cfg(
        None,
        [{"intensity": 1.0, "angle": 0.0}],
        beam_width="2um",
        beam_sg_order=4.0,
        beam_offset="1um",
        kap_bandwidth=0.02,
        pulse_file=str(pulse),
    )
    cfg = _finish(raw)
    light = SpectralCoupledLight(cfg)
    y = np.asarray(cfg["grid"]["y"])
    expected = np.exp(-(((y - 1.0) ** 2 / (2.0 * 2.0**2)) ** 2.0))
    np.testing.assert_allclose(np.asarray(light.beam_envelope_y), expected, rtol=1e-12)
    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    pa = args["drivers"]["E0"]
    # the pulse table scales the source amplitude by 0.25
    s = np.asarray(light.calc_pump_source(0.05, pa))
    raw2 = _beams_cfg(None, [{"intensity": 1.0, "angle": 0.0}], beam_width="2um", beam_sg_order=4.0, beam_offset="1um")
    light2 = SpectralCoupledLight(_finish(raw2))
    s2 = np.asarray(light2.calc_pump_source(0.05, pa))
    np.testing.assert_allclose(np.abs(s), 0.25 * np.abs(s2), rtol=1e-9)
    # KAP: the phase is piecewise constant with correlation time 2 pi / (kap_bandwidth w0)
    w0 = cfg["units"]["derived"]["w0"]
    tau = 2.0 * np.pi / (0.02 * w0)
    phases = [float(light.kap_phase(t, 0)) for t in np.arange(0.0, 20.0 * tau, tau / 4.0)]
    jumps = np.sum(np.abs(np.diff(phases)) > 1e-9)
    assert 15 <= jumps <= 21  # one jump per correlation time
    assert float(light2.kap_phase(0.3, 0)) == 0.0


def test_translator_maps_multi_beam_decks():
    from adept._lpse2d.lpse_deck import translate_parms

    parms = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "lw.enable": "true",
        "lw.spectral.dt": "0.005",
        "simulation.time.end": "1",
        "laser.nBeams": "2",
        "laser.1.intensity": "1e15",
        "laser.1.direction": "0.94 0.34 0",
        "laser.1.phase": "0.5",
        "laser.2.intensity": "3e15",
        "laser.2.direction": "0.94 -0.34 0",
        "laser.2.frequencyShift": "0.01",
        "laser.1.width": "3",
        "laser.1.sgOrder": "4",
        "laser.1.offset": "0 1 0",
        "laser.bandwidth.KAP.frequency": "0.005",
    }
    cfg, report = translate_parms(parms, experiment="x", run="y")
    e0 = cfg["drivers"]["E0"]
    assert len(e0["beams"]) == 2 and e0["beams"][0]["intensity"] == 1e15 and e0["beams"][1]["delta_omega"] == 0.01
    assert abs(e0["beams"][1]["angle"] + e0["beams"][0]["angle"]) < 1e-9
    assert e0["beams"][0]["phase"] == pytest.approx(np.deg2rad(0.5), rel=1e-12)  # LPSE phases are degrees
    assert e0["beam_width"] == "3.0um" and e0["beam_sg_order"] == 4.0 and e0["beam_offset"] == "1.0um"
    assert e0["kap_bandwidth"] == 0.005
    assert not any("direction differs" in u for u in report["unsupported"])


# ---- one normal-incidence beam keeps its frequency shift and phase (PR #372 review) -----------------
# Before the fix the static pump's normal-incidence branch read only the color-level delta_omega /
# phases, the FD evolved pump ignored the beam list, and the translator dropped beams for one beam.
# A beam's shift and phase add to every color's, so a detuned single beam must equal the same shift
# and phase given at the color level -- an identity, checked to rtol 1e-12.

SINGLE_BEAM = [{"intensity": 1.0, "angle": 0.0, "delta_omega": 0.01, "phase": 0.3}]


def _light_wave(ny, delta_omega=0.0, phase=0.0):
    return {
        "delta_omega": jnp.array([delta_omega]),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.full((1, ny), phase),
    }


def test_static_pump_honors_a_single_beam_frequency_shift_and_phase():
    from adept._lpse2d.core.laser import Light

    cfg_plain = _cfg(0.0)
    raw = _beams_cfg(cfg_plain, SINGLE_BEAM)
    raw["terms"]["light"]["pump_depletion"] = False
    raw["terms"]["epw"]["boundary"]["x"] = "periodic"
    cfg_beam = _finish(raw)
    ny = cfg_plain["grid"]["ny"]
    beam, plain = Light(cfg_beam), Light(cfg_plain)
    assert not beam.multi_beam  # still the normal-incidence (speckle-capable) branch
    for t in (0.0, 0.013, 0.05):
        shifted = np.asarray(beam.laser_update(t, None, _light_wave(ny)))
        reference = np.asarray(plain.laser_update(t, None, _light_wave(ny, 0.01, 0.3)))
        np.testing.assert_allclose(shifted, reference, rtol=1e-12, atol=0)
    unshifted = np.asarray(plain.laser_update(0.05, None, _light_wave(ny)))
    assert np.max(np.abs(shifted - unshifted)) > 1e-3 * np.max(np.abs(unshifted))


def test_fd_evolved_pump_honors_a_single_beam_frequency_shift_and_phase():
    from adept._lpse2d.core.light import CoupledLight

    def cfg_for(beams):
        with open("tests/test_lpse2d/configs/srs.yaml") as fi:
            raw = deepcopy(yaml.safe_load(fi))
        raw["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "0.05um", "ymin": "-0.05um"})
        raw["terms"]["light"] = {"solver": "fd", "pump_depletion": True}
        raw["terms"]["epw"]["source"]["noise"] = False
        raw["terms"]["epw"]["boundary"]["x"] = "absorbing"
        if beams is not None:
            raw["drivers"]["E0"]["beams"] = beams
        return _finish(raw)

    cfg_beam, cfg_plain = cfg_for(SINGLE_BEAM), cfg_for(None)
    beam, plain = CoupledLight(cfg_beam), CoupledLight(cfg_plain)
    ny = cfg_plain["grid"]["ny"]
    pump = cfg_plain["drivers"]["E0"]["derived"]
    for t in (0.013, 0.05):
        rows_beam = beam.calc_pump_source(t, {**pump, **_light_wave(ny)})
        rows_plain = plain.calc_pump_source(t, {**pump, **_light_wave(ny, 0.01, 0.3)})
        for a, b in zip(rows_beam, rows_plain, strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=0)


def test_translator_keeps_a_single_beam_frequency_shift_and_phase():
    from adept._lpse2d.lpse_deck import translate_parms

    base = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "lw.enable": "true",
        "lw.spectral.dt": "0.005",
        "simulation.time.end": "1",
        "laser.nBeams": "1",
        "laser.1.intensity": "1e15",
    }
    cfg, _ = translate_parms({**base, "laser.1.frequencyShift": "0.01", "laser.1.phase": "90"}, run="y")
    (b,) = cfg["drivers"]["E0"]["beams"]
    assert b["delta_omega"] == 0.01 and b["phase"] == pytest.approx(np.pi / 2, rel=1e-12)
    # LPSE's wavelengthShift (Angstrom) overrides frequencyShift: dW/W0 = -1e-4 dLambda / lambda_um
    cfg, _ = translate_parms({**base, "laser.1.frequencyShift": "0.01", "laser.1.wavelengthShift": "10"}, run="y")
    assert cfg["drivers"]["E0"]["beams"][0]["delta_omega"] == pytest.approx(-1.0e-3 / 0.351, rel=1e-12)
    # no shift and no phase: the deck stays beam-free, as before
    cfg, _ = translate_parms(base, run="y")
    assert "beams" not in cfg["drivers"]["E0"]


# ---- the combined solver injects the same pump (PR #372 review) -----------------------------------
# The combined solver (needed for simultaneous TPD + SRS) used to build only a normal-incidence
# scalar pump and silently ignore beams, profile, KAP and the pulse table. It now shares
# SmoothPumpInjector with the spectral light solver. Thresholds as in the spectral tests above.


def _combined_beams_cfg(beams, **extra):
    raw = _beams_cfg(None, beams, **extra)
    raw["terms"]["epw"]["solver"] = "combined"
    # both sources on (the combined solver's rule); with no noise and E1 = 0 the unified source
    # is zero, so the pump propagates alone
    raw["terms"]["epw"]["source"].update({"tpd": True, "srs": True, "noise": False})
    return _finish(raw)


def test_combined_solver_pump_source_is_the_spectral_injector(tmp_path):
    from adept._lpse2d.core.combined import CombinedSolver
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight
    from adept._lpse2d.modules.driver import UniformDriver

    pulse = tmp_path / "pulse.txt"
    np.savetxt(pulse, np.array([[0.0, 0.25], [0.05, 1.0], [1.0, 1.0]]))
    cfg = _combined_beams_cfg(
        [{"intensity": 1.0, "angle": 20.0, "phase": 0.3}, {"intensity": 0.5, "angle": -15.0}],
        beam_width="3um",
        beam_sg_order=4.0,
        kap_bandwidth=0.02,
        pulse_file=str(pulse),
    )
    combined = CombinedSolver(cfg)
    spectral = SpectralCoupledLight(cfg)
    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    pa = args["drivers"]["E0"]
    for t in (0.004, 0.03, 0.2):
        np.testing.assert_array_equal(
            np.asarray(combined.calc_pump_source(t, pa)), np.asarray(spectral.calc_pump_source(t, pa))
        )
    # the oblique beams put power in both components (polarization perpendicular to k)
    s = np.asarray(combined.calc_pump_source(0.2, pa))
    assert np.abs(s[..., 0]).max() > 0.1 * np.abs(s[..., 1]).max()


def test_combined_solver_launches_both_oblique_beams():
    from adept._lpse2d.core.combined import CombinedSolver
    from adept._lpse2d.modules.driver import UniformDriver

    cfg = _combined_beams_cfg([{"intensity": 1.0, "angle": 20.0}, {"intensity": 1.0, "angle": -20.0}])
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    solver = CombinedSolver(cfg)
    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    driver_args = {"E0": args["drivers"]["E0"]}
    y = {"E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128), "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128)}
    t, dt = 0.0, cfg["grid"]["dt"]
    for _ in range(int(0.05 / dt)):
        E0, E1, _ = solver(t, y, driver_args)
        y = {"E0": E0, "E1": E1}
        t += dt
    E0 = np.asarray(y["E0"])
    power_ky = np.sum(np.abs(np.fft.fft2(E0[..., 0])) ** 2 + np.abs(np.fft.fft2(E0[..., 1])) ** 2, axis=0)
    ky = np.asarray(cfg["grid"]["ky"])
    k0 = cfg["units"]["derived"]["w0"] / cfg["units"]["derived"]["c"] * np.sqrt(1.0 - float(cfg["density"]["val"]))
    jp = int(np.argmin(np.abs(ky - k0 * np.sin(np.deg2rad(20.0)))))
    jm = int(np.argmin(np.abs(ky + k0 * np.sin(np.deg2rad(20.0)))))
    assert power_ky[jp] + power_ky[jm] > 0.95 * power_ky.sum()
    np.testing.assert_allclose(power_ky[jp], power_ky[jm], rtol=0.05)
