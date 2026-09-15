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


def test_translator_maps_the_test_010_beam_direction():
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    deck = "/home/phil/Desktop/Ergodic-projects/original-lpse/examples/testRuns/test_010/lpse.parms"
    cfg, report = translate_parms(parse_parms(deck), experiment="x", run="test_010")
    assert abs(cfg["drivers"]["E0"]["angle"] - np.degrees(np.arctan2(0.28735, 0.95783))) < 1e-3
    assert not any("direction" in u for u in report["unsupported"])
