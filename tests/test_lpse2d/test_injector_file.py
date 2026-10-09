"""The pump launched from LPSE injector files (drivers.E0.injector_file, plan 2 L.4c)."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._lpse2d.parity import deck_path

NY = 64  # 3.2 um at dx = 0.05 um (7 cells per wavelength)


def _write_injector(path, times, planes):
    """An LPSE injector file (LightSolver::readInjectorFiles): per time, t then the two planes'
    complex values interleaved, single precision -- the layout m201902_createLpseInjector_v02.m writes."""
    rows = []
    for t, pl in zip(times, planes, strict=True):
        flat = np.empty(2 * pl.size)
        flat[0::2] = pl.real.ravel()
        flat[1::2] = pl.imag.ravel()
        rows.append(np.concatenate([[t], flat]))
    np.concatenate(rows).astype("<f4").tofile(path)


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


def _cfg(files, side="min.x", **light):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["grid"].update({"xmax": "20um", "tmax": "60fs", "ymax": "1.6um", "ymin": "-1.6um", "dx": "0.05um"})
    cfg["terms"]["light"] = {"solver": "fd", "pump_depletion": True, **light}
    cfg["terms"]["epw"]["source"].update({"noise": False, "srs": False})
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    # the pump envelope's rise well before t = 0 (srs.yaml holds the pump off until 50 fs)
    cfg["drivers"]["E0"]["envelope"].update({"tw": "41ps", "tc": "20ps"})
    cfg["drivers"]["E0"].update(
        {
            "delta_omega_max": 0.0,
            "offset": "5.025um",
            "turn_on_time": "5fs",
            "injector_file": {"side": side, "files": files},
        }
    )
    return cfg


def test_read_injector_file_layout_and_checks(tmp_path):
    from adept._lpse2d.lpse_deck import read_injector_file

    rng = np.random.default_rng(0)
    planes = rng.normal(size=(2, 2, 5)) + 1j * rng.normal(size=(2, 2, 5))
    path = tmp_path / "inj"
    _write_injector(path, [0.0, 0.05], planes)
    times, read = read_injector_file(path, 5)
    np.testing.assert_allclose(times, [0.0, 0.05], rtol=1e-7)
    np.testing.assert_allclose(read, planes.astype(np.complex64), rtol=1e-6)
    with pytest.raises(ValueError, match="multiple"):
        read_injector_file(path, 6)
    _write_injector(path, [0.01], planes[:1])
    with pytest.raises(ValueError, match="first time"):
        read_injector_file(path, 5)
    _write_injector(path, [0.0, 0.0], planes)
    with pytest.raises(ValueError, match="increase"):
        read_injector_file(path, 5)


@pytest.mark.parametrize("side", ["min.x", "max.x"])
def test_file_source_is_lpse_getInjectorSource(tmp_path, side):
    """The rows are LPSE's second-order getInjectorSource term by term, for all three components
    (the cross terms of the curl-curl included), on the plane's row and the row outside it."""
    from adept._lpse2d.core.light import CoupledLight

    ny = NY
    rng = np.random.default_rng(1)
    files = {}
    v = {}
    for comp in "xyz":
        v[comp] = rng.normal(size=(2, ny)) + 1j * rng.normal(size=(2, ny))
        files[comp] = str(tmp_path / f"inj_{comp}")
        _write_injector(files[comp], [0.0], v[comp][None])
    cfg = _finish(_cfg(files, side))
    light = CoupledLight(cfg)
    e_norm = cfg["units"]["derived"]["e_norm"]
    v = {c: v[c].astype(np.complex64).astype(np.complex128) / e_norm for c in v}
    h = float(cfg["grid"]["dx"])
    hx = h if side == "min.x" else -h  # LPSE flips the sign of hx on a max face
    gamma = light.diffraction_coeff0
    p = int(light.file_rows[1])
    alpha = complex(np.asarray(light.linear_coeff0)[p, 0])

    def dy(f):  # centred y difference numerator, periodic
        return np.roll(f, -1) - np.roll(f, 1)

    def d2y(f):
        return np.roll(f, -1) - 2 * f + np.roll(f, 1)

    V0 = {c: v[c][0] for c in v}
    V1 = {c: v[c][1] for c in v}
    primary = {
        "x": -alpha * V0["x"] - gamma / h**2 * d2y(V0["x"]) + gamma / (4 * hx * h) * dy(V1["y"]),
        "y": -alpha * V0["y"] - gamma / hx**2 * (V1["y"] - 2 * V0["y"]) + gamma / (4 * h * hx) * dy(V1["x"]),
        "z": -alpha * V0["z"] - gamma / hx**2 * (V1["z"] - 2 * V0["z"]) - gamma / h**2 * d2y(V0["z"]),
    }
    secondary = {
        "x": gamma / (4 * hx * h) * dy(V0["y"]),
        "y": -gamma / hx**2 * V0["y"] + gamma / (4 * h * hx) * dy(V0["x"]),
        "z": -gamma / hx**2 * V0["z"],
    }
    pattern = np.asarray(light.file_patterns[0])  # rows (outside, plane)
    for i, c in enumerate("xyz"):
        np.testing.assert_allclose(pattern[1, :, i], primary[c], rtol=1e-10, atol=1e-10 * np.abs(primary[c]).max())
        np.testing.assert_allclose(pattern[0, :, i], secondary[c], rtol=1e-10, atol=1e-10 * np.abs(primary[c]).max())
    assert list(light.file_rows) == ([p - 1, p] if side == "min.x" else [p + 1, p])


@pytest.mark.parametrize("side", ["min.x", "max.x"])
def test_file_injector_launches_a_one_way_plane_wave(tmp_path, side):
    """A file holding the second-order stencil's exact plane wave launches it at the file's
    amplitude into the box and nothing behind the plane (total-field / scattered-field)."""
    from adept._lpse2d.core.light import CoupledLight
    from adept._lpse2d.modules.driver import UniformDriver

    ny = NY
    amp = 1.0e-3  # e E / (m_e w0 c)
    # the discrete wavenumber at the box density: w0^2 (1 - n) / c^2 = (2 - 2 cos(k h)) / h^2
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        n = float(yaml.safe_load(fi)["density"]["val"])
    h = 0.05
    k0 = 2 * np.pi / 0.351 * np.sqrt(1.0 - n)
    kh = np.arccos(1.0 - (k0 * h) ** 2 / 2.0)
    path = tmp_path / "inj_z"
    _write_injector(path, [0.0], (amp * np.array([1.0, np.exp(1j * kh)])[:, None] * np.ones((1, ny)))[None])
    cfg = _finish(_cfg({"z": str(path)}, side))
    light = CoupledLight(cfg)
    assert light.file_injector and not light.fd_general_injector
    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    nx = cfg["grid"]["nx"]
    E0 = jnp.zeros((nx, ny, 3), dtype=jnp.complex128)
    E1 = jnp.zeros_like(E0)
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128)
    t, dt = 0.0, cfg["grid"]["dt"]
    for _ in range(int(0.06 / dt)):
        E0, E1 = light(t, E0, E1, phi_k, args["drivers"]["E0"], None)
        t += dt
    ez = np.abs(np.asarray(E0)[..., 2]).mean(axis=1) * cfg["units"]["derived"]["e_norm"]
    x = np.asarray(cfg["grid"]["x"])
    p = int(light.file_rows[1])
    s = 1 if side == "min.x" else -1
    # clear of the tanh absorbers' skirts (3 um layers, 0.6 um rise)
    inside = (s * (x - x[p]) > 1.0) & (s * (x - x[p]) < 4.0)
    behind = s * (x - x[p]) < -3 * h
    np.testing.assert_allclose(ez[inside], amp, rtol=0.02)
    assert ez[behind].max() < 0.01 * amp
    assert np.all(np.asarray(E0)[..., :2] == 0.0)


def test_file_injector_interpolates_and_cycles_its_times(tmp_path):
    """LPSE's loadInjector time handling: linear between the file times, repeating with the last
    time as the period."""
    from adept._lpse2d.core.light import CoupledLight
    from adept._lpse2d.modules.driver import UniformDriver

    ny = NY
    base = np.ones((2, ny)) * 1.0e-3
    path = tmp_path / "inj_z"
    _write_injector(path, [0.0, 0.01, 0.02], np.stack([base, 2 * base, 3 * base]))
    raw = _cfg({"z": str(path)})
    raw["drivers"]["E0"]["turn_on_time"] = "1e-6fs"
    cfg = _finish(raw)
    light = CoupledLight(cfg)
    _, args = UniformDriver(cfg)({}, {"drivers": {}})
    pa = args["drivers"]["E0"]
    unit = np.asarray(light.file_patterns[0])
    for t, scale in [(0.01, 2.0), (0.005, 1.5), (0.015, 2.5), (0.025, 1.5), (0.0399, 2.99)]:
        np.testing.assert_allclose(np.asarray(light.file_pump_rows(t, pa)), scale * unit, rtol=1e-6)


def test_file_injector_is_refused_where_lpse_refuses_it(tmp_path):
    from adept._lpse2d.helpers import get_derived_quantities, write_units

    path = tmp_path / "inj_z"
    _write_injector(path, [0.0], np.zeros((1, 2, NY)))
    for light, e0 in [
        ({"solver": "spectral"}, {}),
        ({"fd_order": 4}, {}),
        ({}, {"beams": [{"intensity": 1.0, "angle": 0.0}]}),
        ({}, {"angle": 10.0}),
    ]:
        cfg = _cfg({"z": str(path)}, **light)
        cfg["drivers"]["E0"].update(e0)
        write_units(cfg)
        with pytest.raises(ValueError, match="injector_file"):
            get_derived_quantities(cfg)


@pytest.mark.skipif(deck_path("test_019") is None, reason="original-lpse example decks not available")
def test_translator_maps_the_test_019_injector_file():
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    cfg, report = translate_parms(parse_parms(deck_path("test_019")), experiment="x", run="test_019")
    e0 = cfg["drivers"]["E0"]
    assert e0["injector_file"]["side"] == "min.x"
    assert e0["injector_file"]["files"]["z"].endswith("data/injector.E0_z.min.x")
    assert e0["turn_on_time"] == "30.0fs"
    assert not report["unsupported"]
    dx = float(cfg["grid"]["dx"].removesuffix("um"))
    offset = float(e0["offset"].removesuffix("um"))
    # LPSE's primary injector node int(Labc / h) = int(2 / (40 / 1439)) = 71 is adept's i0 + 1
    assert round(offset / dx - 0.5) + 1 == 71
    # the nominal intensity is the file's peak: 1e14 W/cm^2 at the waist, w0 / w = 0.123 at the injector
    # 18 um before the focus
    assert not any("not read" in note for note in report["notes"])
    assert 1.1e13 < float(cfg["units"]["laser intensity"].removesuffix("W/cm^2")) < 1.4e13
