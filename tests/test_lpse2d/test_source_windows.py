"""Source windows and time gates (plan 2 I.3): LPSE ``restrictSourceRange`` on the EPW and IAW
sources, ``suppressSourcesInAbsorbingRegions`` / ``suppressSourcesAtInjectors``, and
``iaw.startEvolvingTime`` / ``stopEvolvingTime``."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._lpse2d.helpers import range_restriction


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


def _cfg(*, epw_window=None, iaw=None, light=None, seed=False, tpd=False):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": 0.2}
    cfg["drivers"]["E0"]["envelope"]["tc"] = "19.95ps"
    cfg["grid"].update(
        {
            "boundary_width": "0.4um",
            "dt": "1fs",
            "dx": "0.04um",
            "xmax": "2.56um",
            "tmax": "0.1ps",
            "ymax": "0.32um",
            "ymin": "-0.32um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": tpd, "srs": not tpd})
    if epw_window is not None:
        cfg["terms"]["epw"]["source_window"] = epw_window
    cfg["terms"]["light"] = {"solver": "fd", "pump_depletion": True, **(light or {})}
    if iaw is not None:
        cfg["terms"]["iaw"] = {"active": True, "solver": "spectral", **iaw}
    if seed:
        cfg["drivers"]["E1"] = {"intensity": "1e12W/cm^2", "offset": "0.8um"}
    return _finish(cfg)


def test_range_restriction_is_lpse_trapezoid():
    x = np.linspace(-5.0, 5.0, 201)
    y = np.linspace(-1.0, 1.0, 5)
    m = range_restriction(x, y, {"width": ["4um", "0um"], "center": ["1um", "0um"], "edge_width": "1um"})
    assert m.shape == (201, 5)
    assert np.all(m == m[:, :1])  # no y restriction
    prof = m[:, 0]
    # flat top of width 4 about +1: [-1, 3]; ramps over 1 um to zero at -2 and 4
    assert np.all(prof[(x >= -1.0) & (x <= 3.0)] == 1.0)
    assert np.all(prof[(x <= -2.0) | (x >= 4.0)] == 0.0)
    np.testing.assert_allclose(prof[(x > -2.0) & (x < -1.0)], x[(x > -2.0) & (x < -1.0)] + 2.0, atol=1e-12)
    np.testing.assert_allclose(prof[(x > 3.0) & (x < 4.0)], 4.0 - x[(x > 3.0) & (x < 4.0)], atol=1e-12)
    # two axes multiply; zero edge width is a box
    m2 = range_restriction(x, y, {"width": ["4um", "1um"], "center": ["1um", "0um"], "edge_width": "0um"})
    assert np.all(m2[:, [0, 4]] == 0.0) and np.all(m2[(x > -1.0) & (x <= 3.0), 2] == 1.0)
    assert np.all(m2[(x <= -1.0) | (x > 3.0), :] == 0.0)


def test_source_mask_zeroes_absorbers_and_injector_rows():
    cfg = _cfg(light={"suppress_sources_in_absorbers": True, "suppress_sources_at_injectors": True}, seed=True)
    mask = np.asarray(cfg["grid"]["epw_source_mask"])
    x = np.asarray(cfg["grid"]["x"])
    assert np.all(mask == mask[:, :1])
    prof = mask[:, 0]
    inside = (x < cfg["grid"]["xmin"] + 0.4) | (x > cfg["grid"]["xmax"] - 0.4)
    assert np.all(prof[inside] == 0.0)
    # the FD pump injector rows (xmin + offset, two rows) and the seed rows (xmax - offset)
    i0 = int(np.argmin(np.abs(x - (cfg["grid"]["xmin"] + cfg["drivers"]["E0"]["derived"]["offset"]))))
    i1 = int(np.argmin(np.abs(x - (cfg["grid"]["xmax"] - 0.8))))
    assert prof[i0] == 0.0 and prof[i0 + 1] == 0.0 and prof[i0 - 1] == 1.0 and prof[i0 + 2] == 1.0
    assert prof[i1] == 0.0 and prof[i1 + 1] == 0.0 and prof[i1 - 1] == 1.0
    # nothing configured: the mask is all ones and the solvers take the fast path
    from adept._lpse2d.core.epw import SpectralEPWSolver

    plain = _cfg()
    assert np.all(np.asarray(plain["grid"]["epw_source_mask"]) == 1.0)
    assert SpectralEPWSolver(plain).source_mask == 1.0


def _run(cfg, n_steps=15, amplitude=1.0e-3):
    from adept._lpse2d.core.vector_field import SplitStep

    step = SplitStep(cfg)
    rng = np.random.default_rng(11)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k *= np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    field = np.fft.ifft2(-1j * kx * phi_k)
    phi_k *= amplitude * cfg["units"]["derived"]["E0_source"] / np.abs(field).max()
    state = {
        "epw": jnp.asarray(phi_k).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 3), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 3), dtype=jnp.complex128).view(jnp.float64),
    }
    if cfg["terms"].get("iaw", {}).get("active", False):
        state["iaw_density"] = jnp.zeros((nx, ny))
        state["iaw_velocity_divergence"] = jnp.zeros((nx, ny))
    pump = {**cfg["drivers"]["E0"]["derived"], "delta_omega": jnp.zeros(1), "phases": jnp.zeros((1, ny))}
    args = {"drivers": {"E0": {**pump, "intensities": 30.0 * jnp.ones((1, ny))}}}
    history = []
    for i in range(n_steps):
        state = step(i * cfg["grid"]["dt"], dict(state), args)
        history.append({k: np.asarray(v) for k, v in state.items()})
    return step, history


def test_zero_window_removes_the_srs_drive_and_full_window_is_bit_identical():
    """A window that vanishes everywhere leaves the EPW to decay freely -- identical to a run
    with the SRS source off -- while an all-ones window reproduces the unwindowed run."""
    everywhere_zero = {"width": ["0.1um", "0um"], "center": ["100um", "0um"], "edge_width": "0um"}
    step_z, hist_z = _run(_cfg(epw_window=everywhere_zero))
    assert np.all(np.asarray(step_z.epw.source_mask) == 0.0)
    cfg_off = _cfg()
    cfg_off["terms"]["epw"]["source"]["srs"] = False
    cfg_off["terms"]["light"] = {"solver": "fd", "pump_depletion": True}
    _, hist_off = _run(cfg_off)
    np.testing.assert_array_equal(hist_z[-1]["epw"], hist_off[-1]["epw"])
    _, hist_plain = _run(_cfg())
    _, hist_ones = _run(_cfg(epw_window={"width": ["100um", "0um"], "center": ["0um", "0um"], "edge_width": "0um"}))
    np.testing.assert_array_equal(hist_ones[-1]["epw"], hist_plain[-1]["epw"])
    assert np.any(hist_plain[-1]["epw"] != hist_z[-1]["epw"])


def test_iaw_window_squares_the_drive_and_time_gates_hold():
    from adept._lpse2d.core.iaw import IonAcousticWave

    half = {"width": ["1.28um", "0um"], "center": ["-0.64um", "0um"], "edge_width": "0.2um"}
    cfg = _cfg(iaw={"source_window": half})
    iaw = IonAcousticWave(cfg)
    mask = np.asarray(cfg["grid"]["iaw_source_mask"])
    np.testing.assert_array_equal(np.asarray(iaw.source_mask_sq), mask**2)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E0 = jnp.ones((nx, ny, 3), dtype=jnp.complex128)
    drive = np.asarray(iaw.ponderomotive_drive(jnp.zeros((nx, ny), dtype=jnp.complex128), E0, 0.0 * E0))
    plain = IonAcousticWave(_cfg(iaw={}))
    drive_plain = np.asarray(plain.ponderomotive_drive(jnp.zeros((nx, ny), dtype=jnp.complex128), E0, 0.0 * E0))
    np.testing.assert_allclose(drive, drive_plain * mask**2, rtol=1e-12)
    assert np.any(mask == 0.0) and np.any(mask == 1.0) and np.any((mask > 0.0) & (mask < 1.0))

    # time gates: no IAW evolution before t_start, none from t_stop on
    _, hist = _run(_cfg(iaw={"t_start": 0.005, "t_stop": 0.010}))
    dens = [h["iaw_density"] for h in hist]
    assert all(np.all(d == 0.0) for d in dens[:5])  # t = 0..4 fs: gated off
    assert np.any(dens[6] != 0.0)  # driven once active
    for i in range(10, len(dens)):
        np.testing.assert_array_equal(dens[i], dens[9])  # frozen from t_stop = 10 fs
    _, hist_free = _run(_cfg(iaw={}))
    assert np.any(hist_free[-1]["iaw_density"] != dens[-1])


def test_lpse_iaw_solvers_window_the_drive_after_the_laplacian():
    """Inventory A23: LPSE forms E2 = K^2 PP and then suppresses it -- injectors, restrictRange squared,
    absorbing regions (getPonderomotivePotential -> suppressSourceInSpecificRegions(pp, false, true)).
    drive_laplacian = mask^2 * lap(PP) to 1e-12 (the same arithmetic), which is not lap(mask^2 PP)."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    half = {"width": ["1.28um", "0um"], "center": ["-0.64um", "0um"], "edge_width": "0.2um"}
    cfg = _cfg(iaw={"source_window": half, "solver": "spectral"})
    iaw = IonAcousticWave(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(3)
    E0 = jnp.asarray(rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3)))
    phi = jnp.zeros((nx, ny), dtype=jnp.complex128)
    k_sq = np.asarray(iaw.k_sq)
    mask_sq = np.asarray(iaw.source_mask_sq)
    pp = np.asarray(iaw.ponderomotive_drive(phi, E0, 0.0 * E0, masked=False))
    want = np.real(np.fft.ifft2(k_sq * np.fft.fft2(pp))) * mask_sq
    got = np.asarray(iaw.drive_laplacian(phi, E0, 0.0 * E0))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
    before = np.real(np.fft.ifft2(k_sq * np.fft.fft2(pp * mask_sq)))
    assert np.abs(got - before).max() > 1e-6 * np.abs(want).max()


def test_tpd_source_is_windowed_after_its_assembly():
    """Inventory A26: LPSE builds TPD_src in k-space (band, 1/k^2 term) and then suppresses it in x-space
    (makeTpdSource_fft -> suppressSourceInSpecificRegions), without re-filtering. The windowed source is
    F(window F^-1(unwindowed source)) to 1e-12 (the same arithmetic), not the source of windowed products."""
    from adept._lpse2d.core.epw import SpectralEPWSolver

    half = {"width": ["1.28um", "0um"], "center": ["-0.64um", "0um"], "edge_width": "0.2um"}
    windowed = SpectralEPWSolver(_cfg(epw_window=half, tpd=True))
    plain = SpectralEPWSolver(_cfg(tpd=True))
    nx, ny = int(windowed.kx.shape[0]), int(windowed.ky.shape[0])
    rng = np.random.default_rng(5)
    phi_k = jnp.asarray(rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * plain.low_pass_filter
    ex, ey = plain.phi_k_to_e_fields(phi_k)
    E0 = jnp.asarray(rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3)))
    base = np.asarray(plain.calc_tpd_source(0.1, phi_k, ex, ey, E0))
    got = np.asarray(windowed.calc_tpd_source(0.1, phi_k, ex, ey, E0))
    window = np.asarray(windowed.source_mask)
    want = np.fft.fft2(window * np.fft.ifft2(base))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_light_sources_are_zeroed_at_the_partner_injector_and_in_their_own_layer():
    """Inventory A27: LPSE zeroes each light field's coupling sources at the *partner* field's injector
    rows (always) and, with suppressSourcesInAbsorbingRegions, inside its own absorbing layer
    (LightSolver::calculateSources). The masks follow the rules exactly; with E1 = 0 the Raman light's
    RHS is its SRS source alone, which vanishes exactly on the pump injector rows -- without the
    transverse projection, which LPSE (like adept) applies after the suppression and which spreads the
    source back (takeTransversePartOfSourceTerms, off by default in LPSE)."""
    from adept._lpse2d.core.light import CoupledLight

    cfg = _cfg(seed=True, light={"transverse_source": False})
    grid = cfg["grid"]
    x = np.asarray(grid["x"])
    m0, m1 = np.asarray(grid["light_source_mask0"]), np.asarray(grid["light_source_mask1"])
    pump_rows = np.where(m1[:, 0] == 0.0)[0]
    seed_rows = np.where(m0[:, 0] == 0.0)[0]
    x_pump = grid["xmin"] + cfg["drivers"]["E0"]["derived"]["offset"]
    x_seed = grid["xmax"] - cfg["drivers"]["E1"]["derived"]["offset"]
    assert pump_rows.size >= 2 and np.all(np.abs(x[pump_rows] - x_pump) < 2.0 * grid["dx"])
    assert seed_rows.size >= 2 and np.all(np.abs(x[seed_rows] - x_seed) < 2.0 * grid["dx"])
    # the own layers only on request
    layered = _cfg(seed=True, light={"suppress_sources_in_absorbers": True})["grid"]
    width = layered["light_boundary_width_um"]
    inside = (x < width) | (x > grid["xmax"] - width)
    assert np.all(np.asarray(layered["light_source_mask0"])[inside] == 0.0)

    light = CoupledLight(cfg)
    nx, ny = grid["nx"], grid["ny"]
    rng = np.random.default_rng(9)
    E0 = jnp.asarray(rng.normal(size=(nx, ny, 3)) + 1j * rng.normal(size=(nx, ny, 3)))
    E1 = jnp.zeros((nx, ny, 3), dtype=jnp.complex128)
    lap = jnp.asarray(rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny)))
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.zeros((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    _, k_e1 = light.coupled_rhs(0.0, E0, E1, lap, pump_args, None, None, None)
    k_e1 = np.asarray(k_e1)
    assert np.all(k_e1[pump_rows] == 0.0)
    assert np.all(np.abs(k_e1[np.setdiff1d(np.arange(nx), pump_rows)]).max(axis=(1, 2)) > 0.0)


def test_translator_maps_source_windows_and_gates():
    from adept._lpse2d.lpse_deck import translate_parms

    base = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "laser.wavelength": "0.351",
        "lw.envelopeDensity": "0.25",
        "simulation.time.end": "1",
        "lw.restrictSourceRange.enable": "true",
        "lw.restrictSourceRange.width": "8 2 0",
        "lw.restrictSourceRange.center": "1 0 0",
        "lw.restrictSourceRange.edgeWidth": "0.5",
        "iaw.enable": "true",
        "iaw.spectral.dt": "0.005",
        "iaw.restrictSourceRange.enable": "true",
        "iaw.restrictSourceRange.width": "6 0 0",
        "iaw.startEvolvingTime": "0.25",
        "iaw.stopEvolvingTime": "0.75",
        "suppressSourcesInAbsorbingRegions": "true",
    }
    cfg, report = translate_parms(base, run="win")
    assert cfg["terms"]["epw"]["source_window"] == {
        "width": ["8.0um", "2.0um"],
        "center": ["1.0um", "0.0um"],
        "edge_width": "0.5um",
    }
    iaw = cfg["terms"]["iaw"]
    assert iaw["source_window"]["width"] == ["6.0um", "0.0um"] and iaw["source_window"]["edge_width"] == "0.0um"
    assert iaw["t_start"] == 0.25 and iaw["t_stop"] == 0.75
    assert cfg["terms"]["light"]["suppress_sources_in_absorbers"] is True
    assert cfg["terms"]["light"]["suppress_sources_at_injectors"] is True  # LPSE's default
    cfg, _ = translate_parms({**base, "suppressSourcesAtInjectors": "false"}, run="win")
    assert cfg["terms"]["light"]["suppress_sources_at_injectors"] is False
    assert not any("restrictSourceRange" in u or "suppressSources" in u for u in report["unsupported"])
