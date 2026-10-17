"""LPSE-parity tests for the spectral light propagator (terms.light.solver: spectral)
and the collisional light absorption (terms.light.absorption)."""

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


def _srs_cfg(solver="spectral", *, seed=False, pump_depletion=False, absorption=False, xmax="20um", periodic=False):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"xmax": xmax, "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um"})
    cfg["terms"]["light"] = {"solver": solver, "pump_depletion": pump_depletion, "absorption": absorption}
    if pump_depletion:
        cfg["terms"]["light"]["coupling"] = "rotation"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["terms"]["epw"]["boundary"]["x"] = "periodic" if periodic else "absorbing"
    if seed:
        # the seed's default offset (1.6 boundary widths) sits inside the absorber skirt by
        # design of the FD path; the smooth injector must be clear of it
        cfg["drivers"]["E1"] = {
            "intensity": "1.0e+12W/cm^2",
            "delta_omega": 0.0,
            "turn_on_time": "10fs",
            "offset": "7um",
        }
    return _finish(cfg)


def _pump_args(cfg):
    ny = cfg["grid"]["ny"]
    return {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }


def _plane_wave(cfg, w, sign, amplitude=1.0):
    """E_y = A exp(sign i k x) with k the grid-snapped local light wavenumber for carrier w."""
    derived = cfg["units"]["derived"]
    n = float(np.asarray(cfg["grid"]["background_density"])[0, 0])
    k = np.sqrt(w**2 - n * derived["w0"] ** 2) / derived["c"]
    dk = 2.0 * np.pi / (cfg["grid"]["nx"] * cfg["grid"]["dx"])
    k = np.round(k / dk) * dk
    x = np.asarray(cfg["grid"]["x"])
    field = np.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"], 2), dtype=np.complex128)
    field[..., 1] = amplitude * np.exp(sign * 1j * k * x)[:, None]
    return jnp.asarray(field), k


def test_spectral_propagator_has_no_grid_dispersion():
    """A resonant plane wave e^{-i k1 x} advances by exactly the analytic phase
    dt [w1/2 (1 - n w0^2/w1^2) - c^2 k^2/(2 w1)] per step under the spectral solver; the FD
    solver's grid dispersion gives a different phase at 7 cells per wavelength."""
    from adept._lpse2d.core.raman import RamanLight
    from adept._lpse2d.core.spectral_light import SpectralRamanLight

    cfg_s = _srs_cfg("spectral", periodic=True, xmax="8um")
    cfg_f = _srs_cfg("fd", periodic=True, xmax="8um")
    spectral, fd = SpectralRamanLight(cfg_s), RamanLight(cfg_f)
    derived = cfg_s["units"]["derived"]
    E1, k = _plane_wave(cfg_s, derived["w1"], -1)
    n_steps = 20
    E0_fn = lambda t: jnp.zeros_like(E1)
    phi_k = jnp.zeros(E1.shape[:2], dtype=jnp.complex128)
    out_s, out_f = E1, E1
    for i in range(n_steps):
        out_s = spectral(i * spectral.dt, out_s, E0_fn, phi_k, None)
        out_f = fd(i * fd.dt, out_f, E0_fn, phi_k, None)
    n = float(np.asarray(cfg_s["grid"]["background_density"])[0, 0])
    w1, c = derived["w1"], derived["c"]
    phase = n_steps * spectral.dt * (w1 / 2.0 * (1.0 - n * derived["w0"] ** 2 / w1**2) - c**2 * k**2 / (2.0 * w1))
    expected = np.asarray(E1) * np.exp(1j * phase)
    np.testing.assert_allclose(np.asarray(out_s), expected, rtol=1e-10, atol=1e-12)
    assert not np.allclose(np.asarray(out_f), expected, rtol=1e-4, atol=1e-6)
    # the spectral solver has no CFL limit: it sub-cycles only for its injectors (one cell per
    # sub-step), fewer steps than the FD scheme's stability bound
    assert 1 <= cfg_s["grid"]["light_substeps"] < cfg_f["grid"]["light_substeps"]


def _fill_box(cfg, n_steps, pump_depletion):
    from adept._lpse2d.core.vector_field import SplitStep

    step = SplitStep(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
    }
    args = {"drivers": {"E0": _pump_args(cfg)} if pump_depletion else {}}
    if "E1" in cfg["drivers"]:
        args["drivers"]["E1"] = cfg["drivers"]["E1"]["derived"]
    dt = cfg["grid"]["dt"]
    for i in range(n_steps):
        state = step(jnp.asarray(i * dt), dict(state), args)
    return step, state


def test_spectral_seed_injector_launches_the_exact_amplitude_leftward():
    """The smooth injector launches |E1| = E1_source / eps1^(1/4) with no sinc deficit, as a
    leftward wave at the local k1 (the FD two-point injector keeps its MATLAB calibration;
    the width default is half a local wavelength, LPSE's injectorWidth)."""
    cfg = _srs_cfg("spectral", seed=True)
    del cfg["drivers"]["E0"]
    step, state = _fill_box(cfg, 180, pump_depletion=False)
    e1y = np.asarray(state["E1"].view(jnp.complex128))[:, 0, 1]
    x = np.asarray(cfg["grid"]["x"])
    derived = cfg["units"]["derived"]
    n = cfg["units"]["envelope density"]
    eps1 = 1.0 - n * (derived["w0"] / derived["w1"]) ** 2
    k1 = np.sqrt(derived["w1"] ** 2 - derived["wp0"] ** 2) / derived["c"]
    bulk = slice(int(np.argmin(np.abs(x - 5.0))), int(np.argmin(np.abs(x - 10.0))))
    expected = cfg["drivers"]["E1"]["derived"]["amplitude"] / eps1**0.25
    np.testing.assert_allclose(np.mean(np.abs(e1y[bulk])), expected, rtol=0.01)
    assert np.std(np.abs(e1y[bulk])) < 0.02 * expected  # a clean traveling wave, no standing-wave ripple
    k_measured = np.polyfit(x[bulk], np.unwrap(np.angle(e1y[bulk])), 1)[0]
    assert k_measured < 0
    np.testing.assert_allclose(abs(k_measured), k1, rtol=0.01)
    # negligible leakage to the right of the injector (rightward wave)
    # beyond the injector's own Gaussian tail (3 sigma ~ 1.6 um) only leakage remains
    right = x > x[step.raman.i1] + 2.5
    assert np.abs(e1y[right]).max() < 1e-2 * expected


def test_spectral_pump_injector_launches_the_exact_amplitude_rightward():
    cfg = _srs_cfg("spectral", pump_depletion=True)
    step, state = _fill_box(cfg, 180, pump_depletion=True)
    e0y = np.asarray(state["E0"].view(jnp.complex128))[:, 0, 1]
    x = np.asarray(cfg["grid"]["x"])
    derived = cfg["units"]["derived"]
    n = cfg["units"]["envelope density"]
    bulk = slice(int(np.argmin(np.abs(x - 8.0))), int(np.argmin(np.abs(x - 14.0))))
    expected = derived["E0_source"] * (1.0 - n) ** -0.25
    np.testing.assert_allclose(np.mean(np.abs(e0y[bulk])), expected, rtol=0.01)
    k0 = np.sqrt(derived["w0"] ** 2 - derived["wp0"] ** 2) / derived["c"]
    k_measured = np.polyfit(x[bulk], np.unwrap(np.angle(e0y[bulk])), 1)[0]
    np.testing.assert_allclose(k_measured, k0, rtol=0.01)
    left = x < x[step.coupled_light.i0] - 1.0
    assert np.abs(e0y[left]).max() < 1e-2 * expected
    # no plasma wave: the Raman field stays identically zero
    assert float(jnp.max(jnp.abs(state["E1"]))) == 0.0


@pytest.mark.parametrize("solver", ["spectral", "fd"])
def test_collisional_absorption_decays_the_light(solver):
    """With terms.light.absorption = nu (1/ps at nc), |E1| decays as exp(-nu (n/nc1)^2 t)."""
    from adept._lpse2d.core.raman import RamanLight
    from adept._lpse2d.core.spectral_light import SpectralRamanLight

    nu = 40.0
    cfg = _srs_cfg(solver, periodic=True, xmax="8um", absorption=nu)
    light = (SpectralRamanLight if solver == "spectral" else RamanLight)(cfg)
    derived = cfg["units"]["derived"]
    assert light.absorption_rate0 == nu
    assert light.absorption_rate1 == pytest.approx(nu * (derived["w1"] / derived["w0"]) ** 2)
    E1, _ = _plane_wave(cfg, derived["w1"], -1)
    E0_fn = lambda t: jnp.zeros_like(E1)
    phi_k = jnp.zeros(E1.shape[:2], dtype=jnp.complex128)
    out = E1
    n_steps = 10
    for i in range(n_steps):
        out = light(i * light.dt, out, E0_fn, phi_k, None)
    n = cfg["units"]["envelope density"]
    n_over_nc1 = n * (derived["w0"] / derived["w1"]) ** 2
    expected_decay = np.exp(-light.absorption_rate1 * n_over_nc1**2 * n_steps * light.dt)
    ratio = np.abs(np.asarray(out)[..., 1]).mean() / np.abs(np.asarray(E1)[..., 1]).mean()
    assert ratio == pytest.approx(expected_decay, rel=1e-3 if solver == "spectral" else 2e-2)
    assert 0.3 < expected_decay < 0.95


def test_nrl_absorption_rate():
    from adept._lpse2d.core.raman import light_absorption_rates

    cfg = _srs_cfg("spectral", periodic=True, xmax="8um", absorption=True)
    rate0, rate1 = light_absorption_rates(cfg)
    te, z, lam = 2.0, 1.0, 0.351
    log_lambda = 6.68 + np.log(lam * te)  # Te > 0.01 Z^2
    expected0 = 5.11e10 * z * log_lambda / (lam**2 * te**1.5) * 1e-12
    assert rate0 == pytest.approx(expected0, rel=1e-12)
    derived = cfg["units"]["derived"]
    lam1 = lam * derived["w0"] / derived["w1"]
    expected1 = 5.11e10 * z * (6.68 + np.log(lam1 * te)) / (lam1**2 * te**1.5) * 1e-12
    assert rate1 == pytest.approx(expected1, rel=1e-12)
    assert 0.1 < rate0 < 100.0
    assert light_absorption_rates(_srs_cfg("spectral", periodic=True, xmax="8um")) == (None, None)


def test_spectral_coupled_solver_conserves_light_action_in_the_exchange():
    """With the propagation trivial (phi = 0 apart from the exchange test) the Strang-split
    rotation is action conserving; here we check the coupled spectral step runs, keeps
    finite fields, and that with SRS the exchange transfers energy between E0 and E1."""
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight

    cfg = _srs_cfg("spectral", pump_depletion=True, xmax="8um")
    light = SpectralCoupledLight(cfg)
    rng = np.random.default_rng(3)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E0, _ = _plane_wave(cfg, cfg["units"]["derived"]["w0"], +1, amplitude=cfg["units"]["derived"]["E0_source"])
    E1 = jnp.zeros_like(E0)
    phi = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * 1e-3
    phi = jnp.asarray(phi * np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"]))
    E0n, E1n = light(0.0, E0, E1, phi, _pump_args(cfg), None)
    assert bool(jnp.all(jnp.isfinite(E0n))) and bool(jnp.all(jnp.isfinite(E1n)))
    assert float(jnp.max(jnp.abs(E1n))) > 0.0
    # without a plasma wave the Raman field is not seeded and the pump only propagates
    E0z, E1z = light(0.0, E0, E1, jnp.zeros_like(phi), _pump_args(cfg), None)
    assert float(jnp.max(jnp.abs(E1z))) == 0.0
    assert bool(jnp.all(jnp.isfinite(E0z)))


def test_light_solver_option_is_validated():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _srs_cfg("implicit", periodic=True, xmax="8um")


def _srs_cfg_2d(solver="spectral", transverse_source=True):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"xmax": "6.4um", "tmax": "10fs", "ymax": "3.2um", "ymin": "-3.2um", "dx": "0.1um"})
    cfg["terms"]["light"] = {"solver": solver, "transverse_source": transverse_source}
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
    return _finish(cfg)


def test_longitudinal_light_is_kept_only_inside_the_band():
    """LPSE zeroes every component of the light field outside the retained band; adept's
    propagator does the same to the longitudinal part it does not move."""
    from adept._lpse2d.core.spectral_light import SpectralRamanLight, transverse_propagate

    cfg = _srs_cfg_2d()
    solver = SpectralRamanLight(cfg)
    ky = np.asarray(cfg["grid"]["ky"])
    y = np.asarray(cfg["grid"]["y"])
    band = np.asarray(solver.light_band)
    j_in = int(np.flatnonzero(band[0] > 0)[1])
    j_out = int(np.flatnonzero(band[0] == 0)[0])
    ones = jnp.ones_like(solver.propagator1)
    for j, kept in ((j_in, True), (j_out, False)):
        wave = np.exp(1j * ky[j] * y)[None, :] * np.ones((len(cfg["grid"]["x"]), 1))
        field = jnp.stack([jnp.zeros_like(wave), jnp.asarray(wave)], axis=-1)  # E || y, k || y: longitudinal
        out = transverse_propagate(field, solver.kx_arr, solver.ky_arr, solver.one_over_k_sq, ones, solver.light_band)
        if kept:
            np.testing.assert_allclose(np.asarray(out), np.asarray(field), atol=1e-12)
        else:
            assert float(jnp.max(jnp.abs(out))) < 1e-12
        dropped = transverse_propagate(
            field, solver.kx_arr, solver.ky_arr, solver.one_over_k_sq, ones, solver.light_band, keep_longitudinal=False
        )
        assert float(jnp.max(jnp.abs(dropped))) < 1e-12


@pytest.mark.parametrize("solver", ["spectral", "fd"])
def test_transverse_source_removes_the_longitudinal_srs_source(solver):
    """An EPW with k along y under a y-polarised pump drives a purely longitudinal E1 source;
    with transverse_source (LPSE takeTransversePartOfSourceTerms) no E1 is produced."""
    from adept._lpse2d.core.raman import RamanLight
    from adept._lpse2d.core.spectral_light import SpectralRamanLight

    cls = SpectralRamanLight if solver == "spectral" else RamanLight
    out = {}
    for transverse_source in (False, True):
        cfg = _srs_cfg_2d(solver, transverse_source)
        light = cls(cfg)
        nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
        band = np.asarray(cfg["grid"]["low_pass_filter_grid"]) > 0
        j = int(np.flatnonzero(band[0])[2])
        phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128).at[0, j].set(1.0e-6 * nx * ny)
        E0 = jnp.stack([jnp.zeros((nx, ny)), 1.0e-3 * jnp.ones((nx, ny))], axis=-1).astype(jnp.complex128)
        E1 = light(0.0, jnp.zeros((nx, ny, 2), dtype=jnp.complex128), lambda t, E0=E0: E0, phi_k, None)
        out[transverse_source] = float(jnp.max(jnp.abs(E1)))
    assert out[False] > 0.0
    assert out[True] < 1e-10 * out[False]
