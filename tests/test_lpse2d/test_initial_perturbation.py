"""``initial_perturbation`` (LPSE ``initialPerturbation``, plan 2 L.5): a plane wave written
into the EPW potential or a light-field component at t = 0."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _module(cfg):
    """A BaseLPSE2D with the lifecycle run up to the initial state, no MLflow."""
    from adept._lpse2d.modules.base import BaseLPSE2D

    module = BaseLPSE2D(deepcopy(cfg))
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    return module


def _cfg(ip, *, solver="separate"):
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    del cfg["drivers"]["E2"]
    cfg["drivers"]["E0"] = {
        "params": {"phases": {"seed": 1}},
        "shape": "uniform",
        "delta_omega_max": 0.0,
        "num_colors": 1,
        "envelope": {
            "tw": "40ps",
            "tr": "0.1ps",
            "tc": "20ps",
            "xr": "0.2um",
            "xw": "1000um",
            "xc": "50um",
            "yr": "0.2um",
            "yw": "1000um",
            "yc": "50um",
        },
    }
    cfg["density"]["val"] = 0.25
    cfg["grid"].update(
        {"dx": "0.04um", "xmax": "2.56um", "ymax": "0.32um", "ymin": "-0.32um", "dt": "1fs", "tmax": "0.02ps"}
    )
    cfg["terms"]["epw"]["damping"]["landau"] = True
    cfg["terms"]["epw"]["solver"] = solver
    if solver == "combined":
        cfg["terms"]["epw"]["source"].update({"tpd": False, "srs": False})
        cfg["terms"]["light"] = {"solver": "spectral"}
    cfg["initial_perturbation"] = ip
    return cfg


def test_plane_wave_in_the_potential_has_the_requested_k_amplitude_and_units():
    from adept._lpse2d.helpers import initial_perturbation_field

    module = _module(_cfg({"field": "epw", "amplitude": 2.0e-6, "wavelength": "0.32um", "direction": [1.0, 0.0]}))
    cfg = module.cfg
    derived = cfg["units"]["derived"]
    wave = initial_perturbation_field(cfg)
    x = np.asarray(cfg["grid"]["x"])
    k = 2.0 * np.pi / 0.32
    # amplitude: e phi / (m_e c^2) = 2e-6 -> phi = 2e-6 / (e_norm x_norm) in this code's units
    expected_amp = 2.0e-6 / (derived["e_norm"] * derived["x_norm"])
    np.testing.assert_allclose(np.abs(wave), expected_amp, rtol=1e-12)
    # the phase advances as exp(i k (x - x_centre))
    phase = np.angle(wave[:, 0] * np.conj(wave[0, 0]))
    np.testing.assert_allclose(np.unwrap(phase), k * (x - x[0]), atol=1e-9)
    # the state holds the k-space potential of it: one mode at kx = +k (0.32 um = 8 cells)
    phi_k = np.asarray(module.state["epw"]).view(np.complex128)
    kx = np.asarray(cfg["grid"]["kx"])
    peak = np.unravel_index(np.argmax(np.abs(phi_k)), phi_k.shape)
    assert kx[peak[0]] == pytest.approx(k, rel=1e-9) and peak[1] == 0
    assert np.abs(phi_k[peak]) == pytest.approx(expected_amp * phi_k.size, rel=1e-9)
    assert np.sum(np.abs(phi_k) > 1e-9 * np.abs(phi_k[peak])) == 1


def test_seeded_mode_free_evolution_follows_the_analytic_landau_rate():
    """The seeded mode has energy sum_k k^2 |phi_k|^2; with no sources it decays at the
    solver's own analytic Landau rate for that k (exact in the spectral split step)."""
    from adept._lpse2d.core.epw import analytic_landau_rate
    from adept._lpse2d.core.vector_field import SplitStep

    module = _module(_cfg({"field": "epw", "amplitude": 1.0e-6, "wavelength": "0.16um"}))
    cfg = module.cfg
    step = SplitStep(cfg)
    state = {k: jnp.asarray(v) for k, v in module.state.items()}
    ny = cfg["grid"]["ny"]
    pump = {**cfg["drivers"]["E0"]["derived"], "delta_omega": jnp.zeros(1), "phases": jnp.zeros((1, ny))}
    args = {"drivers": {"E0": {**pump, "intensities": jnp.zeros((1, ny))}}}
    w0 = float(step.epw.energy(state["epw"].view(jnp.complex128)))
    n = 20
    for i in range(n):
        state = step(i * cfg["grid"]["dt"], dict(state), args)
    w = float(step.epw.energy(state["epw"].view(jnp.complex128)))
    kx = np.asarray(cfg["grid"]["kx"])
    ik = int(np.argmin(np.abs(kx - 2.0 * np.pi / 0.16)))
    gamma = float(np.asarray(analytic_landau_rate(cfg))[ik, 0])
    assert gamma > 0.0
    np.testing.assert_allclose(w / w0, np.exp(-2.0 * gamma * n * cfg["grid"]["dt"]), rtol=1e-8)


def test_light_component_and_combined_field_seeding():
    from adept._lpse2d.helpers import initial_perturbation_field

    ip = {"field": "E1", "component": "z", "amplitude": 1.0e-3, "wavelength": "0.5um", "direction": [1.0, 0.0]}
    module = _module(_cfg(ip))
    e1 = np.asarray(module.state["E1"]).view(np.complex128)
    e0 = np.asarray(module.state["E0"]).view(np.complex128)
    assert e1.shape[-1] == 3
    np.testing.assert_allclose(np.abs(e1[..., 2]), 1.0e-3 / module.cfg["units"]["derived"]["e_norm"], rtol=1e-12)
    assert np.all(e1[..., :2] == 0.0) and np.all(e0 == 0.0)
    assert np.all(np.asarray(module.state["epw"]) == 0.0)

    # combined solver: the potential seed also sets the longitudinal part of E1 = -grad phi
    module = _module(_cfg({"field": "epw", "amplitude": 1.0e-6, "wavelength": "0.32um"}, solver="combined"))
    cfg = module.cfg
    wave = initial_perturbation_field(cfg)
    e1 = np.asarray(module.state["E1"]).view(np.complex128)
    k = 2.0 * np.pi / 0.32
    np.testing.assert_allclose(e1[..., 0], -1j * k * wave, rtol=1e-9, atol=1e-9 * np.abs(wave).max() * k)
    assert np.abs(e1[..., 1]).max() < 1e-12 * np.abs(e1[..., 0]).max() and np.all(e1[..., 2] == 0.0)


def test_envelope_and_oblique_direction():
    from adept._lpse2d.helpers import initial_perturbation_field

    ip = {
        "field": "epw",
        "amplitude": 1.0,
        "wavelength": "0.4um",
        "direction": [1.0, 1.0],
        "envelope_size": ["1.0um", "0um"],
        "envelope_offset": ["0.3um", "0um"],
        "envelope_sg_order": 2.0,
    }
    module = _module(_cfg(ip))
    cfg = module.cfg
    wave = initial_perturbation_field(cfg)
    x = np.asarray(cfg["grid"]["x"])
    y = np.asarray(cfg["grid"]["y"])
    xc, yc = x - 0.5 * (x[0] + x[-1]), y - 0.5 * (y[0] + y[-1])
    k = 2.0 * np.pi / 0.4 / np.sqrt(2.0)
    expected = (
        np.exp(-(((xc[:, None] - 0.3) / 0.5) ** 2))
        * np.exp(1j * k * (xc[:, None] + yc[None, :]))
        / (cfg["units"]["derived"]["e_norm"] * cfg["units"]["derived"]["x_norm"])
    )
    np.testing.assert_allclose(wave, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_translator_maps_initial_perturbation():
    from adept._lpse2d.lpse_deck import translate_parms

    base = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "laser.wavelength": "0.351",
        "lw.envelopeDensity": "0.25",
        "simulation.time.end": "1",
        "initialPerturbation.enable": "true",
        "initialPerturbation.field": "e1_z",
        "initialPerturbation.amplitude": "1e-4",
        "initialPerturbation.wavelength": "0.7",
        "initialPerturbation.direction": "0.6 0.8 0",
        "initialPerturbation.envelopeSize": "4 0 0",
        "initialPerturbation.envelopeOffset": "1 0 0",
    }
    cfg, report = translate_parms(base, run="ip")
    ip = cfg["initial_perturbation"]
    assert ip["field"] == "E1" and ip["component"] == "z" and ip["amplitude"] == 1.0e-4
    assert ip["wavelength"] == "0.7um" and ip["direction"] == [0.6, 0.8]
    assert ip["envelope_size"] == ["4.0um", "0.0um"] and ip["envelope_offset"] == ["1.0um", "0.0um"]
    assert ip["envelope_sg_order"] == 4.0
    assert not any("initialPerturbation" in u for u in report["unsupported"])
    cfg, _ = translate_parms({**base, "initialPerturbation.field": "pots"}, run="ip")
    assert cfg["initial_perturbation"]["field"] == "epw"
