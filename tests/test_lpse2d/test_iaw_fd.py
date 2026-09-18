"""The finite-difference ion-acoustic solver with flow profiles (terms.iaw.solver: fd; LPSE
iaw.solver = fd, iaw.velocityProfile.*) -- plan 2 I.1 / I.2."""

from copy import deepcopy

import jax
import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._lpse2d.parity import deck_path


def _cfg(solver, ny_box="0.02um", flow=None, super_samples=2, landau=0.0):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {"ymax": ny_box, "ymin": f"-{ny_box}", "xmax": "12.8um", "dx": "0.05um", "tmax": "10fs", "dt": "2fs"}
    )
    cfg["terms"]["iaw"] = {
        "active": True,
        "solver": solver,
        "damping": {"landau": landau, "collisions": 0.0},
        "super_samples": super_samples,
    }
    if flow is not None:
        cfg["terms"]["iaw"]["flow"] = flow
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _travelling_wave(cfg, modes):
    """``n, w`` of a +k travelling acoustic wave with ``modes = (mx, my)`` grid modes."""
    x = np.asarray(cfg["grid"]["x"])
    y = np.asarray(cfg["grid"]["y"])
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = 2.0 * np.pi * modes[0] / (nx * cfg["grid"]["dx"])
    ky = 2.0 * np.pi * modes[1] / (ny * cfg["grid"]["dy"]) if ny > 1 else 0.0
    k = np.hypot(kx, ky)
    cs = cfg["units"]["derived"]["cs"]
    phase = kx * x[:, None] + ky * y[None, :]
    return 1e-3 * np.cos(phase), -cs * k * 1e-3 * np.sin(phase), (kx, ky, k)


def _run(cfg, n0, w0, steps):
    from adept._lpse2d.core.iaw import IonAcousticWave
    from adept._lpse2d.core.iaw_fd import FD_STATE_KEYS, upsample

    iaw = IonAcousticWave(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    y = {
        "epw": jnp.zeros((nx, ny), complex),
        "E0": jnp.zeros((nx, ny, 3), complex),
        "E1": jnp.zeros((nx, ny, 3), complex),
        "iaw_density": jnp.asarray(n0),
        "iaw_velocity_divergence": jnp.asarray(w0),
    }
    if iaw.solver == "fd" and iaw.fd.s > 1:
        y[FD_STATE_KEYS[0]] = upsample(jnp.asarray(n0), iaw.fd.s)
        y[FD_STATE_KEYS[1]] = upsample(jnp.asarray(w0), iaw.fd.s)
    step = jax.jit(lambda y, t: iaw(y, t))
    modes = []
    for i in range(steps):
        y = step(y, i * iaw.dt)
        modes.append(np.fft.fft2(np.asarray(y["iaw_density"])))
    return np.array(modes), iaw


def _frequency(modes, index, dt):
    m = modes[:, index[0], index[1]]
    t = np.arange(1, m.size + 1) * dt
    return -np.polyfit(t, np.unwrap(np.angle(m)), 1)[0], abs(m[-1]) / abs(m[0])


def test_ppm_advects_a_bump_exactly_and_conserves_mass():
    from adept._lpse2d.core.iaw_fd import ppm_sweep

    f = np.exp(-(((np.arange(256) - 100.0) / 8.0) ** 2))[:, None]
    g = jnp.asarray(f)
    lam = 0.4 * jnp.ones_like(g)
    for _ in range(250):
        g = ppm_sweep(g, lam, 0)
    g = np.asarray(g)
    np.testing.assert_allclose(g.sum(), f.sum(), rtol=1e-12)
    assert int(np.argmax(g[:, 0])) == 200  # 250 x 0.4 cells
    assert g.max() > 0.95 * f.max()


def test_resampling_is_exact_on_band_limited_fields():
    from adept._lpse2d.core.iaw_fd import downsample, upsample

    a = np.random.default_rng(0).normal(size=(64, 16))
    a_k = np.fft.fft2(a)
    a_k[16:49, :] = 0.0
    a_k[:, 4:13] = 0.0
    a = np.real(np.fft.ifft2(a_k))
    up = np.asarray(upsample(jnp.asarray(a), 2))
    np.testing.assert_allclose(up[::2, ::2], a, atol=1e-13)
    np.testing.assert_allclose(np.asarray(downsample(jnp.asarray(up), 2)), a, atol=1e-13)


@pytest.mark.parametrize(
    "ny_box, flow, modes", [("0.02um", None, (8, 0)), ("0.02um", [0.5, 0.0], (8, 0)), ("1.6um", [0.3, 0.4], (10, 4))]
)
def test_fd_matches_the_spectral_solver_and_the_doppler_shift(ny_box, flow, modes):
    """A travelling acoustic wave keeps its amplitude and runs at ``cs k + k . U`` on both
    solvers (the fd one to 0.3 %, its grid dispersion)."""
    results = {}
    for solver in ("spectral", "fd"):
        cfg = _cfg(solver, ny_box, flow=flow)
        n0, w0, (kx, ky, k) = _travelling_wave(cfg, modes)
        series, iaw = _run(cfg, n0, w0, 300)
        omega, ratio = _frequency(series, modes, iaw.dt)
        cs = cfg["units"]["derived"]["cs"]
        u = np.asarray(flow) * cs if flow else np.zeros(2)
        expected = cs * k + kx * u[0] + ky * u[1]
        results[solver] = omega / expected
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-2)
    np.testing.assert_allclose(results["spectral"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(results["fd"], 1.0, rtol=4e-3)


def test_fd_super_samples_one_and_landau_damping():
    """Without super-sampling the fd fields are the state's own; with the k-space Landau
    damping on w the fd solver follows the exact damped-oscillator propagator (the
    undamped start projects on both damped modes, so the comparison is solver to solver)."""
    results = {}
    for solver in ("spectral", "fd"):
        cfg = _cfg(solver, super_samples=1, landau=0.05)
        n0, w0, _ = _travelling_wave(cfg, (8, 0))
        series, iaw = _run(cfg, n0, w0, 200)
        results[solver] = _frequency(series, (8, 0), iaw.dt)
    np.testing.assert_allclose(results["fd"][0], results["spectral"][0], rtol=5e-3)
    np.testing.assert_allclose(results["fd"][1], results["spectral"][1], rtol=5e-3)
    assert results["fd"][1] < 0.99  # damped


def test_linear_flow_profile_and_the_translator():
    from adept._lpse2d.core.iaw_fd import flow_profile

    flow = {
        "shape": "linear",
        "from_location": ["-4um", "0um"],
        "to_location": ["4um", "0um"],
        "from_mach": -1.0,
        "to_mach": 1.0,
    }
    cfg = _cfg("fd", flow=flow)
    x = np.asarray(cfg["grid"]["x"])
    cs = cfg["units"]["derived"]["cs"]
    u = flow_profile(cfg, x, np.asarray(cfg["grid"]["y"]))
    xc = 0.5 * (x[0] + x[-1])
    mach = u[:, 0, 0] / cs
    np.testing.assert_allclose(np.interp([-6.0, -2.0, 2.0, 6.0], x - xc, mach), [-1.0, -0.5, 0.5, 1.0], atol=0.01)
    assert np.all(u[..., 1] == 0.0)
    # gaussian, decreasing from 2 to 0.5 Mach over 4 um
    cfg["terms"]["iaw"]["flow"] = {
        "shape": "gaussian",
        "from_location": ["0um", "0um"],
        "to_location": ["4um", "0um"],
        "from_mach": 2.0,
        "to_mach": 0.5,
    }
    u = flow_profile(cfg, x, np.asarray(cfg["grid"]["y"]))
    np.testing.assert_allclose(np.interp([0.0, 4.0], x - xc, u[:, 0, 0] / cs), [2.0, 0.5], atol=0.02)
    # a uniform flow is refused by the explicit solver and a profile by the spectral one
    from adept._lpse2d.core.iaw import IonAcousticWave

    with pytest.raises(ValueError, match="fd solver"):
        IonAcousticWave(_cfg("spectral", flow=flow))

    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    if deck_path("test_003") is None:
        pytest.skip("no LPSE decks present")
    cfg, report = translate_parms(parse_parms(deck_path("test_003")), run="test_003")
    iaw = cfg["terms"]["iaw"]
    assert iaw["solver"] == "fd" and iaw["super_samples"] == 2 and iaw["dt_fraction"] == 0.9
    assert iaw["flow"]["shape"] == "linear" and iaw["flow"]["from_mach"] == 1.2 and iaw["flow"]["to_mach"] == 1.6
    assert iaw["flow"]["from_location"] == ["10.0um", "0.0um"] and iaw["flow"]["to_location"] == ["-10.0um", "0.0um"]
    assert not any("iaw.solver = fd" in n for n in report["notes"])


def test_cbet_resonance_localises_at_the_mach_one_layers():
    """Two counter-propagating equal-frequency beams beat statically; on a linear Mach -2 -> +2
    profile the fd IAW solver's response peaks where k . U = -+ cs k, the Mach -+1 layers
    (7.5 and 12.5 um on this box), and stays at the no-flow level at Mach 0 (plan 2 I.2)."""
    from adept import ergoExo

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"].update(
        {
            "ymax": "0.4um",
            "ymin": "-0.4um",
            "xmax": "20um",
            "tmax": "1.5ps",
            "dx": "50nm",
            "dt": "1fs",
            "boundary_width": "2um",
        }
    )
    cfg["terms"]["light"] = {"pump_depletion": True, "solver": "spectral"}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"].update({"noise": False, "srs": False, "tpd": True})  # the EPW stays at zero
    cfg["terms"]["iaw"] = {
        "active": True,
        "solver": "fd",
        "super_samples": 2,
        "damping": {"landau": 0.1, "collisions": 0.0},
        "max_density_perturbation": 0.5,
        "flow": {
            "shape": "linear",
            "from_location": ["-5um", "0um"],
            "to_location": ["5um", "0um"],
            "from_mach": -2.0,
            "to_mach": 2.0,
        },
    }
    cfg["drivers"]["E0"]["beams"] = [{"intensity": 0.5, "angle": 0.0}, {"intensity": 0.5, "angle": 180.0}]
    cfg["drivers"]["E0"]["delta_omega_max"] = 0.0
    cfg["units"]["laser intensity"] = "2.0e+15W/cm^2"
    cfg["save"]["fields"]["t"].update({"tmax": "1.5ps", "dt": "0.5ps"})
    cfg["mlflow"]["run"] = "cbet-localisation"
    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, _, _ = exo(modules)
    result = sol["solver result"]
    x = np.asarray(exo.adept_module.cfg["grid"]["x"])
    n = np.asarray(result.ys["fields"]["iaw_density"])
    assert np.all(np.isfinite(n))
    env = np.sqrt(np.mean(n[-1] ** 2, axis=1))

    def peak(a, b):
        m = (x > a) & (x < b)
        return env[m].max(), x[m][np.argmax(env[m])]

    lower, x_lower = peak(6.0, 9.0)
    mid, _ = peak(9.5, 10.5)
    upper, x_upper = peak(11.0, 14.0)
    assert abs(x_lower - 7.5) < 0.6 and abs(x_upper - 12.5) < 0.6
    assert lower > 2.5 * mid and upper > 2.5 * mid
    assert mid < 0.01  # the uniform standing-wave drive alone (no-flow level ~5e-3)
