"""LPSE-parity tests for the ion-acoustic solver: the spectral (exact damped-oscillator)
propagator, uniform background flow, the full Krall-Trivelpiece ion Landau rate, the
fluctuation-dissipation IAW noise source and the IAW stride."""

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _make_cfg(*, solver="spectral", landau=0.0, collisions=0.0, dt="2fs", stride=1, flow=None, **iaw_extra):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["density"] = {"basis": "uniform", "val": 0.25}
    cfg["grid"].update(
        {
            "boundary_width": "0.2um",
            "dt": dt,
            "dx": "0.1um",
            "xmax": "6.4um",
            "tmax": "20fs",
            "ymax": "0.05um",
            "ymin": "-0.05um",
            "light_substeps": 1,
            "low_pass_filter": 1.0,
        }
    )
    cfg["terms"]["zero_mask"] = True
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
    cfg["terms"]["epw"]["source"] = {"noise": False, "tpd": False, "srs": False}
    cfg["terms"]["epw"]["density_gradient"] = False
    cfg["terms"]["iaw"] = {
        "active": True,
        "solver": solver,
        "boundary": {"x": "periodic", "y": "periodic"},
        "damping": {"collisions": collisions, "landau": landau},
        "stride": stride,
        "flow": flow,
        **iaw_extra,
    }

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _zero_fields(cfg):
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    return {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
    }


def _single_mode_state(cfg, ik, amplitude=0.01):
    x = np.asarray(cfg["grid"]["x"])
    k = float(np.asarray(cfg["grid"]["kx"])[ik])
    density = jnp.asarray(amplitude * np.cos(k * (x - x[0])))[:, None]
    return k, {**_zero_fields(cfg), "iaw_density": density, "iaw_velocity_divergence": jnp.zeros_like(density)}


def _run(solver, state, n_steps):
    for i in range(n_steps):
        state = solver(state, i * solver.dt)
    return state


def test_spectral_propagator_matches_analytic_damped_oscillator():
    """n = cos(kx), w = 0: the exact solution is n(t) = e^{-g t}[cos(b t) + g/b sin(b t)] n0,
    w = -dn/dt, with g = landau cs k and b = sqrt(cs^2 k^2 - g^2). The propagator reproduces
    it to round-off after many steps with cs k dt ~ 1 (far beyond the explicit limit)."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    cfg = _make_cfg(landau=0.2, dt="1.5ps")
    solver = IonAcousticWave(cfg)
    k, state = _single_mode_state(cfg, 4)
    cs = solver.cs
    g = 0.2 * cs * k
    b = np.sqrt((cs * k) ** 2 - g**2)
    assert cs * k * solver.dt > 0.5
    n_steps = 25
    out = _run(solver, state, n_steps)
    t = n_steps * solver.dt
    x = np.asarray(cfg["grid"]["x"])
    n_exact = 0.01 * np.exp(-g * t) * (np.cos(b * t) + g / b * np.sin(b * t)) * np.cos(k * (x - x[0]))
    dn_dt = (
        0.01
        * np.exp(-g * t)
        * (-g * (np.cos(b * t) + g / b * np.sin(b * t)) + (-b * np.sin(b * t) + g * np.cos(b * t)))
        * np.cos(k * (x - x[0]))
    )
    np.testing.assert_allclose(np.asarray(out["iaw_density"])[:, 0], n_exact, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.asarray(out["iaw_velocity_divergence"])[:, 0], -dn_dt, rtol=1e-10, atol=1e-12)


def test_spectral_propagator_is_stable_at_any_dt():
    from adept._lpse2d.core.iaw import IonAcousticWave

    cfg = _make_cfg(landau=0.05, dt="10ps")
    solver = IonAcousticWave(cfg)
    k_max = float(np.abs(np.asarray(cfg["grid"]["kx"])).max())
    assert solver.cs * k_max * solver.dt > 20.0  # omega_max dt far beyond the explicit limit of 2
    _, state = _single_mode_state(cfg, 20)
    out = _run(solver, state, 200)
    n = np.asarray(out["iaw_density"])
    assert np.all(np.isfinite(n)) and np.abs(n).max() <= 0.01 * (1.0 + 1e-9)


def test_explicit_solver_refuses_unstable_dt_but_spectral_accepts_it():
    with pytest.raises(ValueError, match="unstable"):
        _make_cfg(solver="explicit", dt="10ps")
    cfg = _make_cfg(solver="spectral", dt="10ps")
    assert cfg["terms"]["iaw"]["solver"] == "spectral"


def test_explicit_and_spectral_agree_for_small_dt_and_long_wavelength():
    """Undamped, undriven: the explicit kick/drift step converges to the exact propagator as
    dt -> 0; the FD Laplacian differs from k^2 by (k dx)^2/12 (0.1% at k = 2 pi / 6.4 um)."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    cfg_e = _make_cfg(solver="explicit", dt="20fs")
    cfg_s = _make_cfg(solver="spectral", dt="20fs")
    explicit, spectral = IonAcousticWave(cfg_e), IonAcousticWave(cfg_s)
    k, state = _single_mode_state(cfg_e, 1)
    n_steps = 400
    out_e = _run(explicit, state, n_steps)
    out_s = _run(spectral, state, n_steps)
    assert spectral.cs * k * n_steps * spectral.dt > 1.0  # a good fraction of a period
    np.testing.assert_allclose(
        np.asarray(out_e["iaw_density"]), np.asarray(out_s["iaw_density"]), rtol=0, atol=3e-3 * 0.01
    )


def test_flow_doppler_shifts_the_mode_phase():
    """With a uniform flow V0 along x, the exact propagator multiplies every mode by
    e^{-i k V0 dt}: the +k and -k components of cos(kx) pick up opposite phases."""
    from adept._lpse2d.core.iaw import IonAcousticWave

    mach = 0.3
    cfg0 = _make_cfg()
    cfgv = _make_cfg(flow=[mach, 0.0])
    still, moving = IonAcousticWave(cfg0), IonAcousticWave(cfgv)
    ik = 3
    k, state = _single_mode_state(cfg0, ik)
    n0 = np.fft.fft(np.asarray(still(state, 0.0)["iaw_density"])[:, 0])
    nv = np.fft.fft(np.asarray(moving(state, 0.0)["iaw_density"])[:, 0])
    v0 = mach * still.cs
    expected_phase = np.exp(-1j * k * v0 * still.dt)
    assert nv[ik] / n0[ik] == pytest.approx(expected_phase, rel=1e-10)
    assert nv[-ik] / n0[-ik] == pytest.approx(np.conj(expected_phase), rel=1e-10)
    with pytest.raises(ValueError, match="flow"):
        _make_cfg(solver="explicit", flow=[0.1, 0.0])


def test_full_ion_landau_rate_matches_krall_trivelpiece_formula():
    from astropy.units import Quantity as _Q

    from adept._lpse2d.core.iaw import ion_landau_rate

    cfg = _make_cfg(landau=0.1)
    derived = cfg["units"]["derived"]
    kx = np.asarray(cfg["grid"]["kx"])
    k_sq = kx[:, None] ** 2 + np.asarray(cfg["grid"]["ky"])[None, :] ** 2
    simplified = ion_landau_rate(cfg, k_sq)
    np.testing.assert_allclose(simplified, 0.1 * derived["cs"] * np.sqrt(k_sq))

    cfg["terms"]["iaw"]["damping"]["landau_form"] = "full"
    full = ion_landau_rate(cfg, k_sq)
    te = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
    ti = _Q(cfg["units"]["reference ion temperature"]).to("keV").value
    z = cfg["units"]["ionization state"]
    eta = 1.0 + 3.0 * ti / (z * te)
    big_m = derived["mi"] / derived["me"] / (eta * z)
    big_l = 1.0 + k_sq * derived["vte_sq"] / derived["wp0"] ** 2
    w_r = derived["cs"] * np.sqrt(k_sq) / np.sqrt(big_l)
    a = 3.0 / (eta - 1.0)
    w_i = w_r * np.sqrt(np.pi / 8.0) / big_l**1.5 * (a**1.5 * np.exp(-0.5 * a / big_l) + np.sqrt(1.0 / (eta * big_m)))
    np.testing.assert_allclose(full, 0.5 * w_i, rtol=1e-12)
    # Z = 6, Te = 2 Ti: eta = 1.25, a = 12 -> ion Landau dominated, gamma/(cs k) of order 0.1
    band = k_sq > 0
    ratio = full[band] / (derived["cs"] * np.sqrt(k_sq[band]))
    assert np.all((ratio > 0.01) & (ratio < 1.0))


def test_iaw_noise_kick_has_the_lpse_amplitude():
    from adept._lpse2d.core.iaw import IonAcousticWave

    amplitude = 3.0e-7
    cfg = _make_cfg(landau=0.1, collisions=0.02, noise=True, noise_amplitude=amplitude, noise_seed=9)
    solver = IonAcousticWave(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {**_zero_fields(cfg), "iaw_density": jnp.zeros((nx, ny)), "iaw_velocity_divergence": jnp.zeros((nx, ny))}
    out = solver(state, 0.0)
    w_k = np.fft.fft2(np.asarray(out["iaw_velocity_divergence"]))
    gamma = np.asarray(solver.landau_rate) + 0.02
    expected = amplitude * nx * ny * np.sqrt(np.expm1(2.0 * solver.dt * gamma))
    band = np.asarray(solver.filter) > 0
    # the noise is complex in k-space; taking the real part in x-space keeps the
    # Hermitian half, i.e. the +k/-k pair of each real mode (LPSE's own comment)
    assert np.abs(w_k[band]).max() > 0.0
    assert np.abs(w_k[band]).max() <= expected[band].max() * (1.0 + 1e-9)
    assert np.median(np.abs(w_k[band]) / expected[band]) == pytest.approx(0.7, abs=0.15)
    with pytest.raises(ValueError, match="noise"):
        IonAcousticWave(_make_cfg(landau=0.0, collisions=0.0, noise=True))


def test_iaw_stride_advances_every_nth_epw_step():
    from adept._lpse2d.core.vector_field import SplitStep

    cfg = _make_cfg(stride=3, dt="0.5ps")
    step = SplitStep(cfg)
    assert step.iaw.stride == 3 and step.iaw.dt == pytest.approx(3.0 * cfg["grid"]["dt"])
    k, state = _single_mode_state(cfg, 4)
    # with stride > 1 the module's state carries the IAW step's start density (LPSE Nelf_old)
    state["iaw_density_old"] = state["iaw_density"]
    packed = {key: value.view(jnp.float64) for key, value in state.items()}
    dt = cfg["grid"]["dt"]
    same = step(jnp.asarray(dt), dict(packed), {"drivers": {}})
    np.testing.assert_array_equal(np.asarray(same["iaw_density"]), np.asarray(packed["iaw_density"]))
    advanced = step(jnp.asarray(3.0 * dt), dict(packed), {"drivers": {}})
    direct = step.iaw(state, 3.0 * dt)
    np.testing.assert_allclose(np.asarray(advanced["iaw_density"]), np.asarray(direct["iaw_density"]), rtol=1e-12)
    change = np.abs(np.asarray(advanced["iaw_density"]) - np.asarray(packed["iaw_density"])).max()
    assert change > 1e-4  # cs k dt_iaw ~ 1: the mode has rotated appreciably
    with pytest.raises(ValueError, match="stride"):
        _make_cfg(solver="explicit", stride=2)
