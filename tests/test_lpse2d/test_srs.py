"""Tests for the SRS (stimulated Raman scattering) path of the envelope-2d solver.

These are ports of the physics exercised by the `srs_1D` case of lpse-matlab
(m201805_matlabLpse_v11.m), run as small quasi-1D 2D boxes.
"""

import numpy as np
import pytest
import yaml


def _load_cfg():
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)
    return exo, sol


def _run_with_ppo(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)
    return exo, sol, ppo


def _predicted_gamma0(cfg_units, cfg_grid):
    """
    Homogeneous backward-SRS growth rate gamma0 = k vos/4 * wpe / sqrt(w_ek * w_s)
    evaluated at the phase-matched EPW wavenumber k = k0 + k1.
    """
    c = cfg_units["derived"]["c"]
    w0 = cfg_units["derived"]["w0"]
    wp0 = cfg_units["derived"]["wp0"]
    vte_sq = cfg_units["derived"]["vte_sq"]
    e = cfg_units["derived"]["e"]
    me = cfg_units["derived"]["me"]
    E0_source = cfg_units["derived"]["E0_source"]
    n = cfg_units["envelope density"]  # uniform box at the envelope density

    # pump wavenumber as launched by the solver (snapped to the FFT grid)
    dk = 2.0 * np.pi / (cfg_grid["nx"] * cfg_grid["dx"])
    k0 = np.sqrt(w0**2 - wp0**2) / c
    k0 = np.round(k0 / dk) * dk

    # fixed point for the phase-matched EPW wavenumber
    k = k0
    for _ in range(20):
        w_ek = wp0 + 1.5 * k**2 * vte_sq / wp0
        w_s = w0 - w_ek
        k1 = np.sqrt(w_s**2 - wp0**2) / c
        k = k0 + k1

    # local pump field including the density swelling factor
    E0_local = E0_source * (1.0 - n) ** -0.25
    vos = e * E0_local / (me * w0)

    gamma0 = k * vos / 4.0 * wp0 / np.sqrt(w_ek * w_s)
    return gamma0, k1


@pytest.mark.parametrize("ny", ["2d", "1d"])
def test_srs_growth_rate(ny):
    """Noise-seeded homogeneous SRS: the EPW energy should grow at ~2*gamma0.

    The "1d" variant shrinks the y extent below one cell so the box collapses to ny=1,
    which is the cheap true-1D configuration (the MATLAB srs_1D case).
    """
    cfg = _load_cfg()
    if ny == "1d":
        cfg["grid"]["ymax"] = "0.02um"
        cfg["grid"]["ymin"] = "-0.02um"
        cfg["mlflow"]["run"] = "srs-test-1d"
    exo, sol = _run(cfg)

    result = sol["solver result"]
    t = np.array(result.ts["default"])
    e_sq = np.array(result.ys["default"]["e_sq"])

    gamma0, _ = _predicted_gamma0(exo.adept_module.cfg["units"], exo.adept_module.cfg["grid"])

    # fit the log-slope over the last stretch of the run, where exponential growth
    # dominates the noise floor
    fit_window = (t > t[-1] - 0.5) & (t < t[-1] - 0.02)
    slope = np.polyfit(t[fit_window], np.log(e_sq[fit_window]), 1)[0]
    measured_gamma = slope / 2.0  # e_sq ~ |E|^2 grows at 2*gamma

    assert e_sq[-1] > 1e4 * np.min(e_sq[t > 0.1]), "no SRS growth occurred"
    np.testing.assert_allclose(measured_gamma, gamma0, rtol=0.35)


def test_srs_seed_propagation():
    """The Raman seed injector should launch a backward (-x) wave at the local k1."""
    cfg = _load_cfg()

    # seed-only setup: no pump, no noise -- E1 just propagates
    del cfg["drivers"]["E0"]
    cfg["drivers"]["E1"] = {
        "intensity": "1.0e+12W/cm^2",
        "delta_omega": 0.0,
        "turn_on_time": "10fs",
    }
    cfg["grid"]["xmax"] = "20um"
    cfg["grid"]["tmax"] = "0.2ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.2ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.02ps"
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["mlflow"]["run"] = "srs-seed-test"

    exo, sol = _run(cfg)

    result = sol["solver result"]
    e1_raw = np.array(result.ys["fields"]["E1"])
    e1 = e1_raw.view(np.complex64 if e1_raw.dtype == np.float32 else np.complex128)
    e1y_final = e1[-1, :, 0, 1]  # final time, y=first row, y-polarization

    assert np.max(np.abs(e1y_final)) > 0, "seed was not injected"

    dcfg = exo.adept_module.cfg
    x = np.array(dcfg["grid"]["x"])
    derived = dcfg["units"]["derived"]
    wp0, w1, c = derived["wp0"], derived["w1"], derived["c"]
    n = dcfg["units"]["envelope density"]
    k1 = np.sqrt(w1**2 - wp0**2) / c  # seed injected at delta_omega = 0

    # in the bulk (away from the injector near-field and the absorbers) the seed should be
    # a leftward traveling wave exp(-i k1 x)
    bulk = slice(np.argmin(np.abs(x - 6.0)), np.argmin(np.abs(x - 12.0)))
    phase = np.unwrap(np.angle(e1y_final[bulk]))
    k_measured = np.polyfit(x[bulk], phase, 1)[0]
    assert k_measured < 0, f"seed propagates the wrong way: k = {k_measured:.2f} 1/um"
    np.testing.assert_allclose(np.abs(k_measured), k1, rtol=0.05)

    # amplitude calibration of the injector: |E1| = E1_source * sinc(k1 dx) / eps1^(1/4)
    dx = dcfg["grid"]["dx"]
    eps1 = 1.0 - n * (derived["w0"] / w1) ** 2
    expected_amp = dcfg["drivers"]["E1"]["derived"]["amplitude"] * np.sin(k1 * dx) / (k1 * dx) / eps1**0.25
    np.testing.assert_allclose(np.mean(np.abs(e1y_final[bulk])), expected_amp, rtol=0.3)


def _quasi_1d(cfg):
    cfg["grid"]["ymax"] = "0.02um"
    cfg["grid"]["ymin"] = "-0.02um"
    return cfg


def test_pump_injector_calibration():
    """With pump depletion on and no plasma-wave activity, the evolved pump should fill
    the box with the nominal flux and the nominal (swelled) amplitude."""
    cfg = _quasi_1d(_load_cfg())
    cfg["terms"]["light"] = {"pump_depletion": True}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["grid"]["xmax"] = "20um"
    cfg["grid"]["tmax"] = "0.3ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.3ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.05ps"
    cfg["mlflow"]["run"] = "srs-pump-injector-test"

    exo, sol = _run(cfg)
    result = sol["solver result"]
    t = np.array(result.ts["default"])
    steady = t > 0.2  # past the ~0.08 ps fill-in transit

    # the two-point source launches amplitude E_src * sin(k0 dx)/sin(k_grid dx), where
    # k_grid is the FD-dispersion wavenumber -- so the measured physical flux is S^2 of
    # nominal; both probes must agree with that and with each other (no spurious loss)
    dcfg = exo.adept_module.cfg
    derived = dcfg["units"]["derived"]
    n = dcfg["units"]["envelope density"]
    dx = dcfg["grid"]["dx"]
    k0_dx = derived["w0"] / derived["c"] * np.sqrt(1.0 - n) * dx
    kg_dx = np.arccos(1.0 - k0_dx**2 / 2.0)
    S = np.sin(k0_dx) / np.sin(kg_dx)

    incident = np.array(result.ys["default"]["incident_flux"])
    transmitted = np.array(result.ys["default"]["transmitted_flux"])
    np.testing.assert_allclose(np.mean(incident[steady]), S**2, rtol=0.02)
    np.testing.assert_allclose(np.mean(transmitted[steady]), np.mean(incident[steady]), rtol=0.01)

    # bulk amplitude: |E0| = E0_source * S * (1 - n)^(-1/4)
    e0_raw = np.array(result.ys["fields"]["E0"])
    e0 = e0_raw.view(np.complex64 if e0_raw.dtype == np.float32 else np.complex128)
    e0y_final = e0[-1, :, 0, 1]
    x = np.array(dcfg["grid"]["x"])
    bulk = slice(np.argmin(np.abs(x - 8.0)), np.argmin(np.abs(x - 14.0)))
    expected_amp = derived["E0_source"] * S * (1.0 - n) ** -0.25
    np.testing.assert_allclose(np.mean(np.abs(e0y_final[bulk])), expected_amp, rtol=0.03)


def _amplifier_cfg():
    """Seeded Raman-amplifier setup: pump + strong E1 seed, deterministic (no noise)."""
    cfg = _quasi_1d(_load_cfg())
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["drivers"]["E1"] = {
        "intensity": "1.0e+14W/cm^2",
        # seed on the Bohm-Gross-shifted resonance: w_seed = w0 - w_ek(k0+k1) rather
        # than the envelope carrier w1 = w0 - wp0 (n = 0.2, Te = 2 keV => dw1 = -0.0335)
        "delta_omega": -0.0335,
        "turn_on_time": "10fs",
    }
    cfg["grid"]["xmax"] = "40um"
    cfg["grid"]["tmax"] = "0.8ps"
    cfg["save"]["fields"]["t"]["tmax"] = "0.8ps"
    cfg["save"]["fields"]["t"]["dt"] = "0.1ps"
    return cfg


def test_pump_depletion_budget_and_saturation():
    """Raman amplifier with pump depletion: the energy budget must close, and the
    transmitted pump must actually deplete relative to the prescribed-pump run."""
    cfg = _amplifier_cfg()
    cfg["terms"]["light"] = {"pump_depletion": True}
    cfg["mlflow"]["run"] = "srs-depletion-budget-test"
    exo, sol, ppo = _run_with_ppo(cfg)
    ppo_metrics = ppo.get("metrics", {}) if isinstance(ppo, dict) else {}

    result = sol["solver result"]
    t = np.array(result.ts["default"])
    win = t > 0.75 * t[-1]

    inc = np.array(result.ys["default"]["incident_flux"])
    refl = np.array(result.ys["default"]["reflected_flux"])
    trans = np.array(result.ys["default"]["transmitted_flux"])
    backrefl = np.array(result.ys["default"]["backrefl_flux"])
    dissip = np.array(result.ys["default"]["epw_dissipation"])
    bloss = np.array(result.ys["default"]["epw_boundary_loss"])
    W = np.array(result.ys["default"]["epw_energy"])

    # sanity: the amplifier actually amplified and the pump actually depleted
    assert np.mean(refl[win]) > 2.0 * 1e14 / 1.5e15, "seed was not amplified"
    assert np.mean(trans[win]) < 0.97, "pump did not deplete"
    assert np.mean(refl[win]) <= 1.05, "reflectivity exceeds the incident flux"

    # budget closure over the quasi-steady window, all in units of the incident flux:
    # what goes missing between the probes (S_left - S_right) must equal EPW dissipation
    # + absorber losses + EPW energy change. dissip/bloss are total-energy rates already;
    # W is the OSIRIS field-only energy so the total stored EPW energy changes at 2*dW/dt.
    # Convert to flux units: I0_osiris = a0^2/2, energy rates are per ps -> / w0.
    derived = exo.adept_module.cfg["units"]["derived"]
    w0 = derived["w0"]
    a0 = derived["E0_source"] * derived["e_norm"]
    I0_osiris = 0.5 * a0**2
    absorbed_measured = np.mean((inc - refl - trans - backrefl)[win])
    dWdt = np.gradient(W, t)  # OSIRIS (field-only) energy per ps
    absorbed_predicted = np.mean((dissip + bloss + 2.0 * dWdt)[win]) / w0 / I0_osiris
    assert abs(absorbed_measured - absorbed_predicted) < 0.1 * max(np.mean(inc[win]), 1e-30), (
        f"budget does not close: measured {absorbed_measured:.3e} vs predicted {absorbed_predicted:.3e}"
    )

    # the definitional identity R + T + absorbed = 1 from the metrics
    assert "laser_absorbed_frac" in ppo_metrics, "budget metrics missing from post_process"
    total = ppo_metrics["laser_reflectivity"] + ppo_metrics["laser_transmissivity"] + ppo_metrics["laser_absorbed_frac"]
    np.testing.assert_allclose(total, 1.0, atol=1e-6)

    # comparison run with the pump prescribed: transmission cannot deplete there
    cfg2 = _amplifier_cfg()
    cfg2["mlflow"]["run"] = "srs-prescribed-comparison-test"
    exo2, sol2 = _run(cfg2)
    result2 = sol2["solver result"]
    trans2 = np.array(result2.ys["default"]["transmitted_flux"])
    assert np.mean(trans[win]) < np.mean(trans2[win]) - 0.02, "depletion did not reduce transmission"


def test_epw_energy_normalization():
    """The epw_energy save quantity equals the hand-computed OSIRIS-unit energy for a
    synthetic single-mode phi_k."""
    from copy import deepcopy

    import jax.numpy as jnp

    from adept._lpse2d import helpers

    cfg = deepcopy(_quasi_1d(_load_cfg()))
    helpers.write_units(cfg)
    cfg = helpers.get_derived_quantities(cfg)
    cfg["grid"] = helpers.get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = helpers.get_density_profile(cfg)
    cfg = helpers.get_save_quantities(cfg)
    save_func = cfg["save"]["default"]["func"]

    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    ik = 5
    amp = 1e-3
    phi_k = np.zeros((nx, ny), dtype=np.complex128)
    phi_k[ik, 0] = amp
    y = {
        "epw": jnp.array(phi_k).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
    }
    out = save_func(0.0, y, None)

    derived = cfg["units"]["derived"]
    k = cfg["grid"]["kx"][ik]
    # single k-mode: |ex(x)| = k*amp/(nx*ny) everywhere; sum_x mean_y |ex|^2 = nx*(k*amp/(nx*ny))^2
    expected = 0.25 * cfg["grid"]["dx"] * derived["x_norm"] * derived["e_norm"] ** 2 * nx * (k * amp / (nx * ny)) ** 2
    np.testing.assert_allclose(float(out["epw_energy"]), expected, rtol=1e-10)


def test_noise_seed_reproducibility():
    """Two runs with the same explicit noise seed are bit-identical; a different seed is not."""
    e_sq = {}
    for name, seed in [("a", 1234), ("b", 1234), ("c", 4321)]:
        cfg = _quasi_1d(_load_cfg())
        cfg["grid"]["xmax"] = "20um"
        cfg["grid"]["tmax"] = "0.15ps"
        cfg["save"]["fields"]["t"]["tmax"] = "0.15ps"
        cfg["save"]["fields"]["t"]["dt"] = "0.05ps"
        cfg["terms"]["epw"]["source"]["noise_seed"] = seed
        cfg["mlflow"]["run"] = f"srs-seed-repro-{name}"
        _, sol = _run(cfg)
        e_sq[name] = np.array(sol["solver result"].ys["default"]["e_sq"])

    np.testing.assert_array_equal(e_sq["a"], e_sq["b"])
    assert not np.array_equal(e_sq["a"], e_sq["c"]), "different seeds gave identical noise"


if __name__ == "__main__":
    test_srs_seed_propagation()
    test_srs_growth_rate()
