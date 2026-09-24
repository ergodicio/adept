"""Quasilinear evolution of the electron distribution (terms.qle; LPSE qle.*) -- plan 2 K.1."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._lpse2d.parity import deck_path


def _cfg(ny_box="0.02um", xmax="12.8um", nv=121, v_max=0.4, **qle):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {"ymax": ny_box, "ymin": f"-{ny_box}", "xmax": xmax, "dx": "0.05um", "tmax": "10fs", "dt": "1fs"}
    )
    cfg["terms"]["qle"] = {"active": True, "nv": nv, "v_max": v_max, "landau_evolution": True, **qle}
    cfg["terms"]["epw"]["damping"]["landau_form"] = "lpse"
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _line(cfg, k_lambda_d, amplitude, width=2.0):
    """A 7-mode spectral line about ``k lambda_D`` on the ky = 0 row: ``phi_k`` (nx, ny) with
    Fourier coefficients ``amplitude exp(-(dj / width)^2 / 2)``, and the centre index."""
    d = cfg["units"]["derived"]
    lam_d = np.sqrt(d["vte_sq"]) / d["wp0"]
    kx = np.asarray(cfg["grid"]["kx"])
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    i = int(np.argmin(np.abs(kx * lam_d - k_lambda_d)))
    line = np.arange(i - 3, i + 4)
    amps = amplitude * np.exp(-0.5 * ((line - i) / width) ** 2)
    phi_k = np.zeros((nx, ny), dtype=complex)
    phi_k[line, 0] = amps * nx * ny
    return jnp.asarray(phi_k), i, line, amps


@pytest.mark.parametrize("ny_box", ["0.02um", "3.2um"])
def test_landau_rate_of_the_maxwellian_is_the_analytic_one(ny_box):
    from adept._lpse2d.core.epw import analytic_landau_rate
    from adept._lpse2d.core.qle import QuasilinearEvolution, initial_vdf

    cfg = _cfg(ny_box, v_max=0.3)
    qle = QuasilinearEvolution(cfg)
    gamma = np.asarray(qle.landau_rate(jnp.asarray(initial_vdf(cfg))))
    analytic = np.asarray(analytic_landau_rate(cfg))
    valid = np.asarray(qle.k_valid) & (analytic > 1e-3 * cfg["units"]["derived"]["wp0"])
    ratio = gamma[valid] / analytic[valid]
    assert valid.sum() > 20
    assert abs(np.median(ratio) - 1.0) < 0.04
    assert ratio.min() > 0.85 and ratio.max() < 1.05
    # modes whose phase velocity is off the velocity grid are undamped (LPSE)
    assert np.all(gamma[~np.asarray(qle.k_valid)] == 0.0)


def test_diffusion_tensor_conserves_wave_plus_electron_energy():
    """The electron heating ``-n m int v D f' dv`` of a spectral line equals the wave-energy
    loss ``sum 2 gamma_L W_k`` (kinetic energy factor ``d(w eps)/dw``): 1 % at k lambda_D 0.15;
    at 0.25 the O((k lambda_D)^4) convention mismatch (Bohm-Gross vs kinetic) shows."""
    from adept._lpse2d.core.epw import analytic_landau_rate
    from adept._lpse2d.core.qle import QuasilinearEvolution, initial_vdf

    cfg = _cfg(xmax="25.6um", nv=481, v_max=0.6)
    d = cfg["units"]["derived"]
    wpe, vte_sq, e, me = d["wp0"], d["vte_sq"], d["e"], d["me"]
    kx = np.asarray(cfg["grid"]["kx"])
    qle = QuasilinearEvolution(cfg)
    f0 = np.asarray(initial_vdf(cfg))
    fp = np.gradient(f0, qle.dv)
    v = np.asarray(qle.v)
    n_m = wpe**2 * me**2 / (4.0 * np.pi * e**2)  # n_e m_e
    gamma = np.asarray(analytic_landau_rate(cfg))[:, 0]
    for k_lambda_d, tol in ((0.15, 0.03), (0.25, 0.08)):
        phi_k, _, line, amps = _line(cfg, k_lambda_d, 1e-11)
        c = np.asarray(qle.diffusion_tensor(phi_k))[:, 0]
        assert np.all(c >= 0.0) and np.count_nonzero(c) > 4
        heating = -n_m * np.sum(v * c * fp) * qle.dv
        loss = 0.0
        for j, a in zip(line, amps, strict=True):
            k = kx[j]
            omega = np.sqrt(wpe**2 + 3.0 * vte_sq * k**2)
            energy_factor = 1.0 + (wpe**2 + 9.0 * k**2 * vte_sq) / omega**2
            loss += 2.0 * gamma[j] * (k * a) ** 2 / (16.0 * np.pi) * energy_factor
        assert abs(heating / loss - 1.0) < tol, (k_lambda_d, heating / loss)


@pytest.mark.parametrize("ny_box", ["0.02um", "1.6um"])
def test_a_driven_line_flattens_f_and_reduces_its_landau_rate(ny_box):
    from adept._lpse2d.core.epw import analytic_landau_rate
    from adept._lpse2d.core.qle import QuasilinearEvolution, initial_vdf

    cfg = _cfg(ny_box, thermalization_probability=[0.0, 0.0])
    d = cfg["units"]["derived"]
    qle = QuasilinearEvolution(cfg)
    phi_k, i, _, _ = _line(cfg, 0.25, 1e-11)  # w_b / wpe = 0.03
    f = jnp.asarray(initial_vdf(cfg))
    gamma0 = np.asarray(analytic_landau_rate(cfg))[i, 0]
    kx = np.asarray(cfg["grid"]["kx"])
    v_res = np.sqrt(d["wp0"] ** 2 + 3.0 * d["vte_sq"] * kx[i] ** 2) / kx[i]
    v = np.asarray(qle.v)
    j = int(np.argmin(np.abs(v - v_res)))

    def slope(f):
        fr = np.asarray(f) if qle.ndim == 1 else np.asarray(f).sum(axis=1) * qle.dv
        return (fr[j + 2] - fr[j - 2]) / (4.0 * qle.dv)

    slope0 = slope(f)
    for _ in range(5):
        f = qle.evolve_vdf(f, phi_k, 0.02)
    gamma = np.asarray(qle.landau_rate(f))[i, 0]
    assert 0.0 <= gamma / gamma0 < 0.7  # damping reduced by the plateau (to a few % here)
    assert 0.0 < slope(f) / slope0 < 0.85  # the +-2-cell difference smears the narrow plateau
    assert np.all(np.asarray(f) >= 0.0)
    np.testing.assert_allclose(np.sum(np.asarray(f)) * qle.dv**qle.ndim, qle.volume0, rtol=1e-12)


def test_qle_in_a_run_writes_the_vdf_and_evolves_the_srs_mode_damping():
    from adept import ergoExo
    from adept._lpse2d.core.epw import analytic_landau_rate

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"].update({"ymax": "0.02um", "ymin": "-0.02um", "xmax": "40um", "tmax": "0.6ps", "dx": "50nm"})
    cfg["terms"]["qle"] = {"active": True, "nv": 121, "v_max": 0.5, "landau_evolution": True, "update_every": 5}
    cfg["terms"]["epw"]["damping"]["landau_form"] = "lpse"
    cfg["units"]["laser intensity"] = "3.0e+15W/cm^2"
    cfg["terms"]["light"] = {"pump_depletion": True}  # SRS saturates by depleting the pump
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["save"]["fields"]["t"].update({"tmax": "0.6ps", "dt": "0.1ps"})
    cfg["mlflow"]["run"] = "qle-run"
    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, _ = exo(modules)
    result = sol["solver result"]
    dcfg = exo.adept_module.cfg
    vdf = ppo["vdf"]["vdf"].values
    dv = float(np.diff(ppo["vdf"]["vx (um per ps)"].values)[0])
    np.testing.assert_allclose(vdf[-1].sum() * dv, vdf[0].sum() * dv, rtol=1e-10)
    assert np.all(vdf >= 0.0)
    assert "gamma_L" in ppo["k"]
    phi = np.asarray(result.ys["fields"]["epw"]).view(np.complex128)
    assert np.all(np.isfinite(phi))
    gamma = np.asarray(result.ys["fields"]["gamma_L"])
    analytic = np.asarray(analytic_landau_rate(dcfg))
    d = dcfg["units"]["derived"]
    kx = np.asarray(dcfg["grid"]["kx"])
    band = (kx * np.sqrt(d["vte_sq"]) / d["wp0"] > 0.12) & (kx * np.sqrt(d["vte_sq"]) / d["wp0"] < 0.3)
    i = int(np.argmax(np.where(band, np.abs(phi[-1, :, 0]), 0.0)))  # the SRS mode
    assert analytic[i, 0] > 0.0
    ratio = gamma[:, i, 0] / analytic[i, 0]
    np.testing.assert_allclose(ratio[0], 1.0, rtol=1e-12)  # the analytic rate at t = 0
    assert abs(ratio[1] - 1.0) < 0.08  # the Maxwellian rate on the coarse (0.13 vte) velocity grid
    assert 0.0 < ratio[-1] < 0.95 * ratio[1]  # the grown SRS mode has begun to flatten its resonance


def test_validation_and_translator():
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["terms"]["qle"] = {"active": True, "landau_evolution": True}
    cfg["terms"]["hpe"] = {"active": True}
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    with pytest.raises(ValueError, match="one of them"):
        SplitStep(cfg)

    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    if deck_path("test_029") is None:
        pytest.skip("no LPSE decks present")
    parms = parse_parms(deck_path("test_029"))
    cfg, report = translate_parms(parms, run="test_029")
    qle = cfg["terms"]["qle"]
    assert qle["active"] and qle["landau_evolution"] and qle["nv"] == 96 and qle["v_max"] == 0.5
    assert qle["t_start"] == 1.5 and qle["thermalization_probability"] == [1.0, 0.1]
    assert not any("qle" in u for u in report["unsupported"])
