"""Term-level parity of the envelope-2d solver with the original LPSE (C++) code.

Each test pins one of the options added for LPSE parity on the solver's own
coefficients, without running a full simulation:

- the TPD source carries every pump component and LPSE's ``(w0/wp0 - 1)`` factor
  (``tpd_form``), and reduces to the MATLAB form at envelope density n_c/4;
- the TPD pump-depletion term is the transverse part of ``E div E`` (k-space projector);
- the LPSE Landau-damping prefactor, the lower threshold and the multiplier;
- the relativistic (Bessel) rate reduces to the non-relativistic one for c -> infinity;
- the thermal (fluctuation-dissipation) EPW noise reaches the Cerenkov spectrum
  ``<|E_k|^2> = A^2 / (1 + k^2 lambda_D^2)`` per mode, independent of dt;
- the LPSE exponential absorbing-layer profile and ``max_wavenumber`` band cap.
"""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _finish_cfg(cfg):
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


def _base_cfg(envelope_density=0.25, box_density=0.25, **grid):
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["units"]["envelope density"] = envelope_density
    cfg["density"] = {"basis": "uniform", "val": box_density}
    cfg["grid"].update(
        {
            "boundary_width": "0.4um",
            "dt": "1fs",
            "dx": "0.1um",
            "xmax": "6.4um",
            "tmax": "10fs",
            "ymax": "1.6um",
            "ymin": "-1.6um",
        }
    )
    cfg["grid"].update(grid)
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": False})
    cfg["terms"]["epw"]["density_gradient"] = False
    return cfg


def _random_phi(cfg, seed):
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k *= np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    return jnp.asarray(phi_k)


# ------------------------------------------------------------------ TPD source --


def test_tpd_forms_coincide_at_quarter_critical_and_differ_elsewhere():
    """At envelope density 0.25, w0 = 2 wp0 exactly and LPSE's coefficient/factor equal the
    MATLAB prototype's; at 0.23 the LPSE coefficient i e/(4 me w0) is 4% smaller than
    i e/(8 me wp0) and the charge-density factor is (w0/wp0 - 1) instead of 1."""
    from adept._lpse2d.core.epw import SpectralEPWSolver

    for n_env, same in ((0.25, True), (0.23, False)):
        cfg_lpse = _finish_cfg(_base_cfg(envelope_density=n_env, box_density=n_env))
        cfg_matlab = deepcopy(cfg_lpse)
        cfg_matlab["terms"]["epw"]["source"]["tpd_form"] = "matlab"
        lpse = SpectralEPWSolver(cfg_lpse)
        matlab = SpectralEPWSolver(cfg_matlab)
        assert lpse.tpd_form == "lpse" and matlab.tpd_form == "matlab"
        assert lpse.tpd_prefactor == pytest.approx(1j * lpse.e / (4.0 * lpse.w0 * lpse.me))
        assert matlab.tpd_prefactor == pytest.approx(1j * lpse.e / (8.0 * lpse.wp0 * lpse.me))
        assert lpse.tpd_rho_factor == pytest.approx(lpse.w0 / lpse.wp0 - 1.0)
        assert matlab.tpd_rho_factor == 1.0

        phi_k = _random_phi(cfg_lpse, 11)
        rng = np.random.default_rng(12)
        nx, ny = cfg_lpse["grid"]["nx"], cfg_lpse["grid"]["ny"]
        E0 = jnp.asarray(rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2)))
        ex, ey = lpse.phi_k_to_e_fields(phi_k)
        s_lpse = np.asarray(lpse.calc_tpd_source(0.3, phi_k, ex, ey, E0))
        s_matlab = np.asarray(matlab.calc_tpd_source(0.3, phi_k, ex, ey, E0))
        if same:
            np.testing.assert_allclose(s_lpse, s_matlab, rtol=1e-12, atol=1e-12 * np.abs(s_lpse).max())
        else:
            assert not np.allclose(s_lpse, s_matlab, rtol=1e-3)
            # prefactor ratio 2 wp0/w0 = 2 sqrt(0.23) = 0.959; rho factor 1/sqrt(0.23) - 1 = 1.085
            assert abs(lpse.tpd_prefactor / matlab.tpd_prefactor) == pytest.approx(2.0 * np.sqrt(0.23), rel=1e-12)
            assert lpse.tpd_rho_factor == pytest.approx(1.0 / np.sqrt(0.23) - 1.0, rel=1e-12)


def test_tpd_source_uses_both_pump_components():
    """An x-polarized pump drives TPD through E0x . Ex* and kx E0x rho* exactly like the
    y-polarized one drives it through the y terms (LPSE uses the full dot product)."""
    from adept._lpse2d.core.epw import SpectralEPWSolver

    cfg = _finish_cfg(_base_cfg())
    epw = SpectralEPWSolver(cfg)
    phi_k = _random_phi(cfg, 21)
    ex, ey = epw.phi_k_to_e_fields(phi_k)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(22)
    field = jnp.asarray(rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny)))
    zeros = jnp.zeros_like(field)
    E0_x = jnp.stack([field, zeros], axis=-1)
    E0_y = jnp.stack([zeros, field], axis=-1)

    s_x = np.asarray(epw.calc_tpd_source(0.1, phi_k, ex, ey, E0_x))
    s_y = np.asarray(epw.calc_tpd_source(0.1, phi_k, ex, ey, E0_y))
    assert np.abs(s_x).max() > 0.0
    # hand-built reference for the x-polarized pump
    rho = jnp.fft.ifft2(epw.k_sq * phi_k)
    tpd1 = jnp.fft.fft2(field * jnp.conj(ex))
    tpd2 = 1j * epw.kx[:, None] * jnp.fft.fft2(field * jnp.conj(rho)) * epw.one_over_k_sq
    expected = epw.tpd_prefactor * jnp.exp(-1j * (epw.w0 - 2.0 * epw.wp0) * 0.1) * (tpd1 + epw.tpd_rho_factor * tpd2)
    expected = np.asarray(expected * epw.low_pass_filter * epw.zero_mask)
    np.testing.assert_allclose(s_x, expected, rtol=1e-11, atol=1e-11 * np.abs(expected).max())
    # linear in the pump: the sum of the two polarizations is the source of the combined pump
    s_xy = np.asarray(epw.calc_tpd_source(0.1, phi_k, ex, ey, E0_x + E0_y))
    np.testing.assert_allclose(s_xy, s_x + s_y, rtol=1e-11, atol=1e-11 * np.abs(s_xy).max())


# ------------------------------------------------------------ TPD depletion --


def test_tpd_depletion_transverse_projection_is_divergence_free():
    """[E div E]_T has zero divergence mode by mode, the projection is idempotent, and
    switching it off returns the raw E div E for both components."""
    from adept._lpse2d.core.light import CoupledLight

    cfg = _base_cfg()
    cfg["terms"]["light"] = {"pump_depletion": True}
    cfg = _finish_cfg(cfg)
    light = CoupledLight(cfg)
    assert light.tpd_projection
    phi_k = _random_phi(cfg, 31)

    f_t = np.asarray(light.tpd_depletion_vector(phi_k))
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    fx_k, fy_k = np.fft.fft2(f_t[..., 0]), np.fft.fft2(f_t[..., 1])
    div_k = kx * fx_k + ky * fy_k
    assert np.abs(div_k).max() < 1e-10 * (np.abs(kx * fx_k) + np.abs(ky * fy_k)).max()

    # idempotent: projecting the projected field changes nothing
    longitudinal = kx * fx_k + ky * fy_k
    k_sq = kx**2 + ky**2
    k_sq_safe = np.where(k_sq > 0, k_sq, 1.0)
    fx_again = fx_k - np.where(k_sq > 0, kx / k_sq_safe, 0.0) * longitudinal
    np.testing.assert_allclose(fx_again, fx_k, rtol=0, atol=1e-10 * np.abs(fx_k).max())

    cfg_raw = deepcopy(cfg)
    cfg_raw["terms"]["light"]["tpd_projection"] = False
    raw = CoupledLight(cfg_raw)
    f_raw = np.asarray(raw.tpd_depletion_vector(phi_k))
    ex = np.asarray(jnp.fft.ifft2(-1j * jnp.asarray(cfg["grid"]["kx"])[:, None] * phi_k))
    ey = np.asarray(jnp.fft.ifft2(-1j * jnp.asarray(cfg["grid"]["ky"])[None, :] * phi_k))
    rho = np.asarray(jnp.fft.ifft2(light.k_sq * phi_k))
    np.testing.assert_allclose(f_raw[..., 0], ex * rho, rtol=1e-11, atol=1e-11 * np.abs(ex * rho).max())
    np.testing.assert_allclose(f_raw[..., 1], ey * rho, rtol=1e-11, atol=1e-11 * np.abs(ey * rho).max())
    # the projection only removes the longitudinal part: transverse content is unchanged
    fx_raw_k, fy_raw_k = np.fft.fft2(f_raw[..., 0]), np.fft.fft2(f_raw[..., 1])
    transverse_raw = ky * fx_raw_k - kx * fy_raw_k
    transverse_t = ky * fx_k - kx * fy_k
    np.testing.assert_allclose(transverse_t, transverse_raw, rtol=0, atol=1e-10 * np.abs(transverse_raw).max())


def test_tpd_depletion_k_filter_masks_light_band():
    from adept._lpse2d.core.light import CoupledLight

    cfg = _base_cfg()
    cfg["terms"]["light"] = {"pump_depletion": True, "tpd_k_filter": True}
    cfg = _finish_cfg(cfg)
    light = CoupledLight(cfg)
    mask = np.asarray(light.tpd_k_mask)
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    k_max_sq = (1.2 * light.w0 / light.c) ** 2 * (1.0 - 0.25)
    np.testing.assert_array_equal(mask, np.where((kx**2 + ky**2 > 0) & (kx**2 + ky**2 < k_max_sq), 1.0, 0.0))
    f_t = np.asarray(light.tpd_depletion_vector(_random_phi(cfg, 41)))
    for comp in range(2):
        outside = np.fft.fft2(f_t[..., comp]) * (1.0 - mask)
        assert np.abs(outside).max() < 1e-10 * np.abs(np.fft.fft2(f_t[..., comp])).max()


# ---------------------------------------------------------------- Landau --


def test_lpse_landau_form_prefactor_ratio():
    """LPSE's sqrt(1 + 3x^2) vs MATLAB's (1 + 1.5x^2), x = k lambda_D, with identical
    exponents: 0.7% at x = 0.3 and 3.9% at x = 0.5 (hand check in the vault gap audit)."""
    from adept._lpse2d.core.epw import landau_damping_rate

    wp0, vte_sq = 2.0, 0.01
    lam_d = np.sqrt(vte_sq) / wp0
    for x, expected_pct in ((0.3, 0.7), (0.5, 3.8)):
        k_sq = jnp.asarray([[(x / lam_d) ** 2]])
        g_matlab = float(landau_damping_rate(k_sq, wp0, vte_sq, jnp.ones((1, 1)), form="matlab")[0, 0])
        g_lpse = float(landau_damping_rate(k_sq, wp0, vte_sq, jnp.ones((1, 1)), form="lpse")[0, 0])
        ratio = np.sqrt(1.0 + 3.0 * x**2) / (1.0 + 1.5 * x**2)
        assert g_lpse / g_matlab == pytest.approx(ratio, rel=1e-12)
        assert abs(1.0 - ratio) * 100.0 == pytest.approx(expected_pct, abs=0.06)  # 0.70%, 3.79%


def test_landau_threshold_multiplier_and_form_flow_through_the_solver():
    from adept._lpse2d.core.epw import SpectralEPWSolver, analytic_landau_rate, landau_damping_rate

    cfg = _finish_cfg(_base_cfg())
    base = np.asarray(analytic_landau_rate(cfg))
    k_sq = jnp.asarray(cfg["grid"]["kx"])[:, None] ** 2 + jnp.asarray(cfg["grid"]["ky"])[None, :] ** 2
    derived = cfg["units"]["derived"]
    ref = np.asarray(landau_damping_rate(k_sq, derived["wp0"], derived["vte_sq"], jnp.where(k_sq > 0, 1.0, 0.0)))
    np.testing.assert_allclose(base, ref, rtol=1e-12)

    threshold = float(np.median(base[base > 0]))
    cfg2 = deepcopy(cfg)
    cfg2["terms"]["epw"]["damping"].update(
        {"landau_form": "lpse", "landau_lower_threshold": threshold, "landau_multiplier": 0.5}
    )
    rate = np.asarray(analytic_landau_rate(cfg2))
    ref2 = np.asarray(
        landau_damping_rate(k_sq, derived["wp0"], derived["vte_sq"], jnp.where(k_sq > 0, 1.0, 0.0), form="lpse")
    )
    expected = np.where(ref2 > threshold, ref2, 0.0) * 0.5
    np.testing.assert_allclose(rate, expected, rtol=1e-12)
    assert np.any(rate == 0.0) and np.any(rate > 0.0)
    # the solver applies exactly this array
    solver = SpectralEPWSolver(cfg2)
    np.testing.assert_allclose(np.asarray(solver.calc_landau_damping_rate()), expected, rtol=1e-12)


def test_relativistic_landau_rate_reduces_to_nonrelativistic():
    """For c -> infinity (Ve/C -> 0) the Maxwell-Juettner 2-D/3-D rates go to
    sqrt(pi/8) (kde/k)^3 wp omega exp(-omega^2/(2 x^2)) with the *expanded* Bohm-Gross
    omega = 1 + 1.5 x^2 that LPSE's relativistic forms carry (K_nu(a) ~ sqrt(pi/2a) e^{-a});
    that differs from the classical sqrt(1 + 3x^2) form by exp(-1.125 x^2) in the exponent.
    At 2 keV the relativistic rates differ from that limit by a few percent in the band."""
    from adept._lpse2d.core.epw import landau_damping_rate_relativistic

    wp0, vte_sq = 2.0, 0.01
    lam_d = np.sqrt(vte_sq) / wp0
    x = np.linspace(0.25, 0.5, 6)
    k_sq = (x / lam_d) ** 2
    omega = 1.0 + 1.5 * x**2
    classical = np.sqrt(np.pi / 8.0) / x**3 * wp0 * omega * np.exp(-(omega**2) / (2.0 * x**2))
    c_large = np.sqrt(vte_sq) * 1.0e3
    for ndim in (2, 3):
        rel = landau_damping_rate_relativistic(k_sq, wp0, vte_sq, c_large, ndim=ndim)
        np.testing.assert_allclose(rel, classical, rtol=3e-3)
    c_2kev = np.sqrt(vte_sq) / np.sqrt(2.0 / 511.0)
    rel = landau_damping_rate_relativistic(k_sq, wp0, vte_sq, c_2kev, ndim=2)
    ratio = rel / classical
    # relativistic mass reduces the damping of fast modes (v_phi ~ 0.25 c at x = 0.25:
    # -38%) and the correction fades toward x = 0.5 (-5%), monotonically
    assert np.all((ratio > 0.5) & (ratio < 1.0)) and np.all(np.diff(ratio) > 0.0)
    # modes whose phase velocity exceeds c are undamped
    rel_slow = landau_damping_rate_relativistic(np.asarray([(0.02 / lam_d) ** 2]), wp0, vte_sq, c_2kev, ndim=2)
    assert rel_slow[0] == 0.0


# ----------------------------------------------------------------- noise --


def _noise_cfg(model, **source):
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": 0.25}
    cfg["grid"].update(
        {
            "boundary_width": "0.2um",
            "dt": "1fs",
            "dx": "0.05um",
            "xmax": "3.2um",
            "tmax": "10fs",
            "ymax": "0.8um",
            "ymin": "-0.8um",
        }
    )
    cfg["grid"]["low_pass_filter"] = 1.0
    cfg["terms"]["zero_mask"] = True
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["source"] = {
        "noise": True,
        "noise_model": model,
        "noise_seed": 5,
        "tpd": False,
        "srs": False,
        **source,
    }
    cfg["terms"]["epw"]["density_gradient"] = False
    return _finish_cfg(cfg)


def test_flat_noise_model_is_unchanged():
    """noise_model: flat keeps the MATLAB kick dt * amplitude on every retained mode."""
    from adept._lpse2d.core.epw import SpectralEPWSolver, noise_kick_spectrum

    cfg = _noise_cfg("flat", noise_amplitude=3e-9)
    kick = np.asarray(noise_kick_spectrum(cfg))
    band = np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    np.testing.assert_allclose(kick, cfg["grid"]["dt"] * 3e-9 * band, rtol=1e-14)
    solver = SpectralEPWSolver(cfg)
    noise = np.asarray(solver.get_noise(jnp.asarray(4.0 * solver.dt)))
    np.testing.assert_allclose(np.abs(noise), kick, rtol=1e-12)


def test_thermal_noise_requires_damping():
    from adept._lpse2d.core.epw import noise_kick_spectrum

    cfg = _noise_cfg("thermal")
    cfg["terms"]["epw"]["damping"]["landau"] = False
    with pytest.raises(ValueError, match="thermal"):
        noise_kick_spectrum(cfg)


@pytest.mark.parametrize("dt_fs", [1.0, 4.0])
def test_thermal_noise_equilibrates_to_cerenkov_spectrum(dt_fs):
    """Sources off, uniform periodic box: every mode is phi *= e^{-gamma dt}; phi += D e^{i theta},
    whose stationary |E_k|^2 is A^2 / (1 + k^2 lambda_D^2) (x-space amplitude squared) for
    the thermal kick -- for any dt."""
    import jax

    from adept._lpse2d.core.epw import SpectralEPWSolver

    amplitude = 2.0e-3
    cfg = _noise_cfg("thermal", noise_amplitude=amplitude)
    cfg["grid"]["dt"] = dt_fs * 1e-3
    cfg["grid"]["absorbing_boundaries"] = np.ones_like(np.asarray(cfg["grid"]["absorbing_boundaries"]))
    solver = SpectralEPWSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    derived = cfg["units"]["derived"]
    gamma = np.asarray(solver.landau_rate)
    k_sq = np.asarray(solver.k_sq)
    lam_sq = derived["vte_sq"] / derived["wp0"] ** 2

    # compare only modes that relax within the run and are in the retained band
    n_steps, n_burn = 6000, 2000
    relaxed = (2.0 * gamma * solver.dt > 0.01) & (np.asarray(solver.noise_kick) > 0.0)
    assert relaxed.sum() > 100

    E0 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)

    def step(carry, i):
        phi_k, acc = carry
        t = i * solver.dt
        phi_k = solver(t, {"epw": phi_k, "E0": E0, "E1": E0}, None)
        acc = acc + jnp.where(i >= n_burn, k_sq * jnp.abs(phi_k) ** 2, 0.0)
        return (phi_k, acc), None

    phi0 = jnp.zeros((nx, ny), dtype=jnp.complex128)
    (phi_k, acc), _ = jax.lax.scan(jax.jit(step), (phi0, jnp.zeros((nx, ny))), jnp.arange(n_steps))
    measured = np.asarray(acc) / (n_steps - n_burn) / (nx * ny) ** 2
    expected = amplitude**2 / (1.0 + k_sq * lam_sq)
    ratio = measured[relaxed] / expected[relaxed]
    # per-mode samples are exponentially distributed with ~ (n/(1/(2 gamma dt))) effective
    # draws; the band mean is far tighter than any single mode
    assert ratio.mean() == pytest.approx(1.0, abs=0.05), ratio.mean()
    assert np.median(ratio) == pytest.approx(1.0, abs=0.1)


def test_thermal_noise_calibration_scales_with_temperature_and_volume():
    from adept._lpse2d.core.epw import noise_kick_spectrum

    cfg = _noise_cfg("thermal", noise_amplitude=1.0, noise_calibrate=True)
    kick = np.asarray(noise_kick_spectrum(cfg))
    cfg_uncal = _noise_cfg("thermal", noise_amplitude=1.0)
    kick_uncal = np.asarray(noise_kick_spectrum(cfg_uncal))
    derived = cfg["units"]["derived"]
    from astropy.units import Quantity as _Q

    te_kev = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
    energy_scale = derived["massScale"] * derived["spatialScale"] ** 2 / derived["timeScale"] ** 2
    kT = te_kev * 1.602176634e-9 / energy_scale
    lx = cfg["grid"]["nx"] * cfg["grid"]["dx"]
    ly = cfg["grid"]["ny"] * cfg["grid"]["dy"]
    a_thermal = np.sqrt(8.0 * np.pi * kT / (lx * ly * ly))
    band = kick_uncal > 0
    np.testing.assert_allclose(kick[band] / kick_uncal[band], a_thermal, rtol=1e-12)
    assert 0.0 < a_thermal < 1.0


def test_noise_max_wavenumber_masks_the_kick():
    from adept._lpse2d.core.epw import noise_kick_spectrum

    cfg = _noise_cfg("flat", noise_max_wavenumber=2.0)
    kick = np.asarray(noise_kick_spectrum(cfg))
    derived = cfg["units"]["derived"]
    k0 = derived["w0"] / derived["c"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    outside = np.sqrt(kx**2 + ky**2) >= 2.0 * k0
    assert outside.any() and (~outside).sum() > 10
    assert np.all(kick[outside] == 0.0) and np.all(kick[~outside & (kx**2 + ky**2 > 0)] > 0.0)


# --------------------------------------------------- absorbers and band caps --


def test_exp_absorbing_profile_matches_lpse_formula():
    cfg = _base_cfg(boundary_profile="exp", boundary_max_rate=150.0, boundary_lambda=7.0)
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg = _finish_cfg(cfg)
    grid = cfg["grid"]
    rate = np.asarray(grid["absorbing_rate"])
    x = np.asarray(grid["x"])
    L = 0.4
    half = 0.5 * grid["dx"]
    s = np.clip(np.maximum(grid["xmin"] + L + half - x, x - (grid["xmax"] - L - half)), 0.0, L)
    expected = 150.0 * np.expm1(7.0 * s / L) / np.expm1(7.0)
    np.testing.assert_allclose(rate[:, 0], expected, rtol=1e-12, atol=1e-12)
    assert rate[0, 0] == pytest.approx(150.0) and rate[-1, 0] == pytest.approx(150.0)  # edge cells: full rate
    assert (rate[:, 0] > 0).sum() == 2 * round(L / grid["dx"])  # L / dx cells per side, as LPSE
    assert np.all(rate[:, 0] == rate[:, -1])
    np.testing.assert_allclose(np.asarray(grid["absorbing_boundaries"]), np.exp(-rate * grid["dt"]), rtol=1e-14)
    # interior is exactly undamped
    interior = (x > grid["xmin"] + L) & (x < grid["xmax"] - L)
    assert np.all(rate[interior, 0] == 0.0)

    # both axes absorbing: combined by max (LPSE), never by sum
    cfg2 = _base_cfg(boundary_profile="exp", boundary_max_rate=150.0)
    cfg2["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "absorbing"}
    cfg2 = _finish_cfg(cfg2)
    assert np.asarray(cfg2["grid"]["absorbing_rate"]).max() <= 150.0 * (1.0 + 1e-9)


def test_tanh_profile_is_the_default_and_unchanged():
    cfg = _base_cfg()
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    cfg = _finish_cfg(cfg)
    from adept._base_ import get_envelope

    grid = cfg["grid"]
    L = 0.4
    env = get_envelope(L / 5, L / 5, grid["xmin"] + L, grid["xmax"] - L, grid["x"])[:, None]
    expected = np.exp(-1.0e4 * grid["dt"] * (1.0 - env * np.ones((1, grid["ny"]))))
    np.testing.assert_allclose(np.asarray(grid["absorbing_boundaries"]), expected, rtol=1e-13)


def test_max_wavenumber_caps_the_retained_band():
    cfg = _base_cfg()
    cfg["terms"]["epw"]["max_wavenumber"] = 1.5
    cfg = _finish_cfg(cfg)
    derived = cfg["units"]["derived"]
    k0 = derived["w0"] / derived["c"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    k = np.sqrt(kx**2 + ky**2)
    band = np.asarray(cfg["grid"]["low_pass_filter_grid"])
    assert np.all(band[k >= 1.5 * k0] == 0.0)
    assert np.any(band[k < 1.5 * k0] > 0.0)


def test_srs_k_filter_is_optional_and_masks_the_detuned_resonance():
    """The SRS source k-filter cuts at ``scale * k1(n_min)`` for zero detuning. At 0.2 nc
    with envelope density 0.25 the resonant back-scattered Raman wavenumber lies above
    that cutoff, so the default filter removes the SRS mode; ``srs_k_filter: false``
    (LPSE's ``lw.kFilter`` default) passes everything and a larger scale passes it too."""
    from adept._lpse2d.core.epw import SpectralEPWSolver

    def cfg_with(**source):
        cfg = _base_cfg(envelope_density=0.25, box_density=0.2)
        cfg["terms"]["epw"]["source"].update({"tpd": False, "srs": True, **source})
        return _finish_cfg(cfg)

    cfg = cfg_with(srs_k_filter=True)
    d = cfg["units"]["derived"]
    wp = d["w0"] * np.sqrt(0.2)
    k_s = np.sqrt((d["w0"] - wp) ** 2 - wp**2) / d["c"]  # resonant Raman wavenumber at 0.2 nc
    kx = np.asarray(cfg["grid"]["kx"])
    ix = int(np.argmin(np.abs(kx - k_s)))
    assert abs(kx[ix] - k_s) < 0.1 * k_s

    default = SpectralEPWSolver(cfg)
    assert float(default.E1_filter[0, 0, 0]) == 1.0
    assert float(default.E1_filter[ix, 0, 0]) == 0.0, "resonant mode should be masked by the zero-detuning cutoff"

    off = SpectralEPWSolver(cfg_with(srs_k_filter=False))
    assert bool(jnp.all(off.E1_filter == 1.0))

    wide = SpectralEPWSolver(cfg_with(srs_k_filter=True, srs_k_filter_scale=1.5))
    assert float(wide.E1_filter[ix, 0, 0]) == 1.0
