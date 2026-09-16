"""The combined TPD + SRS solver (terms.epw.solver: combined, LPSE lw.solver = combined).

- With the sources off, the longitudinal part of the combined field evolves exactly
  as the separate spectral EPW solver and the transverse part as the wp0-enveloped
  light propagator.
- The unified source's longitudinal part is the separate solver's TPD source plus SRS
  source (as a potential), and its transverse part is the separate Raman coupling;
  the unified pump depletion is the SRS depletion plus the TPD depletion. All of these
  are algebraic identities at envelope density n_c/4, checked on random band-limited
  fields.
- Config validation follows LPSE (TPD and SRS both on or both off).
"""

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


def _cfg(*, solver="combined", sources=True, density=0.2, pump_depletion=False, periodic=True, noise=False, **extra):
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": density}
    cfg["grid"].update(
        {
            "boundary_width": "0.6um",
            "dt": "1fs",
            "dx": "0.1um",
            "xmax": "6.4um",
            "tmax": "10fs",
            "ymax": "1.6um",
            "ymin": "-1.6um",
            "low_pass_filter": 0.6,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic" if periodic else "absorbing", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": True}
    cfg["terms"]["epw"]["density_gradient"] = True
    cfg["terms"]["epw"]["source"].update({"noise": noise, "noise_seed": 1, "tpd": sources, "srs": sources})
    cfg["terms"]["epw"]["solver"] = solver
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": pump_depletion}
    cfg["terms"].update(extra)
    return _finish(cfg)


def _random_phi(cfg, seed, band=None):
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    mask = np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    if band is not None:
        mask = mask * band
    return jnp.asarray(phi_k * mask)


def _random_transverse(cfg, seed, band=None):
    """Random divergence-free (nx, ny, 2) field on the retained band, in x-space."""
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    k_sq = kx**2 + ky**2
    mask = np.asarray(cfg["grid"]["low_pass_filter_grid"]) * np.where(k_sq > 0, 1.0, 0.0)
    if band is not None:
        mask = mask * band
    a_k = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * mask
    # transverse in 2-D: E_k = a_k (-ky, kx)/|k|
    k_safe = np.sqrt(np.where(k_sq > 0, k_sq, 1.0))
    ex_k, ey_k = -ky / k_safe * a_k, kx / k_safe * a_k
    return jnp.asarray(np.stack([np.fft.ifft2(ex_k), np.fft.ifft2(ey_k)], axis=-1))


def _field_from_phi(phi_k, cfg):
    kx = jnp.asarray(cfg["grid"]["kx"])[:, None]
    ky = jnp.asarray(cfg["grid"]["ky"])[None, :]
    return jnp.stack([jnp.fft.ifft2(-1j * kx * phi_k), jnp.fft.ifft2(-1j * ky * phi_k)], axis=-1)


def _phi_from_source_field(source, cfg):
    """i k . S_k / k^2 on the band: the potential source equivalent to a field source."""
    kx = jnp.asarray(cfg["grid"]["kx"])[:, None]
    ky = jnp.asarray(cfg["grid"]["ky"])[None, :]
    k_sq = kx**2 + ky**2
    one_over = jnp.where(k_sq > 0, 1.0 / jnp.where(k_sq > 0, k_sq, 1.0), 0.0)
    band = jnp.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    return 1j * (kx * jnp.fft.fft2(source[..., 0]) + ky * jnp.fft.fft2(source[..., 1])) * one_over * band


def test_free_evolution_matches_the_separate_solvers():
    from adept._lpse2d.core.combined import CombinedSolver
    from adept._lpse2d.core.epw import SpectralEPWSolver

    cfg = _cfg(sources=False, density=0.23)
    combined = CombinedSolver(cfg)
    epw = SpectralEPWSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi0 = _random_phi(cfg, 5)
    e_t0 = _random_transverse(cfg, 6)
    E1 = _field_from_phi(phi0, cfg) + e_t0
    E0 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    # the derived potential of the composed field is the longitudinal input
    np.testing.assert_allclose(np.asarray(combined.potential(E1)), np.asarray(phi0), rtol=1e-10, atol=1e-12)

    n_steps = 7
    y = {"E0": E0, "E1": E1, "epw": phi0}
    phi_sep = phi0
    for i in range(n_steps):
        t = i * cfg["grid"]["dt"]
        _, y["E1"], y["epw"] = combined(t, y, {}, lambda tt: E0)
        phi_sep = epw(t, {"epw": phi_sep, "E0": E0, "E1": E0}, None)
    np.testing.assert_allclose(np.asarray(y["epw"]), np.asarray(phi_sep), rtol=1e-9, atol=1e-12 * np.abs(phi_sep).max())

    # transverse part: exact wp0-enveloped light propagation with the density detuning
    derived = cfg["units"]["derived"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    k_sq = kx**2 + ky**2
    t_total = n_steps * cfg["grid"]["dt"]
    phase = np.exp(
        -1j * t_total * (derived["wp0"] / 2.0 * (0.23 / 0.25 - 1.0) + derived["c"] ** 2 * k_sq / (2.0 * derived["wp0"]))
    )
    e_t = np.asarray(combined.transverse(y["E1"]))
    for comp in range(2):
        expected = np.fft.ifft2(np.fft.fft2(np.asarray(e_t0)[..., comp]) * phase)
        np.testing.assert_allclose(e_t[..., comp], expected, rtol=1e-9, atol=1e-12 * np.abs(expected).max())


def test_unified_source_is_the_separate_sources_at_quarter_critical():
    """At envelope density 0.25 (w0 = 2 wp0, no carrier mismatch) the unified source
    -i e/(4 me w0) [grad(E0 . E1*) + (1 - w0/wp0) E0 (div E1)*] splits exactly into the
    separate solver's terms: longitudinal part = TPD source (from E1_L) + SRS source (from
    E1_T, whose (n/n_env) factor LPSE's combined source does not carry); transverse part =
    the Raman coupling +i e/(4 me w0) rho* E0 (transverse-projected)."""
    from adept._lpse2d.core.combined import CombinedSolver, longitudinal_transverse
    from adept._lpse2d.core.epw import SpectralEPWSolver
    from adept._lpse2d.core.spectral_light import SpectralRamanLight

    n = 0.2
    cfg = _cfg(density=n)
    combined = CombinedSolver(cfg)
    epw = SpectralEPWSolver(cfg)
    raman = SpectralRamanLight(cfg)
    assert combined.rho_factor == pytest.approx(-1.0) and epw.tpd_rho_factor == pytest.approx(1.0)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(8)
    # keep the light part inside the separate SRS source's k-filter (1.2 k1) so that filter is a no-op
    band_light = np.asarray(epw.E1_filter[..., 0])
    phi = _random_phi(cfg, 9)
    e_t = _random_transverse(cfg, 10, band=band_light)
    E1 = _field_from_phi(phi, cfg) + e_t
    E0 = jnp.asarray(rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2)))
    E0 = jnp.fft.ifft2(jnp.fft.fft2(E0, axes=(0, 1)) * combined.E0_filter[..., None], axes=(0, 1))
    t = 0.3

    source = combined.unified_source(t, E0, E1)
    l_k, t_k = longitudinal_transverse(source, combined.kx, combined.ky, combined.one_over_k_sq)
    (lx_k, ly_k), (tx_k, ty_k) = (l_k[..., 0], l_k[..., 1]), (t_k[..., 0], t_k[..., 1])

    # longitudinal part as a potential source vs TPD (E_L) + SRS (E_T) / (n/n_env)
    phi_source = _phi_from_source_field(source, cfg)
    ex, ey = epw.phi_k_to_e_fields(phi)
    # the separate SRS source is returned unmasked (the next step's band filter removes
    # the out-of-band product content); the unified source is band-limited when built
    band = jnp.asarray(combined.band)
    expected = (epw.calc_tpd_source(t, phi, ex, ey, E0) + epw.calc_srs_source(E0, e_t) / (n / 0.25)) * band
    np.testing.assert_allclose(
        np.asarray(phi_source), np.asarray(expected), rtol=1e-9, atol=1e-11 * np.abs(expected).max()
    )

    # transverse part vs the Raman coupling, transverse-projected
    lap_phi = jnp.fft.ifft2(-epw.k_sq * phi)
    coupling = raman.srs_coeff * jnp.conj(lap_phi)[..., None] * E0
    _, c_k = longitudinal_transverse(coupling, combined.kx, combined.ky, combined.one_over_k_sq)
    cx_k, cy_k = c_k[..., 0], c_k[..., 1]
    band = np.asarray(band)
    np.testing.assert_allclose(
        np.asarray(tx_k) * band, np.asarray(cx_k) * band, rtol=1e-9, atol=1e-11 * np.abs(cx_k).max()
    )
    np.testing.assert_allclose(
        np.asarray(ty_k) * band, np.asarray(cy_k) * band, rtol=1e-9, atol=1e-11 * np.abs(cy_k).max()
    )


def test_unified_depletion_is_srs_plus_tpd_depletion_at_quarter_critical():
    from adept._lpse2d.core.combined import CombinedSolver, longitudinal_transverse
    from adept._lpse2d.core.spectral_light import SpectralCoupledLight

    cfg = _cfg(density=0.2, pump_depletion=True, periodic=False)
    combined = CombinedSolver(cfg)
    separate = SpectralCoupledLight(cfg)
    assert combined.w1 == pytest.approx(combined.w0 / 2.0)
    phi = _random_phi(cfg, 11)
    e_t = _random_transverse(cfg, 12)
    E1 = _field_from_phi(phi, cfg) + e_t
    t = 0.2
    unified = combined.unified_depletion(t, E1)
    lap_phi = jnp.fft.ifft2(-combined.k_sq * phi)
    srs_dep = separate.srs_depletion_coeff0 * lap_phi[..., None] * e_t
    _, s_k = longitudinal_transverse(srs_dep, combined.kx, combined.ky, combined.one_over_k_sq)
    sx_k, sy_k = s_k[..., 0], s_k[..., 1]
    srs_dep_t = jnp.stack([jnp.fft.ifft2(sx_k), jnp.fft.ifft2(sy_k)], axis=-1)
    expected = srs_dep_t + separate.calc_tpd_depletion(t, phi)
    np.testing.assert_allclose(
        np.asarray(unified), np.asarray(expected), rtol=1e-9, atol=1e-11 * np.abs(expected).max()
    )


def test_combined_configuration_is_validated_like_lpse():
    with pytest.raises(ValueError, match="both on or both off"):
        cfg = _cfg(sources=False)
        cfg["terms"]["epw"]["source"]["srs"] = True
        from adept._lpse2d.core.combined import CombinedSolver

        CombinedSolver(cfg)
    with pytest.raises(ValueError, match="both on or both off"):
        with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
            raw = yaml.safe_load(fi)
        raw["terms"]["epw"]["solver"] = "combined"
        raw["terms"]["light"] = {"solver": "spectral"}
        raw["terms"]["epw"]["source"].update({"tpd": True, "srs": False})
        _finish(raw)
    with pytest.raises(ValueError, match="spectral"):
        with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
            raw = yaml.safe_load(fi)
        raw["terms"]["epw"]["solver"] = "combined"
        raw["terms"]["epw"]["source"].update({"tpd": True, "srs": True})
        raw["terms"]["light"] = {"solver": "fd"}
        _finish(raw)


def test_combined_split_step_runs_with_pump_depletion_iaw_noise_and_reports_transverse_light():
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import get_default_save_func, get_save_quantities

    cfg = _cfg(
        density=0.22, pump_depletion=True, periodic=False, noise=True, iaw={"active": True, "solver": "spectral"}
    )
    cfg = get_save_quantities(cfg)
    step = SplitStep(cfg)
    assert step.epw_solver == "combined"
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "iaw_density": jnp.zeros((nx, ny)),
        "iaw_velocity_divergence": jnp.zeros((nx, ny)),
    }
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    for i in range(6):
        state = step(jnp.asarray(i * cfg["grid"]["dt"]), dict(state), {"drivers": {"E0": pump_args}})
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in state.values())
    E1 = state["E1"].view(jnp.complex128)
    assert float(jnp.max(jnp.abs(E1))) > 0.0  # noise seeded the longitudinal part
    # the derived potential equals i k . E1_k / k^2
    np.testing.assert_allclose(
        np.asarray(state["epw"].view(jnp.complex128)), np.asarray(step.combined.potential(E1)), rtol=1e-12, atol=0
    )
    out = get_default_save_func(cfg)["func"](0.0, state, None)
    assert "e1_sq" in out and bool(jnp.isfinite(out["e1_sq"]))
    # the pump has started to fill the box
    assert float(jnp.max(jnp.abs(state["E0"].view(jnp.complex128)))) > 0.0
