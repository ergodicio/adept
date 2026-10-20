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
    (lx_k, ly_k), (tx_k, ty_k) = longitudinal_transverse(source, combined.kx, combined.ky, combined.one_over_k_sq)

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
    _, (cx_k, cy_k) = longitudinal_transverse(coupling, combined.kx, combined.ky, combined.one_over_k_sq)
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
    _, (sx_k, sy_k) = longitudinal_transverse(srs_dep, combined.kx, combined.ky, combined.one_over_k_sq)
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


# ---- noise kick at the light sub-step (PR #372 review) ---------------------------------------------
# Tolerances fixed before the runs: the steady state is a mean over ~700 retained modes x 150
# correlated samples, so both the n_sub-independence ratio and the agreement with the analytic
# random-walk level of diagnostics.expected_noise_energy are set at 15 %. Before the fix the kick
# was built for the EPW step and applied every sub-step: ratio 8 / 2 sub-steps ~ 4 for both models.


def _noise_cfg(n_sub, model, landau=True):
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    # n = n_env: the combined field's damping exp(-nu_R dt (n/n_env)^2) is then the uniform rate the
    # noise kick is balanced against; nu_R is the Raman absorption at the combined carrier,
    # terms.light.absorption (wp0/w0)^2 = 80/ps x n_env = 20/ps (the EPW collisions do not act on it)
    cfg["density"] = {"basis": "uniform", "val": 0.25}
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
            "light_substeps": n_sub,
        }
    )
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": landau}
    cfg["terms"]["epw"]["density_gradient"] = True
    cfg["terms"]["epw"]["source"].update(
        {"noise": True, "noise_model": model, "noise_amplitude": 1.0, "noise_seed": 3, "tpd": False, "srs": False}
    )
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False, "absorption": 80.0}
    return _finish(cfg)


def _combined_noise_steady_state(cfg, n_steps=450, n_warm=300):
    """Mean of sum_k k^2 |phi_k|^2 over the last n_steps - n_warm EPW steps from a quiet start."""
    import jax

    from adept._lpse2d.core.combined import CombinedSolver

    solver = CombinedSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    y = {"E0": jnp.zeros((nx, ny, 2), complex), "E1": jnp.zeros((nx, ny, 2), complex)}
    step = jax.jit(lambda t, y: solver(t, y, {}, E0_fn=lambda t: y["E0"]))
    samples = []
    for i in range(n_steps):
        E0, E1, phi_k = step(i * cfg["grid"]["dt"], y)
        y = {"E0": E0, "E1": E1}
        if i >= n_warm:
            samples.append(float(jnp.sum(solver.k_sq * jnp.abs(phi_k) ** 2)))
    return solver, np.mean(samples)


def test_combined_thermal_noise_needs_only_the_raman_absorption():
    """The combined field's damping is the Raman absorption, so thermal noise with the EPW Landau
    damping and collisions both off and terms.light.absorption > 0 is a valid combined deck (PR #372
    review): SplitStep sets up (before the fix it also built the never-stepped separate EPW solver,
    whose thermal noise refused the deck for lack of EPW damping), and the combined noise kick is
    nonzero and balanced against nu_R."""
    from adept._lpse2d.core.epw import noise_step
    from adept._lpse2d.core.vector_field import SplitStep

    cfg = _noise_cfg(1, "thermal", landau=False)
    step = SplitStep(cfg)
    assert step.epw is None
    d = cfg["units"]["derived"]
    assert noise_step(cfg)[1] == pytest.approx(80.0 * (d["wp0"] / d["w0"]) ** 2, rel=1e-12)
    assert float(jnp.max(jnp.abs(step.combined.noise_kick))) > 0.0


@pytest.mark.parametrize("model", ["thermal", "flat"])
def test_combined_noise_steady_state_is_independent_of_the_light_substeps(model):
    """The combined solver adds its noise every light sub-step, so the kick is built for that step
    (LwSolver::addNoise_combinedSolver_spectral(dt)); the seed level cannot depend on n_sub."""
    levels = {}
    for n_sub in (2, 8):
        cfg = _noise_cfg(n_sub, model)
        solver, levels[n_sub] = _combined_noise_steady_state(cfg)
        assert solver.n_sub == n_sub
    assert levels[8] / levels[2] == pytest.approx(1.0, rel=0.15)


@pytest.mark.parametrize("model", ["thermal", "flat"])
def test_combined_noise_matches_the_growth_fit_floor(model):
    """diagnostics.expected_noise_energy integrates the kick the combined solver applies (at its
    light sub-step), so the late-time floor equals the simulated steady state."""
    from adept._lpse2d.core.epw import noise_kick_spectrum, noise_step
    from adept._lpse2d.diagnostics import expected_noise_energy

    cfg = _noise_cfg(4, model)
    dt_noise, nu = noise_step(cfg)
    assert dt_noise == pytest.approx(cfg["grid"]["dt"] / 4, rel=1e-12)
    assert nu == pytest.approx(20.0, rel=1e-12)
    solver, level = _combined_noise_steady_state(cfg)
    np.testing.assert_allclose(
        np.asarray(solver.noise_kick), noise_kick_spectrum(cfg, dt=dt_noise, nu_coll=nu), rtol=1e-12, atol=0
    )
    derived = cfg["units"]["derived"]
    grid = cfg["grid"]
    prefactor = 0.25 * grid["dx"] * derived["x_norm"] * derived["e_norm"] ** 2 / (grid["nx"] * grid["ny"] ** 2)
    assert prefactor * level / expected_noise_energy(cfg, 1.0e3) == pytest.approx(1.0, rel=0.15)


# ---- Raman seed on the combined path (PR #372 review) -----------------------------------------------
# Tolerances fixed before the run, the same as the separate solver's seed test
# (test_spectral_light.test_spectral_seed_injector_launches_the_exact_amplitude_leftward): bulk
# amplitude 1 %, ripple 2 %, wavenumber 1 %. Before the fix the combined solver never read
# drivers.E1, so with the pump off and noise off E1 stayed identically zero. That test's 1 %
# rightward-leakage bound is not carried over: in this geometry (n/n_env = 0.8, region > 3.2 sigma
# right of the injector) the separate spectral solver itself leaves 1.4 % at 180 steps (combined
# 1.3 %), so the bound measures the geometry, not the wiring. The seed source is instead checked
# to equal the separate solver's (an identity at n_env = n_c/4, where w1 = wp0; rtol 1e-12).


def test_combined_seed_injector_launches_the_exact_amplitude_leftward():
    """drivers.E1 seeds the transverse part of the combined field at the combined carrier wp0
    (LPSE's Raman class has W1 = wpe in combined mode): the smooth injector launches
    |E1_y| = A / eps^(1/4), eps = 1 - n/n_env, as a leftward wave at k = sqrt(wp0^2 - wpe^2)/c,
    and the purely transverse seed leaves the derived potential at zero."""
    from adept._lpse2d.core.vector_field import SplitStep

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["units"]["envelope density"] = 0.25
    cfg["density"] = {"basis": "uniform", "val": 0.2}
    cfg["grid"].update({"xmax": "20um", "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um"})
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False}
    cfg["drivers"]["E1"] = {"intensity": "1.0e+12W/cm^2", "delta_omega": 0.0, "turn_on_time": "10fs", "offset": "7um"}
    cfg = _finish(cfg)
    del cfg["drivers"]["E0"]

    step = SplitStep(cfg)
    assert step.epw_solver == "combined" and step.combined.seed_enabled
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
    }
    args = {"drivers": {"E1": cfg["drivers"]["E1"]["derived"]}}
    for i in range(180):
        state = step(jnp.asarray(i * cfg["grid"]["dt"]), dict(state), args)

    e1y = np.asarray(state["E1"].view(jnp.complex128))[:, 0, 1]
    x = np.asarray(cfg["grid"]["x"])
    derived = cfg["units"]["derived"]
    n_over_env = 0.2 / 0.25
    eps = 1.0 - n_over_env
    k_seed = derived["wp0"] * np.sqrt(eps) / derived["c"]
    bulk = slice(int(np.argmin(np.abs(x - 5.0))), int(np.argmin(np.abs(x - 10.0))))
    expected = cfg["drivers"]["E1"]["derived"]["amplitude"] / eps**0.25
    np.testing.assert_allclose(np.mean(np.abs(e1y[bulk])), expected, rtol=0.01)
    assert np.std(np.abs(e1y[bulk])) < 0.02 * expected
    k_measured = np.polyfit(x[bulk], np.unwrap(np.angle(e1y[bulk])), 1)[0]
    assert k_measured < 0
    np.testing.assert_allclose(abs(k_measured), k_seed, rtol=0.01)
    # the seed is transverse: no longitudinal (EPW) part, so the derived potential stays zero
    assert float(jnp.max(jnp.abs(state["epw"]))) < 1e-12 * expected


def test_combined_seed_source_equals_the_separate_seed_source_at_quarter_critical():
    """At n_env = n_c/4 the combined carrier wp0 equals the Raman envelope frequency w1 = w0 - wp0,
    so the combined seed source is the separate spectral solver's, term for term."""
    from adept._lpse2d.core.combined import CombinedSolver
    from adept._lpse2d.core.spectral_light import SpectralRamanLight

    def cfg_for(solver):
        with open("tests/test_lpse2d/configs/srs.yaml") as fi:
            cfg = deepcopy(yaml.safe_load(fi))
        cfg["units"]["envelope density"] = 0.25
        cfg["density"] = {"basis": "uniform", "val": 0.2}
        cfg["grid"].update({"xmax": "20um", "tmax": "10fs", "ymax": "0.5um", "ymin": "-0.5um"})
        cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
        both = solver == "combined"
        cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": both, "srs": True})
        if both:
            cfg["terms"]["epw"]["solver"] = "combined"
        cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False}
        cfg["drivers"]["E1"] = {
            "intensity": "1.0e+12W/cm^2",
            "delta_omega": 0.003,
            "turn_on_time": "10fs",
            "offset": "7um",
            "yw": "0.6um",
        }
        return _finish(cfg)

    combined = CombinedSolver(cfg_for("combined"))
    cfg_s = cfg_for("separate")
    separate = SpectralRamanLight(cfg_s)
    assert combined.wp0 == pytest.approx(separate.w1, rel=1e-12)
    assert combined.i1 == separate.i1
    seed_args = cfg_s["drivers"]["E1"]["derived"]
    for t in (0.004, 0.017, 0.05):
        np.testing.assert_allclose(
            np.asarray(combined.calc_seed_source(t, seed_args)),
            np.asarray(separate.calc_seed_source(t, seed_args)),
            rtol=1e-12,
            atol=0,
        )


def test_combined_seed_injector_refuses_an_evanescent_position():
    """The combined field's light is evanescent where n >= n_env (carrier wp0); setup says so."""
    from adept._lpse2d.core.combined import CombinedSolver

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["grid"].update({"xmax": "20um", "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um"})
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False}
    cfg["drivers"]["E1"] = {"intensity": "1.0e+12W/cm^2", "offset": "7um"}
    cfg = _finish(cfg)  # uniform n = n_env = 0.2
    with pytest.raises(ValueError, match="evanescent"):
        CombinedSolver(cfg)


# ---- the combined carrier wp0 outside the combined solver (PR #372 review round 4) -----------------


def _off_quarter_cfg(solver, n_env, density, *, iaw=None, absorption=False, wavelength=None, seed=False):
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["units"]["envelope density"] = n_env
    if wavelength is not None:
        cfg["units"]["laser_wavelength"] = wavelength
    cfg["density"] = {"basis": "uniform", "val": density}
    cfg["grid"].update({"xmax": "20um", "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um"})
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    combined = solver == "combined"
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": combined, "srs": True})
    cfg["terms"]["epw"]["solver"] = solver
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False, "absorption": absorption}
    if iaw is not None:
        cfg["terms"]["iaw"] = iaw
    if seed:
        cfg["drivers"]["E1"] = {
            "intensity": "1.0e+12W/cm^2",
            "delta_omega": 0.0,
            "turn_on_time": "10fs",
            "offset": "7um",
        }
    return _finish(cfg)


def test_combined_iaw_drive_thermal_term_and_raman_absorption_use_the_combined_carrier():
    """With terms.epw.solver: combined the IAW sees the Raman light at the combined carrier wp0: its
    ponderomotive coefficient is 1/wp0^2, the thermal-filamentation Raman term's critical density is
    n_env and its absorption rate the NRL value at lambda_0 w0/wp0 (a numeric rate scales by
    (wp0/w0)^2). At n_env = 0.2 (w1 = 1.24 wp0). Exact identities, rtol 1e-12, fixed before the
    run; the separate solver keeps w1."""
    from adept._lpse2d.core.iaw import IonAcousticWave
    from adept._lpse2d.core.raman import light_absorption_rates

    iaw = {
        "active": True,
        "solver": "spectral",
        "damping": {"landau": 0.1, "collisions": 0.0},
        "thermal_filamentation": {"laser": True, "raman": True},
    }
    cfg_c = _off_quarter_cfg("combined", 0.2, 0.15, iaw=iaw, absorption=True)
    cfg_s = _off_quarter_cfg("separate", 0.2, 0.15, iaw=iaw, absorption=True)
    d = cfg_c["units"]["derived"]
    assert abs(d["w1"] / d["wp0"] - 1.0) > 0.2
    nx, ny = cfg_c["grid"]["nx"], cfg_c["grid"]["ny"]
    amp = 2.0e-3
    e1 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128).at[..., 1].set(amp)
    zero_phi = jnp.zeros((nx, ny), dtype=jnp.complex128)
    zero_e0 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    for cfg, w_expected in ((cfg_c, d["wp0"]), (cfg_s, d["w1"])):
        wave = IonAcousticWave(cfg)
        assert wave.w_raman == w_expected
        drive = np.asarray(wave.ponderomotive_drive(zero_phi, zero_e0, e1))
        np.testing.assert_allclose(drive, wave.ponderomotive_prefactor * amp**2 / w_expected**2, rtol=1e-12)

    # NRL Raman rate at the combined wavelength lambda_0 w0/wp0 (the pump rate of a deck at that wavelength)
    rate0_c, rate1_c = light_absorption_rates(cfg_c)
    lam_c = 0.351 * d["w0"] / d["wp0"]
    np.testing.assert_allclose(
        rate1_c,
        light_absorption_rates(_off_quarter_cfg("combined", 0.2, 0.15, absorption=True, wavelength=f"{lam_c}um"))[0],
        rtol=1e-12,
    )
    # the thermal Raman/laser coefficient ratio: (rate1 / rate0) / nc1^2, nc1 = n_env combined
    wave_c = IonAcousticWave(cfg_c)
    ratio = np.asarray(wave_c.thermal_coeff["raman"]) / np.asarray(wave_c.thermal_coeff["laser"])
    np.testing.assert_allclose(ratio, (rate1_c / rate0_c) / 0.2**2, rtol=1e-12)
    # a numeric pump rate scales by the carrier ratio squared
    for solver, w in (("combined", d["wp0"]), ("separate", d["w1"])):
        rate0, rate1 = light_absorption_rates(_off_quarter_cfg(solver, 0.2, 0.15, absorption=0.5))
        assert rate0 == 0.5
        np.testing.assert_allclose(rate1, 0.5 * (w / d["w0"]) ** 2, rtol=1e-12)


def test_seeded_combined_deck_above_the_separate_cutoff_sets_up_and_runs():
    """n_env = 0.3: the combined light is cut off at n_env = 0.3 nc, the separate Raman light at
    (w1/w0)^2 = 0.205 nc. A seeded box at 0.25 nc is valid for the combined solver; before the fix
    the separate SRS k-filter check refused the box and the never-stepped separate Raman solver
    refused the seed. Through setup and SplitStep (tolerance fixed before the run): the separate
    light solvers are not built and after 80 steps the seed reaches the bulk at A / eps^(1/4),
    eps = 1 - n/n_env, to 5 %. The separate solver still refuses the same box."""
    from adept._lpse2d.core.vector_field import SplitStep

    cfg = _off_quarter_cfg("combined", 0.3, 0.25, seed=True)
    del cfg["drivers"]["E0"]
    step = SplitStep(cfg)
    assert step.raman is None and step.coupled_light is None and step.combined.seed_enabled
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
    }
    args = {"drivers": {"E1": cfg["drivers"]["E1"]["derived"]}}
    for i in range(80):
        state = step(jnp.asarray(i * cfg["grid"]["dt"]), dict(state), args)
    e1y = np.asarray(state["E1"].view(jnp.complex128))[:, 0, 1]
    x = np.asarray(cfg["grid"]["x"])
    x_inj = float(x[step.combined.i1])
    bulk = (x > x_inj - 4.0) & (x < x_inj - 2.0)
    eps = 1.0 - 0.25 / 0.3
    np.testing.assert_allclose(
        np.mean(np.abs(e1y[bulk])), cfg["drivers"]["E1"]["derived"]["amplitude"] / eps**0.25, rtol=0.05
    )

    with pytest.raises(ValueError, match="Raman critical density"):
        _off_quarter_cfg("separate", 0.3, 0.25, seed=True)


def test_combined_dissipation_diagnostic_uses_the_combined_damping():
    """epw_dissipation in combined mode is the combined solver's own damping of the EPW (PR #372 review):
    the Landau rate in k-space plus the Raman absorption nu_R (n/n_env + dn)^2 on the EPW energy
    density in x-space -- not the EPW collisions, which the combined solver ignores. Uniform box
    (n = 0.15 nc, n_env = 0.2, dn = 0.05), terms.light.absorption 300/ps, EPW collisions on: with
    Landau off, epw_dissipation = 2 (2 - n/n_env) L_c epw_energy, L_c = (1 - exp(-2 nu_R (n/n_env + dn)^2
    dt)) / dt; with Landau on, plus the k-space Landau term (Parseval). Exact identities, rtol 1e-12,
    fixed before the run. Before the fix the first case reported the EPW collisional loss."""
    from adept._lpse2d.core.epw import analytic_landau_rate
    from adept._lpse2d.helpers import get_default_save_func, get_save_quantities

    def run(landau):
        with open("tests/test_lpse2d/configs/srs.yaml") as fi:
            cfg = deepcopy(yaml.safe_load(fi))
        cfg["units"]["envelope density"] = 0.2
        cfg["density"] = {"basis": "uniform", "val": 0.15}
        cfg["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um"})
        cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
        cfg["terms"]["epw"]["damping"] = {"collisions": True, "landau": landau}
        cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
        cfg["terms"]["epw"]["solver"] = "combined"
        cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False, "absorption": 300.0}
        cfg = get_save_quantities(_finish(cfg))
        nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
        rng = np.random.default_rng(13)
        phi_k = 1e-3 * (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny)))
        phi_k = phi_k * np.asarray(cfg["grid"]["low_pass_filter_grid"])
        zeros = jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64)
        state = {
            "epw": jnp.asarray(phi_k).view(jnp.float64),
            "E0": zeros,
            "E1": zeros,
            "iaw_density": jnp.full((nx, ny), 0.05),
        }
        out = get_default_save_func(cfg)["func"](0.0, state, None)
        return cfg, phi_k, out

    cfg, _, out = run(landau=False)
    d, grid = cfg["units"]["derived"], cfg["grid"]
    dt = grid["dt"]
    assert d.get("nu_coll", 0.0) > 0.0  # the EPW collisions are configured, and must not count
    nu_r = 300.0 * (d["wp0"] / d["w0"]) ** 2
    loss_c = (1.0 - np.exp(-2.0 * nu_r * (0.75 + 0.05) ** 2 * dt)) / dt
    factor = 2.0 * (2.0 - 0.75)
    assert float(out["epw_dissipation"]) == pytest.approx(factor * loss_c * float(out["epw_energy"]), rel=1e-12)

    cfg, phi_k, out = run(landau=True)
    grid = cfg["grid"]
    nx, ny = grid["nx"], grid["ny"]
    k_sq = np.asarray(grid["kx"])[:, None] ** 2 + np.asarray(grid["ky"])[None, :] ** 2
    loss_l = (1.0 - np.exp(-2.0 * np.asarray(analytic_landau_rate(cfg)) * dt)) / dt
    assert np.any(loss_l * np.abs(phi_k) > 0.0)
    prefactor = 0.25 * grid["dx"] * d["x_norm"] * d["e_norm"] ** 2
    landau_part = prefactor * factor * np.sum(loss_l * k_sq * np.abs(phi_k) ** 2) / (nx * ny**2)
    want = landau_part + factor * loss_c * float(out["epw_energy"])
    assert float(out["epw_dissipation"]) == pytest.approx(want, rel=1e-12)


def test_the_combined_field_is_damped_at_the_raman_absorption():
    """LPSE damps the combined field with the Raman light's absorption at the combined carrier and the
    field's own critical density n_env, not with the EPW collisions (PR #372 review). Collisions
    off, terms.light.absorption 300/ps (Raman rate at wp0: 300 (wp0/w0)^2), n = 0.15 nc, n_env = 0.2,
    a periodic box: one EPW step multiplies a transverse E1 plane wave by exp(-nu_R dt (n/n_env)^2),
    with a uniform iaw_density dn by exp(-nu_R dt (n/n_env + dn)^2), and noise_step returns nu_R.
    Exact identities, rtol 1e-10, fixed before the run; before the fix the field kept its amplitude."""
    from adept._lpse2d.core.combined import CombinedSolver
    from adept._lpse2d.core.epw import noise_step

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["units"]["envelope density"] = 0.2
    cfg["density"] = {"basis": "uniform", "val": 0.15}
    cfg["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um"})
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}
    cfg["terms"]["epw"]["damping"] = {"collisions": False, "landau": False}
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False, "absorption": 300.0}
    cfg = _finish(cfg)
    d = cfg["units"]["derived"]
    nu_r = 300.0 * (d["wp0"] / d["w0"]) ** 2
    assert noise_step(cfg)[1] == pytest.approx(nu_r, rel=1e-12)
    solver = CombinedSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    x = np.asarray(cfg["grid"]["x"])
    k = float(np.asarray(cfg["grid"]["kx"])[8])
    zero = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    plane = zero.at[..., 1].set(jnp.asarray(1.0e-3 * np.exp(1j * k * x)[:, None] * np.ones((1, ny))))
    epw = jnp.zeros((nx, ny), dtype=jnp.complex128)
    dt = cfg["grid"]["dt"]
    for dn, n_eff in ((None, 0.75), (0.05, 0.75 + 0.05)):
        y = {"epw": epw, "E0": zero, "E1": plane}
        if dn is not None:
            y["iaw_density"] = jnp.full((nx, ny), dn)
        _, E1, _ = solver(0.0, y, {}, lambda t: zero)
        decay = np.abs(np.asarray(E1)[..., 1]) / 1.0e-3
        np.testing.assert_allclose(decay, np.exp(-nu_r * dt * n_eff**2), rtol=1e-10)
