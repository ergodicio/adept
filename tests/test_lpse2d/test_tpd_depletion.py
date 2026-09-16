"""Tests for the reciprocal TPD pump-depletion coupling."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _make_cfg():
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)

    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": 0.25}
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
    cfg["terms"]["light"] = {"pump_depletion": True}
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": False})
    cfg["terms"]["epw"]["density_gradient"] = False

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def test_tpd_only_pump_depletion_configuration_is_supported():
    """TPD does not need SRS merely to select the evolved-pump path."""
    from adept._lpse2d.core.light import CoupledLight
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import get_default_save_func

    cfg = _make_cfg()
    solver = CoupledLight(cfg)

    assert solver.tpd_enabled
    assert not solver.srs_enabled
    assert cfg["grid"]["light_substeps"] >= 1

    # A TPD-only evolved pump still exposes the same four-channel net-flux budget
    # used by post-processing; the Raman channels are identically zero.
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
    }
    out = get_default_save_func(cfg)["func"](0.0, state, None)
    assert all(k in out for k in ("incident_flux", "transmitted_flux", "reflected_flux", "backrefl_flux"))
    assert out["reflected_flux"] == 0.0 and out["backrefl_flux"] == 0.0

    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    advanced = SplitStep(cfg)(jnp.asarray(0.0), state, {"drivers": {"E0": pump_args}})
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in advanced.values())


def test_combined_tpd_srs_iaw_deck_builds_all_fluid_couplings():
    """The production example composes both instabilities, pump feedback, and IAWs."""
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    with open("configs/envelope-2d/tpd-srs-iaw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)

    step = SplitStep(cfg)
    assert step.epw.tpd_enabled and step.epw.srs_enabled
    assert step.pump_depletion and step.coupled_light.tpd_enabled and step.coupled_light.srs_enabled
    assert step.iaw is not None
    assert step.hpe is not None and step.hpe.is_2d


def test_tpd_accepts_box_averaged_2d_particle_feedback():
    """TPD uses one 2D2V ensemble and angle-resolved damping for oblique modes."""
    from adept._lpse2d.core.hpe import HybridParticleEvolution, load_particles
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import (
        get_default_save_func,
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    with open("configs/envelope-2d/tpd-srs-iaw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"].update({"dx": "0.1um", "ymin": "-0.8um", "ymax": "0.8um"})
    cfg["terms"]["hpe"].update({"n_particles": 128, "n_angles": 8, "gather_refine": 1})
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)

    state = load_particles(cfg)
    hpe = HybridParticleEvolution(cfg)
    assert hpe.is_2d
    assert state["u_e"].shape == (128, 2)
    assert state["epw_hist"].shape == (8, cfg["terms"]["hpe"]["nv"])
    assert hpe.mask_res.shape == (cfg["grid"]["nx"], cfg["grid"]["ny"])

    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    state.update(
        {
            "epw": np.zeros((nx, ny), dtype=np.complex128),
            "E0": np.zeros((nx, ny, 2), dtype=np.complex128),
            "E1": np.zeros((nx, ny, 2), dtype=np.complex128),
            "iaw_density": np.zeros((nx, ny), dtype=np.float64),
            "iaw_velocity_divergence": np.zeros((nx, ny), dtype=np.float64),
        }
    )
    packed_state = {key: value.view(np.float64) for key, value in state.items()}
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }
    advanced = SplitStep(cfg)(jnp.asarray(0.0), packed_state, {"drivers": {"E0": pump_args}})
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in advanced.values())
    diagnostics = get_default_save_func(cfg)["func"](jnp.asarray(0.0), advanced, None)
    assert diagnostics["hpe_hist"].shape == (8, cfg["terms"]["hpe"]["nv"])
    assert bool(jnp.isfinite(diagnostics["hpe_mean_energy_keV"]))


def test_tpd_reciprocal_coupling_conserves_wave_energy():
    """The pump term is LPSE's (Follett Eqs 53/55): twice the EPW-side TPD coefficient.

    For the coupling terms alone, d/dt [|E0|^2 + (2 wp0/w0)|E_epw|^2] = 0, i.e. the total
    wave energy at envelope density n_c/4 (two plasmons at wp0 per pump photon at w0).
    This locks the coefficient, complex-conjugation convention, and carrier phase. The
    transverse projection is switched off here because the random test pump is not
    divergence-free; it is checked separately in test_lpse_parity.py.
    """
    from adept._lpse2d.core.epw import SpectralEPWSolver
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg()
    cfg["terms"]["light"]["tpd_projection"] = False
    pump = CoupledLight(cfg)
    epw = SpectralEPWSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]

    rng = np.random.default_rng(7401)
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    # The EPW state is projected into this band every step. Keeping the synthetic
    # state in the same band makes the filtered source operator self-adjoint.
    phi_k *= np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    phi_k = jnp.asarray(phi_k)
    E0 = jnp.asarray(rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2)))
    t = 0.037

    ex, ey = epw.phi_k_to_e_fields(phi_k)
    tpd_epw_rhs = epw.calc_tpd_source(t, phi_k, ex, ey, E0)
    tpd_pump_rhs = pump.calc_tpd_depletion(t, phi_k)
    assert tpd_pump_rhs.shape == (nx, ny, 2)

    pump_energy_rate = 2.0 * jnp.real(jnp.mean(jnp.sum(jnp.conj(E0) * tpd_pump_rhs, axis=-1)))
    epw_energy_rate = 2.0 * jnp.real(jnp.vdot(phi_k * epw.k_sq, tpd_epw_rhs)) / (nx * ny) ** 2
    weight = 2.0 * epw.wp0 / epw.w0
    total_rate = pump_energy_rate + weight * epw_energy_rate

    scale = max(abs(float(pump_energy_rate)), abs(float(weight * epw_energy_rate)))
    assert abs(float(total_rate)) < 1.0e-7 * scale
    # the pump coefficient is exactly twice the EPW source coefficient
    assert pump.tpd_depletion_coeff0 == pytest.approx(2.0 * epw.tpd_prefactor * epw.w0 / epw.w0)
    assert pump.tpd_depletion_coeff0 == pytest.approx(1j * epw.e / (2.0 * epw.w0 * epw.me))


# ------------------------------------------ the projected term's energy ledger (plan 2 N.4) --


def _random_transverse(cfg, seed):
    """Random divergence-free (nx, ny, 2) complex field on the retained band, x-space."""
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    k_sq = kx**2 + ky**2
    mask = np.asarray(cfg["grid"]["low_pass_filter_grid"]) * np.where(k_sq > 0, 1.0, 0.0)
    a_k = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * mask
    k_safe = np.sqrt(np.where(k_sq > 0, k_sq, 1.0))
    return jnp.asarray(np.stack([np.fft.ifft2(-ky / k_safe * a_k), np.fft.ifft2(kx / k_safe * a_k)], axis=-1))


def _longitudinal_part(field, cfg):
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    k_sq = kx**2 + ky**2
    fx_k, fy_k = np.fft.fft2(np.asarray(field[..., 0])), np.fft.fft2(np.asarray(field[..., 1]))
    longitudinal = (kx * fx_k + ky * fy_k) / np.where(k_sq > 0, k_sq, 1.0)
    return np.stack([np.fft.ifft2(kx * longitudinal), np.fft.ifft2(ky * longitudinal)], axis=-1)


def _pair_energy_rates(cfg, phi_k, E0, t=0.037):
    """(pump rate, weighted EPW rate) of the TPD coupling pair for the given states."""
    from adept._lpse2d.core.epw import SpectralEPWSolver
    from adept._lpse2d.core.light import CoupledLight

    pump = CoupledLight(cfg)
    epw = SpectralEPWSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    ex, ey = epw.phi_k_to_e_fields(phi_k)
    tpd_epw_rhs = epw.calc_tpd_source(t, phi_k, ex, ey, E0)
    tpd_pump_rhs = pump.calc_tpd_depletion(t, phi_k)
    pump_rate = 2.0 * float(jnp.real(jnp.mean(jnp.sum(jnp.conj(E0) * tpd_pump_rhs, axis=-1))))
    epw_rate = 2.0 * float(jnp.real(jnp.vdot(phi_k * epw.k_sq, tpd_epw_rhs))) / (nx * ny) ** 2
    return pump_rate, (2.0 * epw.wp0 / epw.w0) * epw_rate


def test_projected_tpd_term_conserves_pair_energy_for_a_transverse_pump():
    """Plan 2 N.4. With ``tpd_projection`` on (the default, LPSE's ``makeExyzDivE``) the pair
    energy ``|E0|^2 + (2 wp0/w0)|E_h|^2`` is conserved to round-off for a divergence-free
    pump, exactly as the unprojected term conserves it for any pump: the projector is
    self-adjoint, so ``<E0, P_T F> = <P_T E0, F> = <E0, F>`` when ``P_T E0 = E0``. For a pump
    with a longitudinal part the mismatch is precisely that part's contribution to the EPW
    source -- the EPW is driven by the whole pump, the projected term can only take energy
    from its transverse part."""
    cfg = _make_cfg()
    assert cfg["terms"]["light"].get("tpd_projection", True)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(7402)
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k = jnp.asarray(phi_k * np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"]))

    E0_t = _random_transverse(cfg, 7403)
    pump_rate, epw_rate = _pair_energy_rates(cfg, phi_k, E0_t)
    scale = max(abs(pump_rate), abs(epw_rate))
    assert scale > 0.0
    assert abs(pump_rate + epw_rate) < 1.0e-9 * scale, (pump_rate, epw_rate)

    # a pump with longitudinal content: mismatch == the longitudinal part's EPW-source rate
    E0_l = jnp.asarray(_longitudinal_part(_random_transverse(cfg, 7404) + 0.0, cfg))  # zero: check helper
    assert float(jnp.abs(E0_l).max()) < 1.0e-9 * float(jnp.abs(E0_t).max())
    E0_any = jnp.asarray(rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2)))
    E0_long = jnp.asarray(_longitudinal_part(E0_any, cfg))
    pump_any, epw_any = _pair_energy_rates(cfg, phi_k, E0_any)
    _, epw_long = _pair_energy_rates(cfg, phi_k, E0_long)
    assert abs(pump_any + epw_any) > 1.0e-3 * max(abs(pump_any), abs(epw_any))  # not conserved
    assert pump_any + epw_any == pytest.approx(epw_long, rel=1.0e-9)
    # ... and the projected pump term does not see the longitudinal part at all
    pump_long, _ = _pair_energy_rates(cfg, phi_k, E0_long)
    assert abs(pump_long) < 1.0e-9 * abs(epw_long)


def test_fd_pump_propagation_and_the_transverse_fields_projection():
    """The FD pump propagator is the discrete curl-curl ``d2y(E0x) - dxdy(E0y)``,
    ``d2x(E0y) - dxdy(E0x)``; with the compact 3-point second differences and the centred
    cross difference its discrete divergence is not identically zero. Free propagation (no
    EPW, no injector, absorber neutralised) of a y-uniform ``E0y`` plane wave -- the shape the
    injector launches -- generates no longitudinal part; a random divergence-free
    band-limited pump acquires a percent-level one within one EPW step (4 % on this 0.1 um
    grid, k0 dx = 1.8), which no propagator moves and which the EPW sources see while the
    projected TPD term cannot return energy from it. ``terms.light.transverse_fields``
    (default on, plan 2 N.4) projects the evolved fields once per EPW step and removes it."""
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg()
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E1 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128)
    pump_args = {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.zeros((1, ny)),  # injector off
        "phases": jnp.zeros((1, ny)),
    }

    def propagate(transverse_fields, E0):
        cfg_i = deepcopy(cfg)
        cfg_i["terms"]["light"]["transverse_fields"] = transverse_fields
        light = CoupledLight(cfg_i)
        assert light.transverse_fields is transverse_fields
        light.sub_boundary = jnp.ones_like(light.sub_boundary)
        E0_new, _ = light(0.0, E0, E1, phi_k, pump_args, None, None)
        return E0_new, np.linalg.norm(_longitudinal_part(E0_new, cfg)) / np.linalg.norm(np.asarray(E0_new))

    x = np.asarray(cfg["grid"]["x"])
    k0 = cfg["units"]["derived"]["w0"] / cfg["units"]["derived"]["c"]
    plane = np.zeros((nx, ny, 2), dtype=np.complex128)
    plane[..., 1] = (1.0e-6 * np.exp(1j * k0 * x))[:, None]
    E0_new, long_plane = propagate(False, jnp.asarray(plane))
    # curl-curl of a y-uniform E0y has no x part (round-off only) and no longitudinal part
    assert float(jnp.abs(E0_new[..., 0]).max()) < 1.0e-14 * float(jnp.abs(E0_new[..., 1]).max())
    assert long_plane < 1.0e-12, long_plane
    E0_proj, _ = propagate(True, jnp.asarray(plane))
    np.testing.assert_allclose(
        np.asarray(E0_proj), np.asarray(E0_new), rtol=0, atol=1e-14 * float(jnp.abs(E0_new).max())
    )

    E0 = _random_transverse(cfg, 7405) * 1.0e-6
    E0_raw, long_raw = propagate(False, E0)
    assert 0.01 < long_raw < 0.1, long_raw
    E0_proj, long_proj = propagate(True, E0)
    assert long_proj < 1.0e-12, long_proj
    # the projection removes only the longitudinal part
    np.testing.assert_allclose(
        np.asarray(E0_proj), np.asarray(E0_raw) - _longitudinal_part(E0_raw, cfg), rtol=0, atol=1e-18
    )
