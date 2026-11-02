"""Tests for the reciprocal TPD pump-depletion coupling."""

from copy import deepcopy

import numpy as np
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
    """The new pump term is the exact discrete reciprocal of the existing TPD term.

    For coupling terms alone, d/dt [|E0|^2 + (wp0/w0)|E_epw|^2] = 0.
    This locks the coefficient, complex-conjugation convention, and carrier phase.
    """
    from adept._lpse2d.core.epw import SpectralEPWSolver
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg()
    pump = CoupledLight(cfg)
    epw = SpectralEPWSolver(cfg)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]

    rng = np.random.default_rng(7401)
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    # The EPW state is projected into this band every step. Keeping the synthetic
    # state in the same band makes the filtered source operator self-adjoint.
    phi_k *= np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    phi_k = jnp.asarray(phi_k)
    E0_y = jnp.asarray(rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny)))
    t = 0.037

    _, ey = epw.phi_k_to_e_fields(phi_k)
    tpd_epw_rhs = epw.calc_tpd_source(t, phi_k, ey, E0_y)
    tpd_pump_rhs = pump.calc_tpd_depletion(t, phi_k)

    pump_energy_rate = 2.0 * jnp.real(jnp.mean(jnp.conj(E0_y) * tpd_pump_rhs))
    epw_energy_rate = 2.0 * jnp.real(jnp.vdot(phi_k * epw.k_sq, tpd_epw_rhs)) / (nx * ny) ** 2
    total_rate = pump_energy_rate + epw.wp0 / epw.w0 * epw_energy_rate

    scale = max(abs(float(pump_energy_rate)), abs(float(epw.wp0 / epw.w0 * epw_energy_rate)))
    assert abs(float(total_rate)) < 1.0e-7 * scale
