"""Stability of the SRS exchange inside the coupled (pump-depletion) light sub-step.

The staggered real/imaginary light update is a leapfrog only for a RHS that is i times
a real operator. The E0 <-> E1 exchange through the complex ``laplacian phi`` is not,
so the explicit scheme amplifies the light pair by 1 + sin^2(arg L) (Omega dt_l)^2 / 2
per sub-step (Omega = e|L| / (4 me sqrt(w0 w1))), whereas the ``rotation`` scheme is an
exact action-conserving map. Both facts are checked here on the solver's own
coefficients, without running a simulation, together with the wiring of the rotation
scheme and the optional light filter into ``CoupledLight.__call__``.
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


def _make_cfg(coupling="rotation", light_filter=None):
    """Small evolved-pump SRS box (uniform density at the envelope density)."""
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"boundary_width": "2um", "xmax": "16um", "tmax": "10fs"})
    cfg["terms"]["light"] = {"pump_depletion": True, "coupling": coupling}
    if light_filter is not None:
        cfg["terms"]["light"]["filter"] = light_filter
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"  # the evolved pump needs an exit
    return _finish_cfg(cfg)


def _make_cfg_tpd_only(coupling="rotation"):
    """Evolved pump with the TPD source only (no Raman light), as in test_tpd_depletion."""
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
    cfg["terms"]["light"] = {"pump_depletion": True, "coupling": coupling}
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": False})
    cfg["terms"]["epw"]["density_gradient"] = False
    return _finish_cfg(cfg)


def _coupled_light(**kwargs):
    from adept._lpse2d.core.light import CoupledLight

    return CoupledLight(_make_cfg(**kwargs))


def _pump_args(cfg):
    ny = cfg["grid"]["ny"]
    return {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }


def _random_fields(cfg, seed):
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E0 = rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2))
    E1 = 0.3 * (rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2)))
    return jnp.asarray(E0), jnp.asarray(E1)


def _explicit_substep_matrix(light, lap, dt_l):
    """One exchange-only sub-step of the explicit scheme at a single grid point, as a
    real 4x4 matrix on (Re E0, Im E0, Re E1, Im E1)."""
    A, B = light.srs_depletion_coeff0, light.srs_coeff

    # the real-part updates use the pair from the start of the sub-step (light.py
    # evaluates both RHS from the same (E0, E1)), so rebuild that ordering exactly
    def step_exact_order(E0, E1):
        k0 = A * lap * E1
        k1 = B * np.conj(lap) * E0
        E0 = E0 + dt_l * np.real(k0)
        E1 = E1 + dt_l * np.real(k1)
        k0 = A * lap * E1
        k1 = B * np.conj(lap) * E0
        return E0 + 1j * dt_l * np.imag(k0), E1 + 1j * dt_l * np.imag(k1)

    M = np.zeros((4, 4))
    for j in range(4):
        v = np.zeros(4)
        v[j] = 1.0
        E0, E1 = step_exact_order(v[0] + 1j * v[1], v[2] + 1j * v[3])
        M[:, j] = [E0.real, E0.imag, E1.real, E1.imag]
    return M


def test_explicit_coupling_substep_is_unstable_for_complex_laplacian():
    light = _coupled_light(coupling="explicit")
    omega_dt = 0.1
    lap_mag = omega_dt / (light.omega_prefactor * light.dt_l)
    for theta in (0.0, 0.25, 0.5):
        lap = lap_mag * np.exp(1j * np.pi * theta)
        rho = np.max(np.abs(np.linalg.eigvals(_explicit_substep_matrix(light, lap, light.dt_l))))
        expected = 1.0 + np.sin(np.pi * theta) ** 2 * omega_dt**2 / 2.0
        assert rho == pytest.approx(expected, abs=2e-4), (theta, rho, expected)
    # the real-laplacian case is a leapfrog (neutrally stable), the imaginary one is not
    rho_real = np.max(np.abs(np.linalg.eigvals(_explicit_substep_matrix(light, lap_mag + 0j, light.dt_l))))
    rho_imag = np.max(np.abs(np.linalg.eigvals(_explicit_substep_matrix(light, 1j * lap_mag, light.dt_l))))
    assert rho_real == pytest.approx(1.0, abs=1e-12)
    assert rho_imag > 1.0 + 0.9 * omega_dt**2 / 2.0


def test_rotation_coupling_conserves_light_action():
    light = _coupled_light()
    rng = np.random.default_rng(0)
    nx, ny = light.k_sq.shape
    E0 = rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2))
    E1 = 0.3 * (rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2)))
    lap = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    # exchange angles from tiny to order unity, well past the explicit limit
    for tau_scale in (1e-3, 1e-1, 1.0, 5.0):
        tau = tau_scale / (light.omega_prefactor * np.abs(lap).max())
        E0n, E1n = light.couple(E0, E1, lap, tau)
        E0n, E1n = np.asarray(E0n), np.asarray(E1n)
        action_before = light.w1 * np.sum(np.abs(E0) ** 2) + light.w0 * np.sum(np.abs(E1) ** 2)
        action_after = light.w1 * np.sum(np.abs(E0n) ** 2) + light.w0 * np.sum(np.abs(E1n) ** 2)
        assert action_after == pytest.approx(action_before, rel=1e-12)
        # pointwise too (the rotation is local)
        pw_before = light.w1 * np.sum(np.abs(E0) ** 2, axis=-1) + light.w0 * np.sum(np.abs(E1) ** 2, axis=-1)
        pw_after = light.w1 * np.sum(np.abs(E0n) ** 2, axis=-1) + light.w0 * np.sum(np.abs(E1n) ** 2, axis=-1)
        assert np.allclose(pw_after, pw_before, rtol=1e-12, atol=0.0)


def test_rotation_matches_explicit_coupling_for_small_angles():
    """For Omega tau -> 0 the exact rotation reduces to the explicit Euler kick
    E0 += tau A L E1, E1 += tau B L* E0 (first order), so the two schemes agree there."""
    light = _coupled_light()
    rng = np.random.default_rng(1)
    nx, ny = light.k_sq.shape
    E0 = rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2))
    E1 = rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2))
    lap = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    tau = 1e-4 / (light.omega_prefactor * np.abs(lap).max())
    E0n, E1n = light.couple(E0, E1, lap, tau)
    E0e = E0 + tau * light.srs_depletion_coeff0 * lap[..., None] * E1
    E1e = E1 + tau * light.srs_coeff * np.conj(lap)[..., None] * E0
    assert np.allclose(np.asarray(E0n), E0e, rtol=0, atol=1e-8 * np.abs(E0).max())
    assert np.allclose(np.asarray(E1n), E1e, rtol=0, atol=1e-8 * np.abs(E1).max())


def test_rotation_scheme_reduces_to_explicit_without_epw():
    """With phi = 0 the exchange rotation is the identity, so one EPW step of the
    Strang-split scheme must reproduce the explicit scheme (propagation, detuning,
    injector and absorbers are shared and untouched)."""
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg(coupling="explicit")
    explicit = CoupledLight(cfg)
    rotation = CoupledLight(_make_cfg(coupling="rotation"))
    assert explicit.coupling == "explicit" and rotation.coupling == "rotation"
    E0, E1 = _random_fields(cfg, 3)
    phi_k = jnp.zeros(explicit.k_sq.shape, dtype=jnp.complex128)
    args = _pump_args(cfg)
    E0e, E1e = explicit(0.0, E0, E1, phi_k, args, None)
    E0r, E1r = rotation(0.0, E0, E1, phi_k, args, None)
    assert bool(jnp.all(jnp.isfinite(E0e))) and bool(jnp.all(jnp.isfinite(E1e)))
    np.testing.assert_allclose(np.asarray(E0r), np.asarray(E0e), rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(np.asarray(E1r), np.asarray(E1e), rtol=1e-13, atol=0.0)


def test_rotation_scheme_conserves_light_action_over_an_epw_step_without_propagation():
    """Exchange + rotation only: the finite-EPW rotation halves commute with nothing but
    themselves, so with the propagation switched off (dt_l -> the rotation angle alone)
    the full sub-step loop is a pure rotation and must conserve w1|E0|^2 + w0|E1|^2."""
    light = _coupled_light()
    E0, E1 = _random_fields(light.cfg, 4)
    rng = np.random.default_rng(5)
    nx, ny = light.k_sq.shape
    lap = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    # angle per half rotation of order 0.3 rad: far beyond the explicit scheme's limit
    lap = lap * 0.3 / (light.omega_prefactor * 0.5 * light.dt_l * np.abs(lap).max())
    E0n, E1n = E0, E1
    for _ in range(light.n_sub):
        E0n, E1n = light.couple(E0n, E1n, jnp.asarray(lap), 0.5 * light.dt_l)
        E0n, E1n = light.couple(E0n, E1n, jnp.asarray(lap), 0.5 * light.dt_l)
    before = light.w1 * np.sum(np.abs(np.asarray(E0)) ** 2) + light.w0 * np.sum(np.abs(np.asarray(E1)) ** 2)
    after = light.w1 * np.sum(np.abs(np.asarray(E0n)) ** 2) + light.w0 * np.sum(np.abs(np.asarray(E1n)) ** 2)
    assert after == pytest.approx(before, rel=1e-11)


def test_rotation_with_srs_off_leaves_raman_field_zero():
    """TPD-only pump depletion has no E0 <-> E1 exchange: the rotation setting must not
    seed E1 from the pump, and the TPD pump term still acts."""
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg_tpd_only(coupling="rotation")
    light = CoupledLight(cfg)
    assert light.tpd_enabled and not light.srs_enabled and light.coupling == "rotation"
    E0, _ = _random_fields(cfg, 6)
    E1 = jnp.zeros_like(E0)
    rng = np.random.default_rng(7)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k = jnp.asarray(phi_k * np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"]))
    E0n, E1n = light(0.0, E0, E1, phi_k, _pump_args(cfg), None)
    assert bool(jnp.all(jnp.isfinite(E0n)))
    assert float(jnp.max(jnp.abs(E1n))) == 0.0
    # the TPD pump term is part of the staggered RHS and changes the pump
    E0z, _ = light(0.0, E0, E1, jnp.zeros_like(phi_k), _pump_args(cfg), None)
    assert float(jnp.max(jnp.abs(E0n - E0z))) > 0.0


def test_light_filter_masks_both_fields_after_the_sub_steps():
    """terms.light.filter multiplies the FFT of the advanced fields by an isotropic
    mask |k| <= filter * pi/dx once per EPW step and does nothing else."""
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg(coupling="rotation")
    plain = CoupledLight(cfg)
    filtered = CoupledLight(_make_cfg(coupling="rotation", light_filter=0.5))
    assert plain.light_filter is None
    mask = np.asarray(filtered.light_filter)[..., 0]
    kx = np.asarray(cfg["grid"]["kx"])
    ky = np.asarray(cfg["grid"]["ky"])
    k_mag = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    np.testing.assert_array_equal(mask, np.where(k_mag <= 0.5 * np.pi / cfg["grid"]["dx"], 1.0, 0.0))
    assert 0.0 < mask.mean() < 1.0

    E0, E1 = _random_fields(cfg, 8)
    rng = np.random.default_rng(9)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi_k = jnp.asarray(rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny)))
    args = _pump_args(cfg)
    E0p, E1p = plain(0.0, E0, E1, phi_k, args, None)
    E0f, E1f = filtered(0.0, E0, E1, phi_k, args, None)
    for unfiltered, out in ((E0p, E0f), (E1p, E1f)):
        expected = np.fft.ifft2(np.fft.fft2(np.asarray(unfiltered), axes=(0, 1)) * mask[..., None], axes=(0, 1))
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_invalid_coupling_and_filter_are_rejected():
    from adept._lpse2d.core.light import CoupledLight

    cfg = _make_cfg(coupling="explicit")
    cfg["terms"]["light"]["coupling"] = "implicit"
    with pytest.raises(ValueError, match=r"terms\.light\.coupling"):
        CoupledLight(cfg)
    cfg["terms"]["light"]["coupling"] = "rotation"
    cfg["terms"]["light"]["filter"] = 0.0
    with pytest.raises(ValueError, match=r"terms\.light\.filter"):
        CoupledLight(cfg)


def test_config_validation_of_light_options():
    """helpers.get_solver_quantities validates terms.light through LightModel and
    refuses the coupled-solver options on the prescribed-pump path."""
    import pydantic

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        base = yaml.safe_load(fi)
    base["grid"].update({"boundary_width": "2um", "xmax": "16um", "tmax": "10fs"})

    cfg = deepcopy(base)
    cfg["terms"]["light"] = {"pump_depletion": True, "coupling": "implicit"}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    with pytest.raises(pydantic.ValidationError):
        _finish_cfg(cfg)

    cfg = deepcopy(base)
    cfg["terms"]["light"] = {"pump_depletion": False, "coupling": "rotation"}
    with pytest.raises(ValueError, match=r"terms\.light\.pump_depletion"):
        _finish_cfg(cfg)

    cfg = deepcopy(base)
    cfg["terms"]["light"] = {"pump_depletion": True, "coupling": "rotation", "filter": 0.7}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg = _finish_cfg(cfg)
    assert cfg["terms"]["light"] == {"pump_depletion": True, "coupling": "rotation", "filter": 0.7}
