"""Stability of the EPW coupling inside the coupled (pump-depletion) light sub-step.

The staggered real/imaginary light update is a leapfrog only for a RHS that is i times
a real operator. The E0 <-> E1 coupling through the complex ``laplacian phi`` is not,
so the explicit scheme amplifies the light pair by 1 + sin^2(arg L) (Omega dt_l)^2 / 2
per sub-step (Omega = e|L| / (4 me sqrt(w0 w1))), whereas the ``rotation`` scheme is an
exact action-conserving map. Both facts are checked here on the solver's own
coefficients, without running a simulation.
"""

import numpy as np
import pytest
import yaml


def _coupled_light():
    from adept import ergoExo
    from adept._lpse2d.core.light import CoupledLight

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["terms"]["light"] = {"pump_depletion": True, "coupling": "rotation"}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"  # the evolved pump needs an exit
    cfg["mlflow"]["experiment"] = "test-lpse2d-light-coupling"
    exo = ergoExo()
    exo.setup(cfg)
    return CoupledLight(exo.cfg)


def _explicit_substep_matrix(light, lap, dt_l):
    """One coupling-only sub-step of the explicit scheme at a single grid point, as a
    real 4x4 matrix on (Re E0, Im E0, Re E1, Im E1)."""
    A, B = light.depletion_coeff0, light.srs_coeff

    def step(E0, E1):
        E0 = E0 + dt_l * np.real(A * lap * E1)
        E1 = E1 + dt_l * np.real(B * np.conj(lap) * E0)
        E0 = E0 + 1j * dt_l * np.imag(A * lap * E1)
        E1 = E1 + 1j * dt_l * np.imag(B * np.conj(lap) * E0)
        return E0, E1

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
    light = _coupled_light()
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
    E0e = E0 + tau * light.depletion_coeff0 * lap[..., None] * E1
    E1e = E1 + tau * light.srs_coeff * np.conj(lap)[..., None] * E0
    assert np.allclose(np.asarray(E0n), E0e, rtol=0, atol=1e-8 * np.abs(E0).max())
    assert np.allclose(np.asarray(E1n), E1e, rtol=0, atol=1e-8 * np.abs(E1).max())


def _coupled_light_tpd():
    from adept import ergoExo
    from adept._lpse2d.core.epw import SpectralEPWSolver
    from adept._lpse2d.core.light import CoupledLight

    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["terms"]["epw"]["source"]["srs"] = True
    cfg["terms"]["epw"]["source"]["tpd"] = True
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["light"] = {"pump_depletion": True, "tpd_depletion": True}
    cfg["drivers"]["E0"]["params"] = {"phases": {"seed": 42}}  # the evolved pump's injector needs them
    cfg["mlflow"]["experiment"] = "test-lpse2d-light-coupling"
    exo = ergoExo()
    exo.setup(cfg)
    return CoupledLight(exo.cfg), SpectralEPWSolver(exo.cfg), exo.cfg


def test_tpd_depletion_conserves_pump_plus_epw_energy():
    """The TPD potential source (epw.py) and the new pump term (light.py) must move
    energy between Int|E0|^2 and Int|grad phi|^2 without creating any, for any EPW
    state and any pump: d/dt Int(|E0|^2 + |grad phi|^2) = 0 for the pair alone."""
    import jax.numpy as jnp

    light, epw, cfg = _coupled_light_tpd()
    rng = np.random.default_rng(2)
    nx, ny = light.k_sq.shape
    # a band-limited, k=0-free EPW potential and a y-polarised pump with structure
    phi_k = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    phi_k = phi_k * np.asarray(epw.low_pass_filter) * np.asarray(epw.zero_mask)
    E0 = np.zeros((nx, ny, 2), dtype=complex)
    E0[..., 1] = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    t = 0.37
    # EPW side: dphi_k/dt from the coded TPD source; energy Int|grad phi|^2 in real space
    ex, ey = epw.phi_k_to_e_fields(jnp.asarray(phi_k))
    src_k = epw.calc_tpd_source(t, jnp.asarray(phi_k), ey, jnp.asarray(E0[..., 1]))
    dex, dey = epw.phi_k_to_e_fields(src_k)
    ex_np, ey_np = np.asarray(ex), np.asarray(ey)
    dW_epw = 2.0 * np.real(np.sum(np.conj(ex_np) * np.asarray(dex) + np.conj(ey_np) * np.asarray(dey)))
    # pump side: the new term alone
    tpd_dep = np.asarray(light.tpd_depletion_term(jnp.asarray(phi_k)))
    dE0y = tpd_dep * np.exp(1j * light.tpd_delta_w * t)
    dW_pump = 2.0 * np.real(np.sum(np.conj(E0[..., 1]) * dE0y))
    assert dW_epw != 0.0
    assert dW_pump == pytest.approx(-dW_epw, rel=1e-10)
