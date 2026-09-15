"""Physical-collision gates for the mixed Hermite--Legendre solver."""

import jax.numpy as jnp
import numpy as np
import pytest

from adept._hermite_legendre_1d.modules import BaseHermiteLegendre1D, _project_legendre
from adept._hermite_legendre_1d.vector_field import (
    ConservingBGKCollision1D,
    DoughertyCollision1D,
    _hermite_function_values,
    _legendre_basis_values,
    legendre_constants,
    mixed_moment_matrix,
)


def _collision_setup(nx: int = 3, model: str = "bgk"):
    Nh, Nl = 8, 16
    alpha, u = np.sqrt(2.0), 0.0
    v_a, v_b = 2.0, 10.0
    leg = legendre_constants(Nl, v_a, v_b)
    moments = mixed_moment_matrix(alpha, u, leg["T_L"], v_b - v_a)
    common = dict(
        nu=1.0,
        Nh=Nh,
        alpha=alpha,
        basis_u=u,
        moment_matrix=moments,
    )
    if model == "bgk":
        collision = ConservingBGKCollision1D(**common)
    elif model == "dougherty":
        collision = DoughertyCollision1D(T_L=leg["T_L"], deriv=leg["deriv"], **common)
    else:
        raise ValueError(model)
    return collision, moments, Nh, Nl, nx, alpha, v_a, v_b


def _state(C: np.ndarray, B: np.ndarray) -> dict:
    Ck = jnp.fft.fft(jnp.asarray(C), axis=-1, norm="forward").astype(jnp.complex128)
    Bk = jnp.fft.fft(jnp.asarray(B), axis=-1, norm="forward").astype(jnp.complex128)
    return {"Ck": Ck.view(jnp.float64), "Bk": Bk.view(jnp.float64)}


def _real_coefficients(state: dict) -> tuple[np.ndarray, np.ndarray]:
    Ck = state["Ck"].view(jnp.complex128)
    Bk = state["Bk"].view(jnp.complex128)
    C = jnp.fft.ifft(Ck, axis=-1, norm="forward").real
    B = jnp.fft.ifft(Bk, axis=-1, norm="forward").real
    return np.asarray(C), np.asarray(B)


def _moments(moment_matrix, C: np.ndarray, B: np.ndarray) -> np.ndarray:
    low = np.stack([C[0], C[1], C[2], B[0], B[1], B[2]], axis=0)
    return np.asarray(moment_matrix) @ low


def test_bgk_preserves_a_sampled_maxwellian():
    collision, _, Nh, Nl, nx, alpha, _, _ = _collision_setup()
    C = np.zeros((Nh, nx))
    C[0] = np.array([0.8, 1.0, 1.2]) / alpha
    B = np.zeros((Nl, nx))

    C_new, B_new = _real_coefficients(collision.apply(_state(C, B), 0.5))

    np.testing.assert_allclose(C_new, C, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(B_new, B, rtol=0.0, atol=1.0e-13)


def test_dougherty_preserves_a_sampled_maxwellian():
    collision, _, Nh, Nl, nx, alpha, _, _ = _collision_setup(model="dougherty")
    C = np.zeros((Nh, nx))
    C[0] = np.array([0.8, 1.0, 1.2]) / alpha
    B = np.zeros((Nl, nx))

    C_new, B_new = _real_coefficients(collision.apply(_state(C, B), 0.5))

    np.testing.assert_allclose(C_new, C, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(B_new, B, rtol=0.0, atol=1.0e-13)


def test_bgk_maxwellian_recurrence_has_requested_moments():
    collision, moment_matrix, _, Nl, _, _, _, _ = _collision_setup()
    density = jnp.asarray([0.8, 1.0, 1.3])
    mean_v = jnp.asarray([-0.3, 0.1, 0.7])
    temperature = jnp.asarray([0.6, 1.0, 1.8])

    C_maxwellian = np.asarray(collision._maxwellian_coefficients(density, mean_v, temperature))
    B = np.zeros((Nl, density.size))
    actual = _moments(moment_matrix, C_maxwellian, B)
    expected = np.stack([density, density * mean_v, 0.5 * density * (temperature + mean_v**2)])

    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-15)


def test_bgk_does_not_project_between_bases():
    collision, _, Nh, Nl, nx, alpha, _, _ = _collision_setup()
    C = np.zeros((Nh, nx))
    C[0] = 1.0 / alpha
    C[3] = np.array([1.0e-3, -2.0e-3, 3.0e-3])
    B = np.zeros((Nl, nx))
    step = 0.2

    C_new, B_new = _real_coefficients(collision.apply(_state(C, B), step))

    np.testing.assert_allclose(B_new, 0.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(C_new[3], np.exp(-step) * C[3], rtol=2.0e-13, atol=1.0e-15)


def test_bgk_conserves_local_moments_and_relaxes_a_beam():
    collision, moment_matrix, Nh, Nl, nx, alpha, v_a, v_b = _collision_setup()
    C = np.zeros((Nh, nx))
    C[0] = 0.99 / alpha

    beam_density = 0.01
    beam = _project_legendre(
        lambda v: beam_density / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (v - 6.0) ** 2),
        Nl,
        v_a,
        v_b,
    )
    B = np.repeat(beam[:, None], nx, axis=1)
    C[3, 1] = 1.0e-3  # make the three spatial points genuinely independent

    before = _moments(moment_matrix, C, B)
    state_new = collision.apply(_state(C, B), 0.2)
    C_new, B_new = _real_coefficients(state_new)
    after = _moments(moment_matrix, C_new, B_new)

    np.testing.assert_allclose(after, before, rtol=0.0, atol=2.0e-13)

    v = np.linspace(-8.0, 12.0, 512)
    psi = _hermite_function_values(Nh, v, 0.0, alpha)
    xi = _legendre_basis_values(Nl, v, v_a, v_b)
    xi[:, (v < v_a) | (v > v_b)] = 0.0
    f = C.T @ psi + B.T @ xi
    f_new = C_new.T @ psi + B_new.T @ xi

    density = before[0]
    mean_v = before[1] / density
    temperature = 2.0 * before[2] / density - mean_v**2
    maxwellian = (
        density[:, None]
        / np.sqrt(2.0 * np.pi * temperature[:, None])
        * np.exp(-((v[None, :] - mean_v[:, None]) ** 2) / (2.0 * temperature[:, None]))
    )

    assert np.linalg.norm(f_new - maxwellian) < np.linalg.norm(f - maxwellian)
    assert np.linalg.norm(C_new - C) > 0.0  # field-particle back reaction reaches the Hermite bulk
    assert np.linalg.norm(B_new - B) > 0.0


def test_dougherty_conserves_local_moments_and_relaxes_a_beam():
    collision, moment_matrix, Nh, Nl, nx, alpha, v_a, v_b = _collision_setup(model="dougherty")
    C = np.zeros((Nh, nx))
    C[0] = 0.99 / alpha
    beam_density = 0.01
    beam = _project_legendre(
        lambda v: beam_density / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (v - 6.0) ** 2),
        Nl,
        v_a,
        v_b,
    )
    B = np.repeat(beam[:, None], nx, axis=1)

    before = _moments(moment_matrix, C, B)
    C_new, B_new = _real_coefficients(collision.apply(_state(C, B), 0.2))
    after = _moments(moment_matrix, C_new, B_new)

    np.testing.assert_allclose(after, before, rtol=0.0, atol=2.0e-13)
    assert np.linalg.norm(C_new - C) > 0.0
    assert np.linalg.norm(B_new - B) > 0.0


@pytest.mark.parametrize("model", ["bgk", "dougherty"])
def test_physical_collision_is_wired_through_solver_config(model):
    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": 2.0 * np.pi,
            "alpha": np.sqrt(2.0),
            "u": 0.0,
            "v_a": 2.0,
            "v_b": 10.0,
            "gamma": 0.5,
            "nu_H": 0.0,
            "nu_L": 0.0,
            "collisions": {"model": model, "nu": 0.2},
            "enforce_conservation": True,
            "field": False,
        },
        "grid": {"Nx": 4, "Nh": 8, "Nl": 16, "tmax": 0.05, "dt": 0.05, "integrator": "split"},
        "initialization": {
            "type": "bump-on-tail",
            "eps": 0.0,
            "n_beam": 0.01,
            "v_drift": 6.0,
            "v_th": 1.0,
        },
        "save": {"default": {"t": {"nt": 2}}},
        "units": {},
    }
    module = BaseHermiteLegendre1D(cfg)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    data = module(trainable_modules={})["solver result"].ys["default"]

    for name in ("mass", "momentum", "energy"):
        values = np.asarray(data[name])
        np.testing.assert_allclose(values, values[0], rtol=0.0, atol=2.0e-12)
