#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Unit tests for the mixed Hermite-Legendre building blocks.

Covers the prediagonalized free-streaming exponentials (Golub-Welsch: streaming
matrix eigenvalues are the Gauss quadrature nodes), the streaming round-trip, and
the analytic parity of the Hermite->Legendre coupling integrals J_{Nh,m} (paper
eqns 27/29/34), which underpin the conservation properties.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pytest
from jax import numpy as jnp

from adept._hermite_legendre_1d.vector_field import (
    StreamingExp1D,
    hermite_cayley_update,
    hermite_force_operator,
    hermite_legendre_coupling_vector,
    hermite_streaming_matrix,
    legendre_cayley_update,
    legendre_constants,
    legendre_force_operator,
    safe_col,
)


def test_hermite_matrix_eigvals_are_gauss_hermite_nodes():
    """Golub-Welsch: eigenvalues of T_H (u=0) equal the Gauss-Hermite nodes."""
    Nh = 24
    T = hermite_streaming_matrix(Nh, u=0.0, alpha=1.0)
    eig = np.sort(np.linalg.eigvalsh(T))
    nodes = np.sort(np.polynomial.hermite.hermgauss(Nh)[0])
    np.testing.assert_allclose(eig, nodes, atol=1e-10)


def test_legendre_matrix_eigvals_are_gauss_legendre_nodes():
    """Eigenvalues of T_L equal the Gauss-Legendre nodes mapped to [v_a, v_b]."""
    Nl, v_a, v_b = 20, -2.5, 2.5
    leg = legendre_constants(Nl, v_a, v_b)
    eig = np.sort(np.linalg.eigvalsh(np.asarray(leg["T_L"])))
    nodes = np.sort(0.5 * (v_b - v_a) * np.polynomial.legendre.leggauss(Nl)[0] + 0.5 * (v_a + v_b))
    np.testing.assert_allclose(eig, nodes, atol=1e-10)


def test_streaming_exp_identity_and_roundtrip():
    """exp(L*0) is identity, and exp(L*s) exp(L*-s) returns the original array."""
    Nh, Nx = 16, 8
    kx = jnp.fft.fftfreq(Nx) * Nx * 2 * jnp.pi / (2 * jnp.pi)
    T = hermite_streaming_matrix(Nh, u=0.3, alpha=1.4)
    exp = StreamingExp1D(T, prefactor=-1j * 1.4, kx_1d=kx)

    rng = np.random.default_rng(0)
    C = jnp.asarray(rng.standard_normal((Nh, Nx)) + 1j * rng.standard_normal((Nh, Nx)))

    np.testing.assert_allclose(np.asarray(exp.apply(C, 0.0)), np.asarray(C), atol=1e-12)
    back = exp.apply(exp.apply(C, 0.37), -0.37)
    np.testing.assert_allclose(np.asarray(back), np.asarray(C), atol=1e-10)


def test_legendre_derivative_matrix_sparsity():
    """sigma_{m,i} is strictly lower triangular and nonzero only when (m-i) is odd."""
    Nl, v_a, v_b = 12, -3.0, 3.0
    deriv = np.asarray(legendre_constants(Nl, v_a, v_b)["deriv"])
    for m in range(Nl):
        for i in range(Nl):
            if i >= m or (m - i) % 2 == 0:
                assert deriv[m, i] == 0.0, (m, i)
            else:
                assert deriv[m, i] != 0.0, (m, i)


def test_collision_profile_conserves_first_three_moments():
    """safe_col is exactly zero for n=0,1,2 (mass/momentum/energy preserved)."""
    col = np.asarray(safe_col(32))
    assert col[0] == 0.0 and col[1] == 0.0 and col[2] == 0.0
    assert np.all(col[3:] > 0.0)
    assert col[-1] == pytest.approx(1.0)  # normalized to 1 at n = N-1


@pytest.mark.parametrize("Nh,vanish", [(25, (0, 2)), (24, (1,))])
def test_coupling_integral_parity(Nh, vanish):
    """J_{Nh,m} parity on a symmetric domain with u=0 (paper eqns 27/29/34):
    odd Nh  -> J_{Nh,0} = J_{Nh,2} = 0 (mass & energy);
    even Nh -> J_{Nh,1} = 0 (momentum)."""
    J = np.asarray(
        hermite_legendre_coupling_vector(
            Nh, Nl=10, alpha=np.sqrt(2.0), u=0.0, v_a=-3.0, v_b=3.0, enforce_conservation=False
        )
    )
    for m in vanish:
        assert abs(J[m]) < 1e-9, (Nh, m, J[m])
    # a non-vanishing entry should actually be present (sanity)
    assert np.max(np.abs(J)) > 1e-6


def test_enforce_conservation_zeros_first_three():
    J = np.asarray(
        hermite_legendre_coupling_vector(
            30, Nl=10, alpha=np.sqrt(2.0), u=0.0, v_a=-2.5, v_b=2.5, enforce_conservation=True
        )
    )
    assert np.all(J[:3] == 0.0)


def test_hermite_cayley_matches_dense_solve():
    Nh, Nx = 9, 5
    alpha, dt = 1.4, 0.17
    rng = np.random.default_rng(2)
    C = rng.standard_normal((Nh, Nx)) + 1j * rng.standard_normal((Nh, Nx))
    E = rng.uniform(-0.8, 0.8, Nx)
    ladder = np.sqrt(2.0 * np.arange(Nh)) / alpha

    actual = np.asarray(hermite_cayley_update(jnp.asarray(C), jnp.asarray(E), dt, jnp.asarray(ladder)))
    G = hermite_force_operator(Nh, alpha)
    expected = np.empty_like(C)
    eye = np.eye(Nh)
    for ix in range(Nx):
        s = 0.5 * dt * E[ix]
        expected[:, ix] = np.linalg.solve(eye - s * G, (eye + s * G) @ C[:, ix])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("all_mode_penalty", [False, True])
def test_legendre_structured_cayley_matches_dense_solve(all_mode_penalty):
    Nl, Nx = 10, 4
    v_a, v_b, dt = -2.0, 3.0, 0.11
    rng = np.random.default_rng(3)
    B = rng.standard_normal((Nl, Nx)) + 1j * rng.standard_normal((Nl, Nx))
    E = rng.uniform(-0.7, 0.7, Nx)
    leg = legendre_constants(Nl, v_a, v_b)
    gamma_vec = np.full(Nl, 0.5)
    if not all_mode_penalty:
        gamma_vec[:3] = 0.0

    actual = np.asarray(
        legendre_cayley_update(
            jnp.asarray(B),
            jnp.asarray(E),
            dt,
            leg["deriv"],
            jnp.asarray(gamma_vec),
            leg["xi_a"],
            leg["xi_b"],
            v_b - v_a,
        )
    )
    G = legendre_force_operator(leg["deriv"], gamma_vec, leg["xi_a"], leg["xi_b"], v_b - v_a)
    expected = np.empty_like(B)
    eye = np.eye(Nl)
    for ix in range(Nx):
        s = 0.5 * dt * E[ix]
        expected[:, ix] = np.linalg.solve(eye - s * G, (eye + s * G) @ B[:, ix])
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_all_mode_legendre_force_is_skew_and_cayley_preserves_norm():
    Nl, Nx = 16, 6
    v_a, v_b = 4.0, 15.0
    leg = legendre_constants(Nl, v_a, v_b)
    gamma_vec = np.full(Nl, 0.5)
    G = legendre_force_operator(leg["deriv"], gamma_vec, leg["xi_a"], leg["xi_b"], v_b - v_a)
    np.testing.assert_allclose(G + G.T, 0.0, atol=2e-13)
    eigenvalues, eigenvectors = np.linalg.eigh(1j * G)

    rng = np.random.default_rng(4)
    B = rng.standard_normal((Nl, Nx)) + 1j * rng.standard_normal((Nl, Nx))
    E = rng.uniform(-2.0, 2.0, Nx)
    updated = np.asarray(
        legendre_cayley_update(
            jnp.asarray(B),
            jnp.asarray(E),
            0.4,
            leg["deriv"],
            jnp.asarray(gamma_vec),
            leg["xi_a"],
            leg["xi_b"],
            v_b - v_a,
            jnp.asarray(eigenvalues),
            jnp.asarray(eigenvectors),
        )
    )
    np.testing.assert_allclose(np.sum(np.abs(updated) ** 2, axis=0), np.sum(np.abs(B) ** 2, axis=0), rtol=2e-11)

    eye = np.eye(Nl)
    expected = np.empty_like(B)
    for ix in range(Nx):
        s = 0.5 * 0.4 * E[ix]
        expected[:, ix] = np.linalg.solve(eye - s * G, (eye + s * G) @ B[:, ix])
    np.testing.assert_allclose(updated, expected, rtol=2e-11, atol=2e-11)
