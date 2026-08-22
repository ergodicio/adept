"""Equation-level regression tests for the anisotropic FLM collision operator."""

import lineax as lx
import numpy as np
import pytest
from jax import numpy as jnp

from adept.vfp1d.fokker_planck import FLMCollisions
from adept.vfp1d.grid import Grid


def _make_collisions(nv: int = 32, nl: int = 3) -> FLMCollisions:
    grid = Grid(
        xmin=0.0,
        xmax=1.0,
        nx=1,
        tmin=0.0,
        tmax=0.1,
        dt=0.01,
        nv=nv,
        vmax=6.0,
        nl=nl,
    )
    return FLMCollisions(Z=1.0, nuee_coeff=1.0, grid=grid, full_aniso_ee=True)


def _raw_tridiagonal_coefficients(collisions: FLMCollisions, f0: jnp.ndarray, il: int):
    """Return row-indexed coefficients before packing the two off-diagonals."""
    i0 = collisions.calc_ros_i(f0, power=0.0)
    jm1 = collisions.calc_ros_j(f0, power=-1.0)
    i2 = collisions.calc_ros_i(f0, power=2.0)

    v = collisions.grid.v[None, :]
    dv = collisions.grid.dv
    tri_i1 = (-i2 + 2.0 * jm1 + 3.0 * i0) / 3.0
    tri_i2 = (i2 + jm1) / 3.0

    diffusion = tri_i2 / v / dv**2
    drift = tri_i1 / v**2 / (2.0 * dv)
    lower = diffusion - drift
    upper = diffusion + drift
    diag = 8.0 * jnp.pi * f0 - 2.0 * diffusion - 0.5 * il * (il + 1) * tri_i1 / v**3
    return diag, lower, upper


def test_flm_b_coefficients_match_tzoufras_operator():
    collisions = _make_collisions(nl=4)
    ell = np.arange(1, 5)
    ll = ell * (ell + 1) / 2.0
    denom_plus = (2 * ell + 1) * (2 * ell + 3)
    denom_minus = (2 * ell + 1) * (2 * ell - 1)

    np.testing.assert_allclose(collisions.b1[1:], (-ll - (ell + 1)) / denom_plus)
    np.testing.assert_allclose(collisions.b2[1:], (-ll + (ell + 2)) / denom_plus)
    np.testing.assert_allclose(collisions.b3[1:], (ll + (ell - 1)) / denom_minus)
    np.testing.assert_allclose(collisions.b4[1:], (ll - ell) / denom_minus)

    # In particular, the f10 J_{-2} coefficient is 2/15, not 4/15.
    np.testing.assert_allclose(collisions.b2[1], 2.0 / 15.0)


@pytest.mark.parametrize("il", [1, 2, 3])
def test_ee_tridiagonal_packs_row_coefficients_and_origin_parity(il):
    collisions = _make_collisions(nv=24, nl=3)
    f0 = jnp.exp(-(collisions.grid.v[None, :] ** 2))
    diag, lower, upper = collisions.get_ee_diagonal_contrib(f0, il=il)
    raw_diag, raw_lower, raw_upper = _raw_tridiagonal_coefficients(collisions, f0, il)

    expected_diag = raw_diag.at[:, 0].add((-1.0 if il % 2 else 1.0) * raw_lower[:, 0])
    np.testing.assert_allclose(diag, expected_diag)
    np.testing.assert_allclose(lower, raw_lower[:, 1:])
    np.testing.assert_allclose(upper, raw_upper[:, :-1])

    matrix = np.asarray(
        lx.TridiagonalLinearOperator(diagonal=diag[0], lower_diagonal=lower[0], upper_diagonal=upper[0]).as_matrix()
    )
    np.testing.assert_allclose(np.diag(matrix, k=-1), np.asarray(raw_lower[0, 1:]))
    np.testing.assert_allclose(np.diag(matrix, k=1), np.asarray(raw_upper[0, :-1]))


def _drifting_maxwellian_collision_residual(nv: int) -> float:
    """Maximum C_ee[f0, f10] for the infinitesimal-drift Maxwellian mode."""
    collisions = _make_collisions(nv=nv, nl=1)
    v = collisions.grid.v
    dv = collisions.grid.dv
    f0 = (jnp.exp(-(v**2)) / jnp.pi**1.5)[None, :]
    f10 = 2.0 * v[None, :] * f0

    diag, lower, upper = collisions.get_ee_diagonal_contrib(f0, il=1)
    operator = lx.TridiagonalLinearOperator(diagonal=diag[0], lower_diagonal=lower[0], upper_diagonal=upper[0])

    padded_f0 = jnp.concatenate([f0[:, 1::-1], f0], axis=1)
    d2dv2 = 0.5 / v * jnp.gradient(jnp.gradient(padded_f0, dv, axis=1), dv, axis=1)[:, 2:]
    ddv = v**-2 * jnp.gradient(padded_f0, dv, axis=1)[:, 2:]
    offdiagonal = collisions.get_ee_offdiagonal_contrib(
        None,
        f10,
        {"ddvf0": ddv, "d2dv2f0": d2dv2, "il": 1},
    )
    residual = operator.mv(f10[0]) + offdiagonal[0]
    return float(jnp.max(jnp.abs(residual)))


def test_drifting_maxwellian_collision_residual_converges_at_velocity_origin():
    """The regular l=1 mode must not develop a 1/dv collision singularity."""
    coarse = _drifting_maxwellian_collision_residual(64)
    fine = _drifting_maxwellian_collision_residual(256)

    assert fine < 0.5 * coarse, (coarse, fine)
    assert fine < 0.02, fine
