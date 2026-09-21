"""Discrete electric-work identities, including the radial truncation boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_vfp2d.test_harmonics import _make_problem
from test_vfp2d.test_magnetic import _finite_magnetic_energy_defect

from adept.vfp2d import TzoufrasVlasov, current, density
from adept.vfp2d.exchange import electron_energy_moment_correction, electron_kinetic_energy_density


@pytest.mark.parametrize("nv", [12, 31])
@pytest.mark.parametrize("m_max", [0, 2])
@pytest.mark.parametrize("finite_tail", [False, True])
def test_electric_work_correction_preserves_density_and_explicit_tail_flux(nv, m_max, finite_tail):
    grid, layout, operator, flm = _make_problem(l_max=3, m_max=m_max, nx=2, ny=3, nv=nv)
    radial = jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(radial)
    for i, (ell, m) in enumerate(layout.pairs):
        if ell:
            coefficient = 0.01 * (i + 1) * (1 + 0.3j * m)
            flm = flm.at[..., i, :].set(coefficient * grid.v**ell * radial)
    if finite_tail:
        flm = flm.at[..., layout.index(1, 0), -1].set(0.003)
        if m_max:
            flm = flm.at[..., layout.index(1, 1), -1].set(-0.002 + 0.001j)
    else:
        flm = flm.at[..., :, -1].set(0.0)
    electric = jnp.broadcast_to(jnp.asarray([0.4, -0.3, 0.2]), (grid.nx, grid.ny, 3))
    electric = electric.at[1, ...].multiply(-0.7)
    uncorrected = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky, conserve_electric_work=False).electric(
        flm, electric
    )
    corrected = jax.jit(operator.electric)(flm, electric)
    contraction = electric[..., 0, None] * jnp.real(flm[..., layout.index(1, 0), :])
    if m_max:
        contraction += 2 * jnp.real(
            (electric[..., 1] + 1j * electric[..., 2])[..., None] * flm[..., layout.index(1, 1), :]
        )
    ghost_velocity = grid.v[-1] + grid.dv
    boundary_number = (2 * jnp.pi / 3) * ghost_velocity**2 * contraction[..., -1]
    boundary_energy = (jnp.pi / 3) * ghost_velocity**4 * contraction[..., -1]
    interior_defect = -(8 * jnp.pi / 3) * grid.dv**3 * jnp.sum(grid.v * contraction, axis=-1)
    physical_work = jnp.sum(current(flm, layout, grid.v, grid.dv) * electric, axis=-1)
    np.testing.assert_allclose(density(corrected, layout, grid.v, grid.dv), boundary_number, atol=5e-14)
    np.testing.assert_allclose(
        density(corrected, layout, grid.v, grid.dv), density(uncorrected, layout, grid.v, grid.dv), atol=5e-14
    )
    np.testing.assert_allclose(
        electron_kinetic_energy_density(uncorrected, layout, grid.v, grid.dv),
        physical_work + boundary_energy + interior_defect,
        atol=5e-14,
    )
    np.testing.assert_allclose(
        electron_kinetic_energy_density(corrected, layout, grid.v, grid.dv),
        physical_work + boundary_energy,
        atol=5e-14,
    )
    np.testing.assert_allclose(corrected[..., 1:, :], uncorrected[..., 1:, :], rtol=1e-14, atol=1e-16)
    assert float(jnp.max(jnp.abs(interior_defect))) > 1e-5
    if finite_tail:
        assert float(jnp.max(jnp.abs(boundary_energy))) > 0.1


def test_electric_work_correction_leaves_zero_distribution_finite():
    grid, _, operator, flm = _make_problem(nx=1, ny=1, nv=8)
    electric = jnp.ones((grid.nx, grid.ny, 3))
    np.testing.assert_array_equal(jax.jit(operator.electric)(flm, electric), flm)


def test_electric_work_correction_requires_nonrelativistic_momentum():
    grid, layout, _, _ = _make_problem(nx=1, ny=1)
    speed = grid.v / jnp.sqrt(1.0 + grid.v**2)
    with pytest.raises(ValueError, match="nonrelativistic"):
        TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky, streaming_speed=speed)
    TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky, streaming_speed=speed, conserve_electric_work=False)


@pytest.mark.parametrize("occupied", [0, 10, 20, 23])
def test_electric_work_correction_skips_monoenergetic_distributions_under_jit(occupied):
    grid, layout, operator, flm = _make_problem(l_max=2, nx=1, ny=1, nv=24)
    flm = flm.at[..., layout.index(0, 0), occupied].set(1.0)
    flm = flm.at[..., layout.index(1, 0), occupied].set(0.01)
    electric = jnp.asarray([[[0.4, 0.0, 0.0]]])
    correction = jax.jit(operator.electric_work_correction)(flm, electric)
    np.testing.assert_array_equal(correction, jnp.zeros_like(flm))


@pytest.mark.parametrize("minority_weight", [1.0, 1e-4, 1e-8])
def test_narrow_positive_radial_distributions_retain_density_and_requested_work(minority_weight):
    grid, layout, _, flm = _make_problem(l_max=2, nx=1, ny=1, nv=24)
    flm = flm.at[..., layout.index(0, 0), 10].set(1.0)
    flm = flm.at[..., layout.index(0, 0), 11].set(minority_weight)
    requested_work = jnp.asarray([[0.003]])
    correction = jax.jit(
        lambda f: electron_energy_moment_correction(
            f,
            layout,
            grid.v,
            grid.dv,
            requested_work,
            1.0,
        )
    )(flm)
    assert bool(jnp.all(jnp.isfinite(correction)))
    np.testing.assert_allclose(density(correction, layout, grid.v, grid.dv), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        electron_kinetic_energy_density(correction, layout, grid.v, grid.dv),
        requested_work,
        atol=1e-14,
    )


def test_electric_work_correction_does_not_conceal_momentum_discretization_error():
    grid, layout, operator, flm = _make_problem(l_max=2, nx=1, ny=1, nv=24)
    f0 = jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(f0)
    electric = jnp.asarray([[[0.4, -0.3, 0.2]]])
    rate = operator.electric(flm, electric)
    measured = current(rate, layout, grid.v, grid.dv, charge=1.0)
    analytic = -density(flm, layout, grid.v, grid.dv)[..., None] * electric
    interior = -(4 * jnp.pi / 3) * grid.dv**3 * jnp.sum(f0) * electric
    boundary = (2 * jnp.pi / 3) * (grid.v[-1] + grid.dv) ** 3 * f0[-1] * electric
    np.testing.assert_allclose(measured, analytic + interior + boundary, atol=5e-14)
    assert float(jnp.max(jnp.abs(measured - analytic))) > 1e-3


def test_corrected_finite_magnetic_energy_budget_remains_resolved_at_longer_time():
    original = _finite_magnetic_energy_defect(32, 0.01, final_time=2.0, conserve_electric_work=False)
    coarse = _finite_magnetic_energy_defect(32, 0.01, final_time=2.0)
    time_fine = _finite_magnetic_energy_defect(32, 0.005, final_time=2.0)
    radial_fine = _finite_magnetic_energy_defect(64, 0.005, final_time=2.0)
    for result in (coarse, time_fine, radial_fine):
        np.testing.assert_allclose(result["raw"] - result["projection"], result["accounted"], atol=2e-12)
        assert result["quasineutrality"] < 1e-12
        # Less than 0.1% of the initial transverse magnetic perturbation energy;
        # raw/projection contributions remain individually measured above.
        assert abs(result["accounted"]) < 1e-3
    assert abs(original["accounted"]) / abs(coarse["accounted"]) > 100.0
    assert abs(coarse["accounted"] - time_fine["accounted"]) < 1e-6
    assert abs(time_fine["accounted"] - radial_fine["accounted"]) < 4e-5
