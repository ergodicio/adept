"""Verification for explicit and implicit VFP2D field responses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adept.vfp2d import Grid, HarmonicLayout, Maxwell2D, OSHUNImplicitStep, TzoufrasVlasov, current
from adept.vfp2d.harmonics import complex_to_real, real_to_complex


def _problem(nx=6, ny=4, nv=48, dt=1.0e-3):
    grid = Grid(
        xmin=0.0,
        xmax=2.0 * np.pi,
        nx=nx,
        ymin=0.0,
        ymax=2.0 * np.pi,
        ny=ny,
        vmax=7.0,
        nv=nv,
        dt=dt,
        l_max=2,
    )
    layout = HarmonicLayout(2)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    maxwell = Maxwell2D(grid.kx, grid.ky, c=5.0)
    radial = jnp.exp(-(grid.v**2))
    radial /= 4.0 * jnp.pi * jnp.sum(radial * grid.v**2) * grid.dv
    flm = jnp.zeros((nx, ny, layout.size, nv), dtype=jnp.complex128)
    flm = flm.at[..., layout.index(0, 0), :].set(radial)
    step = OSHUNImplicitStep(vlasov, maxwell, layout, grid.v, grid.dv, dt)
    return grid, layout, vlasov, maxwell, step, flm


def test_relative_permittivity_slows_only_the_explicit_ampere_response():
    grid, _layout, _vlasov, physical, _step, _flm = _problem()
    relaxed = Maxwell2D(grid.kx, grid.ky, c=5.0, relative_permittivity=1.0e6)
    x, y = grid.x[:, None], grid.y[None, :]
    electric = jnp.stack(
        (
            jnp.sin(x) * jnp.ones_like(y),
            jnp.ones_like(x) * jnp.cos(y),
            jnp.sin(x) * jnp.cos(y),
        ),
        axis=-1,
    )
    magnetic = jnp.stack(
        (
            jnp.ones_like(x) * jnp.sin(y),
            jnp.cos(x) * jnp.ones_like(y),
            jnp.cos(x) * jnp.sin(y),
        ),
        axis=-1,
    )
    plasma_current = 0.03 * electric

    physical_dedt, physical_dbdt = physical(electric, magnetic, plasma_current)
    relaxed_dedt, relaxed_dbdt = relaxed(electric, magnetic, plasma_current)

    np.testing.assert_allclose(relaxed_dedt, physical_dedt / 1.0e6, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(relaxed_dbdt, physical_dbdt, rtol=0.0, atol=0.0)


def test_oshun_response_tensor_matches_direct_three_field_perturbations():
    grid, _layout, _vlasov, _maxwell, step, flm = _problem()
    baseline, response = step.current_response(flm)
    x, y = grid.x[:, None], grid.y[None, :]
    direction = jnp.stack(
        (
            0.3 + 0.1 * jnp.cos(x) * jnp.ones_like(y),
            -0.2 * jnp.ones_like(x) * jnp.sin(y),
            0.05 * jnp.sin(x) * jnp.cos(y),
        ),
        axis=-1,
    )
    epsilon = 1.0e-7
    perturbed = flm + step.electric_increment(flm, epsilon * direction)
    measured = (current(perturbed, step.layout, step.v, step.dv) - baseline) / epsilon
    predicted = jnp.einsum("...ij,...j->...i", response, direction)

    np.testing.assert_allclose(measured, predicted, rtol=2e-9, atol=2e-12)


def test_oshun_direct_solve_enforces_ampere_current_without_f1_projection():
    grid, layout, _vlasov, maxwell, step, flm = _problem()
    x, y = grid.x[:, None], grid.y[None, :]
    magnetic = jnp.zeros((grid.nx, grid.ny, 3))
    magnetic = magnetic.at[..., 2].set(0.002 * jnp.sin(x) * jnp.cos(y))
    flm = flm.at[..., layout.index(1, 0), :].set(1.0e-4 * flm[..., layout.index(0, 0), :])
    f2_before = flm[..., layout.index(2, 0), :]

    updated, electric, residual = jax.jit(step.solve_electric_field)(flm, magnetic)

    assert jnp.all(jnp.isfinite(electric))
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(
        current(updated, layout, grid.v, grid.dv),
        maxwell.c2 * maxwell.curl(magnetic),
        rtol=2e-13,
        atol=2e-13,
    )
    assert jnp.max(jnp.abs(updated[..., layout.index(2, 0), :] - f2_before)) > 0.0


def test_oshun_step_uses_explicit_faraday_and_enforces_the_new_ampere_target():
    grid, layout, _vlasov, maxwell, step, flm = _problem()
    x, y = grid.x[:, None], grid.y[None, :]
    electric = jnp.stack(
        (
            jnp.zeros((grid.nx, grid.ny)),
            jnp.zeros((grid.nx, grid.ny)),
            1.0e-4 * jnp.sin(x) * jnp.cos(y),
        ),
        axis=-1,
    )
    magnetic = jnp.zeros_like(electric).at[..., 2].set(2.0e-5 * jnp.sin(x) * jnp.cos(y))

    result = jax.jit(step)(0.0, {"flm": flm, "e": electric, "b": magnetic}, {})

    expected_magnetic = magnetic - step.dt * maxwell.curl(electric)
    np.testing.assert_allclose(result["b"], expected_magnetic, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        current(result["flm"], layout, grid.v, grid.dv),
        maxwell.c2 * maxwell.curl(result["b"]),
        rtol=2e-12,
        atol=2e-12,
    )


def test_oshun_step_is_jittable_and_leaves_a_uniform_equilibrium_unchanged():
    grid, _layout, _vlasov, _maxwell, step, flm = _problem()
    field = jnp.zeros((grid.nx, grid.ny, 3))
    state = {"flm": flm, "e": field, "b": field}

    result = jax.jit(step)(0.0, state, {})

    np.testing.assert_allclose(result["flm"], flm, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result["e"], 0.0, atol=2e-13)
    np.testing.assert_allclose(result["b"], 0.0, atol=2e-13)


@pytest.mark.parametrize("real_storage", [False, True])
def test_oshun_nyquist_transport_preserves_real_m_zero_at_every_stage(real_storage):
    grid, layout, vlasov, maxwell, _, flm = _problem(nx=8)
    step = OSHUNImplicitStep(vlasov, maxwell, layout, grid.v, grid.dv, grid.dt, real_storage=real_storage)
    checkerboard = (-1.0) ** jnp.arange(grid.nx)
    flm = flm * (1.0 + 0.05 * checkerboard[:, None, None, None])
    field = jnp.zeros((grid.nx, grid.ny, 3))
    m_zero = np.flatnonzero(layout.m == 0)
    assert jnp.min(flm[..., layout.index(0, 0), :].real) > 0.0
    # Exercise a nonzero imaginary Nyquist derivative, not a uniform null case.
    assert jnp.max(jnp.abs(vlasov.streaming(flm)[..., m_zero, :].imag)) > 1e-4

    rate1 = step._non_electric_rate(flm, field)
    midpoint = flm + 0.5 * grid.dt * rate1
    rate2 = step._non_electric_rate(midpoint, field)
    transported = step._non_electric_step(flm, field)
    for stage in (rate1, midpoint, rate2, transported):
        np.testing.assert_array_equal(stage[..., m_zero, :].imag, 0.0)
    # Projecting only the final state would still allow a spurious f00/f20 update.
    np.testing.assert_allclose(transported, flm, rtol=0.0, atol=2e-14)

    state = {"flm": complex_to_real(flm) if real_storage else flm, "e": field, "b": field}
    result = jax.jit(step)(0.0, state, {})
    final_f = real_to_complex(result["flm"]) if real_storage else result["flm"]
    np.testing.assert_array_equal(final_f[..., m_zero, :].imag, 0.0)
    np.testing.assert_allclose(final_f, flm, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(result["e"], 0.0, atol=2e-14)
    np.testing.assert_allclose(result["b"], 0.0, atol=2e-14)
