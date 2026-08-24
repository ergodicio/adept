"""Verification for explicit and implicit VFP2D field responses."""

import jax
import jax.numpy as jnp
import numpy as np

from adept.vfp2d import Grid, HarmonicLayout, Maxwell2D, OSHUNImplicitStep, TzoufrasVlasov, current


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
