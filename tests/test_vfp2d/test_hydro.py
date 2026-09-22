"""Verification tests for the conservative ion-fluid backbone."""

import jax
import jax.numpy as jnp
import numpy as np

from adept.vfp2d import (
    IonEuler2D,
    conserved_to_primitive,
    euler_flux,
    hllc_flux,
    primitive_to_conserved,
)


def _constant_primitive(shape, values):
    return jnp.broadcast_to(jnp.asarray(values), (*shape, 5))


def test_primitive_round_trip_and_directional_fluxes():
    primitive = jnp.asarray(
        [
            [1.2, 0.7, -0.2, 0.3, 2.5],
            [0.4, -0.1, 0.8, -0.5, 0.2],
        ]
    )
    conserved = primitive_to_conserved(primitive, gamma=1.4)
    np.testing.assert_allclose(conserved_to_primitive(conserved, gamma=1.4), primitive, rtol=1e-14, atol=1e-14)

    flux_x = euler_flux(primitive, normal_axis=0, gamma=1.4)
    flux_y = euler_flux(primitive, normal_axis=1, gamma=1.4)
    np.testing.assert_allclose(flux_x[..., 0], conserved[..., 1])
    np.testing.assert_allclose(flux_y[..., 0], conserved[..., 2])
    np.testing.assert_allclose(flux_x[..., 2], conserved[..., 2] * primitive[..., 1])
    np.testing.assert_allclose(flux_y[..., 1], conserved[..., 1] * primitive[..., 2])


def test_hllc_is_consistent_and_resolves_a_stationary_contact():
    state = jnp.asarray([1.3, 0.4, -0.2, 0.1, 0.9])
    np.testing.assert_allclose(hllc_flux(state, state, 0, 1.4), euler_flux(state, 0, 1.4), atol=1e-14)
    np.testing.assert_allclose(hllc_flux(state, state, 1, 1.4), euler_flux(state, 1, 1.4), atol=1e-14)

    left = jnp.asarray([1.0, 0.0, 0.3, -0.2, 1.0])
    right = jnp.asarray([0.125, 0.0, 0.3, -0.2, 1.0])
    expected = jnp.asarray([0.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(hllc_flux(left, right, 0, 1.4), expected, atol=2e-14)


def test_uniform_flow_is_exact_for_periodic_and_outflow_boundaries():
    primitive = _constant_primitive((9, 7), [0.8, 0.6, -0.25, 0.1, 1.4])
    conserved = primitive_to_conserved(primitive, gamma=1.4)
    for boundaries in (("periodic", "periodic"), ("outflow", "outflow")):
        solver = IonEuler2D(dx=0.2, dy=0.3, gamma=1.4, boundaries=boundaries)
        np.testing.assert_allclose(solver.rhs(conserved), 0.0, atol=2e-14)
        np.testing.assert_allclose(solver.step(conserved, 0.01), conserved, rtol=2e-14, atol=2e-14)


def test_periodic_step_conserves_every_cell_integrated_quantity():
    nx, ny = 18, 14
    x = (jnp.arange(nx) + 0.5) / nx
    y = (jnp.arange(ny) + 0.5) / ny
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    primitive = jnp.stack(
        (
            1.0 + 0.08 * jnp.sin(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * yy),
            0.2 + 0.04 * jnp.cos(2.0 * jnp.pi * yy),
            -0.1 + 0.03 * jnp.sin(2.0 * jnp.pi * xx),
            0.05 * jnp.cos(2.0 * jnp.pi * (xx + yy)),
            1.0 + 0.05 * jnp.cos(2.0 * jnp.pi * xx),
        ),
        axis=-1,
    )
    conserved = primitive_to_conserved(primitive, gamma=5.0 / 3.0)
    solver = IonEuler2D(dx=1.0 / nx, dy=1.0 / ny)
    advanced = solver.step(conserved, 0.2 * solver.cfl_timestep(conserved))
    np.testing.assert_allclose(jnp.sum(advanced, axis=(0, 1)), jnp.sum(conserved, axis=(0, 1)), rtol=2e-14, atol=2e-13)


def _advect_density_wave(nx):
    x = (jnp.arange(nx) + 0.5) / nx
    rho = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x)
    primitive = jnp.stack((rho, jnp.full_like(x, 0.7), jnp.zeros_like(x), jnp.zeros_like(x), jnp.ones_like(x)), axis=-1)
    conserved = primitive_to_conserved(primitive[:, None, :], gamma=1.4)
    solver = IonEuler2D(dx=1.0 / nx, dy=1.0, gamma=1.4)
    final_time = 0.2
    stable_dt = float(solver.cfl_timestep(conserved, cfl=0.35))
    steps = int(np.ceil(final_time / stable_dt))
    dt = final_time / steps
    advance = jax.jit(lambda state: solver.step(state, dt))
    for _ in range(steps):
        conserved = advance(conserved)
    exact = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * ((x - 0.7 * final_time) % 1.0))
    return float(jnp.mean(jnp.abs(conserved[:, 0, 0] - exact)))


def test_smooth_density_advection_converges_at_second_order():
    coarse_error = _advect_density_wave(32)
    fine_error = _advect_density_wave(64)
    observed_order = np.log2(coarse_error / fine_error)
    assert observed_order > 1.75


def _advance_sod(normal_axis):
    normal_cells, transverse_cells = 80, 4
    normal = (jnp.arange(normal_cells) + 0.5) / normal_cells
    rho = jnp.where(normal < 0.5, 1.0, 0.125)
    pressure = jnp.where(normal < 0.5, 1.0, 0.1)
    primitive_1d = jnp.stack((rho, jnp.zeros_like(rho), jnp.zeros_like(rho), jnp.zeros_like(rho), pressure), axis=-1)
    if normal_axis == 0:
        primitive = jnp.broadcast_to(primitive_1d[:, None, :], (normal_cells, transverse_cells, 5))
        solver = IonEuler2D(
            dx=1.0 / normal_cells,
            dy=1.0 / transverse_cells,
            gamma=1.4,
            boundaries=("outflow", "periodic"),
        )
    else:
        primitive = jnp.broadcast_to(primitive_1d[None, :, :], (transverse_cells, normal_cells, 5))
        solver = IonEuler2D(
            dx=1.0 / transverse_cells,
            dy=1.0 / normal_cells,
            gamma=1.4,
            boundaries=("periodic", "outflow"),
        )
    conserved = primitive_to_conserved(primitive, gamma=1.4)
    final_time = 0.08
    stable_dt = float(solver.cfl_timestep(conserved, cfl=0.3))
    steps = int(np.ceil(final_time / stable_dt))
    dt = final_time / steps
    advance = jax.jit(lambda state: solver.step(state, dt))
    for _ in range(steps):
        conserved = advance(conserved)
    return conserved_to_primitive(conserved, gamma=1.4)


def test_sod_tube_is_positive_and_rotation_invariant():
    result_x = _advance_sod(normal_axis=0)
    result_y = _advance_sod(normal_axis=1)
    rotated_y = jnp.swapaxes(result_y, 0, 1)
    rotated_y = rotated_y.at[..., 1].set(result_y.swapaxes(0, 1)[..., 2])
    rotated_y = rotated_y.at[..., 2].set(result_y.swapaxes(0, 1)[..., 1])

    assert jnp.all(result_x[..., 0] > 0.0)
    assert jnp.all(result_x[..., 4] > 0.0)
    assert jnp.max(result_x[..., 1]) > 0.5
    np.testing.assert_allclose(result_x, rotated_y, rtol=3e-12, atol=3e-12)
