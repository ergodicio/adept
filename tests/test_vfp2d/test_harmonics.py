"""Equation-level tests for the arbitrary-f_lm VFP-2D core."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from adept.vfp1d.fokker_planck import FLMCollisions
from adept.vfp1d.grid import Grid as VFP1DGrid
from adept.vfp2d import (
    AnisotropicCollisions,
    Grid,
    HarmonicLayout,
    HouLiFilter2D,
    KineticOhm2D,
    Maxwell2D,
    TzoufrasVlasov,
    cartesian_l2,
    conservative_f00_positivity,
    current,
    density,
    nernst_velocity,
    project_current_moment,
    scalar_velocity_moment,
    tensor_velocity_moment,
    vector_velocity_moment,
)


def _make_problem(l_max=3, m_max=None, nx=16, ny=12, nv=10):
    grid = Grid(
        xmin=0.0,
        xmax=2.0 * np.pi,
        nx=nx,
        ymin=0.0,
        ymax=2.0 * np.pi,
        ny=ny,
        vmax=4.0,
        nv=nv,
        dt=0.01,
        l_max=l_max,
        m_max=m_max,
    )
    layout = HarmonicLayout(l_max, m_max)
    operator = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    flm = jnp.zeros((nx, ny, layout.size, nv), dtype=jnp.complex128)
    return grid, layout, operator, flm


def test_layout_is_compact_and_supports_independent_m_truncation():
    layout = HarmonicLayout(l_max=4, m_max=2)
    assert layout.pairs == (
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
    )
    assert layout.index(3, 2) == 8
    assert layout.index(3, 3) == -1


def test_isotropic_spatial_gradient_drives_f10_and_f11_with_tzoufras_coefficients():
    grid, layout, operator, flm = _make_problem()
    profile = jnp.cos(grid.x)[:, None, None] + 0.7 * jnp.sin(2.0 * grid.y)[None, :, None]
    flm = flm.at[..., layout.index(0, 0), :].set(profile)

    result = operator.streaming(flm)
    expected_f10 = jnp.broadcast_to(grid.v[None, None, :] * jnp.sin(grid.x)[:, None, None], (grid.nx, grid.ny, grid.nv))
    expected_f11 = jnp.broadcast_to(
        -0.7 * grid.v[None, None, :] * jnp.cos(2.0 * grid.y)[None, :, None],
        (grid.nx, grid.ny, grid.nv),
    )

    np.testing.assert_allclose(result[..., layout.index(1, 0), :], expected_f10, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result[..., layout.index(1, 1), :], expected_f11, rtol=1e-12, atol=1e-12)


def test_hidden_z_gradient_drives_imaginary_f11_with_tzoufras_coefficient():
    grid, layout, operator, flm = _make_problem(nx=3, ny=2)
    f00 = jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(f00)
    alpha = 0.07
    dfdz = alpha * flm

    result = operator.streaming(flm, dfdz=dfdz)
    expected_f11 = jnp.broadcast_to(
        0.5j * alpha * grid.v * f00,
        (grid.nx, grid.ny, grid.nv),
    )

    np.testing.assert_allclose(result[..., layout.index(1, 0), :], 0.0, atol=1e-14)
    np.testing.assert_allclose(result[..., layout.index(1, 1), :], expected_f11, rtol=1e-12, atol=1e-12)


def test_isotropic_electric_push_matches_equations_20_and_21():
    grid, layout, operator, flm = _make_problem(nx=3, ny=2)
    f00 = jnp.exp(-(grid.v**2))[None, None, :]
    flm = flm.at[..., layout.index(0, 0), :].set(f00)
    e = jnp.broadcast_to(jnp.asarray([0.4, -0.3, 0.2]), (grid.nx, grid.ny, 3))

    result = operator.electric(flm, e)
    derivative = operator.ddv(jnp.broadcast_to(f00, (grid.nx, grid.ny, grid.nv)), ell=0)

    np.testing.assert_allclose(result[..., layout.index(1, 0), :], 0.4 * derivative, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result[..., layout.index(1, 1), :], 0.5 * (-0.3 - 0.2j) * derivative, rtol=1e-12, atol=1e-12
    )


def test_origin_regularity_annihilates_g_for_f_l_proportional_to_v_l():
    grid, layout, operator, flm = _make_problem(l_max=2, nx=2, ny=2, nv=24)
    i10 = layout.index(1, 0)
    flm = flm.at[..., i10, :].set(grid.v)
    g, _ = operator.gh(flm)
    np.testing.assert_allclose(g[..., i10, :-1], 0.0, atol=1e-13)


def test_density_and_current_use_tzoufras_moment_normalization():
    grid, layout, _operator, flm = _make_problem(l_max=1, nx=2, ny=3)
    f00 = jnp.exp(-(grid.v**2))
    f10 = 0.2 * jnp.exp(-(grid.v**2))
    f11 = (0.1 - 0.05j) * jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(f00)
    flm = flm.at[..., layout.index(1, 0), :].set(f10)
    flm = flm.at[..., layout.index(1, 1), :].set(f11)

    n = density(flm, layout, grid.v, grid.dv)
    j = current(flm, layout, grid.v, grid.dv)
    radial2 = jnp.sum(f00 * grid.v**2) * grid.dv
    radial3 = jnp.sum(jnp.exp(-(grid.v**2)) * grid.v**3) * grid.dv

    np.testing.assert_allclose(n, 4.0 * np.pi * radial2)
    expected = -(4.0 * np.pi / 3.0) * radial3 * jnp.asarray([0.2, 0.2, 0.1])
    np.testing.assert_allclose(j, jnp.broadcast_to(expected, j.shape))


def test_conservative_f00_positivity_removes_undershoots_and_preserves_density():
    grid, layout, _operator, flm = _make_problem(l_max=1, nx=2, ny=3)
    f00 = jnp.exp(-(grid.v**2))
    f00 = f00.at[-3:].set(jnp.asarray([-0.02, -0.01, -0.005]))
    flm = flm.at[..., layout.index(0, 0), :].set(f00)
    flm = flm.at[..., layout.index(1, 0), :].set(0.1 * jnp.exp(-(grid.v**2)))

    density_before = density(flm, layout, grid.v, grid.dv)
    projected = conservative_f00_positivity(flm, layout, grid.v, grid.dv)

    assert jnp.all(jnp.real(projected[..., layout.index(0, 0), :]) >= 0.0)
    np.testing.assert_allclose(density(projected, layout, grid.v, grid.dv), density_before, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        projected[..., layout.index(1, 0), :],
        flm[..., layout.index(1, 0), :],
        rtol=0.0,
        atol=0.0,
    )


def test_hou_li_filter_damps_only_grid_scale_configuration_modes():
    nx, ny = 32, 24
    x = jnp.arange(nx)[:, None, None, None]
    y = jnp.arange(ny)[None, :, None, None]
    smooth = jnp.exp(2j * jnp.pi * (2 * x / nx + 3 * y / ny))
    checkerboard = (-1.0) ** (x + y)
    value = smooth + checkerboard

    filtered = HouLiFilter2D(nx, ny, alpha=36.0, order=36)(value)

    # The low Fourier mode is unaffected to roundoff, while the Nyquist mode
    # is attenuated by exp(-alpha) on each of the two axes.
    np.testing.assert_allclose(filtered, smooth, rtol=1e-12, atol=1e-12)


def test_partitioned_x_filter_removes_checkerboard_and_preserves_smooth_mode():
    nx, ny = 32, 8
    x = jnp.arange(nx)[:, None]
    smooth = jnp.exp(2j * jnp.pi * x / nx) * jnp.ones((1, ny))
    checkerboard = (-1.0) ** x * jnp.ones((1, ny))
    mesh = Mesh(np.asarray(jax.devices()), ("x",))

    filtered = HouLiFilter2D(nx, ny, dimensions=("x",), mesh=mesh)(smooth + checkerboard)

    np.testing.assert_allclose(filtered, smooth, rtol=2e-7, atol=2e-7)


def test_joglekar_velocity_moments_and_l2_tensor_mapping():
    grid, layout, _operator, flm = _make_problem(l_max=2, nx=1, ny=1, nv=64)
    radial = jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(radial)
    flm = flm.at[..., layout.index(1, 0), :].set(0.2 * radial)
    flm = flm.at[..., layout.index(1, 1), :].set((0.1 - 0.05j) * radial)
    flm = flm.at[..., layout.index(2, 0), :].set(0.3 * radial)
    flm = flm.at[..., layout.index(2, 1), :].set((0.04 - 0.02j) * radial)
    flm = flm.at[..., layout.index(2, 2), :].set((0.01 - 0.03j) * radial)

    l2 = cartesian_l2(flm, layout)[0, 0]
    np.testing.assert_allclose(l2[0, 0], 0.3 * radial)
    np.testing.assert_allclose(l2[0, 1], 0.12 * radial)
    np.testing.assert_allclose(l2[0, 2], 0.06 * radial)
    np.testing.assert_allclose(l2[1, 1], -0.09 * radial)
    np.testing.assert_allclose(l2[1, 2], 0.18 * radial)
    np.testing.assert_allclose(jnp.trace(l2, axis1=0, axis2=1), 0.0, atol=1e-15)

    ne = density(flm, layout, grid.v, grid.dv)
    expected_v2 = 4.0 * np.pi * jnp.sum(radial * grid.v**4) * grid.dv / ne
    np.testing.assert_allclose(scalar_velocity_moment(flm, layout, grid.v, grid.dv, 2), expected_v2)
    vector = vector_velocity_moment(flm, layout, grid.v, grid.dv, 0)
    expected_prefactor = 4.0 * np.pi / (3.0 * ne) * jnp.sum(radial * grid.v**3) * grid.dv
    np.testing.assert_allclose(vector, expected_prefactor[..., None] * jnp.asarray([0.2, 0.2, 0.1]))

    tensor = tensor_velocity_moment(flm, layout, grid.v, grid.dv, 0)
    assert tensor.shape == (1, 1, 3, 3)
    np.testing.assert_allclose(jnp.trace(tensor, axis1=-2, axis2=-1), 0.0, atol=1e-15)


def test_nernst_velocity_matches_prl_moment_definition():
    grid, layout, _operator, flm = _make_problem(l_max=1, nx=2, ny=1, nv=48)
    radial = jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(radial)
    flm = flm.at[..., layout.index(1, 0), :].set(0.04 * radial)
    measured = nernst_velocity(flm, layout, grid.v, grid.dv)
    v3 = scalar_velocity_moment(flm, layout, grid.v, grid.dv, 3)
    vv3 = vector_velocity_moment(flm, layout, grid.v, grid.dv, 3)
    expected = (
        vv3 / (2.0 * v3[..., None])
        + current(flm, layout, grid.v, grid.dv) / density(flm, layout, grid.v, grid.dv)[..., None]
    )
    np.testing.assert_allclose(measured, expected)


def test_current_projection_enforces_ampere_moment_without_changing_f00_or_f2():
    grid, layout, _operator, flm = _make_problem(l_max=2, nx=3, ny=2, nv=48)
    radial = jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(radial)
    flm = flm.at[..., layout.index(2, 1), :].set((0.02 + 0.01j) * radial)
    target = jnp.broadcast_to(jnp.asarray([0.03, -0.02, 0.01]), (grid.nx, grid.ny, 3))
    projected = project_current_moment(flm, layout, grid.v, grid.dv, target)
    np.testing.assert_allclose(current(projected, layout, grid.v, grid.dv), target, atol=2e-15)
    np.testing.assert_allclose(projected[..., layout.index(0, 0), :], flm[..., layout.index(0, 0), :])
    np.testing.assert_allclose(projected[..., layout.index(2, 1), :], flm[..., layout.index(2, 1), :])


def test_kinetic_ohm_hidden_density_gradient_generates_prl_ez_source():
    grid, layout, _operator, flm = _make_problem(l_max=2, nx=4, ny=3, nv=64)
    radial = jnp.exp(-(grid.v**2))
    flm = flm.at[..., layout.index(0, 0), :].set(radial)
    ohm = KineticOhm2D(layout, grid.v, grid.dv, grid.kx, grid.ky)
    b = jnp.zeros((grid.nx, grid.ny, 3))
    dndz = 0.07 * jnp.ones((grid.nx, grid.ny))
    electric, terms = ohm(flm, b, hidden_dndz=dndz)

    ne = density(flm, layout, grid.v, grid.dv)
    v3 = scalar_velocity_moment(flm, layout, grid.v, grid.dv, 3)
    v5 = scalar_velocity_moment(flm, layout, grid.v, grid.dv, 5)
    expected_ez = -dndz * v5 / (6.0 * ne * v3)
    np.testing.assert_allclose(electric[..., :2], 0.0, atol=2e-14)
    np.testing.assert_allclose(electric[..., 2], expected_ez)
    np.testing.assert_allclose(terms["tensor_pressure"], 0.0, atol=2e-14)


def test_periodic_streaming_conserves_total_particle_number():
    grid, layout, operator, flm = _make_problem(nx=10, ny=8)
    x, y, v = grid.x[:, None, None], grid.y[None, :, None], grid.v[None, None, :]
    flm = flm.at[..., layout.index(0, 0), :].set((1.0 + 0.1 * jnp.cos(x) * jnp.sin(y)) * jnp.exp(-(v**2)))
    flm = flm.at[..., layout.index(1, 0), :].set(0.02 * jnp.sin(x) * jnp.exp(-(v**2)))
    flm = flm.at[..., layout.index(1, 1), :].set(0.01j * jnp.cos(y) * jnp.exp(-(v**2)))

    dndt = density(operator.streaming(flm), layout, grid.v, grid.dv)
    np.testing.assert_allclose(jnp.sum(dndt), 0.0, atol=2e-12)


def test_maxwell_curl_keeps_divergence_of_b_constant():
    grid, _layout, _operator, _flm = _make_problem()
    maxwell = Maxwell2D(grid.kx, grid.ky, c=3.0)
    e = jnp.zeros((grid.nx, grid.ny, 3)).at[..., 2].set(jnp.sin(grid.x)[:, None] * jnp.cos(2.0 * grid.y)[None, :])
    b = jnp.zeros_like(e)
    _dedt, dbdt = maxwell(e, b, jnp.zeros_like(e))
    divergence = maxwell.ddx(dbdt[..., 0]) + maxwell.ddy(dbdt[..., 1])
    np.testing.assert_allclose(divergence, 0.0, atol=2e-12)


def test_operator_is_jittable_and_differentiable():
    grid, layout, operator, flm = _make_problem(l_max=2, nx=4, ny=4, nv=6)
    flm = flm.at[..., layout.index(0, 0), :].set(jnp.exp(-(grid.v**2)))
    b = jnp.zeros((grid.nx, grid.ny, 3))

    def loss(amplitude):
        e = jnp.zeros_like(b).at[..., 0].set(amplitude)
        return jnp.sum(jnp.abs(operator(flm, e, b)) ** 2)

    eager = loss(0.2)
    compiled = jax.jit(loss)(0.2)
    gradient = jax.grad(loss)(0.2)
    np.testing.assert_allclose(compiled, eager, rtol=1e-12)
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_relativistic_momentum_mode_uses_p_over_gamma_for_streaming_and_current():
    grid, layout, _operator, flm = _make_problem(l_max=1, nx=8, ny=2)
    speed = grid.v / jnp.sqrt(1.0 + grid.v**2)
    operator = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky, streaming_speed=speed)
    flm = flm.at[..., layout.index(0, 0), :].set(jnp.cos(grid.x)[:, None, None])
    result = operator.streaming(flm)
    expected = jnp.broadcast_to(speed[None, None, :] * jnp.sin(grid.x)[:, None, None], (grid.nx, grid.ny, grid.nv))
    np.testing.assert_allclose(result[..., layout.index(1, 0), :], expected, atol=1e-12)

    flm = flm.at[..., layout.index(1, 0), :].set(1.0)
    measured = current(flm, layout, grid.v, grid.dv, streaming_speed=speed)[..., 0]
    expected_jx = -(4.0 * np.pi / 3.0) * jnp.sum(grid.v**2 * speed) * grid.dv
    np.testing.assert_allclose(measured, expected_jx)


def test_anisotropic_collisions_apply_correct_l_dependent_damping_to_every_m():
    grid, layout, _operator, flm = _make_problem(l_max=3, nx=2, ny=2, nv=8)
    collision_grid = VFP1DGrid(
        xmin=0.0,
        xmax=1.0,
        nx=grid.nx * grid.ny,
        tmin=0.0,
        tmax=0.1,
        dt=0.01,
        nv=grid.nv,
        vmax=grid.vmax,
        nl=layout.l_max,
    )
    nuee = 0.2
    base = FLMCollisions(Z=1.0, nuee_coeff=nuee, grid=collision_grid, full_aniso_ee=False)
    collisions = AnisotropicCollisions(base, layout)
    for i, (ell, _m) in enumerate(layout.pairs):
        if ell > 0:
            flm = flm.at[..., i, :].set(1.0 + 0.25j)

    dt = 0.03
    result = jax.jit(collisions)(flm, 1.0, 1.0, dt)
    np.testing.assert_allclose(result[..., layout.index(0, 0), :], 0.0)
    for i, (ell, _m) in enumerate(layout.pairs):
        if ell == 0:
            continue
        ei_rate = 0.5 * ell * (ell + 1) / grid.v**3
        expected = (1.0 + 0.25j) / (1.0 + dt * nuee * base.Z_nuei_scaling * ei_rate)
        np.testing.assert_allclose(result[..., i, :], jnp.broadcast_to(expected, result[..., i, :].shape))


def test_full_anisotropic_ee_collision_solve_supports_complex_m_modes():
    grid, layout, _operator, flm = _make_problem(l_max=2, nx=2, ny=2, nv=12)
    collision_grid = VFP1DGrid(
        xmin=0.0,
        xmax=1.0,
        nx=grid.nx * grid.ny,
        tmin=0.0,
        tmax=0.1,
        dt=0.01,
        nv=grid.nv,
        vmax=grid.vmax,
        nl=layout.l_max,
    )
    base = FLMCollisions(Z=1.0, nuee_coeff=0.01, grid=collision_grid, full_aniso_ee=True)
    collisions = AnisotropicCollisions(base, layout)
    flm = flm.at[..., layout.index(0, 0), :].set(jnp.exp(-(grid.v**2)))
    flm = flm.at[..., layout.index(1, 1), :].set((0.01 + 0.02j) * jnp.exp(-(grid.v**2)))
    result = jax.jit(collisions)(flm, 1.0, 1.0, 1e-3)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.max(jnp.abs(jnp.imag(result[..., layout.index(1, 1), :]))) > 0.0
