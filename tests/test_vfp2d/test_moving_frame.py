"""Equation-level tests for the ion peculiar-velocity representation."""

import jax
import jax.numpy as jnp
import numpy as np

from adept.vfp2d import (
    Grid,
    HarmonicLayout,
    IonFrameVlasov,
    TzoufrasVlasov,
    density,
    scalar_velocity_moment,
    tensor_velocity_moment,
)


def _make_problem(l_max=3, nx=4, ny=3, nv=96, vmax=7.0):
    grid = Grid(
        xmin=0.0,
        xmax=2.0 * np.pi,
        nx=nx,
        ymin=0.0,
        ymax=2.0 * np.pi,
        ny=ny,
        vmax=vmax,
        nv=nv,
        dt=0.01,
        l_max=l_max,
    )
    layout = HarmonicLayout(l_max)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    frame = IonFrameVlasov(vlasov)
    f = jnp.zeros((nx, ny, layout.size, nv), dtype=jnp.complex128)
    f = f.at[..., layout.index(0, 0), :].set(jnp.exp(-(grid.v**2)))
    return grid, layout, vlasov, frame, f


def test_angular_galerkin_transform_round_trips_physical_coefficients():
    _grid, layout, _vlasov, frame, f = _make_problem(l_max=9, nx=1, ny=1, nv=4)
    rng = np.random.default_rng(42)
    # Equal-amplitude normalized harmonics correspond to stored coefficients
    # scaled by the inverse norm of ADEPT's unnormalized Legendre basis.
    scale = np.asarray(frame.angular.coefficient_scale)
    coefficients = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) / scale[None, None, :, None]
    for index, (_ell, m) in enumerate(layout.pairs):
        if m == 0:
            coefficients[..., index, :] = coefficients[..., index, :].real

    reconstructed = frame.angular.reconstruct(jnp.asarray(coefficients))
    projected = frame.angular.project(reconstructed)
    np.testing.assert_allclose(
        projected * scale[None, None, :, None],
        coefficients * scale[None, None, :, None],
        rtol=2e-12,
        atol=2e-12,
    )


def test_sparse_deformation_matches_dense_galerkin_oracle():
    _grid, layout, _vlasov, frame, f = _make_problem(l_max=5, nx=2, ny=1, nv=18)
    rng = np.random.default_rng(123)
    coefficients = rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)
    for index, (_ell, m) in enumerate(layout.pairs):
        if m == 0:
            coefficients[..., index, :] = coefficients[..., index, :].real
    f = jnp.asarray(coefficients)
    gradient = jnp.asarray(rng.normal(scale=0.07, size=(*f.shape[:2], 3, 3)))

    sparse = jax.jit(frame.deformation)(f, gradient)
    dense = frame.deformation_reference(f, gradient)
    np.testing.assert_allclose(sparse, dense, rtol=4e-12, atol=4e-12)

    number_real_harmonics = (layout.l_max + 1) ** 2
    dense_entries = 2 * 9 * number_real_harmonics**2
    assert frame.sparse_angular.nnz < 0.3 * dense_entries


def test_bulk_advection_is_conservative_for_every_harmonic_and_speed_cell():
    grid, layout, _vlasov, frame, f = _make_problem(l_max=2, nx=12, ny=10, nv=8)
    xx, yy = jnp.meshgrid(grid.x, grid.y, indexing="ij")
    f = f.at[..., layout.index(0, 0), :].multiply(1.0 + 0.1 * jnp.cos(xx)[..., None])
    f = f.at[..., layout.index(1, 0), :].set(0.03 * jnp.sin(yy)[..., None] * jnp.exp(-(grid.v**2)))
    velocity = jnp.stack((0.2 + 0.04 * jnp.sin(yy), -0.1 + 0.03 * jnp.cos(xx), 0.02 * jnp.sin(xx + yy)), axis=-1)

    rate = frame.bulk_advection(f, velocity)
    np.testing.assert_allclose(jnp.sum(rate, axis=(0, 1)), 0.0, atol=2e-13)


def test_velocity_gradient_uses_component_by_derivative_index_order():
    grid, _layout, _vlasov, frame, _f = _make_problem(l_max=1, nx=12, ny=10, nv=8)
    xx, yy = jnp.meshgrid(grid.x, grid.y, indexing="ij")
    velocity = jnp.stack((jnp.sin(xx), jnp.cos(yy), jnp.sin(xx + 2.0 * yy)), axis=-1)
    gradient = frame.velocity_gradient(velocity)
    expected = jnp.stack(
        (
            jnp.stack((jnp.cos(xx), jnp.zeros_like(xx), jnp.zeros_like(xx)), axis=-1),
            jnp.stack((jnp.zeros_like(yy), -jnp.sin(yy), jnp.zeros_like(yy)), axis=-1),
            jnp.stack((jnp.cos(xx + 2.0 * yy), 2.0 * jnp.cos(xx + 2.0 * yy), jnp.zeros_like(xx)), axis=-1),
        ),
        axis=-2,
    )
    np.testing.assert_allclose(gradient, expected, rtol=2e-12, atol=2e-12)


def test_uniform_maxwellian_is_exact_under_galilean_translation():
    _grid, _layout, _vlasov, frame, f = _make_problem(l_max=3)
    velocity = jnp.broadcast_to(jnp.asarray([0.7, -0.25, 0.15]), (*f.shape[:2], 3))
    zeros = jnp.zeros_like(velocity)
    gradient = jnp.zeros((*f.shape[:2], 3, 3))

    rate = frame(
        f,
        electric_field=zeros,
        magnetic_field=zeros,
        ion_velocity=velocity,
        velocity_gradient=gradient,
        material_acceleration=zeros,
    )
    np.testing.assert_allclose(rate, 0.0, atol=2e-13)


def test_isotropic_compression_preserves_particles_and_heats_adiabatically():
    grid, layout, _vlasov, frame, f = _make_problem(l_max=2, nx=2, ny=2, nv=192, vmax=8.0)
    alpha = -0.2
    gradient = jnp.broadcast_to(alpha * jnp.eye(3), (*f.shape[:2], 3, 3))
    deformation = jax.jit(frame.deformation)(f, gradient)
    theta = 3.0 * alpha

    np.testing.assert_allclose(density(deformation, layout, grid.v, grid.dv), 0.0, atol=3e-13)

    # A homogeneous fluid element also receives -div(u f) = -theta*f from
    # conservative bulk transport. Together the moments obey n~rho and
    # T~rho^(2/3), the ideal collisionally isotropic electron limit.
    local_rate = deformation - theta * f
    number = density(f, layout, grid.v, grid.dv)
    number_rate = density(local_rate, layout, grid.v, grid.dv)
    moment2 = scalar_velocity_moment(f, layout, grid.v, grid.dv, power=2)
    f00_rate = jnp.real(local_rate[..., layout.index(0, 0), :])
    moment2_numerator_rate = 4.0 * jnp.pi * jnp.sum(f00_rate * grid.v**4, axis=-1) * grid.dv
    moment2_rate = moment2_numerator_rate / number - moment2 * number_rate / number
    temperature = moment2 / 3.0
    temperature_rate = moment2_rate / 3.0

    np.testing.assert_allclose(number_rate, -theta * number, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(temperature_rate, -2.0 * alpha * temperature, rtol=2e-3, atol=2e-5)


def test_prescribed_shear_generates_the_expected_pressure_anisotropy():
    grid, layout, _vlasov, frame, f = _make_problem(l_max=2, nx=2, ny=2, nv=192, vmax=8.0)
    strain_rate = 0.12
    strain = jnp.diag(jnp.asarray([strain_rate, -strain_rate, 0.0]))
    gradient = jnp.broadcast_to(strain, (*f.shape[:2], 3, 3))
    deformation = frame.deformation(f, gradient)

    np.testing.assert_allclose(density(deformation, layout, grid.v, grid.dv), 0.0, atol=3e-13)
    generated_tensor = tensor_velocity_moment(f + deformation, layout, grid.v, grid.dv, power=0)
    moment2 = scalar_velocity_moment(f, layout, grid.v, grid.dv, power=2)
    expected = -(2.0 / 3.0) * moment2[..., None, None] * gradient
    np.testing.assert_allclose(generated_tensor, expected, rtol=3e-3, atol=3e-5)


def test_deformation_preserves_particles_for_an_anisotropic_distribution():
    grid, layout, _vlasov, frame, f = _make_problem(l_max=4, nx=2, ny=2, nv=48)
    rng = np.random.default_rng(19)
    for index, (ell, m) in enumerate(layout.pairs[1:], start=1):
        radial = grid.v**ell * jnp.exp(-(grid.v**2))
        coefficient = rng.normal() if m == 0 else rng.normal() + 1j * rng.normal()
        f = f.at[..., index, :].set(0.01 * coefficient * radial)
    gradient = jnp.broadcast_to(
        jnp.asarray(
            [
                [0.08, -0.03, 0.02],
                [0.04, -0.05, 0.01],
                [-0.02, 0.03, 0.06],
            ]
        ),
        (*f.shape[:2], 3, 3),
    )

    rate = frame.deformation(f, gradient)
    np.testing.assert_allclose(density(rate, layout, grid.v, grid.dv), 0.0, atol=2e-12)


def test_rigid_frame_rotation_matches_the_existing_angular_rotation_operator():
    grid, layout, vlasov, frame, f = _make_problem(l_max=4, nx=2, ny=2, nv=32)
    rng = np.random.default_rng(7)
    for index, (ell, m) in enumerate(layout.pairs):
        radial = grid.v**ell * jnp.exp(-(grid.v**2))
        coefficient = rng.normal() if m == 0 else rng.normal() + 1j * rng.normal()
        f = f.at[..., index, :].set(coefficient * radial)

    angular_frequency = 0.17
    rotation_gradient = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -angular_frequency],
            [0.0, angular_frequency, 0.0],
        ]
    )
    gradient = jnp.broadcast_to(rotation_gradient, (*f.shape[:2], 3, 3))
    magnetic_field = jnp.broadcast_to(jnp.asarray([-angular_frequency, 0.0, 0.0]), (*f.shape[:2], 3))

    np.testing.assert_allclose(
        frame.deformation(f, gradient),
        vlasov.magnetic(f, magnetic_field),
        rtol=3e-12,
        atol=3e-12,
    )


def test_accelerating_frame_cancels_a_uniform_force_equilibrium():
    _grid, _layout, _vlasov, frame, f = _make_problem(l_max=3)
    force = jnp.broadcast_to(jnp.asarray([0.15, -0.08, 0.04]), (*f.shape[:2], 3))
    zeros = jnp.zeros_like(force)
    gradient = jnp.zeros((*f.shape[:2], 3, 3))

    rate = frame(
        f,
        electric_field=force,
        magnetic_field=zeros,
        ion_velocity=zeros,
        velocity_gradient=gradient,
        material_acceleration=-force,
    )
    np.testing.assert_allclose(rate, 0.0, atol=2e-13)
