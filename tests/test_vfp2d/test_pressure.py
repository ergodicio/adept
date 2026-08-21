"""Electron-pressure feedback and pressure-work budget tests."""

import jax
import jax.numpy as jnp
import numpy as np

from adept.vfp2d import (
    ElectronPressureCoupling,
    Grid,
    HarmonicLayout,
    IonFrameVlasov,
    TzoufrasVlasov,
    electron_kinetic_energy_density,
    electron_pressure_tensor,
    primitive_to_conserved,
    scalar_velocity_moment,
)


def _make_problem(nx=16, ny=12, nv=160):
    grid = Grid(
        xmin=0.0,
        xmax=2.0 * np.pi,
        nx=nx,
        ymin=0.0,
        ymax=2.0 * np.pi,
        ny=ny,
        vmax=8.0,
        nv=nv,
        dt=1.0e-3,
        l_max=2,
    )
    layout = HarmonicLayout(2)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    frame = IonFrameVlasov(vlasov)
    x, y = grid.x[:, None], grid.y[None, :]
    electron_density = 1.0 + 0.08 * jnp.cos(x) * jnp.sin(y)
    radial = jnp.exp(-(grid.v**2))
    radial /= 4.0 * jnp.pi * jnp.sum(radial * grid.v**2) * grid.dv
    f = jnp.zeros((nx, ny, layout.size, nv), dtype=jnp.complex128)
    f = f.at[..., layout.index(0, 0), :].set(electron_density[..., None] * radial)
    anisotropy = 0.018 * jnp.sin(2.0 * x - y)
    f = f.at[..., layout.index(2, 0), :].set(anisotropy[..., None] * grid.v**2 * radial)
    f = f.at[..., layout.index(2, 1), :].set((0.5 - 0.3j) * anisotropy[..., None] * grid.v**2 * radial)

    ion_velocity = jnp.stack(
        jnp.broadcast_arrays(
            0.12 * jnp.sin(x) * jnp.cos(y),
            -0.09 * jnp.cos(x) * jnp.sin(y),
            0.05 * jnp.sin(x + y),
        ),
        axis=-1,
    )
    ion_primitive = jnp.concatenate(
        (
            (100.0 * electron_density)[..., None],
            ion_velocity,
            (0.2 * electron_density)[..., None],
        ),
        axis=-1,
    )
    return grid, layout, frame, f, primitive_to_conserved(ion_primitive)


def test_pressure_tensor_recovers_scalar_f0_and_traceless_f2_parts():
    grid, layout, _frame, f, _ions = _make_problem(nx=4, ny=3)
    isotropic = f.at[..., layout.index(2, 0), :].set(0.0)
    isotropic = isotropic.at[..., layout.index(2, 1), :].set(0.0)
    pressure = electron_pressure_tensor(isotropic, layout, grid.v, grid.dv)
    scalar_pressure = (
        4.0
        * jnp.pi
        * jnp.sum(jnp.real(isotropic[..., layout.index(0, 0), :]) * grid.v**2, axis=-1)
        * grid.dv
        * scalar_velocity_moment(isotropic, layout, grid.v, grid.dv, power=2)
        / 3.0
    )
    expected = scalar_pressure[..., None, None] * jnp.eye(3)
    np.testing.assert_allclose(pressure, expected, rtol=2e-13, atol=2e-13)

    full_pressure = electron_pressure_tensor(f, layout, grid.v, grid.dv)
    np.testing.assert_allclose(full_pressure, jnp.swapaxes(full_pressure, -1, -2), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(jnp.trace(full_pressure, axis1=-2, axis2=-1), 3.0 * scalar_pressure, rtol=2e-13)


def test_pressure_feedback_closes_periodic_deformation_work_budget():
    grid, _layout, frame, f, ions = _make_problem()
    pressure = ElectronPressureCoupling(frame)
    electron_rate, ion_rate, diagnostics = jax.jit(pressure)(f, ions)
    velocity = ions[..., 1:4] / ions[..., :1]
    deformation_rate = frame.deformation(f, frame.velocity_gradient(velocity))
    measured_electron_work = electron_kinetic_energy_density(
        deformation_rate,
        frame.layout,
        grid.v,
        grid.dv,
    )

    local_defect = jnp.max(jnp.abs(measured_electron_work - diagnostics["electron_deformation_work"]))
    work_scale = jnp.max(jnp.abs(diagnostics["electron_deformation_work"]))
    assert local_defect / work_scale < 3.0e-3
    np.testing.assert_allclose(jnp.sum(ion_rate[..., 1:4], axis=(0, 1)), 0.0, atol=3e-13)
    total_work = jnp.sum(measured_electron_work + ion_rate[..., 4]) * grid.dx * grid.dy
    np.testing.assert_allclose(total_work, 0.0, atol=3e-12)
    np.testing.assert_allclose(
        electron_kinetic_energy_density(electron_rate, frame.layout, grid.v, grid.dv) + ion_rate[..., 4],
        0.0,
        atol=3e-13,
    )


def test_pressure_work_radial_defect_converges_at_second_order():
    defects = []
    for nv in (80, 160):
        grid, _layout, frame, f, ions = _make_problem(nx=8, ny=6, nv=nv)
        pressure = ElectronPressureCoupling(frame)
        _electron_rate, _ion_rate, diagnostics = pressure(f, ions)
        velocity = ions[..., 1:4] / ions[..., :1]
        deformation_rate = frame.deformation(f, frame.velocity_gradient(velocity))
        measured_work = electron_kinetic_energy_density(
            deformation_rate,
            frame.layout,
            grid.v,
            grid.dv,
        )
        defects.append(jnp.max(jnp.abs(measured_work - diagnostics["electron_deformation_work"])))

    assert defects[0] / defects[1] > 3.8
