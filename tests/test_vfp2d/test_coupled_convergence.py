"""Nonlinear conservation and refinement gate for coupled VFP-2D."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from adept.vfp2d import (
    CoupledIonKineticStep,
    ElectronPressureCoupling,
    Grid,
    HarmonicLayout,
    IonEuler2D,
    IonFrameVlasov,
    KineticOhm2D,
    KineticOhmStep,
    Maxwell2D,
    TzoufrasVlasov,
    coupled_invariants,
    primitive_to_conserved,
)

ENERGY_DEFECT_TOLERANCE = 3.1e-9


@cache
def _run_nonlinear_benchmark(dt=1.0e-3, nx=8, nv=64):
    final_time = 4.0e-3
    grid = Grid(
        xmin=0.0,
        xmax=2.0 * np.pi,
        nx=nx,
        ymin=0.0,
        ymax=2.0 * np.pi,
        ny=nx,
        vmax=7.0,
        nv=nv,
        dt=dt,
        l_max=2,
    )
    layout = HarmonicLayout(2)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    maxwell = Maxwell2D(grid.kx, grid.ky, c=5.0)
    electron_step = KineticOhmStep(
        vlasov,
        maxwell,
        KineticOhm2D(layout, grid.v, grid.dv, grid.kx, grid.ky),
        layout,
        grid.v,
        grid.dv,
        dt,
        ion_frame=IonFrameVlasov(vlasov),
    )

    x, y = grid.x[:, None], grid.y[None, :]
    electron_density = 1.0 + 0.08 * jnp.cos(x) * jnp.cos(y)
    electron_temperature = 0.5 * (1.0 + 0.08 * jnp.sin(x - 2.0 * y))
    radial = jnp.exp(-(grid.v[None, None, :] ** 2) / (2.0 * electron_temperature[..., None]))
    radial_density = 4.0 * jnp.pi * jnp.sum(radial * grid.v**2, axis=-1) * grid.dv
    radial *= electron_density[..., None] / radial_density[..., None]
    flm = jnp.zeros((nx, nx, layout.size, nv), dtype=jnp.complex128)
    flm = flm.at[..., layout.index(0, 0), :].set(radial)

    ion_mass = 100.0
    ion_primitive = jnp.concatenate(
        (
            (ion_mass * electron_density)[..., None],
            jnp.zeros((nx, nx, 3)),
            (0.2 * electron_density)[..., None],
        ),
        axis=-1,
    )
    ions = primitive_to_conserved(ion_primitive)
    field = jnp.zeros((nx, nx, 3))
    state = {
        "flm": flm,
        "e": field,
        "b": field,
        "ions": ions,
        "current_projection_energy": jnp.zeros((nx, nx)),
    }
    coupled = CoupledIonKineticStep(
        electron_step,
        IonEuler2D(grid.dx, grid.dy),
        dt,
        pressure=ElectronPressureCoupling(electron_step.ion_frame),
    )

    def invariants(current_state):
        return coupled_invariants(
            current_state["flm"],
            current_state["ions"],
            current_state["b"],
            layout,
            grid.v,
            grid.dv,
            dx=grid.dx,
            dy=grid.dy,
            ion_mass=ion_mass,
            ion_charge=1.0,
            light_speed=5.0,
            current_projection_energy=current_state["current_projection_energy"],
        )

    initial = invariants(state)
    advance = jax.jit(coupled)
    for step in range(round(final_time / dt)):
        state = advance(step * dt, state, {})
    final = invariants(state)
    energy_scale = initial["total_energy"]
    raw_energy_error = (final["total_energy"] - initial["total_energy"]) / energy_scale
    accounted_energy_error = (final["accounted_total_energy"] - initial["accounted_total_energy"]) / energy_scale
    projection_energy = final["current_projection_energy"] / energy_scale
    ion_velocity = state["ions"][..., 1:4] / state["ions"][..., :1]
    return tuple(
        float(value)
        for value in (
            raw_energy_error,
            accounted_energy_error,
            projection_energy,
            jnp.max(jnp.linalg.norm(ion_velocity, axis=-1)),
            jnp.max(jnp.abs(state["b"][..., 2])),
        )
    )


def test_nonlinear_energy_budget_is_below_tolerance_and_second_order_in_time():
    coarse = _run_nonlinear_benchmark(dt=2.0e-3)
    medium = _run_nonlinear_benchmark(dt=1.0e-3)
    fine = _run_nonlinear_benchmark(dt=5.0e-4)

    for raw_error, accounted_error, projection_energy, _ion_velocity, _magnetic_field in (coarse, medium, fine):
        np.testing.assert_allclose(raw_error - projection_energy, accounted_error, atol=2e-16)
        assert abs(accounted_error) < ENERGY_DEFECT_TOLERANCE

    ion_order = np.log2(abs(coarse[3] - medium[3]) / abs(medium[3] - fine[3]))
    magnetic_order = np.log2(abs(coarse[4] - medium[4]) / abs(medium[4] - fine[4]))
    assert ion_order > 1.8
    assert magnetic_order > 1.8
    assert fine[3] > 0.0
    assert fine[4] > 0.0


def test_nonlinear_energy_defect_converges_under_spatial_and_radial_refinement():
    radial_coarse = _run_nonlinear_benchmark(nv=24)
    radial_fine = _run_nonlinear_benchmark(nv=96)
    spatial_coarse = _run_nonlinear_benchmark(nx=6)
    spatial_fine = _run_nonlinear_benchmark(nx=12)

    assert abs(radial_coarse[1]) / abs(radial_fine[1]) > 8.0
    assert abs(radial_fine[1]) < 2.0e-9
    assert abs(spatial_coarse[1]) / abs(spatial_fine[1]) > 8.0
    assert abs(spatial_fine[1]) < 1.0e-9
