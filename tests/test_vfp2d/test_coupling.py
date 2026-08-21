"""Coupled local-limit tests for the Gate 2a production split."""

import jax
import jax.numpy as jnp
import numpy as np

from adept.vfp2d import (
    CoupledIonKineticStep,
    ElectronIonExchange,
    Grid,
    HarmonicLayout,
    IonEuler2D,
    IonFrameVlasov,
    KineticOhm2D,
    KineticOhmStep,
    Maxwell2D,
    TzoufrasVlasov,
    coupled_invariants,
    density,
    primitive_to_conserved,
)


def _make_problem(nx=6, ny=4, nv=48, dt=2.0e-3):
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
    ohm = KineticOhm2D(layout, grid.v, grid.dv, grid.kx, grid.ky)
    stationary_step = KineticOhmStep(vlasov, maxwell, ohm, layout, grid.v, grid.dv, dt)
    moving_step = KineticOhmStep(
        vlasov,
        maxwell,
        ohm,
        layout,
        grid.v,
        grid.dv,
        dt,
        ion_frame=IonFrameVlasov(vlasov),
    )

    profile = 1.0 + 0.05 * jnp.cos(grid.x)[:, None] * jnp.ones((1, grid.ny))
    radial = jnp.exp(-(grid.v**2))
    radial /= 4.0 * jnp.pi * jnp.sum(radial * grid.v**2) * grid.dv
    flm = jnp.zeros((nx, ny, layout.size, nv), dtype=jnp.complex128)
    flm = flm.at[..., layout.index(0, 0), :].set(profile[..., None] * radial)
    field = jnp.zeros((nx, ny, 3))

    ion_mass = 100.0
    ion_primitive = jnp.concatenate(
        (
            (ion_mass * profile)[..., None],
            jnp.zeros((nx, ny, 3)),
            (0.2 * jnp.ones_like(profile))[..., None],
        ),
        axis=-1,
    )
    ions = primitive_to_conserved(ion_primitive)
    hydro = IonEuler2D(grid.dx, grid.dy)
    return grid, layout, maxwell, stationary_step, moving_step, hydro, flm, field, ions, ion_mass


def test_zero_velocity_coupling_reproduces_frozen_ion_kinetic_step():
    _grid, _layout, _maxwell, stationary, moving, hydro, flm, field, ions, _ion_mass = _make_problem()
    state = {"flm": flm, "e": field, "b": field}
    reference = stationary(0.0, state)
    coupled = CoupledIonKineticStep(moving, hydro, moving.dt, evolve_ions=False)
    result = jax.jit(coupled)(0.0, {**state, "ions": ions})

    for key in ("flm", "e", "b"):
        np.testing.assert_allclose(result[key], reference[key], rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(result["ions"], ions, rtol=0.0, atol=0.0)


def test_lab_electric_field_recovers_ideal_bulk_magnetic_advection():
    grid, _layout, maxwell, stationary, moving, _hydro, flm, field, _ions, _ion_mass = _make_problem(nx=16)
    magnetic = field.at[..., 2].set(jnp.sin(grid.x)[:, None])
    ion_velocity = jnp.broadcast_to(jnp.asarray([0.3, -0.1, 0.0]), magnetic.shape)
    hidden = jnp.zeros(magnetic.shape[:-1])
    stationary_e, _stationary_terms = stationary.electric_field(flm, magnetic, {}, hidden_dndz=hidden)
    moving_e, terms = moving.electric_field(
        flm,
        magnetic,
        {"ion_velocity": ion_velocity},
        hidden_dndz=hidden,
    )

    bulk_e = moving_e - stationary_e
    np.testing.assert_allclose(bulk_e, -jnp.cross(ion_velocity, magnetic), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(terms["bulk"], bulk_e, rtol=2e-15, atol=2e-15)
    magnetic_rate = -maxwell.curl(bulk_e)
    expected_bz_rate = -ion_velocity[..., 0] * jnp.cos(grid.x)[:, None]
    np.testing.assert_allclose(magnetic_rate[..., 2], expected_bz_rate, rtol=2e-12, atol=2e-12)
    divergence_rate = maxwell.ddx(magnetic_rate[..., 0]) + maxwell.ddy(magnetic_rate[..., 1])
    np.testing.assert_allclose(divergence_rate, 0.0, atol=2e-13)


def test_midpoint_temperature_exchange_preserves_coupled_energy():
    grid, layout, _maxwell, _stationary, moving, hydro, flm, field, ions, ion_mass = _make_problem()
    exchange = ElectronIonExchange(layout, grid.v, grid.dv, ion_mass=ion_mass)
    coupled = CoupledIonKineticStep(moving, hydro, moving.dt, exchange=exchange, evolve_ions=False)
    before = coupled_invariants(
        flm,
        ions,
        field,
        layout,
        grid.v,
        grid.dv,
        dx=grid.dx,
        dy=grid.dy,
        ion_mass=ion_mass,
        ion_charge=1.0,
        light_speed=5.0,
    )
    updated_f, updated_ions = jax.jit(coupled._exchange)(
        0.0,
        flm,
        ions,
        {"ei_temperature_relaxation_rate": 0.4},
    )
    after = coupled_invariants(
        updated_f,
        updated_ions,
        field,
        layout,
        grid.v,
        grid.dv,
        dx=grid.dx,
        dy=grid.dy,
        ion_mass=ion_mass,
        ion_charge=1.0,
        light_speed=5.0,
    )

    np.testing.assert_allclose(after["electron_number"], before["electron_number"], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(after["ion_number"], before["ion_number"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(after["total_energy"], before["total_energy"], rtol=2e-14, atol=2e-14)
    assert jnp.all(jnp.abs(density(updated_f - flm, layout, grid.v, grid.dv)) < 2e-14)
    assert after["electron_energy"] < before["electron_energy"]
    assert after["ion_energy"] > before["ion_energy"]
