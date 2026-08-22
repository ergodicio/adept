"""Conservative moment tests for finite-mass electron--ion relaxation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adept.vfp2d import (
    ElectronIonExchange,
    Grid,
    HarmonicLayout,
    IonFrameVlasov,
    TzoufrasVlasov,
    VelocityFrameRemap,
    density,
    electron_kinetic_energy_density,
    electron_momentum_density,
    primitive_to_conserved,
    scalar_velocity_moment,
)


def _make_state(nx=2, ny=2, nv=160, l_max=2, ion_mass=100.0, ion_temperature=0.15):
    grid = Grid(
        xmin=0.0,
        xmax=1.0,
        nx=nx,
        ymin=0.0,
        ymax=1.0,
        ny=ny,
        vmax=8.0,
        nv=nv,
        dt=0.01,
        l_max=l_max,
    )
    layout = HarmonicLayout(l_max)
    f = jnp.zeros((nx, ny, layout.size, nv), dtype=jnp.complex128)
    f00 = jnp.exp(-(grid.v**2))
    normalization = 4.0 * jnp.pi * jnp.sum(f00 * grid.v**2) * grid.dv
    f = f.at[..., layout.index(0, 0), :].set(f00 / normalization)

    ion_primitive = jnp.broadcast_to(
        jnp.asarray([ion_mass, 0.0, 0.0, 0.0, ion_temperature]),
        (nx, ny, 5),
    )
    ions = primitive_to_conserved(ion_primitive, gamma=5.0 / 3.0)
    exchange = ElectronIonExchange(layout, grid.v, grid.dv, ion_mass=ion_mass)
    return grid, layout, exchange, f, ions


@pytest.mark.parametrize(("ion_temperature", "energy_sign"), [(0.15, -1.0), (0.85, 1.0)])
def test_temperature_relaxation_preserves_density_and_total_energy(ion_temperature, energy_sign):
    grid, layout, exchange, f, ions = _make_state(ion_temperature=ion_temperature)
    dfdt, diondt, diagnostics = jax.jit(exchange)(
        f,
        ions,
        momentum_relaxation_rate=0.0,
        temperature_relaxation_rate=0.4,
    )

    np.testing.assert_allclose(density(dfdt, layout, grid.v, grid.dv), 0.0, atol=2e-14)
    np.testing.assert_allclose(diagnostics["momentum_residual"], 0.0, atol=2e-14)
    np.testing.assert_allclose(diagnostics["energy_residual"], 0.0, atol=2e-14)
    np.testing.assert_allclose(
        electron_kinetic_energy_density(dfdt, layout, grid.v, grid.dv) + diondt[..., 4],
        0.0,
        atol=2e-14,
    )
    assert jnp.all(energy_sign * diagnostics["electron_energy_rate"] > 0.0)
    assert jnp.all(energy_sign * diagnostics["ion_energy_rate"] < 0.0)
    np.testing.assert_allclose(diondt[..., 0], 0.0, atol=0.0)

    electron_temperature_rate = 2.0 * electron_kinetic_energy_density(dfdt, layout, grid.v, grid.dv) / 3.0
    expected = -0.4 * (diagnostics["electron_temperature"] - diagnostics["ion_temperature"])
    np.testing.assert_allclose(electron_temperature_rate, expected, rtol=2e-13, atol=2e-13)


def test_momentum_relaxation_is_equal_opposite_and_energy_neutral():
    grid, layout, exchange, f, ions = _make_state()
    i10, i11 = layout.index(1, 0), layout.index(1, 1)
    radial = grid.v * jnp.real(f[..., layout.index(0, 0), :])
    f = f.at[..., i10, :].set(0.08 * radial)
    f = f.at[..., i11, :].set((0.03 - 0.02j) * radial)
    momentum_before = electron_momentum_density(f, layout, grid.v, grid.dv)

    dfdt, diondt, diagnostics = exchange(
        f,
        ions,
        momentum_relaxation_rate=0.35,
        temperature_relaxation_rate=0.0,
    )

    np.testing.assert_allclose(
        electron_momentum_density(dfdt, layout, grid.v, grid.dv),
        -0.35 * momentum_before,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(diagnostics["momentum_residual"], 0.0, atol=2e-14)
    np.testing.assert_allclose(diagnostics["energy_residual"], 0.0, atol=2e-14)
    np.testing.assert_allclose(diondt[..., 1:4], 0.35 * momentum_before, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(electron_kinetic_energy_density(dfdt, layout, grid.v, grid.dv), 0.0, atol=2e-14)
    np.testing.assert_allclose(diondt[..., 4], 0.0, atol=2e-14)


def test_velocity_frame_remap_preserves_lab_moments_exactly():
    grid, layout, _exchange, f, _ions = _make_state(nx=3, ny=2, nv=192, l_max=3)
    f00 = jnp.real(f[..., layout.index(0, 0), :])
    f = f.at[..., layout.index(1, 0), :].set(0.06 * grid.v * f00)
    f = f.at[..., layout.index(1, 1), :].set((-0.025 + 0.035j) * grid.v * f00)
    f = f.at[..., layout.index(2, 0), :].set(0.015 * grid.v**2 * f00)

    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    remap = VelocityFrameRemap(IonFrameVlasov(vlasov))
    old_velocity = jnp.broadcast_to(jnp.asarray([0.12, -0.07, 0.04]), (*f.shape[:2], 3))
    velocity_change = jnp.broadcast_to(jnp.asarray([0.018, -0.011, 0.009]), old_velocity.shape)

    old_density = density(f, layout, grid.v, grid.dv)
    old_relative_momentum = electron_momentum_density(f, layout, grid.v, grid.dv)
    old_relative_energy = electron_kinetic_energy_density(f, layout, grid.v, grid.dv)
    translated = jax.jit(remap)(f, velocity_change)
    new_relative_momentum = electron_momentum_density(translated, layout, grid.v, grid.dv)
    new_relative_energy = electron_kinetic_energy_density(translated, layout, grid.v, grid.dv)

    target_momentum = old_relative_momentum - old_density[..., None] * velocity_change
    target_energy = old_relative_energy - jnp.sum(velocity_change * old_relative_momentum, axis=-1)
    target_energy += 0.5 * old_density * jnp.sum(velocity_change**2, axis=-1)
    np.testing.assert_allclose(density(translated, layout, grid.v, grid.dv), old_density, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(new_relative_momentum, target_momentum, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(new_relative_energy, target_energy, rtol=3e-13, atol=3e-13)

    new_velocity = old_velocity + velocity_change
    old_lab_momentum = old_relative_momentum + old_density[..., None] * old_velocity
    new_lab_momentum = new_relative_momentum + old_density[..., None] * new_velocity
    old_lab_energy = old_relative_energy + jnp.sum(old_velocity * old_relative_momentum, axis=-1)
    old_lab_energy += 0.5 * old_density * jnp.sum(old_velocity**2, axis=-1)
    new_lab_energy = new_relative_energy + jnp.sum(new_velocity * new_relative_momentum, axis=-1)
    new_lab_energy += 0.5 * old_density * jnp.sum(new_velocity**2, axis=-1)
    np.testing.assert_allclose(new_lab_momentum, old_lab_momentum, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(new_lab_energy, old_lab_energy, rtol=3e-13, atol=3e-13)


def test_combined_exchange_moves_temperatures_toward_equilibrium():
    grid, layout, exchange, f, ions = _make_state(ion_temperature=0.1)
    dfdt, diondt, diagnostics = exchange(
        f,
        ions,
        momentum_relaxation_rate=0.0,
        temperature_relaxation_rate=0.2,
    )
    dt = 0.01
    updated_f = f + dt * dfdt
    updated_ions = ions + dt * diondt
    electron_temperature_after = scalar_velocity_moment(updated_f, layout, grid.v, grid.dv, power=2) / 3.0
    ion_internal_energy = updated_ions[..., 4]
    ion_temperature_after = (5.0 / 3.0 - 1.0) * ion_internal_energy
    difference_before = diagnostics["electron_temperature"] - diagnostics["ion_temperature"]
    difference_after = electron_temperature_after - ion_temperature_after

    assert jnp.all(jnp.abs(difference_after) < jnp.abs(difference_before))
