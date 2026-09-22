"""External-source budgets and magnetic geometry for driven periodic buffers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    density,
    primitive_to_conserved,
)
from adept.vfp2d.reservoir import SOURCE_INVARIANTS, DrivenReservoirStep, boundary_buffer


def _problem(rate_scale=1.0, magnetic=True):
    grid = Grid(
        xmin=-np.pi,
        xmax=np.pi,
        nx=12,
        ymin=-np.pi,
        ymax=np.pi,
        ny=10,
        vmax=7.0,
        nv=32,
        dt=1.0e-4,
        l_max=2,
    )
    layout = HarmonicLayout(2)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    maxwell = Maxwell2D(grid.kx, grid.ky, c=2.0)
    electron = KineticOhmStep(
        vlasov,
        maxwell,
        KineticOhm2D(layout, grid.v, grid.dv, grid.kx, grid.ky),
        layout,
        grid.v,
        grid.dv,
        grid.dt,
        ion_frame=IonFrameVlasov(vlasov),
    )
    coupled = CoupledIonKineticStep(electron, IonEuler2D(grid.dx, grid.dy), grid.dt)
    radial = jnp.exp(-(grid.v**2))
    radial /= 4.0 * jnp.pi * jnp.sum(grid.v**2 * radial) * grid.dv
    shape = (grid.nx, grid.ny)
    f = jnp.zeros((*shape, layout.size, grid.nv), dtype=jnp.complex128)
    f = f.at[..., layout.index(0, 0), :].set(radial)
    primitive = jnp.zeros((*shape, 5)).at[..., 0].set(100.0).at[..., 4].set(0.3)
    field = jnp.zeros((*shape, 3))
    state = {
        "flm": f,
        "ions": primitive_to_conserved(primitive),
        "e": field,
        "b": field,
        "current_projection_energy": jnp.zeros(shape),
        **DrivenReservoirStep.initial_ledger(field),
    }
    target_primitive = primitive.at[..., 0].set(120.0).at[..., 1].set(0.02).at[..., 4].set(0.5)
    potential = field.at[..., 2].set(0.01 * jnp.cos(grid.x)[:, None] * jnp.cos(grid.y)[None, :])
    target_b = maxwell.curl(potential)
    target = {
        **state,
        "ions": primitive_to_conserved(target_primitive),
        "flm": electron._project(1.2 * f, target_b),
        "b": target_b,
    }
    rate = rate_scale * boundary_buffer(grid, x_width=1.6, y_width=1.6)
    driven = DrivenReservoirStep(coupled, target, rate, ion_mass=100.0, ion_charge=1.0, magnetic=magnetic)
    return grid, driven, state, target


def test_particle_mixture_is_quasineutral_and_has_the_exact_ion_relaxation():
    grid, driven, state, target = _problem()
    dt = 0.2
    result = jax.jit(driven.source)(0.0, state, None, dt)
    alpha = -jnp.expm1(-driven.rate * dt)
    np.testing.assert_allclose(
        result["ions"],
        state["ions"] + alpha[..., None] * (target["ions"] - state["ions"]),
        atol=2e-14,
    )
    ne = density(result["flm"], driven.electrons.layout, grid.v, grid.dv)
    np.testing.assert_allclose(ne, result["ions"][..., 0] / 100.0, rtol=3e-13, atol=3e-13)
    assert np.max(np.abs(np.asarray(result["ions"] - state["ions"]))) > 0.01


def test_measured_source_ledgers_close_each_invariant_including_magnetic_work():
    _grid, driven, state, _target = _problem()
    before = driven._invariants(state)
    result = jax.jit(driven.source)(0.0, state, None, 0.3)
    after = driven._invariants(result)
    for name in SOURCE_INVARIANTS:
        np.testing.assert_allclose(after[name] - result[f"reservoir_{name}"], before[name], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        result["reservoir_total_energy"],
        result["reservoir_electron_energy"] + result["reservoir_ion_energy"] + result["reservoir_magnetic_energy"],
        rtol=2e-12,
        atol=2e-12,
    )
    assert result["reservoir_magnetic_energy"] > 0
    np.testing.assert_array_equal(result["current_projection_energy"], state["current_projection_energy"])
    np.testing.assert_allclose(result["reservoir_magnetic_field_change"], result["b"] - state["b"], atol=1e-16)
    second = driven.source(0.3, result, None, 0.2)
    np.testing.assert_allclose(second["reservoir_magnetic_field_change"], second["b"] - state["b"], atol=1e-16)


def test_particle_and_momentum_injection_match_the_prescribed_reservoir():
    grid, driven, state, _target = _problem(magnetic=False)
    duration = 0.3
    result = jax.jit(driven.source)(0.0, state, None, duration)
    integrated_fraction = grid.dx * grid.dy * jnp.sum(-jnp.expm1(-driven.rate * duration))
    # The reservoir changes ni=ne from 1 to 1.2 and injects ions at ux=.02.
    # With zero mean current, electrons carry the same bulk velocity and add
    # the known 1/100 electron-to-ion mass correction to integrated momentum.
    expected_particles = 0.2 * integrated_fraction
    expected_px = (120.0 + 1.2) * 0.02 * integrated_fraction
    np.testing.assert_allclose(result["reservoir_electron_number"], expected_particles, rtol=3e-12, atol=3e-13)
    np.testing.assert_allclose(result["reservoir_ion_number"], expected_particles, rtol=3e-12, atol=3e-13)
    np.testing.assert_allclose(result["reservoir_total_momentum"], [expected_px, 0.0, 0.0], rtol=3e-12, atol=3e-13)
    np.testing.assert_array_equal(result["b"], state["b"])


def test_nonuniform_magnetic_drive_is_divergence_free_and_preserves_mean_flux():
    _grid, driven, state, _target = _problem()
    result = jax.jit(driven.source)(0.0, state, None, 0.5)
    maxwell = driven.electrons.maxwell
    b = result["b"]
    divergence = maxwell.ddx(b[..., 0]) + maxwell.ddy(b[..., 1])
    assert jnp.max(jnp.abs(b)) > 1e-4
    np.testing.assert_allclose(divergence, 0.0, atol=3e-15)
    np.testing.assert_allclose(jnp.mean(b, axis=(0, 1)), jnp.mean(state["b"], axis=(0, 1)), atol=2e-17)


def test_zero_rate_preserves_nonzero_current_residual_and_has_no_source_budget():
    _grid, driven, state, _target = _problem(rate_scale=0.0)
    state["flm"] = (
        state["flm"]
        .at[..., driven.electrons.layout.index(1, 0), :]
        .set(0.01 * state["flm"][..., 0, :] * driven.electrons.v)
    )
    result = jax.jit(driven.source)(0.0, state, None, 1.0)
    for name in ("flm", "ions", "b"):
        np.testing.assert_allclose(result[name], state[name], rtol=3e-13, atol=3e-14)
    for name in SOURCE_INVARIANTS:
        np.testing.assert_allclose(result[f"reservoir_{name}"], 0.0, atol=3e-12)


def test_source_wrapper_retains_ledgers_across_physical_steps():
    _grid, driven, state, _target = _problem()
    advance = jax.jit(driven)
    first = advance(0.0, state)
    second = advance(driven.dt, first)
    assert set(second) == set(state)
    assert second["reservoir_ion_number"] > first["reservoir_ion_number"] > 0
    for value in second.values():
        assert jnp.all(jnp.isfinite(value))


def test_buffer_has_compact_support_and_rejects_domain_filling_widths():
    grid, _driven, _state, _target = _problem()
    mask = np.asarray(boundary_buffer(grid, x_width=1.6, y_width=0.0))
    np.testing.assert_array_equal(mask[4:8], 0.0)
    np.testing.assert_allclose(mask, mask[::-1], atol=1e-15)
    assert np.min(mask) >= 0.0 and np.max(mask) <= 1.0
    with pytest.raises(ValueError, match="half-box"):
        boundary_buffer(grid, x_width=np.pi)
    with pytest.raises(ValueError, match="positive"):
        boundary_buffer(grid)


def test_reservoir_density_reaches_both_collision_half_steps_instead_of_stale_args():
    grid, original, state, target = _problem(rate_scale=2.0e4, magnetic=False)
    # Inject a stationary, constant-pressure contact: hydro cannot change its
    # density before the kinetic midpoint, so the collision density is analytic.
    target = {
        **target,
        "ions": state["ions"].at[..., 0].set(120.0),
    }
    electron = original.electrons
    seen_density = []

    def record_collision(flm, *, Z, ni, dt, **heating):
        seen_density.append(np.asarray(ni))
        return flm

    electron.collisions = record_collision
    coupled = CoupledIonKineticStep(
        electron,
        original.step.hydro,
        grid.dt,
        exchange=ElectronIonExchange(electron.layout, grid.v, grid.dv, ion_mass=100.0),
    )
    driven = DrivenReservoirStep(coupled, target, original.rate, ion_mass=100.0, ion_charge=1.0, magnetic=False)
    stale_density = jnp.full((grid.nx, grid.ny), 0.125)
    # Eager invocation records the actual arrays delivered through
    # KineticOhmStep._collide, after the reservoir and physical hydro half-step.
    result = driven(0.0, state, {"ni": stale_density, "Z": 1.0})
    expected_density = 1.0 + 0.2 * (-jnp.expm1(-0.5 * grid.dt * driven.rate))
    assert len(seen_density) == 2
    assert np.max(expected_density) > 1.1
    for collision_density in seen_density:
        np.testing.assert_allclose(collision_density, expected_density, rtol=3e-13, atol=3e-14)
        assert not np.allclose(collision_density, stale_density)
    assert result["reservoir_ion_number"] > 0.0


def test_noop_preserves_a_weak_accumulated_magnetic_source_ledger():
    _grid, driven, state, _target = _problem(rate_scale=0.0, magnetic=False)
    state["b"] = jnp.full_like(state["b"], 1.0e-4)
    state["reservoir_magnetic_field_change"] = jnp.full_like(state["b"], 1.0e-24)
    # A zero increment must preserve existing injection even when it is smaller
    # than one ulp of the live field. Adding B before subtracting old B loses it.
    result = driven.source(0.0, state, None, 1.0)
    np.testing.assert_array_equal(result["b"], state["b"])
    np.testing.assert_array_equal(result["reservoir_magnetic_field_change"], state["reservoir_magnetic_field_change"])
