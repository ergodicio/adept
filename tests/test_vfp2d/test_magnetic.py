"""Magnetic force/work gates; the wave test isolates the ideal magnetic split."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_vfp2d.test_coupling import _make_problem

from adept.vfp2d import (
    CoupledIonKineticStep,
    ElectronIonExchange,
    ElectronPressureCoupling,
    IonMagneticCoupling,
    Maxwell2D,
    coupled_invariants,
    current,
    density,
    primitive_to_conserved,
)


@pytest.mark.parametrize("finite_difference", [False, True])
def test_magnetic_pressure_tension_and_periodic_work_balance(finite_difference):
    grid, _, maxwell, _, _, _, _, zeros, _, _ = _make_problem(nx=24, ny=16)
    if finite_difference:
        maxwell = Maxwell2D(grid.kx, grid.ky, c=5.0, dx=grid.dx, dy=grid.dy)
    magnetic = zeros.at[..., 0].set(0.4).at[..., 1].set(0.03 * jnp.sin(grid.x)[:, None])
    source = IonMagneticCoupling(maxwell)
    by_derivative = maxwell.ddx(magnetic[..., 1])
    expected = jnp.stack(
        (
            -25.0 * magnetic[..., 1] * by_derivative,
            25.0 * magnetic[..., 0] * by_derivative,
            jnp.zeros_like(by_derivative),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(source.force(magnetic), expected, atol=2e-14)
    velocity = zeros.at[..., 0].set(0.1 * jnp.cos(grid.x)[:, None])
    velocity = velocity.at[..., 1].set(0.2 * jnp.cos(grid.x)[:, None])
    ions = primitive_to_conserved(
        jnp.concatenate((jnp.ones((*zeros.shape[:-1], 1)), velocity, jnp.ones((*zeros.shape[:-1], 1))), axis=-1)
    )
    rate, diagnostics = source(magnetic, ions)
    magnetic_rate = maxwell.curl(jnp.cross(velocity, magnetic))
    magnetic_work = maxwell.c2 * jnp.sum(magnetic * magnetic_rate)
    ion_work = jnp.sum(rate[..., 4])
    assert abs(float(ion_work)) > 1e-3
    np.testing.assert_allclose(magnetic_work + ion_work, 0.0, atol=2e-13)
    np.testing.assert_allclose(diagnostics["ion_magnetic_work"], rate[..., 4])
    np.testing.assert_allclose(jnp.sum(rate[..., 1:4], axis=(0, 1)), 0.0, atol=2e-13)


def test_magnetic_kick_preserves_ion_internal_energy_and_electron_lab_moments():
    grid, layout, maxwell, _, moving, hydro, flm, zeros, ions, mass = _make_problem(nx=16)
    magnetic = zeros.at[..., 0].set(0.2).at[..., 1].set(0.01 * jnp.sin(grid.x)[:, None])
    flm = moving._project(flm, magnetic)
    coupled = CoupledIonKineticStep(moving, hydro, moving.dt)
    source_dt = 0.1
    new_f, new_ions = coupled._exchange(0.0, flm, ions, {}, source_dt, magnetic)
    force = jnp.cross(maxwell.c2 * maxwell.curl(magnetic), magnetic)
    np.testing.assert_allclose(new_ions[..., 1:4] - ions[..., 1:4], source_dt * force, atol=2e-16)
    kinetic_before = jnp.sum(ions[..., 1:4] ** 2, axis=-1) / (2.0 * ions[..., 0])
    kinetic_after = jnp.sum(new_ions[..., 1:4] ** 2, axis=-1) / (2.0 * new_ions[..., 0])
    np.testing.assert_allclose(new_ions[..., 4] - kinetic_after, ions[..., 4] - kinetic_before, atol=2e-15)
    np.testing.assert_allclose(
        density(new_f, layout, grid.v, grid.dv), density(flm, layout, grid.v, grid.dv), atol=2e-14
    )
    common = {
        "layout": layout,
        "v": grid.v,
        "dv": grid.dv,
        "dx": grid.dx,
        "dy": grid.dy,
        "ion_mass": mass,
        "ion_charge": 1.0,
        "light_speed": 5.0,
    }
    before = coupled_invariants(flm, ions, magnetic, **common)
    after = coupled_invariants(new_f, new_ions, magnetic, **common)
    np.testing.assert_allclose(after["electron_energy"], before["electron_energy"], atol=2e-13)
    np.testing.assert_allclose(after["quasineutrality_linf"], before["quasineutrality_linf"], atol=2e-14)
    assert float(after["ion_energy"] - before["ion_energy"]) > 0.0


def test_electron_and_magnetic_pressure_balance_has_no_net_ion_force():
    grid, _, maxwell, _, moving, _, flm, zeros, ions, mass = _make_problem(nx=24, ny=4, nv=64)
    magnetic = zeros.at[..., 1].set(0.1 + 0.01 * jnp.sin(grid.x)[:, None])
    pressure = ElectronPressureCoupling(moving.ion_frame)
    # A fixed-temperature Maxwellian can carry the prescribed scalar pressure
    # through its density, avoiding a velocity-discretization temperature error.
    target_pressure = 0.5 - 0.5 * maxwell.c2 * magnetic[..., 1] ** 2
    old_pressure = pressure.pressure_tensor(flm)[..., 0, 0]
    flm = flm * (target_pressure / old_pressure)[..., None, None]
    ions = ions.at[..., 0].set(mass * density(flm, moving.layout, grid.v, grid.dv))
    _, pressure_rate, _ = pressure(flm, ions)
    magnetic_rate, _ = IonMagneticCoupling(maxwell)(magnetic, ions)
    np.testing.assert_allclose(pressure_rate[..., 1:4] + magnetic_rate[..., 1:4], 0.0, atol=3e-14)


def test_coupled_step_accelerates_ions_with_magnetic_tension():
    grid, layout, maxwell, _, moving, hydro, flm, zeros, ions, _ = _make_problem(nx=16, nv=32, dt=1e-5)
    flm = jnp.broadcast_to(flm[:1, :1], flm.shape)
    ions = jnp.broadcast_to(ions[:1, :1], ions.shape)
    magnetic = zeros.at[..., 0].set(0.02).at[..., 1].set(0.001 * jnp.sin(grid.x)[:, None])
    flm = moving._project(flm, magnetic)
    result = jax.jit(CoupledIonKineticStep(moving, hydro, moving.dt))(
        0.0,
        {"flm": flm, "ions": ions, "e": zeros, "b": magnetic},
        {},
    )
    force = jnp.cross(maxwell.c2 * maxwell.curl(magnetic), magnetic)
    measured = (result["ions"][..., 1:4] - ions[..., 1:4]) / moving.dt
    # The Hall field rotates B during the step, creating an O(dt) correction
    # to the force. Compare against the initial force on its nonzero scale.
    assert float(jnp.max(jnp.abs(measured - force)) / jnp.max(jnp.abs(force))) < 3e-6
    np.testing.assert_allclose(
        current(result["flm"], layout, grid.v, grid.dv), maxwell.c2 * maxwell.curl(result["b"]), atol=2e-14
    )
    assert bool(jnp.all(jnp.isfinite(result["flm"])))


def test_frozen_ions_preserve_magnetic_kinetic_regression():
    grid, layout, _, stationary, moving, hydro, flm, zeros, ions, mass = _make_problem(nx=12, nv=32, dt=1e-5)
    magnetic = zeros.at[..., 0].set(0.02).at[..., 1].set(0.001 * jnp.sin(grid.x)[:, None])
    state = {"flm": flm, "e": zeros, "b": magnetic}
    reference = stationary(0.0, state)
    result = jax.jit(
        CoupledIonKineticStep(
            moving,
            hydro,
            moving.dt,
            evolve_ions=False,
            pressure=ElectronPressureCoupling(moving.ion_frame),
            exchange=ElectronIonExchange(layout, grid.v, grid.dv, ion_mass=mass),
        )
    )(
        0.0,
        {**state, "ions": ions},
        {"ei_momentum_relaxation_rate": 0.4, "ei_temperature_relaxation_rate": 0.4},
    )
    np.testing.assert_array_equal(result["ions"], ions)
    for key in ("flm", "e", "b"):
        np.testing.assert_allclose(result[key], reference[key], atol=3e-12, rtol=3e-12)


def test_collisions_receive_updated_midpoint_ion_density(monkeypatch):
    _, layout, _, _, moving, hydro, flm, zeros, ions, mass = _make_problem(nx=6, nv=24)
    exchange = ElectronIonExchange(layout, moving.v, moving.dv, ion_mass=mass)
    coupled = CoupledIonKineticStep(moving, hydro, moving.dt, exchange=exchange)
    seen_density = []
    original = moving._collide

    def record_density(t, f, args, dt):
        seen_density.append(args["ni"])
        return original(t, f, args, dt)

    monkeypatch.setattr(moving, "_collide", record_density)
    coupled(0.0, {"flm": flm, "ions": ions, "e": zeros, "b": zeros}, {"ni": 99.0})
    _, midpoint_ions = coupled._hydro_half_step(flm, ions)
    expected = midpoint_ions[..., 0] / mass
    assert len(seen_density) == 2
    for measured in seen_density:
        np.testing.assert_allclose(measured, expected, atol=2e-15)


def _ideal_alfven_wave(steps):
    """Exercise force plus ideal induction alone for one full Alfven period."""
    grid, _, maxwell, _, _, _, _, zeros, _, _ = _make_problem(nx=16, ny=4, nv=16)
    source = IonMagneticCoupling(maxwell)
    rho, b0, amplitude = 25.0, 1.0, 0.01
    magnetic = zeros.at[..., 0].set(b0)
    magnetic = magnetic.at[..., 1].set(amplitude * jnp.cos(grid.x)[:, None])
    magnetic = magnetic.at[..., 2].set(amplitude * jnp.sin(grid.x)[:, None])
    velocity = -magnetic.at[..., 0].set(0.0) * np.sqrt(maxwell.c2 / rho)
    ions = primitive_to_conserved(
        jnp.concatenate((rho * jnp.ones((*zeros.shape[:-1], 1)), velocity, jnp.ones((*zeros.shape[:-1], 1))), axis=-1)
    )
    dt = 2.0 * np.pi / steps  # v_A = c B0 / sqrt(rho) = 1.

    def kick(i, b):
        first, _ = source(b, i)
        midpoint, _ = source(b, i + 0.25 * dt * first)
        return i + 0.5 * dt * midpoint

    initial_energy = jnp.sum(ions[..., 4] + 0.5 * maxwell.c2 * jnp.sum(magnetic**2, axis=-1))

    def advance(_, state):
        i, b, max_energy_error = state
        i = kick(i, b)
        u = i[..., 1:4] / i[..., :1]

        def induction(field):
            return maxwell.curl(jnp.cross(u, field))

        k1 = induction(b)
        k2 = induction(b + 0.5 * dt * k1)
        k3 = induction(b + 0.5 * dt * k2)
        k4 = induction(b + dt * k3)
        b = b + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        i = kick(i, b)
        energy = jnp.sum(i[..., 4] + 0.5 * maxwell.c2 * jnp.sum(b**2, axis=-1))
        return i, b, jnp.maximum(max_energy_error, jnp.abs(energy - initial_energy))

    new_ions, new_b, max_energy_error = jax.jit(
        lambda i, b: jax.lax.fori_loop(0, steps, advance, (i, b, jnp.asarray(0.0)))
    )(ions, magnetic)
    # Normalize by the perturbation energy, never by the large guide field or
    # background thermal reservoir; this makes the gate sensitive to wave work.
    wave_energy = jnp.sum(
        0.5 * rho * jnp.sum(velocity**2, axis=-1) + 0.5 * maxwell.c2 * jnp.sum(magnetic[..., 1:] ** 2, axis=-1)
    )
    phase_error = jnp.sqrt(jnp.mean((new_b[..., 1:] - magnetic[..., 1:]) ** 2)) / amplitude
    return float(phase_error), float(max_energy_error / wave_energy)


def test_ideal_magnetic_split_propagates_alfven_wave_and_converges():
    coarse_phase, coarse_energy = _ideal_alfven_wave(64)
    fine_phase, fine_energy = _ideal_alfven_wave(128)
    assert coarse_phase / fine_phase > 3.8
    assert fine_phase < 5e-4
    assert coarse_energy / fine_energy > 8.0
    assert fine_energy < 1e-6


@cache
def _finite_magnetic_energy_defect(nv, dt, final_time=0.5, conserve_electric_work=True):
    """Expose finite-radial-grid work errors on a nonzero magnetic reservoir.

    This is a convergence diagnostic, not a claim that a percent-level defect
    is acceptable for experiment modelling. Unlike the zero-field short gate,
    it normalizes to the field perturbation that can accelerate the plasma.
    """
    grid, layout, maxwell, _, moving, hydro, flm, zeros, ions, mass = _make_problem(
        nx=12,
        ny=4,
        nv=nv,
        dt=dt,
    )
    moving.vlasov.conserve_electric_work = bool(conserve_electric_work)
    flm = jnp.broadcast_to(flm[:1, :1], flm.shape)
    ions = jnp.broadcast_to(ions[:1, :1], ions.shape)
    magnetic = zeros.at[..., 0].set(0.2)
    magnetic = magnetic.at[..., 1].set(0.01 * jnp.cos(grid.x)[:, None])
    magnetic = magnetic.at[..., 2].set(0.01 * jnp.sin(grid.x)[:, None])
    flm = moving._project(flm, magnetic)
    coupled = CoupledIonKineticStep(
        moving,
        hydro,
        dt,
        pressure=ElectronPressureCoupling(moving.ion_frame),
    )
    state = {
        "flm": flm,
        "e": zeros,
        "b": magnetic,
        "ions": ions,
        "current_projection_energy": jnp.zeros(zeros.shape[:-1]),
    }

    def invariants(s):
        return coupled_invariants(
            s["flm"],
            s["ions"],
            s["b"],
            layout,
            grid.v,
            grid.dv,
            dx=grid.dx,
            dy=grid.dy,
            ion_mass=mass,
            ion_charge=1.0,
            light_speed=5.0,
            current_projection_energy=s["current_projection_energy"],
        )

    initial = invariants(state)
    wave_energy = 0.5 * maxwell.c2 * grid.dx * grid.dy * jnp.sum(magnetic[..., 1:] ** 2)
    advance = jax.jit(coupled)
    for i in range(round(final_time / dt)):
        state = advance(i * dt, state, {})
    final = invariants(state)
    return {
        "raw": float((final["total_energy"] - initial["total_energy"]) / wave_energy),
        "accounted": float((final["accounted_total_energy"] - initial["accounted_total_energy"]) / wave_energy),
        "projection": float(final["current_projection_energy"] / wave_energy),
        "quasineutrality": float(final["quasineutrality_linf"]),
    }


def test_uncorrected_finite_magnetic_energy_defect_decreases_under_radial_refinement():
    coarse = _finite_magnetic_energy_defect(32, 0.01, conserve_electric_work=False)
    time_fine = _finite_magnetic_energy_defect(32, 0.005, conserve_electric_work=False)
    radial_fine = _finite_magnetic_energy_defect(96, 0.005, conserve_electric_work=False)
    for result in (coarse, time_fine, radial_fine):
        np.testing.assert_allclose(result["raw"] - result["projection"], result["accounted"], atol=2e-12)
        assert result["quasineutrality"] < 1e-12
    assert abs(coarse["accounted"] - time_fine["accounted"]) < 1e-5
    assert abs(time_fine["accounted"]) / abs(radial_fine["accounted"]) > 8.0
    # Declared bounded-problem gate: less than 1% of the transverse magnetic
    # perturbation energy at t=0.5, with projection work reported separately.
    assert abs(radial_fine["accounted"]) < 0.01
