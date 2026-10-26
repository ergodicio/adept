"""Gate 0b benchmarks for the conservative ion-fluid backbone."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adept.vfp2d import IonEuler2D, conserved_to_primitive, primitive_to_conserved

GAMMA = 1.4


def _pressure_function(pressure, density, reference_pressure, sound_speed):
    if pressure > reference_pressure:
        a = 2.0 / ((GAMMA + 1.0) * density)
        b = (GAMMA - 1.0) * reference_pressure / (GAMMA + 1.0)
        root = np.sqrt(a / (pressure + b))
        value = (pressure - reference_pressure) * root
        derivative = root * (1.0 - 0.5 * (pressure - reference_pressure) / (pressure + b))
        return value, derivative
    exponent = (GAMMA - 1.0) / (2.0 * GAMMA)
    ratio = pressure / reference_pressure
    value = 2.0 * sound_speed * (ratio**exponent - 1.0) / (GAMMA - 1.0)
    derivative = ratio ** (-(GAMMA + 1.0) / (2.0 * GAMMA)) / (density * sound_speed)
    return value, derivative


def _star_state(left, right):
    rho_l, velocity_l, pressure_l = left
    rho_r, velocity_r, pressure_r = right
    sound_l = np.sqrt(GAMMA * pressure_l / rho_l)
    sound_r = np.sqrt(GAMMA * pressure_r / rho_r)
    guess = max(
        1.0e-12,
        0.5 * (pressure_l + pressure_r) - 0.125 * (velocity_r - velocity_l) * (rho_l + rho_r) * (sound_l + sound_r),
    )
    pressure = guess
    for _ in range(30):
        value_l, derivative_l = _pressure_function(pressure, rho_l, pressure_l, sound_l)
        value_r, derivative_r = _pressure_function(pressure, rho_r, pressure_r, sound_r)
        updated = pressure - (value_l + value_r + velocity_r - velocity_l) / (derivative_l + derivative_r)
        updated = max(updated, 1.0e-12)
        if abs(updated - pressure) < 1.0e-12 * (updated + pressure):
            pressure = updated
            break
        pressure = updated
    value_l, _ = _pressure_function(pressure, rho_l, pressure_l, sound_l)
    value_r, _ = _pressure_function(pressure, rho_r, pressure_r, sound_r)
    velocity = 0.5 * (velocity_l + velocity_r + value_r - value_l)
    return pressure, velocity


def _sample_riemann(similarity, left, right, star_pressure, star_velocity):
    rho_l, velocity_l, pressure_l = left
    rho_r, velocity_r, pressure_r = right
    sound_l = np.sqrt(GAMMA * pressure_l / rho_l)
    sound_r = np.sqrt(GAMMA * pressure_r / rho_r)
    g_ratio = (GAMMA - 1.0) / (GAMMA + 1.0)

    if similarity <= star_velocity:
        if star_pressure > pressure_l:
            shock = velocity_l - sound_l * np.sqrt(
                (GAMMA + 1.0) * star_pressure / (2.0 * GAMMA * pressure_l) + (GAMMA - 1.0) / (2.0 * GAMMA)
            )
            if similarity <= shock:
                return left
            star_density = rho_l * (star_pressure / pressure_l + g_ratio) / (g_ratio * star_pressure / pressure_l + 1.0)
            return star_density, star_velocity, star_pressure
        head = velocity_l - sound_l
        star_sound = sound_l * (star_pressure / pressure_l) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
        tail = star_velocity - star_sound
        if similarity <= head:
            return left
        if similarity >= tail:
            star_density = rho_l * (star_pressure / pressure_l) ** (1.0 / GAMMA)
            return star_density, star_velocity, star_pressure
        velocity = 2.0 * (sound_l + 0.5 * (GAMMA - 1.0) * velocity_l + similarity) / (GAMMA + 1.0)
        sound = 2.0 * (sound_l + 0.5 * (GAMMA - 1.0) * (velocity_l - similarity)) / (GAMMA + 1.0)
        ratio = sound / sound_l
        return rho_l * ratio ** (2.0 / (GAMMA - 1.0)), velocity, pressure_l * ratio ** (2.0 * GAMMA / (GAMMA - 1.0))

    if star_pressure > pressure_r:
        shock = velocity_r + sound_r * np.sqrt(
            (GAMMA + 1.0) * star_pressure / (2.0 * GAMMA * pressure_r) + (GAMMA - 1.0) / (2.0 * GAMMA)
        )
        if similarity >= shock:
            return right
        star_density = rho_r * (star_pressure / pressure_r + g_ratio) / (g_ratio * star_pressure / pressure_r + 1.0)
        return star_density, star_velocity, star_pressure
    head = velocity_r + sound_r
    star_sound = sound_r * (star_pressure / pressure_r) ** ((GAMMA - 1.0) / (2.0 * GAMMA))
    tail = star_velocity + star_sound
    if similarity >= head:
        return right
    if similarity <= tail:
        star_density = rho_r * (star_pressure / pressure_r) ** (1.0 / GAMMA)
        return star_density, star_velocity, star_pressure
    velocity = 2.0 * (-sound_r + 0.5 * (GAMMA - 1.0) * velocity_r + similarity) / (GAMMA + 1.0)
    sound = 2.0 * (sound_r - 0.5 * (GAMMA - 1.0) * (velocity_r - similarity)) / (GAMMA + 1.0)
    ratio = sound / sound_r
    return rho_r * ratio ** (2.0 / (GAMMA - 1.0)), velocity, pressure_r * ratio ** (2.0 * GAMMA / (GAMMA - 1.0))


def _exact_riemann(coordinates, time, left, right):
    star_pressure, star_velocity = _star_state(left, right)
    return np.asarray(
        [
            _sample_riemann((coordinate - 0.5) / time, left, right, star_pressure, star_velocity)
            for coordinate in coordinates
        ]
    )


def _advance_tube(left, right, normal_axis, cells=160, final_time=0.006):
    transverse_cells = 3
    coordinate = (jnp.arange(cells) + 0.5) / cells
    rho = jnp.where(coordinate < 0.5, left[0], right[0])
    velocity = jnp.where(coordinate < 0.5, left[1], right[1])
    pressure = jnp.where(coordinate < 0.5, left[2], right[2])
    zeros = jnp.zeros_like(rho)
    if normal_axis == 0:
        primitive_1d = jnp.stack((rho, velocity, zeros, zeros, pressure), axis=-1)
        primitive = jnp.broadcast_to(primitive_1d[:, None, :], (cells, transverse_cells, 5))
        solver = IonEuler2D(1.0 / cells, 1.0 / transverse_cells, gamma=GAMMA, boundaries=("outflow", "periodic"))
    else:
        primitive_1d = jnp.stack((rho, zeros, velocity, zeros, pressure), axis=-1)
        primitive = jnp.broadcast_to(primitive_1d[None, :, :], (transverse_cells, cells, 5))
        solver = IonEuler2D(1.0 / transverse_cells, 1.0 / cells, gamma=GAMMA, boundaries=("periodic", "outflow"))
    conserved = primitive_to_conserved(primitive, gamma=GAMMA)
    stable_dt = float(solver.cfl_timestep(conserved, cfl=0.25))
    steps = int(np.ceil(final_time / stable_dt))
    dt = final_time / steps
    advance = jax.jit(lambda state: solver.step(state, dt))
    for _ in range(steps):
        conserved = advance(conserved)
    return coordinate, conserved_to_primitive(conserved, gamma=GAMMA)


@pytest.mark.parametrize(
    ("left", "right", "relative_l1_tolerance"),
    [
        ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.035),
        ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.12),
    ],
)
def test_sod_and_strong_shock_tubes_match_exact_solution_in_both_directions(
    left,
    right,
    relative_l1_tolerance,
):
    coordinate, result_x = _advance_tube(left, right, normal_axis=0)
    _coordinate, result_y = _advance_tube(left, right, normal_axis=1)
    rotated_y = jnp.swapaxes(result_y, 0, 1)
    rotated_y = rotated_y.at[..., 1].set(result_y.swapaxes(0, 1)[..., 2])
    rotated_y = rotated_y.at[..., 2].set(result_y.swapaxes(0, 1)[..., 1])
    np.testing.assert_allclose(result_x, rotated_y, rtol=5e-12, atol=5e-12)

    exact = _exact_riemann(np.asarray(coordinate), 0.006, left, right)
    numerical = np.asarray(result_x[:, 0])[:, (0, 1, 4)]
    scales = np.ptp(exact, axis=0)
    relative_l1 = np.mean(np.abs(numerical - exact), axis=0) / np.maximum(scales, 1.0e-12)
    assert np.max(relative_l1) < relative_l1_tolerance
    assert jnp.all(result_x[..., 0] > 0.0)
    assert jnp.all(result_x[..., 4] > 0.0)


def _isentropic_vortex(n, time):
    length = 10.0
    coordinate = -5.0 + (jnp.arange(n) + 0.5) * length / n
    x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    center_x = -2.0 + time
    center_y = -2.0 + 0.5 * time
    dx = (x - center_x + 0.5 * length) % length - 0.5 * length
    dy = (y - center_y + 0.5 * length) % length - 0.5 * length
    radius_squared = dx**2 + dy**2
    beta = 5.0
    envelope = jnp.exp(0.5 * (1.0 - radius_squared))
    delta_u = -beta * dy * envelope / (2.0 * jnp.pi)
    delta_v = beta * dx * envelope / (2.0 * jnp.pi)
    temperature = 1.0 - (GAMMA - 1.0) * beta**2 * jnp.exp(1.0 - radius_squared) / (8.0 * GAMMA * jnp.pi**2)
    density = temperature ** (1.0 / (GAMMA - 1.0))
    pressure = density**GAMMA
    return jnp.stack((density, 1.0 + delta_u, 0.5 + delta_v, jnp.zeros_like(x), pressure), axis=-1)


def _vortex_error(n):
    final_time = 0.2
    solver = IonEuler2D(10.0 / n, 10.0 / n, gamma=GAMMA)
    conserved = primitive_to_conserved(_isentropic_vortex(n, 0.0), gamma=GAMMA)
    stable_dt = float(solver.cfl_timestep(conserved, cfl=0.35))
    steps = int(np.ceil(final_time / stable_dt))
    dt = final_time / steps
    advance = jax.jit(lambda state: solver.step(state, dt))
    for _ in range(steps):
        conserved = advance(conserved)
    numerical_density = conserved[..., 0]
    exact_density = _isentropic_vortex(n, final_time)[..., 0]
    return float(jnp.mean(jnp.abs(numerical_density - exact_density)))


def test_translating_isentropic_vortex_converges_at_second_order():
    coarse_error = _vortex_error(24)
    fine_error = _vortex_error(48)
    assert np.log2(coarse_error / fine_error) > 1.7


def test_sedov_blast_conserves_energy_and_remains_radially_symmetric():
    n = 48
    length = 1.0
    dx = length / n
    coordinate = -0.5 + (jnp.arange(n) + 0.5) * dx
    x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    density = jnp.ones((n, n))
    pressure = jnp.full((n, n), 1.0e-5)
    central = (jnp.abs(x) < dx) & (jnp.abs(y) < dx)
    pressure = pressure + central * (GAMMA - 1.0) / (4.0 * dx**2)
    primitive = jnp.stack(
        (density, jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x), pressure),
        axis=-1,
    )
    conserved = primitive_to_conserved(primitive, gamma=GAMMA)
    initial_energy = jnp.sum(conserved[..., 4]) * dx**2
    solver = IonEuler2D(dx, dx, gamma=GAMMA, boundaries=("outflow", "outflow"))
    final_time = 0.02
    time = 0.0
    advance = jax.jit(lambda state, timestep: solver.step(state, timestep))
    while time < final_time:
        dt = min(float(solver.cfl_timestep(conserved, cfl=0.3)), final_time - time)
        conserved = advance(conserved, dt)
        time += dt

    result = conserved_to_primitive(conserved, gamma=GAMMA)
    final_energy = jnp.sum(conserved[..., 4]) * dx**2
    np.testing.assert_allclose(final_energy, initial_energy, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(result, jnp.swapaxes(result, 0, 1)[..., [0, 2, 1, 3, 4]], rtol=3e-11, atol=3e-11)
    assert jnp.all(result[..., 0] > 0.0)
    assert jnp.all(result[..., 4] > 0.0)

    radial_velocity = (result[..., 1] * x + result[..., 2] * y) / jnp.maximum(
        jnp.sqrt(x**2 + y**2),
        dx / 2.0,
    )
    kinetic_weight = 0.5 * conserved[..., 0] * radial_velocity**2
    angle = jnp.arctan2(y, x)
    cartesian_anisotropy = jnp.abs(jnp.sum(kinetic_weight * jnp.cos(4.0 * angle))) / jnp.sum(kinetic_weight)
    assert cartesian_anisotropy < 0.05
