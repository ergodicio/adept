"""Quantitative collisional-local-limit tests for Gate 2b."""

import jax.numpy as jnp
import numpy as np

from adept.vfp1d.fokker_planck import FLMCollisions
from adept.vfp1d.grid import Grid as CollisionGrid
from adept.vfp2d import (
    AnisotropicCollisions,
    Grid,
    HarmonicLayout,
    KineticOhm2D,
    Maxwell2D,
    vector_velocity_moment,
)


def _biermann_error(nxy: int) -> float:
    grid = Grid(
        xmin=0.0,
        xmax=2.0 * np.pi,
        nx=nxy,
        ymin=0.0,
        ymax=2.0 * np.pi,
        ny=nxy,
        vmax=7.0,
        nv=96,
        dt=1.0e-3,
        l_max=2,
    )
    layout = HarmonicLayout(2)
    x, y = grid.x[:, None], grid.y[None, :]
    log_density = 0.12 * jnp.sin(x) + 0.08 * jnp.cos(y)
    electron_density = jnp.exp(log_density)
    temperature = 0.5 + 0.04 * jnp.cos(x + y) + 0.03 * jnp.sin(2.0 * x - y)

    radial = jnp.exp(-(grid.v[None, None, :] ** 2) / (2.0 * temperature[..., None]))
    radial_density = 4.0 * jnp.pi * jnp.sum(radial * grid.v**2, axis=-1) * grid.dv
    radial *= electron_density[..., None] / radial_density[..., None]
    flm = jnp.zeros((nxy, nxy, layout.size, grid.nv), dtype=jnp.complex128)
    flm = flm.at[..., layout.index(0, 0), :].set(radial)
    magnetic = jnp.zeros((nxy, nxy, 3))

    electric, _terms = KineticOhm2D(layout, grid.v, grid.dv, grid.kx, grid.ky)(flm, magnetic)
    magnetic_rate = -Maxwell2D(grid.kx, grid.ky, c=1.0).curl(electric)
    dtemperature_dx = -0.04 * jnp.sin(x + y) + 0.06 * jnp.cos(2.0 * x - y)
    dtemperature_dy = -0.04 * jnp.sin(x + y) - 0.03 * jnp.cos(2.0 * x - y)
    dlog_density_dx = 0.12 * jnp.cos(x) + jnp.zeros_like(y)
    dlog_density_dy = -0.08 * jnp.sin(y) + jnp.zeros_like(x)
    classical_biermann = dtemperature_dx * dlog_density_dy - dtemperature_dy * dlog_density_dx
    return float(jnp.max(jnp.abs(magnetic_rate[..., 2] - classical_biermann)))


def test_local_maxwellian_biermann_generation_converges_to_classical_rate():
    coarse_error = _biermann_error(12)
    fine_error = _biermann_error(16)
    assert coarse_error / fine_error > 50.0
    assert fine_error < 2.0e-6


def _local_eh_heat_flux(nv: int) -> tuple[float, float]:
    vmax = 7.0
    dv = vmax / nv
    v = (jnp.arange(nv) + 0.5) * dv
    layout = HarmonicLayout(1)
    temperature = 0.5
    electron_density = 1.0
    temperature_gradient = 0.03
    ionization = 6.0
    collision_coefficient = 0.4

    f0 = jnp.exp(-(v**2) / (2.0 * temperature))
    f0 *= electron_density / (4.0 * jnp.pi * jnp.sum(f0 * v**2) * dv)
    temperature_derivative = f0 * (-1.5 / temperature + v**2 / (2.0 * temperature**2))
    velocity_derivative = -(v / temperature) * f0
    thermal_force = -2.5 * temperature_gradient
    streaming_source = -v * temperature_derivative * temperature_gradient + thermal_force * velocity_derivative

    steady_step = 1.0e12
    flm = jnp.zeros((1, 1, layout.size, nv), dtype=jnp.complex128)
    flm = flm.at[..., layout.index(0, 0), :].set(f0)
    flm = flm.at[..., layout.index(1, 0), :].set(steady_step * streaming_source)
    collision_grid = CollisionGrid(
        xmin=0.0,
        xmax=1.0,
        nx=1,
        tmin=0.0,
        tmax=1.0,
        dt=0.1,
        nv=nv,
        vmax=vmax,
        nl=1,
    )
    operator = FLMCollisions(
        Z=ionization,
        nuee_coeff=collision_coefficient,
        grid=collision_grid,
        full_aniso_ee=False,
    )
    updated = AnisotropicCollisions(operator, layout)(
        flm,
        Z=jnp.ones((1, 1)),
        ni=jnp.full((1, 1), electron_density / ionization),
        dt=steady_step,
    )
    heat_flux = 0.5 * electron_density * vector_velocity_moment(updated, layout, v, dv, power=2)[0, 0, 0]

    z_star = (ionization + 4.2) / (ionization + 0.24)
    expected = (
        -16.0
        * (2.0 * temperature) ** 2.5
        * temperature_gradient
        / (np.sqrt(np.pi) * collision_coefficient * ionization * z_star)
    )
    return float(heat_flux), float(expected)


def test_epperlein_haines_local_heat_flux_recovers_spitzer_harm_scaling():
    coarse, expected = _local_eh_heat_flux(12)
    fine, _expected = _local_eh_heat_flux(16)
    coarse_error = abs(coarse - expected)
    fine_error = abs(fine - expected)
    assert coarse_error / fine_error > 40.0
    np.testing.assert_allclose(fine, expected, rtol=2.0e-6)
