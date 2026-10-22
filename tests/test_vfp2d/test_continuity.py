"""Shared mass-flux and time-stage regressions for moving-ion continuity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adept.vfp2d import (
    CoupledIonKineticStep,
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
from adept.vfp2d.harmonics import HouLiFilter2D, complex_to_real, real_to_complex


def _problem(nx=12, ny=10, nv=32):
    grid = Grid(xmin=0.0, xmax=2 * np.pi, nx=nx, ymin=0.0, ymax=2 * np.pi, ny=ny, vmax=8.0, nv=nv, dt=0.002, l_max=2)
    layout = HarmonicLayout(2)
    x, y = grid.x[:, None], grid.y[None, :]
    ne = 1.0 + 0.15 * jnp.cos(x) * jnp.cos(y)
    temperature = 0.4 * (1.0 + 0.15 * jnp.sin(x + y))
    radial = jnp.exp(-(grid.v**2) / (2 * temperature[..., None]))
    radial *= ne[..., None] / (4 * jnp.pi * jnp.sum(radial * grid.v**2, axis=-1)[..., None] * grid.dv)
    f = jnp.zeros((nx, ny, layout.size, nv), dtype=jnp.complex128).at[..., 0, :].set(radial)
    f = f.at[..., layout.index(2, 1), :].set((0.01 + 0.02j) * jnp.sin(x)[..., None] * radial)
    # Z=4, mi=100. Nonuniform density, temperature and flow exercise both
    # directions, both upwind branches, and differing radial slope limiters.
    primitive = jnp.zeros((nx, ny, 5)).at[..., 0].set(25 * ne).at[..., 4].set(0.1 * ne)
    primitive = primitive.at[..., 1].set(0.15 * jnp.sin(x))
    primitive = primitive.at[..., 2].set(-0.12 * jnp.sin(y))
    ions = primitive_to_conserved(primitive)
    return grid, layout, f, ions


def test_shared_flux_preserves_charge_and_every_global_passive_component():
    grid, layout, f, ions = _problem()
    hydro = IonEuler2D(grid.dx, grid.dy)
    ion_rate, electron_rate = jax.jit(hydro.rhs_with_passive)(ions, f)
    np.testing.assert_allclose(
        density(electron_rate, layout, grid.v, grid.dv), ion_rate[..., 0] / 25, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(jnp.sum(electron_rate, axis=(0, 1)), 0.0, atol=2e-15)

    next_ions, next_f = jax.jit(hydro.step_with_passive)(ions, f, 0.1)
    np.testing.assert_allclose(next_ions, hydro.step(ions, 0.1), atol=0.0, rtol=2e-15)
    np.testing.assert_allclose(density(next_f, layout, grid.v, grid.dv), next_ions[..., 0] / 25, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(jnp.sum(next_f, axis=(0, 1)), jnp.sum(f, axis=(0, 1)), rtol=0.0, atol=2e-14)
    assert jnp.min(jnp.real(next_f[..., 0, :])) >= 0.0
    assert jnp.max(jnp.abs(next_ions[..., 0] - ions[..., 0])) > 1e-3


def test_shared_flux_transports_an_existing_charge_error_instead_of_erasing_it():
    grid, layout, f, ions = _problem()
    f *= (1.0 + 0.03 * jnp.sin(grid.x)[:, None])[..., None, None]
    hydro = IonEuler2D(grid.dx, grid.dy)
    next_ions, next_f = jax.jit(hydro.step_with_passive)(ions, f, 0.1)
    charge = density(next_f, layout, grid.v, grid.dv) - next_ions[..., 0] / 25
    assert jnp.max(jnp.abs(charge)) > 0.02
    np.testing.assert_allclose(
        jnp.sum(density(next_f, layout, grid.v, grid.dv)),
        jnp.sum(density(f, layout, grid.v, grid.dv)),
        rtol=0.0,
        atol=2e-13,
    )


def test_shared_passive_transport_refines_at_second_order():
    errors = []
    for nx in (24, 48, 96):
        dx = 2 * np.pi / nx
        x = (jnp.arange(nx) + 0.5) * dx
        ions = primitive_to_conserved(jnp.zeros((nx, 2, 5)).at[..., 0].set(1.0).at[..., 1].set(0.4).at[..., 4].set(0.1))
        passive = jnp.broadcast_to((1 + 0.2 * jnp.sin(x))[:, None, None], (nx, 2, 1))
        hydro = IonEuler2D(dx, 1.0)
        steps = int(np.ceil(1.0 / (0.2 * dx)))
        advance = jax.jit(hydro.step_with_passive)
        state = ions, passive
        for _ in range(steps):
            state = advance(*state, 1.0 / steps)
        expected = 1 + 0.2 * jnp.sin(x - 0.4)
        errors.append(float(jnp.mean(jnp.abs(state[1][:, 0, 0] - expected))))
    assert errors[0] / errors[1] > 3.0, errors
    assert errors[1] / errors[2] > 3.0, errors


@pytest.mark.parametrize("nx,ny", [(8, 10), (9, 7)])
@pytest.mark.parametrize("axis", [0, 1])
def test_harmonic_spatial_derivative_preserves_components_and_resolved_modes(nx, ny, axis):
    grid, layout, _f, _ions = _problem(nx=nx, ny=ny)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    n = (nx, ny)[axis]
    coordinate = grid.x[:, None] if axis == 0 else grid.y[None, :]
    mode = (n - 1) // 2  # Highest resolved mode, including odd grids.
    smooth = (1.0 + 2.0j) * jnp.cos(mode * coordinate) * jnp.ones((nx, ny))
    expected = -(1.0 + 2.0j) * mode * jnp.sin(mode * coordinate) * jnp.ones((nx, ny))
    if n % 2 == 0:
        checker = (-1.0) ** jnp.arange(n)
        checker = checker[:, None] if axis == 0 else checker[None, :]
        smooth += (0.7 - 0.3j) * checker
    np.testing.assert_allclose(vlasov.spatial_derivative(smooth, axis), expected, rtol=0.0, atol=5e-14)


def test_ampere_projected_streaming_has_zero_number_rate_at_transverse_nyquist():
    grid, layout, f, _ions = _problem(nx=8, ny=8)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    maxwell = Maxwell2D(grid.kx, grid.ky, c=2.0)
    electrons = KineticOhmStep(
        vlasov,
        maxwell,
        KineticOhm2D(layout, grid.v, grid.dv, grid.kx, grid.ky),
        layout,
        grid.v,
        grid.dv,
        grid.dt,
    )
    # curl(B) has only Jz: no electron number flux in the resolved x/y plane.
    # Its imaginary f11 coefficient must not leak into Re(d_y f11) at Nyquist.
    checker = (-1.0) ** jnp.arange(grid.ny)
    magnetic = jnp.zeros((grid.nx, grid.ny, 3)).at[..., 1].set(jnp.sin(grid.x)[:, None] * checker[None, :])
    projected = electrons._project(f, magnetic)
    assert jnp.max(jnp.abs(electrons._target_current(magnetic)[..., 2])) > 1.0
    np.testing.assert_allclose(density(vlasov.streaming(projected), layout, grid.v, grid.dv), 0.0, atol=2e-14)


@pytest.mark.parametrize("real_storage,filtered", [(False, False), (True, True)])
def test_compressive_coupled_steps_preserve_charge_with_filter_and_storage(real_storage, filtered):
    grid, layout, f, ions = _problem(nx=8, ny=8)
    vlasov = TzoufrasVlasov(layout, grid.v, grid.dv, grid.kx, grid.ky)
    maxwell = Maxwell2D(grid.kx, grid.ky, c=2.0)
    electrons = KineticOhmStep(
        vlasov,
        maxwell,
        KineticOhm2D(layout, grid.v, grid.dv, grid.kx, grid.ky),
        layout,
        grid.v,
        grid.dv,
        grid.dt,
        ion_frame=IonFrameVlasov(vlasov),
        real_storage=real_storage,
        enforce_f00_positivity=True,
        spatial_filter=HouLiFilter2D(grid.nx, grid.ny, order=4) if filtered else None,
    )
    coupled = CoupledIonKineticStep(electrons, IonEuler2D(grid.dx, grid.dy), grid.dt, ion_mass=100.0)
    field = jnp.zeros((grid.nx, grid.ny, 3))
    potential = field.at[..., 2].set(0.001 * jnp.cos(grid.x)[:, None] * jnp.cos(grid.y)[None, :])
    magnetic = maxwell.curl(potential)
    f = electrons._project(f, magnetic)
    state = {"flm": complex_to_real(f) if real_storage else f, "ions": ions, "e": field, "b": magnetic}
    advance = jax.jit(coupled)
    for step in range(12):
        state = advance(step * grid.dt, state, {"Z": 4.0})
    final_f = real_to_complex(state["flm"]) if real_storage else state["flm"]
    np.testing.assert_allclose(
        density(final_f, layout, grid.v, grid.dv), state["ions"][..., 0] / 25, rtol=0.0, atol=2e-12
    )
    assert jnp.max(jnp.abs(state["ions"][..., 0] - ions[..., 0])) > 1e-3
    for value in state.values():
        assert jnp.all(jnp.isfinite(value))
