"""Implementation checks for explicit Strang splitting and its wrapper."""

from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._vlasov1d.modules import BaseVlasov1D
from adept._vlasov1d.solvers.vector_field import StrangIntegrator, VlasovMaxwell


def _build(field="poisson", edfdv="lagrange7"):
    filename = "boltzmann_iaw.yaml" if field == "poisson-boltzmann" else "resonance.yaml"
    cfg = yaml.safe_load((Path(__file__).parent / "configs" / filename).read_text())
    cfg["terms"].update(time="strang", field=field, edfdv=edfdv)
    cfg["terms"]["fokker_planck"]["is_on"] = False
    cfg["terms"]["krook"]["is_on"] = False
    cfg["drivers"] = {"ex": {}, "ey": {}}
    cfg["grid"].update(nx=32, nv=128, dt=0.2, tmax=0.8)
    if cfg["terms"].get("species"):
        for species in cfg["terms"]["species"]:
            species["nv"] = 128
    cfg["save"] = {"fields": {"t": {"nt": 3}}}
    module = BaseVlasov1D(cfg)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    return module


def test_strang_constant_acceleration_matches_characteristics():
    """The two half-streams center a constant kick correctly in both x and v."""
    module = _build()
    grid = module.simulation.grid
    step = StrangIntegrator(module.cfg, grid)
    x = grid.x
    v = module.cfg["grid"]["species_grids"]["electron"]["v"]
    k = 2 * jnp.pi / (grid.xmax - grid.xmin)
    distribution = lambda x, v: (1 + 0.1 * jnp.cos(k * x)) * (1 + 0.02 * v**2)
    f = {"electron": distribution(x[:, None], v[None, :])}
    zeros = jnp.zeros_like(x)
    step.field_solve = lambda **kwargs: (zeros, zeros)
    external_e = jnp.full_like(x, 0.3)
    _, actual = step(f, jnp.zeros(grid.nx + 2), [external_e, external_e], zeros)
    accel = -0.3  # electron q/m
    expected = distribution(
        x[:, None] - v[None, :] * grid.dt + 0.5 * accel * grid.dt**2,
        v[None, :] - accel * grid.dt,
    )
    np.testing.assert_allclose(actual["electron"][:, 8:-8], expected[:, 8:-8], atol=2e-11, rtol=2e-11)


def test_strang_wrapper_uses_midpoint_forcing_and_saves_endpoint_driver():
    """A time-linear force gives the exact impulse; saved de is at the new time."""
    module = _build()
    grid = module.simulation.grid
    wrapper = VlasovMaxwell(module.cfg, grid, module.simulation.drivers)
    zeros = jnp.zeros_like(grid.x)
    wrapper.vpfp.vlasov_poisson.field_solve = lambda **kwargs: (zeros, zeros)
    wrapper.total_dex = lambda t, args: jnp.full_like(grid.x, t)
    v = module.cfg["grid"]["species_grids"]["electron"]["v"]
    state = dict(module.state)
    state["electron"] = jnp.broadcast_to(1 + 0.02 * v**2, (grid.nx, v.size))
    t = 0.7
    result = jax.jit(wrapper)(t, state, {})
    shift = -grid.dt * (t + grid.dt / 2)
    expected = jnp.broadcast_to(1 + 0.02 * (v - shift) ** 2, state["electron"].shape)
    np.testing.assert_allclose(result["electron"][:, 8:-8], expected[:, 8:-8], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(result["de"], t + grid.dt, atol=1e-14)


@pytest.mark.parametrize("field", ["poisson", "poisson-boltzmann"])
def test_strang_returns_field_from_final_distribution(field):
    """The saved self-consistent field is synchronized with the returned f."""
    module = _build(field)
    grid = module.simulation.grid
    step = StrangIntegrator(module.cfg, grid)
    k = 2 * jnp.pi / (grid.xmax - grid.xmin)
    f = {
        name: module.state[name] * (1 + 0.05 * jnp.cos(k * grid.x))[:, None]
        for name in module.cfg["grid"]["species_grids"]
    }
    zeros = jnp.zeros_like(grid.x)
    a = jnp.zeros(grid.nx + 2)
    e, f_new = step(f, a, [zeros, zeros], zeros)
    _, expected = step.field_solve(f_dict=f_new, a=a, prev_ex=None, dt=None)
    np.testing.assert_allclose(e, expected, atol=1e-14)
    assert jnp.max(jnp.abs(e)) > 0


@pytest.mark.parametrize("field", ["poisson", "poisson-boltzmann"])
@pytest.mark.parametrize("edfdv", ["lagrange7", "cubic-spline", "exponential"])
def test_strang_solver_smoke(field, edfdv):
    """Both electrostatic closures can run through the public solver lifecycle."""
    module = _build(field, edfdv)
    module.init_diffeqsolve()
    result = module({})["solver result"]
    assert int(result.stats["num_steps"]) > 0
    for values in jax.tree.leaves(result.ys):
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize("field", ["ampere", "hampere"])
def test_strang_rejects_unsupported_field_evolution(field):
    with pytest.raises(NotImplementedError, match="strang requires field"):
        StrangIntegrator({"terms": {"field": field}}, None)


def test_strang_sharded_matches_serial():
    """Both x-sharded kicks and v-sharded half-streams compose correctly."""
    module = _build()
    grid = module.simulation.grid
    sharded_cfg = {**module.cfg, "grid": {**module.cfg["grid"], "parallel": ("x", "v")}}
    serial = StrangIntegrator(module.cfg, grid)
    sharded = StrangIntegrator(sharded_cfg, grid)
    k = 2 * jnp.pi / (grid.xmax - grid.xmin)
    f = {"electron": module.state["electron"] * (1 + 0.05 * jnp.cos(k * grid.x))[:, None]}
    zeros = jnp.zeros_like(grid.x)
    args = (f, jnp.zeros(grid.nx + 2), [zeros, zeros], zeros)
    expected = jax.jit(serial)(*args)
    actual = jax.jit(sharded)(*args)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
