"""The EPW step in LPSE's order (inventory A11; ZakharovSolver::evolve): the ion-acoustic waves with the
fields at t, the Langmuir waves with the light at t and the ion density at the step's middle, then the
light with the EPW potential and the ion density linearly interpolated to each light sub-step's middle
(LightSolver::computeDynamicE0 at time + dt / 2, interpolateSourcesInTime).

The checks compose the solvers by hand and compare with the step; the tolerances (1e-12 relative, the
Linear values exact) are fixed in advance: the same operations in the same order up to round-off.
"""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._lpse2d.core.timeline import Linear, as_linear


def _cfg(iaw=None, **light):
    from adept._lpse2d.helpers import get_density_profile, get_derived_quantities, get_solver_quantities, write_units

    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["grid"].update({"xmax": "8um", "dx": "0.1um", "tmax": "10fs", "dt": "2fs"})
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["terms"]["light"] = {"solver": "fd", **light}
    if iaw is not None:
        cfg["terms"]["iaw"] = {"active": True, "solver": "spectral", **iaw}
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _state(cfg, seed=0):
    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]

    def c(*shape, scale=1.0):
        return jnp.asarray(scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape)))

    y = {"epw": c(nx, ny, scale=1e-3), "E0": c(nx, ny, 3, scale=1e-2), "E1": c(nx, ny, 3, scale=1e-4)}
    if cfg["terms"].get("iaw", {}).get("active"):
        y["iaw_density"] = jnp.asarray(1e-3 * rng.normal(size=(nx, ny)))
        y["iaw_velocity_divergence"] = jnp.asarray(1e-3 * rng.normal(size=(nx, ny)))
        if int(cfg["terms"]["iaw"].get("stride", 1)) > 1:
            y["iaw_density_old"] = jnp.asarray(1e-3 * rng.normal(size=(nx, ny)))
    return y


def _pack(y):
    return {k: v.view(jnp.float64) for k, v in y.items()}


def _unpack(y, complex_keys=("epw", "E0", "E1")):
    return {k: (v.view(jnp.complex128) if k in complex_keys else v) for k, v in y.items()}


def test_linear_values():
    a, b = jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 6.0])
    tl = Linear(a, b, 0.5, 1.0)
    np.testing.assert_array_equal(tl.at(0.0), a + 0.5 * (b - a))
    np.testing.assert_array_equal(tl.at(1.0), b)
    np.testing.assert_array_equal(tl.substep(1, 4), a + (0.5 + 0.375 * 0.5) * (b - a))
    np.testing.assert_array_equal(Linear(a, b, interpolate=False).at(0.3), b)
    assert as_linear(a).constant and as_linear(None).at(0.5) is None
    assert as_linear(tl) is tl


def test_light_reads_the_potential_at_each_substep_middle():
    """n light sub-steps with a Linear potential = n single sub-steps, each with the potential held at
    its sub-step's middle; without interpolation = the pass with the new potential."""
    from adept._lpse2d.core.raman import RamanLight

    cfg = _cfg()
    light = RamanLight(cfg)
    n = light.n_sub
    assert n > 1
    y = _state(cfg)
    phi_new = y["epw"] * (1.3 + 0.2j)
    tl = Linear(y["epw"], phi_new)

    def e0_fn(t):
        return y["E0"]

    got = light(0.0, y["E1"], e0_fn, tl, None)
    light.n_sub = 1
    want = y["E1"]
    for i in range(n):
        want = light(i * light.dt_l, want, e0_fn, tl.substep(i, n), None)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * float(jnp.max(jnp.abs(want))))
    light.n_sub = n
    np.testing.assert_allclose(
        light(0.0, y["E1"], e0_fn, Linear(y["epw"], phi_new, interpolate=False), None),
        light(0.0, y["E1"], e0_fn, phi_new, None),
        rtol=1e-12,
    )


def test_step_advances_iaw_then_epw_with_light_at_t_then_light():
    """One SplitStep = IAW (fields at t) -> EPW (light at t, ion density at mid-step) -> light (the
    potential and the ion density interpolated per sub-step)."""
    from adept._lpse2d.core.vector_field import SplitStep

    cfg = _cfg(iaw={})
    step = SplitStep(cfg)
    y = _state(cfg)
    args = {"drivers": {}}
    got = _unpack(step(0.0, _pack(dict(y)), args))

    iaw_out = step.iaw_step(dict(y), 0.0)
    old, new = y["iaw_density"], iaw_out["iaw_density"]
    epw_in = {**y, **{k: v for k, v in iaw_out.items() if k.startswith("iaw_")}, "iaw_density": 0.5 * (old + new)}
    phi_new, _ = step.epw.advance(0.0, epw_in, args)
    light_y = {**y, "epw": phi_new, "iaw_density": new}
    light_y = step.light_split_step(0.0, light_y, {}, Linear(y["epw"], phi_new), Linear(old, new))
    np.testing.assert_allclose(got["iaw_density"], new, rtol=1e-12)
    np.testing.assert_allclose(got["epw"], phi_new, rtol=1e-12, atol=1e-12 * float(jnp.max(jnp.abs(phi_new))))
    for k in ("E0", "E1"):
        scale = float(jnp.max(jnp.abs(light_y[k])))
        np.testing.assert_allclose(got[k], light_y[k], rtol=1e-12, atol=1e-12 * scale)


@pytest.mark.parametrize("index", [0, 1])
def test_strided_iaw_interpolates_across_its_step(index):
    """stride 2: the IAW advances on even EPW steps, remembering the density it started from; the
    waves read it at fractions (m + s) / 2 of the IAW step (m = 0, 1)."""
    from adept._lpse2d.core.vector_field import SplitStep

    cfg = _cfg(iaw={"stride": 2})
    step = SplitStep(cfg)
    y = _state(cfg)
    t = index * step.dt
    out, (old, new, f0, f1) = step.iaw_first(dict(y), t)
    if index == 0:
        np.testing.assert_array_equal(old, y["iaw_density"])
        assert not np.allclose(new, y["iaw_density"])
    else:
        np.testing.assert_array_equal(old, y["iaw_density_old"])
        np.testing.assert_array_equal(new, y["iaw_density"])
    np.testing.assert_array_equal(out["iaw_density_old"], old)
    assert (float(f0), float(f1)) == (index / 2, (index + 1) / 2)
