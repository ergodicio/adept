"""
The EPW noise source draws a fresh random phase every step, so its contributions add in quadrature.
With the increment proportional to dt the accumulated noise power comes out proportional to dt, which
makes the seed level the instability grows out of depend on the timestep. Scaling the increment by
sqrt(dt) instead makes it dt-independent.

These tests drive the real solver with TPD off, so the only physics left is the noise source
balanced against Landau and collisional damping, and measure how the resulting floor moves with dt.
"""

import re

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import yaml
from jax import numpy as jnp

from adept._lpse2d.core.epw import SpectralEPWSolver
from adept._lpse2d.helpers import (
    get_density_profile,
    get_derived_quantities,
    get_solver_quantities,
    write_units,
)

CONFIG_PATH = "tests/test_lpse2d/configs/tpd.yaml"
T_END_PS = 0.3


def _build(dt_fs: float, scaling: str | None) -> dict:
    with open(CONFIG_PATH) as fi:
        cfg = yaml.safe_load(fi)

    # coarse and short: this test is about the dt scaling of the floor, not about resolving TPD
    cfg["grid"]["dx"] = "150nm"
    cfg["grid"]["ymax"] = "1um"
    cfg["grid"]["ymin"] = "-1um"
    cfg["grid"]["dt"] = f"{dt_fs}fs"
    cfg["terms"]["epw"]["source"]["tpd"] = False
    cfg["terms"]["epw"]["source"]["noise"] = True
    if scaling is not None:
        cfg["terms"]["epw"]["source"]["noise_scaling"] = scaling
    # the absorbing sponge would otherwise dominate the amplitude and mask the effect
    cfg["terms"]["epw"]["boundary"] = {"x": "periodic", "y": "periodic"}

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _noise_floor(dt_fs: float, scaling: str | None, seed: int = 0) -> float:
    """RMS |phi_k| after the noise source and the damping have come into balance."""
    cfg = _build(dt_fs, scaling)
    solver = SpectralEPWSolver(cfg)
    solver.noise_seed = seed  # randomised at construction; pin it so runs are comparable

    dt = cfg["grid"]["dt"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    E0 = jnp.zeros((nx, ny, 2), dtype=jnp.complex128)

    @jax.jit
    def step(phi_k, t):
        return solver(t, {"epw": phi_k, "E0": E0}, None)

    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128)
    for n in range(int(T_END_PS / dt)):
        phi_k = step(phi_k, jnp.array(n * dt))

    return float(jnp.sqrt(jnp.mean(jnp.abs(phi_k) ** 2)))


def _mean_floor(dt_fs: float, scaling: str | None) -> float:
    return float(np.mean([_noise_floor(dt_fs, scaling, seed) for seed in (0, 1, 2)]))


def test_default_noise_floor_scales_as_sqrt_dt():
    """
    The current behaviour, pinned so the fix has something to be measured against: a fresh phase per
    step means the floor grows as sqrt(dt) rather than staying put.
    """
    coarse = _mean_floor(20.0, None)
    fine = _mean_floor(2.5, None)

    # eight-fold change in dt -> sqrt(8) = 2.83x in the floor
    assert coarse / fine == pytest.approx(np.sqrt(8.0), rel=0.25)


def test_sqrt_dt_noise_floor_is_dt_independent():
    """With the sqrt(dt) increment the injected noise power no longer depends on the timestep."""
    coarse = _mean_floor(20.0, "sqrt_dt")
    fine = _mean_floor(2.5, "sqrt_dt")

    assert coarse / fine == pytest.approx(1.0, rel=0.2)


def test_dt_scaling_is_the_default():
    """Decks that do not opt in keep the behaviour they had before."""
    assert _noise_floor(5.0, None) == _noise_floor(5.0, "dt")


def test_sqrt_dt_raises_the_seed_level():
    """
    Switching scaling raises the seed by sqrt(dt)/dt, which brings the instability onset forward.
    Worth pinning because it is the visible consequence of flipping the flag.
    """
    cfg = _build(5.0, "sqrt_dt")
    dt = cfg["grid"]["dt"]
    assert SpectralEPWSolver(cfg).noise_increment == pytest.approx(np.sqrt(dt))
    assert SpectralEPWSolver(_build(5.0, "dt")).noise_increment == pytest.approx(dt)


def test_unknown_noise_scaling_is_rejected():
    with pytest.raises(ValueError, match=re.escape("Unknown terms.epw.source.noise_scaling")):
        SpectralEPWSolver(_build(5.0, "per_step"))
