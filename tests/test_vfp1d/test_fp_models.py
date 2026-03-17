#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Unit tests for Fokker-Planck model classes in adept.vfp1d.fokker_planck.
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from fp_relaxation import problems
from fp_relaxation.registry import VelocityGrid

from adept.driftdiffusion import _find_self_consistent_beta_single
from adept.vfp1d.fokker_planck import CoulombianKernel, FastVFP

CASES = [
    pytest.param(
        {
            "nv": 512,
            "vmax": 16.0,
            "f_fn": partial(problems.maxwellian, T=1.0),
            "beta_mode": "analytic",
            "beta": 0.5,
            "v_min": 5.0,
        },
        id="maxwellian_high_vmax_analytic_beta",
    ),
    pytest.param(
        {
            "nv": 256,
            "vmax": 8.0,
            "f_fn": partial(problems.maxwellian, T=1.0),
            "beta_mode": "sc0",
            "v_min": 5.0,
        },
        id="maxwellian_low_vmax_sc0_beta",
    ),
    pytest.param(
        {
            "nv": 256,
            "vmax": 8.0,
            "f_fn": partial(problems.supergaussian, m=5, T=1.0),
            "beta_mode": "sc0",
            "v_min": 5.0,
        },
        id="supergaussian_sc0_beta",
    ),
    pytest.param(
        {
            "nv": 256,
            "vmax": 8.0,
            "f_fn": partial(problems.two_temperature, T_cold=0.5, T_hot=2.0, frac_cold=0.7),
            "beta_mode": "sc0",
            "v_min": 6.9,  # 5*v_th: hot component (T_hot=2) tail still O(1%) at v=5
        },
        id="two_temperature_sc0_beta",
    ),
]


@pytest.mark.parametrize("case", CASES)
def test_coulombian_high_v_limit_matches_fast_vfp(case):
    """CoulombianKernel D converges to FastVFP D = 1/(2βv) at high velocity.

    For any unit-density distribution, the Coulombian lower integral converges
    to (4π/3) ∫ v⁴ f dv / v = T/v = D_FastVFP as v → ∞ (upper tail negligible).

    Error is O(dv²) from midpoint quadrature of the cumulative lower integral.
    """
    grid = VelocityGrid(nv=case["nv"], vmax=case["vmax"], spherical=True)
    f0 = case["f_fn"](grid)

    if case["beta_mode"] == "analytic":
        beta = case["beta"]
    else:
        beta = _find_self_consistent_beta_single(f0, grid.v, grid.dv, spherical=True, max_steps=0)

    coulombian = CoulombianKernel(v=grid.v, dv=grid.dv)
    fastvfp = FastVFP(v=grid.v, dv=grid.dv)

    beta_arr = jnp.array([beta])
    D_coulombian = coulombian.compute_D(f0[None, :], beta_arr)[0]
    D_fastvfp = fastvfp.compute_D(f0[None, :], beta_arr)[0]

    # Use cell-center index to find threshold, then slice edges from there.
    # Edge i lies between cells i and i+1, so D[start_idx:] covers v ≳ v_min.
    v_min = case["v_min"]
    start_idx = int(jnp.argwhere(grid.v >= v_min)[0, 0])

    np.testing.assert_allclose(
        D_coulombian[start_idx:],
        D_fastvfp[start_idx:],
        rtol=5e-4,
        atol=0.0,
        err_msg=f"CoulombianKernel D does not match FastVFP D = 1/(2βv) for v >= {v_min} (beta={beta:.4f})",
    )
