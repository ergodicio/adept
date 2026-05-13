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

from adept.driftdiffusion import AbstractKernelBasedModel, LogMeanFlux, _find_self_consistent_beta_single
from adept.vfp1d.fokker_planck import AsymptoticLocal, CoulombianKernel, FastVFP

# ============================================================================
# Fixtures
# ============================================================================

KERNEL_MODELS: list[type[AbstractKernelBasedModel]] = [CoulombianKernel, AsymptoticLocal]


@pytest.fixture
def grid():
    return VelocityGrid(nv=256, vmax=8.0, spherical=True)


@pytest.fixture(
    params=[
        pytest.param(partial(problems.shifted_maxwellian, v_shift=1.8, T=0.162), id="shifted_maxwellian"),
        pytest.param(partial(problems.supergaussian, m=5, T=1.0), id="supergaussian"),
        pytest.param(partial(problems.two_temperature, T_cold=0.5, T_hot=2.0, frac_cold=0.7), id="two_temperature"),
    ]
)
def f(request, grid):
    return request.param(grid)


# ============================================================================
# Tests: CoulombianKernel high-v limit matches FastVFP
# ============================================================================

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


# ============================================================================
# Tests: LogMeanFlux + kernel-derived C conserves energy (bilinear symmetry)
# ============================================================================


@pytest.mark.parametrize("model_cls", KERNEL_MODELS, ids=lambda c: c.__name__)
def test_logmean_flux_conserves_energy(model_cls, f, grid):
    """LogMeanFlux operator with kernel-derived C and D conserves energy.

    The LogMeanFlux scheme uses log-mean interpolation for f at edges,
    which is the same interpolation used in compute_D. This ensures the
    kernel's bilinear symmetry (Σ K[A]·B·dε = Σ K[B]·A·dε) carries
    through to the full tridiagonal operator.

    We verify that the spherical energy moment of the operator increment
    vanishes: Σ v⁴ · (f - op.mv(f)) ≈ 0, i.e. the implicit step
    conserves energy in the dt → 0 limit.
    """
    model = model_cls(v=grid.v, dv=grid.dv)
    scheme = LogMeanFlux(dv=grid.dv)

    beta = _find_self_consistent_beta_single(
        f,
        grid.v,
        grid.dv,
        spherical=True,
        max_steps=0,
    )
    C_edge, D = model.compute_C_and_D(f[None, :], beta[None])
    C_edge, D = C_edge[0], D[0]

    nu = 1.0 / grid.v**2
    dt = 1.0
    op = scheme.get_operator(C_edge=C_edge, D=D, nu=nu, dt=dt, f=f)

    # df/dt = (f - op.mv(f)) / dt; energy = Σ v⁴·f·dv (spherical)
    increment = jnp.array(f) - jnp.array(op.mv(f))
    energy_change = jnp.sum(increment * grid.v**4 * grid.dv)
    total_energy = jnp.sum(jnp.array(f) * grid.v**4 * grid.dv)

    # With v_edge (arithmetic mean) for the D conversion, the velocity-space
    # operator is mathematically equivalent to energy-space Buet, so energy
    # conservation holds to machine precision.
    np.testing.assert_allclose(
        float(energy_change / total_energy),
        0.0,
        atol=1e-12,
        err_msg=f"{model_cls.__name__}: LogMeanFlux operator does not conserve energy",
    )
