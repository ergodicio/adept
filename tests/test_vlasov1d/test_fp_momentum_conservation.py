#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""
Momentum conservation of the Dougherty operator where n(x) != 1.

The Dougherty drag must center on the *mean velocity* u(x) = ∫v f dv / n(x).
A drag centered on the raw first moment ∫v f dv = n*u (the historical bug)
breaks momentum conservation exactly where the density deviates from 1, e.g.
for bump-on-tail or large-amplitude density perturbations with collisions on.

This test applies the collision operator in isolation (no advection, no
fields) to a drifting Maxwellian with a strong density modulation and checks
per-cell conservation of density, momentum, and energy.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from jax import numpy as jnp

from adept._vlasov1d.solvers.pushers.fokker_planck import Collisions


def _make_cfg(nx, nv, vmax, fp_type):
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    return {
        "grid": {
            "species_grids": {"electron": {"v": v, "dv": dv, "nv": nv, "vmax": vmax}},
            "species_params": {"electron": {"charge": -1.0, "mass": 1.0, "charge_to_mass": -1.0, "T0": 1.0}},
        },
        "terms": {
            "fokker_planck": {"is_on": True, "type": fp_type},
            "krook": {"is_on": False},
        },
    }


# Energy tolerance is per-scheme: Chang-Cooper preserves the discrete equilibrium
# to ~1e-6, plain central differencing only to ~1e-3 at this resolution.
@pytest.mark.parametrize("fp_type,energy_rtol", [("dougherty", 5e-3), ("chang_cooper_dougherty", 1e-5)])
def test_dougherty_momentum_conservation_with_density_perturbation(fp_type, energy_rtol):
    nx, nv, vmax = 16, 512, 6.4
    cfg = _make_cfg(nx, nv, vmax, fp_type)
    v = np.array(cfg["grid"]["species_grids"]["electron"]["v"])
    dv = cfg["grid"]["species_grids"]["electron"]["dv"]

    # Strong density modulation and a finite drift: exactly the regime where
    # centering the drag on n*u instead of u destroys momentum conservation.
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    n_prof = 1.0 + 0.5 * np.sin(x)
    u0 = 0.5
    f = n_prof[:, None] * np.exp(-((v[None, :] - u0) ** 2) / 2.0)
    f = f / (np.sum(np.exp(-((v - u0) ** 2) / 2.0)) * dv)

    f = jnp.array(f)
    collisions = Collisions(cfg)

    nu = jnp.ones(nx)
    dt = 0.1
    f_out = f
    for _ in range(50):
        f_out = collisions(nu, None, f_out, dt)

    n0_ = np.sum(np.array(f), axis=1) * dv
    n1 = np.sum(np.array(f_out), axis=1) * dv
    p0 = np.sum(np.array(f) * v[None, :], axis=1) * dv
    p1 = np.sum(np.array(f_out) * v[None, :], axis=1) * dv
    e0 = np.sum(np.array(f) * v[None, :] ** 2, axis=1) * dv
    e1 = np.sum(np.array(f_out) * v[None, :] ** 2, axis=1) * dv

    # The historical n*u-centered drag gives O(50%) momentum errors here;
    # the corrected operator conserves to discretization level (~1e-7).
    np.testing.assert_allclose(n1, n0_, rtol=1e-10, err_msg="density not conserved")
    np.testing.assert_allclose(p1, p0, rtol=1e-6, err_msg="momentum not conserved where n(x) != 1")
    np.testing.assert_allclose(e1, e0, rtol=energy_rtol, err_msg="energy not conserved")

    # And the distribution must still be centered on u0, not n(x)*u0
    u_final = p1 / n1
    np.testing.assert_allclose(u_final, u0, atol=1e-6)
