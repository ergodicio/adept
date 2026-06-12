#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Lenard-Bernstein collision model: exact per-mode decay rates.

Hermite functions are eigenfunctions of the LB operator with eigenvalue
-nu*n. With physics.collision_model = "lenard-bernstein", modes n >= 3 of a
force-free, spatially-uniform state must decay as exp(-nu*n*t) exactly (the
DiagonalExp applies it analytically), while n = 0, 1, 2 are untouched — the
standard conserving truncation so collisions conserve density, momentum,
and energy (USER: "typically, folks avoid having any of the collision
operators hit the first 3 modes/moments").
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D, _collision_profile


def test_collision_profiles():
    prof_lb = np.asarray(_collision_profile(8, "lenard-bernstein"))
    np.testing.assert_allclose(prof_lb, [0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    prof_hc = np.asarray(_collision_profile(8, "hypercollision"))
    assert prof_hc[0] == prof_hc[1] == prof_hc[2] == 0.0
    np.testing.assert_allclose(prof_hc[-1], 1.0)


def test_lenard_bernstein_decay_rates():
    nu = 1.0e-3
    tmax = 50.0
    Nn = 8

    cfg = {
        "physics": {
            "Lx": 10.0,
            "alpha_e": 1.0,
            "alpha_i": 1.0,
            "nu": nu,
            "collision_model": "lenard-bernstein",
            "static_ions": True,
            "c_light": 1.0,
        },
        "grid": {"Nn": Nn, "Ni": 2, "Nx": 32, "tmax": tmax, "dt": 0.1},
        "drivers": {},
        "save": {"hermite": {"t": {"tmin": 0.0, "tmax": tmax, "nt": 11}}},
    }

    module = BaseHermitePoisson1D(cfg)
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()

    # Spatially-uniform, force-free state: seed every mode at k=0.
    # (Uniform C_n produces no density perturbation -> E stays 0; free
    # streaming at k=0 is inert -> evolution is purely collisional.)
    import jax.numpy as jnp

    Ck = module.state["Ck_electrons"].view(jnp.complex128)
    for n in range(Nn):
        Ck = Ck.at[n, 0].set(1.0)
    module.state["Ck_electrons"] = Ck.view(jnp.float64)

    module.init_diffeqsolve()
    sol = module(None)["solver result"]

    Ck_t = np.asarray(sol.ys["hermite"]["electrons"]).view(np.complex128)  # (nt, Nn, Nx)
    t = np.asarray(sol.ts["hermite"])

    amp = np.abs(Ck_t[:, :, 0])  # k=0 component of each mode vs time
    for n in range(3, Nn):
        expected = np.exp(-nu * n * t)
        np.testing.assert_allclose(amp[:, n], expected, rtol=1e-8,
                                   err_msg=f"mode n={n} LB decay mismatch")
    # n=0,1,2 (density, momentum, energy) must be exactly conserved
    for n in range(3):
        np.testing.assert_allclose(amp[:, n], 1.0, rtol=1e-10,
                                   err_msg=f"conserved moment n={n} was damped")
