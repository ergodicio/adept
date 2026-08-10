#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Implicit-midpoint (AD-JFNK) gate for the mixed Hermite-Legendre solver.

`integrator: implicit` advances the FULL right-hand side with the implicit-midpoint
rule y1 = y0 + dt F((y0+y1)/2), solved by Jacobian-free Newton-Krylov: the Newton
linear systems use a matrix-free GMRES whose Jacobian-vector products are exact
autodiff JVPs (jax.linearize) -- the Jacobian is never assembled. Implicit midpoint is
A-stable (no CFL) and conserves quadratic invariants, so unlike the explicit and IMEX
paths it stays stable into the saturated/long-time regime and conserves mass exactly
and energy to the nonlinear-solve tolerance.

This gates a two-stream run at dt=0.05 -- a step at which the explicit Lawson path is
violently unstable -- staying finite, conserving mass to machine precision, and holding
energy far better than an explicit step of the same size could.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from adept._hermite_legendre_1d.modules import BaseHermiteLegendre1D


def _run_two_stream(integrator: str, dt: float, tmax: float = 10.0, Nh: int = 48, Nl: int = 48):
    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": 4.0 * np.pi,
            "alpha": np.sqrt(2.0),
            "u": 0.0,
            "v_a": -2.5,
            "v_b": 2.5,
            "gamma": 0.5,
            "nu_H": 0.0,
            "nu_L": 1.0,
            "enforce_conservation": True,
            "field": True,
        },
        "grid": {"Nx": 48, "Nh": Nh, "Nl": Nl, "tmax": tmax, "dt": dt, "integrator": integrator},
        "initialization": {"type": "two-stream", "eps": 0.05, "mode": 1},
        "save": {"default": {"t": {"nt": 40}}},
        "units": {},
    }
    m = BaseHermiteLegendre1D(cfg)
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    return m(trainable_modules={})["solver result"].ys["default"]


def test_implicit_midpoint_stable_and_conserving():
    d = _run_two_stream("implicit", dt=0.05)
    energy = np.asarray(d["energy"])
    mass = np.asarray(d["mass"])

    assert np.all(np.isfinite(energy)), "implicit midpoint went non-finite"
    # implicit midpoint conserves mass exactly and energy to the JFNK solve tolerance
    assert np.max(np.abs(mass - mass[0])) / abs(mass[0]) < 1e-11, "mass not conserved"
    assert np.max(np.abs(energy - energy[0])) / abs(energy[0]) < 1e-4, "energy drift too large"
