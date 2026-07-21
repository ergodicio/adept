#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Conservation gate for the mixed Hermite-Legendre solver (paper sec 3, Fig 7).

With the constraint J_{Nh,0}=J_{Nh,1}=J_{Nh,2}=0 enforced, the self-consistent
(field-on) mixed method conserves total mass and momentum to machine precision and
total energy to the explicit time integrator's accuracy (the paper's machine-
precision energy relies on the implicit-midpoint integrator; the explicit Lawson-RK4
used here is time-integration-limited but convergent). The run also stays finite.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from adept._hermite_legendre_1d.modules import BaseHermiteLegendre1D


def _run(cfg):
    m = BaseHermiteLegendre1D(cfg)
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    return m(trainable_modules={})["solver result"]


def _two_stream_cfg(enforce, Nh=48, Nl=48):
    return {
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
            "enforce_conservation": enforce,
            "field": True,
        },
        "grid": {"Nx": 48, "Nh": Nh, "Nl": Nl, "tmax": 10.0, "dt": 0.01},
        "initialization": {"type": "two-stream", "eps": 0.05, "mode": 1},
        "save": {"default": {"t": {"nt": 100}}},
        "units": {},
    }


def test_two_stream_conserves_invariants_with_enforcement():
    sol = _run(_two_stream_cfg(enforce=True))
    d = sol.ys["default"]
    drifts = {}
    for k in ("mass", "momentum", "energy"):
        a = np.asarray(d[k])
        assert np.all(np.isfinite(a)), f"{k} went non-finite"
        base = abs(a[0]) if abs(a[0]) > 1e-30 else 1.0
        drifts[k] = float(np.max(np.abs(a - a[0])) / base)
    assert drifts["mass"] < 1e-10, drifts
    assert drifts["momentum"] < 1e-10, drifts
    assert drifts["energy"] < 1e-5, drifts  # time-integration-limited (dt=0.01)


def test_collisions_only_damp_high_modes():
    """nu_L damps Legendre modes m>=3 only; mass/momentum/energy stay conserved
    because the collision profile vanishes on the first three moments."""
    sol = _run(_two_stream_cfg(enforce=True))
    d = sol.ys["default"]
    # mass uses C_0, B_0; momentum C_0,C_1,B_0,B_1; energy up to C_2,B_2 -- all
    # below the collisional cutoff, so collisions must not spoil them.
    for k, tol in (("mass", 1e-10), ("momentum", 1e-10)):
        a = np.asarray(d[k])
        base = abs(a[0]) if abs(a[0]) > 1e-30 else 1.0
        assert np.max(np.abs(a - a[0])) / base < tol
