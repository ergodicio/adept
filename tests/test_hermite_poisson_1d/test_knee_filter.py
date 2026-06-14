#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Scale-selective knee absorber (drivers.hermite_filter profile="knee").

A tanh step in absolute n: ~0 below n_knee, ~1 above. Built so a low
Lenard-Bernstein nu can preserve the resonant EPW mode (n ~ vphi^2/2) while
the filamentation cascade just above it is strongly damped — a separation
the default (n/Nn)^order Hou-Li profile cannot make at large Nn.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D


def _diag_e(filter_cfg):
    cfg = {
        "physics": {"Lx": 10.0, "alpha_e": 1.0, "alpha_i": 1.0, "nu": 0.0,
                    "static_ions": True, "c_light": 1.0},
        "grid": {"Nn": 1024, "Ni": 2, "Nx": 32, "tmax": 1.0, "dt": 0.1},
        "drivers": {"hermite_filter": filter_cfg},
        "save": {},
    }
    m = BaseHermitePoisson1D(cfg)
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    # the electron DiagonalExp holds the realized absorber column
    return m.diffeqsolve_quants["terms"].vector_field.combined_exp.linear_e.diagonal


def test_knee_profile_separates_resonance_from_cascade():
    diag = _diag_e({"enabled": True, "profile": "knee",
                    "strength": 0.1, "n_knee": 45.0, "width": 8.0})
    col = np.asarray(diag.hou_li_col_1d)
    assert diag.hou_li_strength == 0.1
    # ~0 at the resonance (n=24), ~1 in the cascade (n>=70)
    assert col[24] < 0.02
    assert col[70] > 0.95
    assert abs(col[45] - 0.5) < 1e-9   # tanh midpoint at the knee
    # monotonic
    assert np.all(np.diff(col) >= -1e-12)


def test_houli_profile_still_default():
    diag = _diag_e({"enabled": True, "strength": 36.0, "order": 8})
    col = np.asarray(diag.hou_li_col_1d)
    # (n/Nn)^8: negligible at moderate n, ~1 only at the truncation edge
    assert col[24] < 1e-10
    assert col[512] < 0.01
    np.testing.assert_allclose(col[-1], 1.0)


def test_knee_preserves_low_modes_dynamically():
    """A pure resonance-band excitation (n=24) is barely damped; a cascade-band
    excitation (n=80) decays fast — over a short force-free, k=0 run."""
    import copy

    base = {
        "physics": {"Lx": 10.0, "alpha_e": 1.0, "alpha_i": 1.0, "nu": 0.0,
                    "static_ions": True, "c_light": 1.0},
        "grid": {"Nn": 1024, "Ni": 2, "Nx": 16, "tmax": 20.0, "dt": 0.1},
        "drivers": {"hermite_filter": {"enabled": True, "profile": "knee",
                                       "strength": 0.1, "n_knee": 45.0, "width": 8.0}},
        "save": {"hermite": {"t": {"tmin": 0.0, "tmax": 20.0, "nt": 3}}},
    }

    decays = {}
    for nmode in (24, 80):
        cfg = copy.deepcopy(base)
        m = BaseHermitePoisson1D(cfg)
        m.get_derived_quantities()
        m.get_solver_quantities()
        m.init_state_and_args()
        Ck = m.state["Ck_electrons"].view(jnp.complex128)
        Ck = Ck.at[nmode, 0].set(1.0)  # k=0 -> streaming inert, isolate the absorber
        m.state["Ck_electrons"] = Ck.view(jnp.float64)
        m.init_diffeqsolve()
        sol = m(None)["solver result"]
        Ck_t = np.asarray(sol.ys["hermite"]["electrons"]).view(np.complex128)
        decays[nmode] = abs(Ck_t[-1, nmode, 0]) / abs(Ck_t[0, nmode, 0])

    assert decays[24] > 0.95      # resonance preserved
    assert decays[80] < 0.2       # cascade strongly damped
