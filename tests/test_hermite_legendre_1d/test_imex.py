#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""IMEX stability gate for the mixed Hermite-Legendre solver.

The explicit Lawson-RK4 step has a CFL-like limit set by the (stiff) E.d_v f Lorentz
force, whose spectral operator norm scales as ~Nl^2/width * |E|; for the two-stream
benchmark it blows up by t~20 at dt=0.01. The IMEX integrator advances that force with
a frozen-E Backward-Euler substep (unconditionally stable), so the same run stays
finite and well-behaved at a step several times larger. This test gates that:

  - explicit (lawson) at dt=0.01 goes non-finite before t=35 (the failure IMEX fixes),
  - IMEX at dt=0.01 stays finite through t=35, conserves mass (the force leaves the
    C_0/B_0 moments untouched), and saturates to a physically reasonable field energy.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from adept._hermite_legendre_1d.modules import BaseHermiteLegendre1D


def _run_two_stream(integrator: str, dt: float, tmax: float = 35.0, Nh: int = 85, Nl: int = 171):
    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": 4.0 * np.pi, "alpha": np.sqrt(2.0), "u": 0.0, "v_a": -2.5, "v_b": 2.5,
            "gamma": 0.5, "nu_H": 0.0, "nu_L": 1.0, "enforce_conservation": True, "field": True,
        },
        "grid": {"Nx": 64, "Nh": Nh, "Nl": Nl, "tmax": tmax, "dt": dt, "integrator": integrator},
        "initialization": {"type": "two-stream", "eps": 0.01, "mode": 1},
        "save": {"default": {"t": {"nt": 80}}},
        "units": {},
    }
    m = BaseHermiteLegendre1D(cfg)
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    return m(trainable_modules={})["solver result"].ys["default"]


def test_explicit_blows_up_at_large_dt():
    """The failure mode that motivates IMEX: explicit two-stream is unstable at dt=0.01."""
    d = _run_two_stream("lawson", dt=0.01)
    assert not np.all(np.isfinite(np.asarray(d["energy"]))), "expected explicit blow-up at dt=0.01"


def test_imex_stable_at_large_dt():
    d = _run_two_stream("imex", dt=0.01)
    energy = np.asarray(d["energy"])
    ee = np.asarray(d["e_energy"])
    mass = np.asarray(d["mass"])

    assert np.all(np.isfinite(energy)), "IMEX went non-finite at dt=0.01"
    # mass is untouched by the Lorentz force (G_C/G_B leave the C_0/B_0 moments alone)
    assert np.max(np.abs(mass - mass[0])) / abs(mass[0]) < 1e-10, "IMEX broke mass conservation"
    # the instability grew and saturated to a finite field energy comparable to the
    # converged explicit value (~0.33); generous bounds (Backward Euler is dissipative).
    assert ee.max() > 1e-2, "two-stream field energy did not grow under IMEX"
    assert 0.1 < ee[-1] < 1.0, f"saturated field energy out of range: {ee[-1]:.3f}"
