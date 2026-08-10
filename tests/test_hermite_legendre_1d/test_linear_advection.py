#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Linear-advection physics gate for the mixed Hermite-Legendre solver.

With phi = 0 the Vlasov equation reduces to advection d_t f + v d_x f = 0, whose
exact solution is f(x, v, t) = f(x - v t, v, 0). Running the full lifecycle on a
coarse grid and reconstructing f = f0 + df from the spectral coefficients should
match the analytic solution at a time before the spectral velocity recurrence
(paper sec 4.2, Fig 3). Also gates that mass/momentum/energy are conserved to
machine precision under pure advection (all k=0 moments are time-invariant).
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from adept._hermite_legendre_1d.modules import BaseHermiteLegendre1D
from adept._hermite_legendre_1d.vector_field import _hermite_function_values, _legendre_basis_values


def _run(cfg):
    m = BaseHermiteLegendre1D(cfg)
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    return m, m(trainable_modules={})["solver result"]


def test_linear_advection_matches_analytic():
    Lx = 2.0 * np.pi
    alpha = np.sqrt(2.0)
    Nh = Nl = 48
    Nx = 48
    v_a, v_b = -6.0, 6.0
    t_check = 2.0

    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": Lx,
            "alpha": alpha,
            "u": 0.0,
            "v_a": v_a,
            "v_b": v_b,
            "gamma": 0.5,
            "nu_H": 0.0,
            "nu_L": 0.0,
            "enforce_conservation": True,
            "field": False,
        },
        "grid": {"Nx": Nx, "Nh": Nh, "Nl": Nl, "tmax": t_check, "dt": 0.01},
        "initialization": {"type": "linear-advection", "eps": 1.0, "mode": 1},
        "save": {"hermite": {"t": {"nt": 2}}, "legendre": {"t": {"nt": 2}}},
        "units": {},
    }
    m, sol = _run(cfg)
    x = np.asarray(m.cfg["grid"]["x"])

    Ck = np.asarray(sol.ys["hermite"]["Ck"])[-1]  # (Nh, Nx) k-space
    Bk = np.asarray(sol.ys["legendre"]["Bk"])[-1]  # (Nl, Nx)
    C = np.fft.ifft(Ck, axis=-1, norm="forward").real  # (Nh, Nx)
    B = np.fft.ifft(Bk, axis=-1, norm="forward").real  # (Nl, Nx)

    v = np.linspace(v_a + 0.1, v_b - 0.1, 120)
    psi = _hermite_function_values(Nh, v, u=0.0, alpha=alpha)  # (Nh, len(v))
    xi = _legendre_basis_values(Nl, v, v_a, v_b)  # (Nl, len(v))

    f = C.T @ psi + B.T @ xi  # (Nx, len(v))
    XX, VV = np.meshgrid(x, v, indexing="ij")
    f_exact = (1.0 + np.cos(XX - VV * t_check)) / np.sqrt(2.0 * np.pi) * np.exp(-(VV**2) / 2.0)

    rel_l2 = np.linalg.norm(f - f_exact) / np.linalg.norm(f_exact)
    assert rel_l2 < 0.02, f"linear advection rel L2 = {rel_l2:.4f}"


def test_advection_conserves_moments_to_machine_precision():
    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": 2.0 * np.pi,
            "alpha": np.sqrt(2.0),
            "u": 0.0,
            "v_a": -6.0,
            "v_b": 6.0,
            "gamma": 0.5,
            "nu_H": 0.0,
            "nu_L": 0.0,
            "enforce_conservation": True,
            "field": False,
        },
        "grid": {"Nx": 48, "Nh": 48, "Nl": 48, "tmax": 5.0, "dt": 0.02},
        "initialization": {"type": "linear-advection", "eps": 1.0, "mode": 1},
        "save": {"default": {"t": {"nt": 60}}},
        "units": {},
    }
    _, sol = _run(cfg)
    d = sol.ys["default"]
    for k in ("mass", "momentum", "energy"):
        a = np.asarray(d[k])
        base = abs(a[0]) if abs(a[0]) > 1e-30 else 1.0
        assert np.max(np.abs(a - a[0])) / base < 1e-11, f"{k} not conserved under advection"
