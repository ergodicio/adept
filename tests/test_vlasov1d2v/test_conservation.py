"""Conservation and stationarity gates for the v_par collision operator.

D1 gates: density machine-exact, momentum at the 1e-8 level, and ENERGY AT
PARITY with the vlasov-1d operator on the same problem. The CD-paired
Dougherty has a known O(dv^2) secular energy drift at a sampled Maxwellian
(measured 4.2e-4 per collision time at nv=256, x16 smaller at nv=1024;
~7e-6/collision-time at the production nv=2048) which the 2V operator
inherits exactly by construction — the parity gate pins that, and the D2
operator will improve on it (Chang-Cooper pairing / moment restoration).
"""

import numpy as np

from adept.vlasov1d import BaseVlasov1D
from adept.vlasov1d2v import BaseVlasov1D2V

from .utils import base_config, run_module


def _scalars(res):
    return {k: np.asarray(v) for k, v in res.ys["default"].items()}


def test_collisional_conservation_parity_with_1d():
    """Beam relaxation: n exact, P at 1e-8, E drift no worse than vlasov-1d."""
    beam = {
        "noise_seed": 7,
        "noise_type": "gaussian",
        "noise_val": 0.0,
        "v0": 1.5,
        "T0": 0.5,
        "m": 2.0,
        "basis": "uniform",
        "baseline": 0.1,
        "bump_or_trough": "bump",
        "center": 0.0,
        "rise": 25.0,
        "bump_height": 0.0,
        "width": 1.0e5,
    }

    drifts = {}
    for solver, cls in [("vlasov-1d", BaseVlasov1D), ("vlasov-1d2v", BaseVlasov1D2V)]:
        cfg = base_config(nx=4, nv=256, nvperp=16, tmax=100.0, dt=0.25, nu=1.0e-2, a0=0.0)
        cfg["drivers"]["ex"] = {}
        cfg["density"]["species-beam"] = dict(beam)
        cfg["solver"] = solver
        mod, res = run_module(cls, cfg)
        s = _scalars(res)
        en = 0.5 * s["mean_P_electron"] + 0.5 * s["mean_e2"]
        if solver == "vlasov-1d2v":
            en = en + 0.5 * s["mean_Pperp_electron"]
        # centered third moment (skewness) from the fields save: relaxes to ~0
        q_centered = np.asarray(res.ys["fields"]["electron"]["q"]).mean(axis=1)
        drifts[solver] = {
            "n": abs(s["mean_n_electron"][-1] - s["mean_n_electron"][0]) / s["mean_n_electron"][0],
            "j": abs(s["mean_j_electron"][-1] - s["mean_j_electron"][0]),
            "e": abs(en[-1] - en[0]) / en[0],
            "q0": abs(q_centered[0]),
            "q1": abs(q_centered[-1]),
        }

    d2, d1 = drifts["vlasov-1d2v"], drifts["vlasov-1d"]
    assert d2["n"] < 1e-12, f"density drift {d2['n']:.2e}"
    assert d2["j"] < 1e-8, f"momentum drift {d2['j']:.2e}"
    assert d2["e"] < 1.5 * d1["e"] + 1e-9, f"energy drift {d2['e']:.2e} vs 1D {d1['e']:.2e}"
    # the beam must actually relax: skewness decays at ~3 nu (nu*t = 1 here)
    assert d2["q1"] < 0.2 * d2["q0"], f"beam did not relax: |q| {d2['q0']:.2e} -> {d2['q1']:.2e}"


def test_nodrag_maxwellian_stationarity():
    """dougherty_nodrag: Maxwellian stationary up to the (nu*dt)^2 splitting residual.

    The implicit solve advances (I + dt nu L)^-1 while the Maxwellian
    diffusion is subtracted explicitly, so the cancellation at f = f_M is
    exact only to O((nu dt)^2) per step — identical to the vlasov-1d
    operator (measured 1.13e-3 at nu dt = 2.5e-3 for BOTH modules; the pawl
    production runs sit at nu dt = 5e-5 where this is ~1e-8-level). Assert
    the quadratic scaling and an absolute bound at production-like nu dt.
    """
    drift = {}
    for nu in (1.0e-2, 1.0e-3):
        cfg = base_config(nx=4, nv=256, nvperp=16, tmax=25.0, dt=0.25, nu=nu, fp_type="dougherty_nodrag", a0=0.0)
        cfg["drivers"]["ex"] = {}
        mod, res = run_module(BaseVlasov1D2V, cfg)
        f_series = np.asarray(res.ys["electron.main"])
        drift[nu] = np.abs(f_series[-1] - f_series[0]).max() / f_series[0].max()

    # second-order in nu dt: 10x smaller nu (same number of steps => 10x fewer
    # collision times) must give ~100x smaller drift; allow generous slack
    assert drift[1.0e-3] < 0.05 * drift[1.0e-2], f"drift not O((nu dt)^2): {drift}"
    assert drift[1.0e-3] < 3e-5, f"nodrag Maxwellian drift {drift[1.0e-3]:.2e}"
