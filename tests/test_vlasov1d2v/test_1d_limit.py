"""1D-limit equivalence: the 2V marginal must reproduce the vlasov-1d solve.

For a v_perp-separable initial condition f = F(x, v_par) M(v_perp), every
operator in the 2V solve (spectral advection, marginal-fed field solve,
v_par collisions with marginal-moment coefficients) preserves separability,
so the marginal F(t, x, v_par) must match the 1D solver's f(t, x, v) to
floating-point accumulation. This validates the whole module composition:
grids, weights, pushers, field coupling, collisions, saves, diagnostics.
"""

import numpy as np
import pytest

from adept.vlasov1d import BaseVlasov1D
from adept.vlasov1d2v import BaseVlasov1D2V

from .utils import base_config, marginal, run_module


@pytest.mark.parametrize("fp_type", ["dougherty", "dougherty_nodrag"])
def test_marginal_matches_vlasov1d(fp_type):
    """Driven, collisional run: 2V marginal == 1D solve to ~1e-10."""
    cfg2v = base_config(fp_type=fp_type)
    cfg1d = base_config(fp_type=fp_type)
    cfg1d["solver"] = "vlasov-1d"

    mod2, res2 = run_module(BaseVlasov1D2V, cfg2v)
    mod1, res1 = run_module(BaseVlasov1D, cfg1d)

    wperp = np.asarray(mod2.cfg["grid"]["species_grids"]["electron"]["wperp"])

    # Full-f species save (rank-4) vs 1D species save (rank-3)
    f2 = marginal(res2.ys["electron.main"], wperp)
    f1 = np.asarray(res1.ys["electron.main"])
    scale = np.abs(f1).max()
    assert np.abs(f2 - f1).max() / scale < 1e-9, f"marginal mismatch {np.abs(f2 - f1).max() / scale:.2e}"

    # Electric field time series
    e2 = np.asarray(res2.ys["fields"]["e"])
    e1 = np.asarray(res1.ys["fields"]["e"])
    e_scale = np.abs(e1).max()
    assert e_scale > 1e-4, "driver did not couple"
    assert np.abs(e2 - e1).max() / e_scale < 1e-8, f"E-field mismatch {np.abs(e2 - e1).max() / e_scale:.2e}"

    # Marginal dfdt diagnostics vs the 1D diagnostics
    for diag in ["diag-vlasov-dfdt", "diag-fp-dfdt"]:
        if diag in res2.ys:
            d2 = np.asarray(res2.ys[diag])
            d1 = np.asarray(res1.ys[diag])
            dscale = np.abs(d1).max() + 1e-300
            assert np.abs(d2 - d1).max() / dscale < 1e-6, f"{diag} mismatch"


def test_dfdt_diags_present_and_marginal_shaped():
    """The dfdt diagnostics must be emitted with marginal (t, x, v_par) shape."""
    cfg = base_config(nx=16, nv=128, nvperp=8, tmax=20.0)
    cfg["save"]["diag-vlasov-dfdt"] = {"t": {"tmin": 0.0, "tmax": 20.0, "nt": 5}}
    cfg["save"]["diag-fp-dfdt"] = {"t": {"tmin": 0.0, "tmax": 20.0, "nt": 5}}
    mod, res = run_module(BaseVlasov1D2V, cfg)
    assert res.ys["diag-vlasov-dfdt"].shape == (5, 16, 128)
    assert res.ys["diag-fp-dfdt"].shape == (5, 16, 128)
    # FP dfdt must integrate to ~zero density change (conservative operator)
    dv = mod.cfg["grid"]["species_grids"]["electron"]["dv"]
    dn = np.abs(np.asarray(res.ys["diag-fp-dfdt"]).sum(axis=-1) * dv)
    assert dn.max() < 1e-10
