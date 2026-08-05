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

    # NOTE: the 2V diagnostics are time-INTEGRALS while the 1D ones are sampled
    # rates, so they are not directly comparable; the marginal-f and E-field
    # checks above already establish the 1D-limit equivalence. The accumulators
    # get their own (stronger) identity test below.


def test_cumulative_diags_telescope_to_the_marginal_change():
    """acc_vlasov + acc_fp must equal F(t) - F(0) exactly, by construction.

    The two accumulators partition every step's change in the marginal between
    the Vlasov and collision substeps, so their sum telescopes to the total
    change. This is the identity that makes the differenced accumulators a
    valid (and alias-free) decomposition of d(Delta_eps)/dt.
    """
    tmax, nt = 20.0, 5
    cfg = base_config(nx=16, nv=128, nvperp=8, tmax=tmax)
    cfg["save"]["electron"]["main"]["t"] = {"tmin": 0.0, "tmax": tmax, "nt": nt}
    cfg["save"]["diag-vlasov-cumulative"] = {"t": {"tmin": 0.0, "tmax": tmax, "nt": nt}}
    cfg["save"]["diag-fp-cumulative"] = {"t": {"tmin": 0.0, "tmax": tmax, "nt": nt}}

    mod, res = run_module(BaseVlasov1D2V, cfg)

    assert res.ys["diag-vlasov-cumulative"].shape == (nt, 16, 128)
    assert res.ys["diag-fp-cumulative"].shape == (nt, 16, 128)

    grid = mod.cfg["grid"]["species_grids"]["electron"]
    wperp = np.asarray(grid["wperp"])
    F = marginal(res.ys["electron.main"], wperp)  # (t, x, v_par)
    acc = np.asarray(res.ys["diag-vlasov-cumulative"]) + np.asarray(res.ys["diag-fp-cumulative"])

    lhs = acc - acc[0]
    rhs = F - F[0]
    scale = np.abs(rhs).max()
    assert scale > 0, "nothing evolved"
    assert np.abs(lhs - rhs).max() / scale < 1e-10, (
        f"accumulators do not telescope: {np.abs(lhs - rhs).max() / scale:.2e}"
    )

    # the FP accumulator must carry no density (conservative operator)
    dn = np.abs(np.asarray(res.ys["diag-fp-cumulative"]).sum(axis=-1) * grid["dv"])
    assert dn.max() < 1e-10
