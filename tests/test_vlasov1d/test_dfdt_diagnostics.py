"""Regression test for the dfdt diagnostics (``diag-vlasov-dfdt`` / ``diag-fp-dfdt``).

``VlasovPoissonFokkerPlanck`` once emitted per-species diagnostic keys
(``diag-vlasov-dfdt-electron``) while ``init_state_and_args`` and the save
machinery use the unsuffixed keys (``diag-vlasov-dfdt`` / ``diag-fp-dfdt``), so
enabling either diagnostic made diffrax raise "The vector field inside ODETerm
must return a pytree with the same structure as y0".  No stock config enables
these diagnostics, so the mismatch went unnoticed.

This runs a few steps of a tiny simulation with both diagnostics enabled and
checks that the solve completes and the saved dfdt arrays are populated.  The
initial condition is a noisy super-Gaussian so that both the Vlasov term
(advection of the density noise) and the Fokker-Planck term (relaxation toward
a Maxwellian) produce a nonzero df/dt from the first step.
"""

import numpy as np
import pytest
import yaml

from adept import ergoExo

NX = 8
NV = 64
TMAX = 1.0
NT_SAVE = 3


@pytest.fixture
def cfg():
    with open("tests/test_vlasov1d/configs/fokker_planck_conservation.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["grid"]["nx"] = NX
    cfg["grid"]["nv"] = NV
    cfg["grid"]["tmax"] = TMAX

    # Density noise -> spatial gradients -> nonzero Vlasov df/dt.
    # Super-Gaussian (m=3) initial condition -> nonzero Fokker-Planck df/dt.
    cfg["density"]["species-background"]["noise_val"] = 1.0e-3
    cfg["density"]["species-background"]["m"] = 3.0
    cfg["terms"]["fokker_planck"]["space"]["baseline"] = 1.0e-2

    cfg["diagnostics"] = {"diag-vlasov-dfdt": True, "diag-fp-dfdt": True}

    t_save = {"tmin": 0.0, "tmax": TMAX, "nt": NT_SAVE}
    cfg["save"] = {
        "fields": {"t": dict(t_save)},
        "diag-vlasov-dfdt": {"t": dict(t_save)},
        "diag-fp-dfdt": {"t": dict(t_save)},
    }

    cfg["mlflow"] = {"experiment": "vlasov1d-test-dfdt-diags", "run": "dfdt-diagnostics"}
    return cfg


def test_dfdt_diagnostics_run_and_save(cfg):
    """Both dfdt diagnostics solve to completion and save populated arrays."""
    exo = ergoExo()
    exo.setup(cfg)
    result, datasets, _ = exo(None)

    solver_result = result["solver result"]
    for key in ("diag-vlasov-dfdt", "diag-fp-dfdt"):
        assert key in solver_result.ys
        arr = np.asarray(solver_result.ys[key])
        assert arr.shape == (NT_SAVE, NX, NV)
        assert np.all(np.isfinite(arr))
        # The t=0 snapshot is the zero-initialized state; every later snapshot
        # must contain actual df/dt data.
        assert np.all(np.abs(arr[1:]).max(axis=(1, 2)) > 0.0), f"{key} saves are all zero"

        # The diagnostic also flows through the dist-save post-processing.
        assert key in datasets["dists"]
        assert key in datasets["dists"][key].data_vars
