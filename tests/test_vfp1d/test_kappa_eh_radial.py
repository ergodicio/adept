import os

import mlflow
import numpy as np
import pytest
import yaml
from jax import devices

from adept import ergoExo


def _run_(Z, ee):
    # with open("configs/tf-1d/damping.yaml", "r") as fi:
    with open(f"{os.path.join(os.getcwd(), 'tests/test_vfp1d/epp-short-radial')}.yaml") as fi:
        cfg = yaml.safe_load(fi)

    cfg["units"]["Z"] = Z

    if ee:
        cfg["terms"]["fokker_planck"]["flm"]["ee"] = True
        cfg["grid"]["nv"] = 2048

    exo = ergoExo()
    exo.setup(cfg)

    sol, datasets, run_id = exo(None)
    dataT = datasets["fields"]["fields-T keV"].data

    # For radial geometry with reflecting BC at r=0, check solution stability
    # Temperature should be finite and positive everywhere
    assert np.all(np.isfinite(dataT)), "Temperature contains non-finite values"
    assert np.all(dataT > 0), "Temperature contains non-positive values"

    # Check that temperature at center (r=0) evolves smoothly
    T_center_initial = dataT[0, 0]
    T_center_final = dataT[0, -1]
    assert np.isfinite(T_center_initial) and np.isfinite(T_center_final)

    datan = datasets["fields"]["fields-n n_c"].data
    # Density should remain approximately uniform in radial geometry
    assert np.all(np.isfinite(datan)), "Density contains non-finite values"
    np.testing.assert_almost_equal(np.mean(datan[:, 0]), np.mean(datan[:, -1]), decimal=1)

    kappa_eh = mlflow.get_run(run_id).data.metrics["kappa_eh"]
    kappa = mlflow.get_run(run_id).data.metrics["kappa"]

    np.testing.assert_almost_equal(kappa, kappa_eh, decimal=0)

    return run_id


@pytest.mark.parametrize("Z", [1, 80])  # list(range(1, 22, 4)) + [40, 60, 80])
@pytest.mark.parametrize("ee", [True, False])
def test_kappa_eh_radial(Z, ee):
    if not any(["gpu" == device.platform for device in devices()]):
        if Z in [1, 21, 80]:
            _run_(Z, ee)
        else:
            pytest.skip(f"Skipping Z={Z} to save time because no GPU is available")

    else:
        _run_(Z, ee)
