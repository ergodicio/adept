import os, yaml, mlflow, numpy as np, pytest

from adept import ergoExo
import pytest
from jax import devices


def _run_(Z, ee):
    # with open("configs/tf-1d/damping.yaml", "r") as fi:
    with open(f"{os.path.join(os.getcwd(), 'tests/test_vfp1d/epp-short')}.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["units"]["Z"] = Z

    if ee:
        cfg["terms"]["fokker_planck"]["flm"]["ee"] = True
        cfg["grid"]["nv"] = 2048

    exo = ergoExo()
    exo.setup(cfg)
    sol, datasets, run_id = exo(None)
    dataT = datasets["fields"]["fields-T keV"].data
    np.testing.assert_almost_equal(np.mean(dataT[-4, :]), np.mean(dataT[4, :]), decimal=5)

    datan = datasets["fields"]["fields-n n_c"].data
    np.testing.assert_almost_equal(np.mean(datan[-4, :]), np.mean(datan[4, :]), decimal=5)

    kappa_eh = mlflow.get_run(run_id).data.metrics["kappa_eh"]
    kappa = mlflow.get_run(run_id).data.metrics["kappa"]

    np.testing.assert_almost_equal(kappa, kappa_eh, decimal=0)

    return run_id


@pytest.mark.parametrize("Z", list(range(1, 21, 4)) + [40, 60, 80])
@pytest.mark.parametrize("ee", [True, False])
def test_kappa_eh(Z, ee):
    if not any(["gpu" == device.platform for device in devices()]):
        pytest.skip(f"Skipping Z={Z} to save time because no GPU is available")
        if Z in [1, 21, 80]:
            _run_(Z, ee)

    else:
        _run_(Z, ee)
