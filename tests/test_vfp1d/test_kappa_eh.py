import os

import numpy as np
import pytest
import yaml
from jax import devices

import adept.patched_mlflow as mlflow
from adept import ergoExo


def _run_(Z, ee, config_name="epp-short", tags=None):
    # with open("configs/tf-1d/damping.yaml", "r") as fi:
    with open(f"{os.path.join(os.getcwd(), 'tests/test_vfp1d', config_name)}.yaml") as fi:
        cfg = yaml.safe_load(fi)

    cfg["units"]["Z"] = Z

    if ee:
        cfg["terms"]["fokker_planck"]["flm"]["ee"] = True
        cfg["grid"]["nv"] = 2048

    exo = ergoExo()
    exo.setup(cfg)
    if tags is not None:
        with mlflow.start_run(run_id=exo.mlflow_run_id, nested=exo.mlflow_nested):
            mlflow.set_tags(tags)

    sol, datasets, run_id = exo(None)
    dataT = datasets["fields"]["fields-T keV"].data
    np.testing.assert_almost_equal(np.mean(dataT[-4, :]), np.mean(dataT[4, :]), decimal=5)

    datan = datasets["fields"]["fields-n n_c"].data
    np.testing.assert_almost_equal(np.mean(datan[-4, :]), np.mean(datan[4, :]), decimal=5)

    kappa_eh = mlflow.get_run(run_id).data.metrics["kappa_eh"]
    kappa = mlflow.get_run(run_id).data.metrics["kappa"]

    np.testing.assert_almost_equal(kappa, kappa_eh, decimal=0)

    return run_id


@pytest.mark.parametrize("Z", list(range(1, 22, 4)) + [40, 60, 80])
@pytest.mark.parametrize("ee", [True, False])
def test_kappa_eh(Z, ee, tags):
    if not any(["gpu" == device.platform for device in devices()]):
        if Z in [1, 21, 80]:
            _run_(Z, ee, tags=tags)
        else:
            pytest.skip(f"Skipping Z={Z} to save time because no GPU is available")

    else:
        _run_(Z, ee)


@pytest.mark.parametrize("Z", list(range(1, 22, 4)) + [40, 60, 80])
@pytest.mark.parametrize("ee", [True, False])
def test_kappa_eh_reflective(Z, ee, tags):
    if not any(["gpu" == device.platform for device in devices()]):
        if Z in [1, 21, 80]:
            _run_(Z, ee, config_name="epp-short-reflective", tags=tags)
        else:
            pytest.skip(f"Skipping Z={Z} to save time because no GPU is available")

    else:
        _run_(Z, ee, config_name="epp-short-reflective")
