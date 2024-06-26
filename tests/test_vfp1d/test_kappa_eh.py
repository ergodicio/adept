import os, yaml, mlflow, numpy as np, pytest

from utils.runner import run


def _run_(Z, ee):
    # with open("configs/tf-1d/damping.yaml", "r") as fi:
    with open(f"{os.path.join(os.getcwd(), 'tests/test_vfp1d/epp-short')}.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["units"]["Z"] = Z

    if ee:
        cfg["terms"]["fokker_planck"]["flm"]["ee"] = True
        cfg["grid"]["nv"] = 2048

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        result, datasets = run(cfg)

    result, state, args = result
    dataT = datasets["fields"]["fields-T keV"].data
    np.testing.assert_almost_equal(np.mean(dataT[-4, :]), np.mean(dataT[4, :]), decimal=5)

    datan = datasets["fields"]["fields-n n_c"].data
    np.testing.assert_almost_equal(np.mean(datan[-4, :]), np.mean(datan[4, :]), decimal=5)

    kappa_eh = mlflow.get_run(mlflow_run.info.run_id).data.metrics["kappa_eh"]
    kappa = mlflow.get_run(mlflow_run.info.run_id).data.metrics["kappa"]

    np.testing.assert_almost_equal(kappa, kappa_eh, decimal=0)

    return mlflow_run.info.run_id


@pytest.mark.parametrize("Z", list(range(1, 21, 4)) + [40, 60, 80])
@pytest.mark.parametrize("ee", [True, False])
def test_kappa_eh(Z, ee):
    _run_(Z, ee)
