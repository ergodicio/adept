import os, yaml, mlflow, numpy as np

from utils.runner import run


def _run_(Z):
    # with open("configs/tf-1d/damping.yaml", "r") as fi:
    with open(f"{os.path.join(os.getcwd(), 'tests/test_vfp1d/epp-short')}.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["units"]["Z"] = Z
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # modify config
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        result, datasets = run(cfg)

    kappa_eh = mlflow.get_run(mlflow_run.info.run_id).data.metrics["kappa_eh"]
    kappa = mlflow.get_run(mlflow_run.info.run_id).data.metrics["kappa"]
    return kappa, kappa_eh


def test_kappa_eh():
    actual = []
    desired = []
    for Z in list(range(1, 21, 4)) + [40, 60, 80]:
        kappa, kappa_eh = _run_(Z)
        actual.append(kappa)
        desired.append(kappa_eh)

    np.testing.assert_allclose(actual, desired, rtol=0.2)
