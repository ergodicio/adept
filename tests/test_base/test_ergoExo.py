import yaml

import adept.patched_mlflow as mlflow
from adept import ergoExo


def test_reuse_config_dict(tags):
    with open("tests/test_base/configs/example.yaml") as file:
        cfg = yaml.safe_load(file)

    exo = ergoExo()
    exo.setup(cfg)
    with mlflow.start_run(run_id=exo.mlflow_run_id, nested=exo.mlflow_nested):
        mlflow.set_tags(tags)

    exo = ergoExo()
    exo.setup(cfg)
    with mlflow.start_run(run_id=exo.mlflow_run_id, nested=exo.mlflow_nested):
        mlflow.set_tags(tags)
