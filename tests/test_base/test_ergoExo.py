import yaml
from adept import ergoExo


def test_reuse_config_dict():
    with open("tests/test_base/configs/example.yaml") as file:
        cfg = yaml.safe_load(file)

    exo = ergoExo()
    exo.setup(cfg)

    exo = ergoExo()
    exo.setup(cfg)
