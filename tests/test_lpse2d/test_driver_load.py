"""Loading a pretrained E0 driver from ``drivers.E0.file``.

``load`` reads a local ``.pkl`` or ``.eqx`` off the filesystem. It used to also accept an
``s3://`` URI and fetch it with boto3; that path is gone, along with adept's cloud SDK dependency.
Callers that keep drivers in object storage download them first and pass the local path.

The remote-URI case is worth a test because the failure is otherwise silent-ish and misleading:
``s3://.../laser.eqx`` matches the ``"eqx" in filename`` dispatch, so without an explicit check it
reaches ``open()`` and surfaces as a missing local file rather than as "this is a URI".
"""

import pickle
import re

import numpy as np
import pytest
import yaml

from adept._lpse2d.modules import driver
from adept._lpse2d.modules.base import BaseLPSE2D

CONFIG_PATH = "tests/test_lpse2d/configs/tpd.yaml"


@pytest.fixture
def cfg():
    """A config carried far enough through the module lifecycle for a driver to be constructible."""
    with open(CONFIG_PATH) as fi:
        cfg = yaml.safe_load(fi)
    cfg["drivers"]["E0"]["params"] = {"phases": {"seed": 0}}
    module = BaseLPSE2D(cfg)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    return module.cfg


def test_load_local_pkl(cfg, tmp_path):
    """A local .pkl overrides the freshly constructed driver's intensities and phases."""
    DriverModule = driver.choose_driver("uniform")
    intensities = np.asarray(DriverModule(cfg).intensities) * 3.0
    phases = np.asarray(DriverModule(cfg).phases) + 0.5

    path = tmp_path / "used_driver.pkl"
    with open(path, "wb") as f:
        pickle.dump({"E0": {"intensities": intensities, "phases": phases}}, f)

    cfg["drivers"]["E0"]["file"] = str(path)
    loaded = driver.load(cfg, DriverModule)

    np.testing.assert_allclose(np.asarray(loaded.intensities), intensities)
    np.testing.assert_allclose(np.asarray(loaded.phases), phases)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://public-ergodic-continuum/181417/abc123/artifacts/laser.eqx",
        "s3://bucket/run/used_driver.pkl",
        "gs://bucket/laser.eqx",
        "https://example.invalid/laser.eqx",
    ],
)
def test_remote_uri_is_rejected(cfg, uri):
    """A URI must fail as a URI, not as a missing file -- see the module docstring."""
    cfg["drivers"]["E0"]["file"] = uri
    with pytest.raises(ValueError, match="must be a local path"):
        driver.load(cfg, driver.choose_driver("uniform"))


def test_unsupported_suffix_still_raises(cfg, tmp_path):
    """The pre-existing suffix guard is unchanged by the remote-URI check in front of it."""
    cfg["drivers"]["E0"]["file"] = str(tmp_path / "laser.bin")
    with pytest.raises(NotImplementedError, match=re.escape("Must be .pkl or .eqx")):
        driver.load(cfg, driver.choose_driver("uniform"))
