from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow
import numpy as np
import pytest

from numpy import testing

# from utils.runner import run
from adept.theory.electrostatic import get_roots_to_electrostatic_dispersion, get_nlfs
from adept import ergoExo


def _real_part_():
    with open("tests/test_lpse2d/configs/epw.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    exo = ergoExo()
    _ = exo.setup(cfg)

    with open("tests/test_lpse2d/configs/epw.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)
    # modify config
    rand_k0 = np.random.uniform(10, 30)
    rand_scalar = np.random.uniform(0.5, 2.0)
    lambda_w = round(float(2 * np.pi / rand_k0), 2)
    k0 = 2 * np.pi / lambda_w
    w0 = (
        1.5
        * k0**2.0
        * exo.adept_module.cfg["units"]["derived"]["vte_sq"]
        / exo.adept_module.cfg["units"]["derived"]["wp0"]
    )
    cfg["drivers"]["E2"]["k0"] = float(k0)
    cfg["drivers"]["E2"]["w0"] = w0 * rand_scalar
    cfg["grid"]["xmax"] = f"{10*float(2 * np.pi / k0)}um"

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)

    # get damping rate out of ppo
    # ppo["fields"]["epw"] = ppo["fields"]["epw"].view(np.complex128)
    flds = ppo["x"]
    slc = np.fft.fft(np.real(flds["phi"][:, :, 0]).data, axis=1)[:, 10]
    dt = flds.coords["t (ps)"].data[2] - flds.coords["t (ps)"].data[1]
    amplitude_envelope, instantaneous_frequency_smooth = get_nlfs(slc, dt)

    actual = np.mean(instantaneous_frequency_smooth[100:160])

    testing.assert_allclose(desired=w0, actual=actual, rtol=0.1)


def _imaginary_part_():
    with open("tests/test_lpse2d/configs/epw.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    exo = ergoExo()
    _ = exo.setup(cfg)

    with open("tests/test_lpse2d/configs/epw.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)
    # modify config
    rand_k0 = np.random.uniform(10, 30)
    lambda_w = round(float(2 * np.pi / rand_k0), 2)
    k0 = 2 * np.pi / lambda_w
    cfg["drivers"]["E2"]["k0"] = float(k0)
    cfg["drivers"]["E2"]["w0"] = (
        1.5
        * k0**2.0
        * exo.adept_module.cfg["units"]["derived"]["vte_sq"]
        / exo.adept_module.cfg["units"]["derived"]["wp0"]
    )
    cfg["grid"]["xmax"] = f"{10*float(2 * np.pi / k0)}um"
    # cfg["grid"]["tmax"] = 200.0
    # cfg["save"]["t"]["nt"] = 512

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)

    # get damping rate out of ppo
    ppo["fields"]["epw"] = ppo["fields"]["epw"].view(np.complex128)

    mlflow.log_metrics({"actual damping rate": actual, "desired damping rate": desired})
    testing.assert_almost_equal(desired=desired, actual=actual, decimal=3)


@pytest.mark.parametrize("test_func", [_real_part_, _imaginary_part_])
def test_epw_frequency(test_func):
    test_func()


if __name__ == "__main__":
    test_epw_frequency(_real_part_, True)
    test_epw_frequency(_real_part_, False)
    test_epw_frequency(_imaginary_part_, False)
