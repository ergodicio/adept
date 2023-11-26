from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml, mlflow
import numpy as np
import pytest

from numpy import testing
from utils.runner import run
from theory.electrostatic import get_roots_to_electrostatic_dispersion, get_nlfs


def _real_part_(kinetic):
    with open("tests/test_lpse2d/configs/epw.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment("test-epw-frequency")

    # modify config
    rand_k0 = np.random.uniform(0.15, 0.22)
    cfg["drivers"]["E2"]["k0"] = float(rand_k0)
    cfg["grid"]["xmax"] = float(2 * np.pi / rand_k0)
    cfg["grid"]["tmax"] = 6000.0
    cfg["save"]["t"]["tmax"] = 5900.0
    cfg["save"]["t"]["nt"] = 4096

    cfg["terms"]["epw"]["kinetic real part"] = kinetic

    with mlflow.start_run(run_name=f'oscillation-{round(cfg["drivers"]["E2"]["k0"], 4)}') as mlflow_run:
        result, datasets = run(cfg)

        kflds, flds = datasets

        if kinetic:
            desired = np.real(get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)) - 1
        else:
            desired = 1.5 * rand_k0**2.0

        slc = np.fft.fft(np.real(flds["phi"][:, :, 0]).data, axis=1)[:, 1]
        dt = flds.coords["t"].data[2] - flds.coords["t"].data[1]
        amplitude_envelope, instantaneous_frequency_smooth = get_nlfs(slc, dt)
        actual = np.mean(instantaneous_frequency_smooth[1024:-1024])
        mlflow.log_metrics({"actual wepw": actual, "desired wepw": desired})

    testing.assert_almost_equal(desired=desired, actual=actual, decimal=2)


def _imaginary_part_(kinetic):
    with open("tests/test_lpse2d/configs/epw.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment("test-epw-frequency")

    # modify config
    rand_k0 = np.random.uniform(0.26, 0.4)
    cfg["drivers"]["E2"]["k0"] = float(rand_k0)
    cfg["grid"]["xmax"] = float(2 * np.pi / rand_k0)
    cfg["grid"]["tmax"] = 200.0
    cfg["save"]["t"]["nt"] = 512

    with mlflow.start_run(run_name=f'damping-{round(cfg["drivers"]["E2"]["k0"], 4)}') as mlflow_run:
        result, datasets = run(cfg)

        kflds, flds = datasets

        desired = np.imag(get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0))
        slc = np.abs(flds["phi"][-128:, 1, 0].data)
        dt = flds.coords["t"].data[2] - flds.coords["t"].data[1]
        actual = np.mean(np.gradient(slc) / dt / slc)
        mlflow.log_metrics({"actual damping rate": actual, "desired damping rate": desired})
    testing.assert_almost_equal(desired=desired, actual=actual, decimal=3)


@pytest.mark.parametrize("test_func, kinetic", [(_real_part_, True), (_real_part_, False), (_imaginary_part_, False)])
def test_epw_frequency(test_func, kinetic):
    test_func(kinetic)


if __name__ == "__main__":
    test_epw_frequency(_real_part_, True)
    test_epw_frequency(_real_part_, False)
    test_epw_frequency(_imaginary_part_, False)
