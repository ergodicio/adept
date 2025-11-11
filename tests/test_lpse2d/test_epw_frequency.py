import jax 

jax.config.update("jax_enable_x64", False)

import mlflow
import numpy as np
import pytest
import yaml
from numpy import testing

from adept import ergoExo
from adept.electrostatic import get_nlfs


def _real_part_():
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)

    exo = ergoExo()
    _ = exo.setup(cfg)
    cfg["mlflow"]["run"] = "epw-test-real"
    rand_k0 = np.random.uniform(5, 25)
    # rand_scalar = np.random.uniform(0.5, 2.0)
    lambda_w = round(float(2 * np.pi / rand_k0), 2)
    k0 = 2 * np.pi / lambda_w
    w0 = (
        1.5
        * k0**2.0
        * exo.adept_module.cfg["units"]["derived"]["vte_sq"]
        / exo.adept_module.cfg["units"]["derived"]["wp0"]
    )
    cfg["drivers"]["E2"]["k0"] = float(k0)
    cfg["drivers"]["E2"]["w0"] = w0
    cfg["grid"]["xmax"] = f"{10 * float(2 * np.pi / k0)}um"
    cfg["grid"]["tmax"] = "6ps"
    cfg["save"]["fields"]["t"]["tmax"] = "6ps"
    cfg["save"]["fields"]["t"]["dt"] = "10fs"

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)

    # get damping rate out of ppo
    # ppo["fields"]["epw"] = ppo["fields"]["epw"].view(np.complex128)
    flds = ppo["x"]
    slc = np.fft.fft(np.real(flds["phi"][:, :, 0]).data, axis=1)[:, 10]
    dt = flds.coords["t (ps)"].data[2] - flds.coords["t (ps)"].data[1]
    amplitude_envelope, instantaneous_frequency_smooth = get_nlfs(slc, dt)

    # select slice 65-85% of the simulation
    mid_slice = slice(
        int(len(instantaneous_frequency_smooth) * 0.65),
        int(len(instantaneous_frequency_smooth) * 0.85),
    )
    actual = np.mean(instantaneous_frequency_smooth[mid_slice])
    from matplotlib import pyplot as plt

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax[0].plot(instantaneous_frequency_smooth[32:-32])
    # np.real(flds["phi"][:, :, 2]).plot(ax=ax[1])
    mlflow.log_metrics({"actual frequency": actual, "desired frequency": w0}, run_id=mlrunid)
    testing.assert_allclose(desired=w0, actual=actual, rtol=0.1)


def _imaginary_part_():
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)

    exo = ergoExo()
    _ = exo.setup(cfg)
    # modify config
    cfg["mlflow"]["run"] = "epw-test-imaginary"
    rand_k0 = np.random.uniform(22, 40)
    lambda_w = round(float(2 * np.pi / rand_k0), 2)
    k0 = 2 * np.pi / lambda_w
    cfg["drivers"]["E2"]["k0"] = float(k0)
    cfg["drivers"]["E2"]["w0"] = (
        1.5
        * k0**2.0
        * exo.adept_module.cfg["units"]["derived"]["vte_sq"]
        / exo.adept_module.cfg["units"]["derived"]["wp0"]
    )
    cfg["grid"]["xmax"] = f"{10 * float(2 * np.pi / k0)}um"
    cfg["terms"]["epw"]["damping"]["landau"] = True

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)

    wp0 = exo.cfg["units"]["derived"]["wp0"]
    vte_sq = exo.cfg["units"]["derived"]["vte_sq"]
    desired = -(
        np.sqrt(np.pi / 8)
        * (1.0 + 1.5 * k0**2.0 * (vte_sq / wp0**2))
        * wp0**4
        * k0**-3.0
        / vte_sq**1.5
        * np.exp(-(1.5 + 0.5 * wp0**2 * k0**-2.0 / vte_sq))
    )

    # get damping rate out of ppo
    flds = ppo["x"]
    amplitude_envelope = np.abs(np.fft.fft(np.real(flds["phi"][:, :, 0]).data, axis=1)[:, 10])
    dt = flds.coords["t (ps)"].data[2] - flds.coords["t (ps)"].data[1]

    # middle 20% of points
    mid_slice = slice(int(len(amplitude_envelope) * 0.75), int(len(amplitude_envelope) * 0.85))
    actual = np.mean((np.gradient(amplitude_envelope) / amplitude_envelope)[mid_slice]) / dt

    from matplotlib import pyplot as plt

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax[0].semilogy(flds.coords["t (ps)"].data[32: -32], amplitude_envelope[32:-32])
    # plot midslice
    # ax[0].semilogy(flds.coords["t (ps)"].data[mid_slice], amplitude_envelope[mid_slice])
    # np.real(flds["phi"][:, :, 2]).plot(ax=ax[1])

    mlflow.log_metrics({"actual damping rate": actual, "desired damping rate": desired}, run_id=mlrunid)
    testing.assert_allclose(desired=desired, actual=actual, rtol=0.35)


@pytest.mark.parametrize("test_func", [_real_part_, _imaginary_part_])
def test_epw_frequency(test_func):
    test_func()


if __name__ == "__main__":
    test_epw_frequency(_real_part_)
    test_epw_frequency(_imaginary_part_)
