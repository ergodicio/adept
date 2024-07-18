import yaml, mlflow, os
from adept.lpse2d.helpers import calc_threshold_intensity
from adept import ergoExo
import numpy as np


def run_once(L, Te, I0):

    with open("tests/test_lpse2d/configs/tpd.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["units"]["laser intensity"] = f"{I0}e14 W/cm^2"
    cfg["density"]["gradient scale length"] = f"{L}um"
    cfg["units"]["reference electron temperature"] = f"{Te}keV"
    cfg["mlflow"]["run"] = f"{I0}W/cm^2"  # I0
    cfg["mlflow"]["experiment"] = f"threshold-L={L}um, Te={Te}keV"

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, ppo, mlrunid = exo(modules)
    es = ppo["metrics"]["log10_total_e_sq"]
    return es


def test_threshold():
    if "CPU_ONLY" in os.environ:
        pass
    else:
        L = round(np.random.uniform(200, 1000))
        Te = round(np.random.uniform(1, 4), 2)
        c = 3e8
        lam0 = 351e-9
        w0 = 2 * np.pi * c / lam0 / 1e12  # 1/ps
        It = calc_threshold_intensity(Te, L, w0)
        I_scan = np.linspace(0.2, 2, 10) * It
        I_scan = np.round(I_scan, 2)
        ess = []

        for I0 in I_scan:
            es = run_once(L, Te, I0)
            ess.append(es)

        actual = I_scan[np.amax(np.gradient(np.gradient(ess)))]
        np.testing.assert_allclose(actual, desired=It, rtol=0.25)  # it is 25% because of the resolution of the scan
