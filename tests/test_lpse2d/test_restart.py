"""Checkpoint / restart (LPSE --restart): a run split in two reproduces the unbroken run."""

from copy import deepcopy

import numpy as np
import yaml


def _cfg(tmax_fs, checkpoint=None, restart=None, run="restart-test"):
    with open("tests/test_lpse2d/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {"tmax": f"{tmax_fs}fs", "dt": "1fs", "xmax": "8um", "dx": "0.1um", "ymax": "0.2um", "ymin": "-0.2um"}
    )
    cfg["save"]["fields"]["t"].update({"tmin": "0fs", "tmax": f"{tmax_fs}fs", "dt": "5fs"})
    cfg["terms"]["epw"]["source"]["noise"] = True
    cfg["terms"]["epw"]["source"]["noise_seed"] = 7
    cfg["mlflow"]["run"] = run
    if checkpoint is not None:
        cfg["save"]["checkpoint"] = checkpoint
    if restart is not None:
        cfg["restart"] = {"file": restart}
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    _, ppo, _ = exo(modules)
    return ppo


def test_split_run_matches_unbroken_run(tmp_path):
    ckpt = str(tmp_path / "ckpt.npz")
    _run(_cfg(10, checkpoint=ckpt, run="restart-first"))
    saved = np.load(ckpt)
    t_ckpt = float(saved["t"])
    assert 0.0099 < t_ckpt < 0.0121 and "epw" in saved.files  # the solver's end time (tmax rounded to steps)

    resumed = _run(_cfg(20, restart=ckpt, run="restart-second"))
    unbroken = _run(_cfg(20, run="restart-unbroken"))

    t_res = resumed["series"]["t (ps)"].values
    assert t_res[0] >= t_ckpt - 1e-12  # the series starts at the restart time
    for name in ("phi", "ex"):
        a = resumed["x"][name].isel({"t (ps)": -1}).values
        b = unbroken["x"][name].isel({"t (ps)": -1}).values
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * np.max(np.abs(b)))
    e_res = float(resumed["series"]["epw_energy"].values[-1])
    e_unb = float(unbroken["series"]["epw_energy"].values[-1])
    assert e_unb > 0.0
    np.testing.assert_allclose(e_res, e_unb, rtol=1e-9)
