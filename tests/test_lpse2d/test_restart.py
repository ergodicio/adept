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
    # the per-step noise stream is what the restart must reproduce; the flat model needs no damping
    # (the thermal default refuses an undamped EPW, as LPSE)
    cfg["terms"]["epw"]["source"]["noise_model"] = "flat"
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


def test_kap_histories_are_prefix_stable_as_the_run_grows():
    """Each beam's KAP transition times and phases depend on kap_seed and the beam index only, so a
    longer table only appends (PR #378 review): the phase of every beam at every time inside the
    shorter horizon is identical (exact) between tables drawn for 10 fs and 1 ps."""
    from adept._lpse2d.core.kap import KapPhases

    short = KapPhases(0.2, 5370.0, 3, 0.01, seed=11)  # mean dwell 2 / (0.2 w0) = 1.9 fs
    long = KapPhases(0.2, 5370.0, 3, 1.0, seed=11)
    for beam in range(3):
        for t in np.linspace(0.0, 0.01, 401):
            assert float(short.phase(t, beam)) == float(long.phase(t, beam))
    assert len({float(long.phase(t, 0)) for t in np.linspace(0.0, 0.01, 401)}) > 2  # it does jump


def _kap_cfg(tmax_fs, checkpoint=None, restart=None, run="kap-restart", probe=False):
    """A static TPD pump with KAP bandwidth 0.05 (mean dwell ~7 fs) driving the EPW, flat noise."""
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["density"] = {"basis": "uniform", "val": 0.24}
    cfg["grid"].update(
        {"tmax": f"{tmax_fs}fs", "dt": "1fs", "xmax": "8um", "dx": "0.1um", "ymax": "0.4um", "ymin": "-0.4um"}
    )
    cfg["save"]["fields"]["t"].update({"tmin": "0fs", "tmax": f"{tmax_fs}fs", "dt": "5fs"})
    # the full-grid field save (tpd.yaml's interpolated x / y save grid writes non-finite fields, on
    # ergodicio/main as well -- a separate issue)
    cfg["save"]["fields"].pop("x", None)
    cfg["save"]["fields"].pop("y", None)
    cfg["terms"]["epw"]["source"].update({"noise": True, "noise_seed": 7, "noise_model": "flat", "tpd": True})
    cfg["drivers"]["E0"].update({"kap_bandwidth": 0.05, "kap_seed": 3, "params": {"phases": {"seed": 42}}})
    cfg["mlflow"]["run"] = run
    if probe:
        # default start (the grid's t = 0), every 3 fs: the 10 fs restart falls between samples
        cfg["save"]["light_spectrum"] = [{"field": "E0", "interval": "3fs", "x": ["-1um", "1um"]}]
    if checkpoint is not None:
        cfg["save"]["checkpoint"] = checkpoint
    if restart is not None:
        cfg["restart"] = {"file": restart}
    return cfg


def test_kap_split_run_matches_unbroken_run(tmp_path):
    """A KAP-bandwidth pump resumed from a checkpoint with a larger tmax reproduces the unbroken run
    (PR #378 review; the KAP table used to be redrawn for the new horizon, changing the pump phases).
    The tolerance of the split-run test above, fixed before the run."""
    ckpt = str(tmp_path / "ckpt.npz")
    _run(_kap_cfg(10, checkpoint=ckpt, run="kap-restart-first"))
    resumed = _run(_kap_cfg(20, restart=ckpt, run="kap-restart-second"))
    unbroken = _run(_kap_cfg(20, run="kap-restart-unbroken"))
    for name in ("phi", "ex"):
        a = resumed["x"][name].isel({"t (ps)": -1}).values
        b = unbroken["x"][name].isel({"t (ps)": -1}).values
        assert np.max(np.abs(b)) > 0.0
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * np.max(np.abs(b)))


def test_restart_rebases_the_light_spectrum_probes(tmp_path):
    """A resumed run with save.light_spectrum (a list of probes, here with the default start) starts
    each probe at its first own sample inside the resumed interval (PR #378 review: the probes kept
    their t = 0 start and the resumed solve was rejected). With a 3 fs interval and a 10 fs restart
    the resumed probe samples exactly the unbroken run's times from 12 fs on, with the same values
    (rtol 1e-9, the split-run tolerance, fixed before the run)."""
    ckpt = str(tmp_path / "ckpt.npz")
    _run(_kap_cfg(10, checkpoint=ckpt, run="probe-restart-first", probe=True))
    resumed = _run(_kap_cfg(20, restart=ckpt, run="probe-restart-second", probe=True))["light_spectrum"][0]
    unbroken = _run(_kap_cfg(20, run="probe-restart-unbroken", probe=True))["light_spectrum"][0]
    t_res = resumed["t (ps)"].values
    t_unb = unbroken["t (ps)"].values
    np.testing.assert_allclose(t_res[0], 0.012, atol=1e-12)
    later = t_unb >= t_res[0] - 1e-12
    np.testing.assert_allclose(t_res, t_unb[later], rtol=0, atol=1e-12)
    a, b = resumed["e_y"].values, unbroken["e_y"].values[later]
    assert np.max(np.abs(b)) > 0.0
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * np.max(np.abs(b)))
