"""ω-k plot layout (full box vs burst) and the aliasing flag.

A coarse full-box dump cadence (e.g. every 120 steps of 0.178/ω0, Nyquist 0.147 ω0)
cannot resolve the drive; the canned ω-k figure must say so, and full-box and burst
series must land in ``omega_k/<comp>/{full_box,burst_<tag>}.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from adept.osiris import plots as oplt


def _series(dt: float, n: int = 64) -> xr.DataArray:
    t = np.arange(n) * dt
    x = np.linspace(0.0, 20.0, 64, endpoint=False)
    data = np.sin(1.0 * t[:, None] - 1.0 * x[None, :]).astype(np.float32)
    return xr.DataArray(data, dims=("t", "x1"), coords={"t": t, "x1": x})


def test_omega_k_relpath_and_burst_tags() -> None:
    assert oplt.omega_k_relpath("e1") == "omega_k/e1/full_box"
    assert oplt.omega_k_relpath("e2-wk1") == "omega_k/e2/burst_wk1"
    assert oplt.omega_k_relpath("e2-line-x2-0001") == "omega_k/e2/line-x2-0001"
    assert oplt.omega_k_relpath("e1-savg") == "omega_k/e1/savg"
    assert oplt.is_burst_tag("wk5") and not oplt.is_burst_tag("")
    assert not oplt.is_burst_tag("line-x1-0003") and not oplt.is_burst_tag("savg")


def test_drive_omega_code_and_kwargs() -> None:
    cfg = {"units": {"derived": {"w_laser": "5.3665e15 rad/s", "wp0": "5.3668e15 rad/s"}}}
    w = oplt.drive_omega_code(cfg)
    assert w is not None and abs(w - 5.3665 / 5.3668) < 1e-12
    assert oplt.drive_omega_code({"units": {"derived": {"wp0": 1.0}}}) is None
    assert oplt.drive_omega_code(None) is None
    kw = oplt.canned_plot_kwargs({"v_th": 0.1}, cfg)
    assert abs(kw["omega_resolve"] - w) < 1e-12
    assert "omega_resolve" not in oplt.canned_plot_kwargs({"v_th": 0.1})


def test_figure_flags_an_aliasing_cadence() -> None:
    coarse = oplt.plot_omega_k_figure(_series(21.36), omega_resolve=1.0)
    title = coarse._suptitle.get_text()
    assert "ALIASED" in title and "Nyquist = 0.1471" in title
    fine = oplt.plot_omega_k_figure(_series(0.534), omega_resolve=1.0)
    assert "ALIASED" not in fine._suptitle.get_text()
    # without a drive frequency only the Nyquist is stated
    plain = oplt.plot_omega_k_figure(_series(21.36))
    assert "ALIASED" not in plain._suptitle.get_text()
    plt.close("all")


def _write_dump(path: Path, name: str, data: np.ndarray, t: float, it: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["TIME"] = np.array([t])
        f.attrs["ITER"] = np.array([it], dtype="int32")
        f.attrs["NAME"] = np.array([name.encode()], dtype="S256")
        f.attrs["LABEL"] = np.array([name.encode()], dtype="S256")
        f.attrs["UNITS"] = np.array([b"a.u."], dtype="S256")
        f.attrs["TIME UNITS"] = np.array([rb"1 / \omega_p"], dtype="S256")
        ax = f.create_group("AXIS")
        d = ax.create_dataset("AXIS1", data=np.array([0.0, 20.0]))
        d.attrs["NAME"] = np.array([b"x1"], dtype="S256")
        d.attrs["LONG_NAME"] = np.array([b"x_1"], dtype="S256")
        d.attrs["UNITS"] = np.array([rb"c / \omega_p"], dtype="S256")
        sim = f.create_group("SIMULATION")
        sim.attrs["DT"] = np.array([0.05])
        sim.attrs["NDIMS"] = np.array([1], dtype="int32")
        sim.attrs["NX"] = np.array([data.size], dtype="int32")
        sim.attrs["XMIN"] = np.array([0.0])
        sim.attrs["XMAX"] = np.array([20.0])
        f.create_dataset(name, data=data.astype("float32"))


def test_save_canned_plots_full_box_and_burst_paths(tmp_path: Path) -> None:
    run = tmp_path / "run"
    x = np.linspace(0.0, 20.0, 64, endpoint=False)
    for k in range(24):  # full box, coarse
        t = 21.36 * k
        _write_dump(run / "MS/FLD/e2" / f"e2-{k:06d}.h5", "e2", np.sin(t - x), t, 120 * k)
    for k in range(64):  # burst, fine
        t = 1000.0 + 0.534 * k
        _write_dump(run / "MS/FLD/e2-wk1" / f"e2-{k:06d}.h5", "e2", np.sin(t - x), t, 3 * k)
    written = oplt.save_canned_plots(run, tmp_path / "plots", omega_resolve=1.0)
    assert written["omega_k/e2/full_box"] == tmp_path / "plots/omega_k/e2/full_box.png"
    assert written["omega_k/e2/burst_wk1"] == tmp_path / "plots/omega_k/e2/burst_wk1.png"
    assert written["omega_k/e2/full_box"].exists() and written["omega_k/e2/burst_wk1"].exists()
