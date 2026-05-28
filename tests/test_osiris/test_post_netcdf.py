"""post.collect converts OSIRIS diagnostics to netCDF time-series.

These tests synthesize a tiny OSIRIS-shaped run so they're self-contained
(no dependency on a real run on disk).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from adept.osiris import io as oio
from adept.osiris import post as opost


def _write_dump(path: Path, name: str, data: np.ndarray, t: float, it: int, axes) -> None:
    """Write one OSIRIS-style HDF5 dump.

    ``axes`` is a list of ``(name, long_name, units, min, max)`` in OSIRIS
    (Fortran) order, i.e. the last entry is the fastest-varying numpy axis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["TIME"] = np.array([t])
        f.attrs["ITER"] = np.array([it], dtype="int32")
        f.attrs["NAME"] = np.array([name.encode()], dtype="S256")
        f.attrs["LABEL"] = np.array([name.encode()], dtype="S256")
        f.attrs["UNITS"] = np.array([b"a.u."], dtype="S256")
        f.attrs["TIME UNITS"] = np.array([rb"1 / \omega_p"], dtype="S256")
        f.attrs["TYPE"] = np.array([b"grid"], dtype="S4")
        ax = f.create_group("AXIS")
        for i, (an, ln, un, lo, hi) in enumerate(axes, start=1):
            d = ax.create_dataset(f"AXIS{i}", data=np.array([lo, hi], dtype="float64"))
            d.attrs["NAME"] = np.array([an.encode()], dtype="S256")
            d.attrs["LONG_NAME"] = np.array([ln.encode()], dtype="S256")
            d.attrs["UNITS"] = np.array([un.encode()], dtype="S256")
            d.attrs["TYPE"] = np.array([b"linear"], dtype="S6")
        sim = f.create_group("SIMULATION")
        sim.attrs["DT"] = np.array([0.05])
        sim.attrs["NDIMS"] = np.array([1], dtype="int32")
        sim.attrs["NX"] = np.array([data.shape[-1]], dtype="int32")
        sim.attrs["XMIN"] = np.array([axes[-1][3]])
        sim.attrs["XMAX"] = np.array([axes[-1][4]])
        f.create_dataset(name, data=data.astype("float32"))


def _make_run(root: Path, n_steps: int = 4, nx: int = 8) -> Path:
    """Build a run dir with an FLD/e1 field diagnostic over ``n_steps`` dumps."""
    run_dir = root / "run"
    e1 = run_dir / "MS" / "FLD" / "e1"
    rng = np.random.default_rng(0)
    for k in range(n_steps):
        it = k * 10
        _write_dump(
            e1 / f"e1-{it:06d}.h5",
            "e1",
            rng.standard_normal(nx),
            t=k * 0.5,
            it=it,
            axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)],
        )
    (run_dir / "os-stdin").write_text("node_conf\n{\n}\n")
    (run_dir / "stdout.log").write_text("ok\n")
    (run_dir / "stderr.log").write_text("")
    return run_dir


def test_save_run_datasets_writes_full_timeseries(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, n_steps=4, nx=8)
    out = tmp_path / "binary"
    written = oio.save_run_datasets(run_dir, out)

    assert written == [out / "FLD" / "e1.nc"]
    ds = xr.open_dataset(out / "FLD" / "e1.nc", engine="h5netcdf")
    try:
        assert "e1" in ds.data_vars
        assert ds.sizes == {"t": 4, "x1": 8}  # every time slice kept
        assert list(ds["t"].values) == [0.0, 0.5, 1.0, 1.5]
        assert list(ds["iter"].values) == [0, 10, 20, 30]
        # axis metadata moved onto the coordinate (netCDF-serializable)
        assert ds["x1"].attrs["units"] == r"c / \omega_p"
    finally:
        ds.close()


def test_collect_emits_netcdf_not_h5(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, n_steps=3, nx=8)
    run_output = {"solver result": {"run_dir": str(run_dir), "wall_time": 1.0, "exit_code": 0}}
    cfg = {"osiris": {"deck": str(run_dir / "os-stdin")}, "output": {}}

    td = tmp_path / "td"
    td.mkdir()
    result = opost.collect(run_output, cfg, str(td))

    # No raw OSIRIS dumps uploaded; the diagnostic is a netCDF time-series.
    assert not list(td.rglob("*.h5"))
    assert (td / "binary" / "FLD" / "e1.nc").exists()
    # Deck + logs still copied; metrics still produced.
    assert (td / "os-stdin").exists()
    assert (td / "stdout.log").exists()
    assert result["metrics"]["final_iter"] == 20.0


def test_collect_respects_diagnostics_whitelist(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, n_steps=2, nx=8)
    # Add a second diagnostic that should be filtered out.
    _write_dump(
        run_dir / "MS" / "FLD" / "e2" / "e2-000000.h5",
        "e2",
        np.zeros(8),
        t=0.0,
        it=0,
        axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)],
    )
    run_output = {"solver result": {"run_dir": str(run_dir), "wall_time": 1.0, "exit_code": 0}}
    cfg = {"osiris": {"deck": str(run_dir / "os-stdin")}, "output": {"diagnostics_to_log": ["e1"]}}

    td = tmp_path / "td"
    td.mkdir()
    opost.collect(run_output, cfg, str(td))

    assert (td / "binary" / "FLD" / "e1.nc").exists()
    assert not (td / "binary" / "FLD" / "e2.nc").exists()
