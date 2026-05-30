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


def _write_raw_dump(
    path: Path, quantities: dict[str, np.ndarray], t: float, it: int
) -> None:
    """Write one OSIRIS-style RAW (particle) HDF5 dump.

    ``quantities`` maps a per-particle quantity name (``"p1"``, ``"x1"``, ...)
    to its 1-D array. RAW dumps have NO ``AXIS`` group; each quantity is a
    top-level 1-D dataset, all the same length, plus TIME / ITER root attrs and
    a SIMULATION group.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["TIME"] = np.array([t])
        f.attrs["ITER"] = np.array([it], dtype="int32")
        f.attrs["NAME"] = np.array([b"RAW"], dtype="S256")
        f.attrs["LABEL"] = np.array([b"RAW"], dtype="S256")
        f.attrs["TIME UNITS"] = np.array([rb"1 / \omega_p"], dtype="S256")
        f.attrs["TYPE"] = np.array([b"particles"], dtype="S16")
        for qn, arr in quantities.items():
            d = f.create_dataset(qn, data=arr.astype("float32"))
            d.attrs["UNITS"] = np.array([b"a.u."], dtype="S256")
            d.attrs["LONG_NAME"] = np.array([qn.encode()], dtype="S256")
        sim = f.create_group("SIMULATION")
        sim.attrs["DT"] = np.array([0.05])
        sim.attrs["NDIMS"] = np.array([1], dtype="int32")


def test_load_raw_h5_returns_particle_dataset(tmp_path: Path) -> None:
    p = tmp_path / "MS" / "RAW" / "species_1" / "RAW-species_1-000030.h5"
    npart = 7
    quants = {
        q: np.arange(npart, dtype="float64") for q in ("ene", "p1", "p2", "p3", "q", "x1")
    }
    _write_raw_dump(p, quants, t=1.5, it=30)

    ds = oio.load_raw_h5(p)
    assert isinstance(ds, xr.Dataset)
    assert set(ds.dims) == {"pidx"}
    assert ds.sizes["pidx"] == npart
    assert set(ds.data_vars) == set(quants)
    assert ds.attrs["iter"] == 30
    assert ds.attrs["time"] == 1.5
    assert ds["p1"].attrs["units"] == "a.u."


def test_load_grid_h5_still_works_for_grid_dumps(tmp_path: Path) -> None:
    # A normal grid dump (single dataset + AXIS) must still load as a DataArray.
    p = tmp_path / "e1-000000.h5"
    _write_dump(
        p,
        "e1",
        np.arange(8, dtype="float64"),
        t=0.0,
        it=0,
        axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)],
    )
    da = oio.load_grid_h5(p)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("x1",)
    assert da.sizes["x1"] == 8


def test_load_raw_series_handles_variable_particle_counts(tmp_path: Path) -> None:
    raw = tmp_path / "MS" / "RAW" / "species_1"
    # Two dumps with DIFFERENT particle counts (raw_fraction sampling).
    _write_raw_dump(
        raw / "RAW-species_1-000000.h5",
        {q: np.zeros(5) for q in ("p1", "x1")},
        t=0.0,
        it=0,
    )
    _write_raw_dump(
        raw / "RAW-species_1-000100.h5",
        {q: np.ones(9) for q in ("p1", "x1")},
        t=5.0,
        it=100,
    )

    ds = oio.load_raw_series(raw)
    assert ds.sizes["pidx"] == 14  # 5 + 9 — no equal-shape assumption
    assert sorted(set(ds["iter"].values.tolist())) == [0, 100]
    assert (ds["t"].values[:5] == 0.0).all()
    assert (ds["t"].values[5:] == 5.0).all()


def test_save_run_datasets_routes_raw_to_netcdf(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, n_steps=2, nx=8)  # FLD/e1 grid diagnostic
    raw = run_dir / "MS" / "RAW" / "species_1"
    _write_raw_dump(
        raw / "RAW-species_1-000000.h5",
        {q: np.arange(4, dtype="float64") for q in ("ene", "p1", "x1", "q")},
        t=0.0,
        it=0,
    )
    out = tmp_path / "binary"
    oio.save_run_datasets(run_dir, out)

    # Both the grid and the RAW diagnostic produced netCDF — no crash on RAW.
    assert (out / "FLD" / "e1.nc").exists()
    assert (out / "RAW" / "species_1.nc").exists()
    ds = xr.open_dataset(out / "RAW" / "species_1.nc", engine="h5netcdf")
    try:
        assert "pidx" in ds.dims
        assert "p1" in ds.data_vars
    finally:
        ds.close()


def test_save_run_datasets_skips_unloadable_diagnostic(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, n_steps=2, nx=8)  # good FLD/e1
    # A corrupt diagnostic: a .h5 file that isn't valid HDF5.
    bad = run_dir / "MS" / "FLD" / "bad"
    bad.mkdir(parents=True)
    (bad / "bad-000000.h5").write_bytes(b"not an hdf5 file")

    out = tmp_path / "binary"
    written = oio.save_run_datasets(run_dir, out)  # must not raise

    assert (out / "FLD" / "e1.nc").exists()  # good diagnostic still produced
    assert not (out / "FLD" / "bad.nc").exists()
    assert (out / "FLD" / "e1.nc") in written
