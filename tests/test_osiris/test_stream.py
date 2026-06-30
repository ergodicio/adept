"""Incremental / concurrent HDF5 -> NetCDF conversion (adept.osiris.stream).

These tests synthesize a tiny OSIRIS-shaped run (no real run on disk) and check
that the streaming converter — both the at-job-end Stage A path and the
concurrent Stage B watcher — produces a NetCDF that is equivalent, for every
downstream consumer, to the in-memory batch path in ``io.save_run_datasets``.
"""

from __future__ import annotations

import stat
import time
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from adept.osiris import io as oio
from adept.osiris import post as opost
from adept.osiris import runner as orunner
from adept.osiris import stream as ostream


def _write_dump(path: Path, name: str, data: np.ndarray, t: float, it: int, axes) -> None:
    """Write one OSIRIS-style HDF5 grid dump.

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
        ax = f.create_group("AXIS")
        for i, (an, ln, un, lo, hi) in enumerate(axes, start=1):
            d = ax.create_dataset(f"AXIS{i}", data=np.array([lo, hi], dtype="float64"))
            d.attrs["NAME"] = np.array([an.encode()], dtype="S256")
            d.attrs["LONG_NAME"] = np.array([ln.encode()], dtype="S256")
            d.attrs["UNITS"] = np.array([un.encode()], dtype="S256")
        sim = f.create_group("SIMULATION")
        sim.attrs["DT"] = np.array([0.05])
        sim.attrs["NDIMS"] = np.array([len(axes)], dtype="int32")
        sim.attrs["NX"] = np.array(list(reversed(data.shape)), dtype="int32")
        sim.attrs["XMIN"] = np.array([a[3] for a in axes])
        sim.attrs["XMAX"] = np.array([a[4] for a in axes])
        f.create_dataset(name, data=data.astype("float32"))


def _write_field(run_dir: Path, comp: str, n_steps: int, nx: int, seed: int = 0) -> Path:
    """Write an ``MS/FLD/<comp>`` 1-D field diagnostic over ``n_steps`` dumps."""
    rng = np.random.default_rng(seed)
    d = run_dir / "MS" / "FLD" / comp
    for k in range(n_steps):
        _write_dump(
            d / f"{comp}-{k * 10:06d}.h5",
            comp,
            rng.standard_normal(nx),
            t=k * 0.5,
            it=k * 10,
            axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)],
        )
    return d


def _write_phasespace_autoscaled(run_dir: Path, n_steps: int, nx: int, npmom: int) -> Path:
    """Phase space whose momentum bounds grow each dump (if_ps_p_auto)."""
    rng = np.random.default_rng(7)
    d = run_dir / "MS" / "PHA" / "p1x1" / "electrons"
    for k in range(n_steps):
        p_hi = 1.0 + 0.5 * k
        _write_dump(
            d / f"p1x1-electrons-{k * 10:06d}.h5",
            "p1x1",
            rng.standard_normal((npmom, nx)),
            t=k * 0.5,
            it=k * 10,
            axes=[
                ("x1", "x_1", r"c / \omega_p", 0.0, 10.0),
                ("p1", "p_1", r"m_e c", -p_hi, p_hi),
            ],
        )
    return d


def _assert_series_equiv(got: xr.DataArray, ref: xr.DataArray) -> None:
    """``got`` (streamed) is interchangeable with ``ref`` (raw load_series)."""
    assert got.name == ref.name
    assert got.dims == ref.dims
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got.values, ref.values)  # float32 -> float32, exact
    np.testing.assert_allclose(got["t"].values, ref["t"].values)
    assert list(got["iter"].values) == list(ref["iter"].values)
    for dim in ref.dims:
        if dim == "t":
            continue
        np.testing.assert_allclose(oio.physical_axis(got, dim), oio.physical_axis(ref, dim))
    assert got.attrs.get("axis_units") == ref.attrs.get("axis_units")
    assert got.attrs.get("autoscaled_dims") == ref.attrs.get("autoscaled_dims")


# --- Stage A: streaming finalize -----------------------------------------


def test_streaming_grid_matches_load_series(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_field(run_dir, "e1", n_steps=5, nx=8)

    dest = tmp_path / "stream" / "e1.nc"
    ostream.convert_diagnostic_streaming(diag, dest)

    streamed = oio.load_series_nc(dest)
    raw = oio.load_series(diag)
    _assert_series_equiv(streamed, raw)
    # A fixed spatial axis is not autoscaled even though the streamer always
    # records its (constant) bounds.
    assert "autoscaled_dims" not in streamed.attrs


def test_streaming_matches_batch_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_field(run_dir, "e1", n_steps=6, nx=12, seed=3)

    batch = oio.save_run_datasets(run_dir, tmp_path / "batch")[0]
    stream = tmp_path / "stream" / "e1.nc"
    ostream.convert_diagnostic_streaming(diag, stream)

    a = oio.load_series_nc(stream)
    b = oio.load_series_nc(batch)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_allclose(a["t"].values, b["t"].values)
    assert list(a["iter"].values) == list(b["iter"].values)
    assert a.attrs.get("long_name") == b.attrs.get("long_name")
    assert a.attrs.get("units") == b.attrs.get("units")


def test_streaming_preserves_autoscale_bounds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_phasespace_autoscaled(run_dir, n_steps=4, nx=8, npmom=6)

    dest = tmp_path / "stream" / "p1x1.nc"
    ostream.convert_diagnostic_streaming(diag, dest)

    streamed = oio.load_series_nc(dest)
    raw = oio.load_series(diag)
    _assert_series_equiv(streamed, raw)
    assert streamed.attrs.get("autoscaled_dims") == ["p1"]
    np.testing.assert_allclose(
        oio.physical_axis(streamed, "p1", it=-1),
        np.linspace(-2.5, 2.5, streamed.sizes["p1"]),
    )


def test_streaming_compresses_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_field(run_dir, "e1", n_steps=8, nx=64)
    dest = tmp_path / "stream" / "e1.nc"
    ostream.convert_diagnostic_streaming(diag, dest)
    enc = xr.open_dataset(dest, engine="h5netcdf")["e1"].encoding
    assert enc.get("zlib") or (enc.get("compression") == "gzip")


# --- restart / resume -----------------------------------------------------


def test_writer_resumes_after_reopen(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_field(run_dir, "e1", n_steps=6, nx=8)
    dumps = oio._sort_dumps(diag)
    dest = tmp_path / "stream" / "e1.nc"

    # First "segment": write the first 3 dumps, then close (simulates a crash
    # after a checkpoint).
    template = oio.load_grid_h5(dumps[0])
    w = ostream.StreamWriter(dest, template, source_dir=diag)
    w.append(template)
    for p in dumps[1:3]:
        w.append(oio.load_grid_h5(p))
    assert w.n_written == 3
    w.close()

    # Restart: reopen and continue from where it left off.
    w2 = ostream.StreamWriter(dest, oio.load_grid_h5(dumps[0]), source_dir=diag)
    assert w2.n_written == 3  # picked up the on-disk slots
    for p in dumps[3:]:
        w2.append(oio.load_grid_h5(p))
    w2.close()

    _assert_series_equiv(oio.load_series_nc(dest), oio.load_series(diag))


# --- Stage B: concurrent watcher -----------------------------------------


def test_watcher_holds_back_in_progress_dump(tmp_path: Path) -> None:
    """A non-final scan must not read the highest-numbered (still-writing) dump."""
    run_dir = tmp_path / "run"
    _write_field(run_dir, "e1", n_steps=5, nx=8)

    conv = ostream.StreamConverter(run_dir, tmp_path / "binary", poll_s=0.01)
    conv._scan_once(final=False)
    # 5 dumps present -> 4 are "safe", the newest is held back.
    ds = xr.open_dataset(tmp_path / "binary" / "FLD" / "e1.nc", engine="h5netcdf")
    try:
        assert ds.sizes["t"] == 4
    finally:
        ds.close()

    completed = conv.finalize()  # picks up the last dump and closes
    assert "FLD/e1" in completed
    _assert_series_equiv(
        oio.load_series_nc(tmp_path / "binary" / "FLD" / "e1.nc"),
        oio.load_series(run_dir / "MS" / "FLD" / "e1"),
    )


def test_watcher_thread_streams_dumps_as_they_appear(tmp_path: Path) -> None:
    """End-to-end Stage B: dumps written over time while the thread polls."""
    run_dir = tmp_path / "run"
    diag = run_dir / "MS" / "FLD" / "e1"
    rng = np.random.default_rng(0)

    conv = ostream.StreamConverter(run_dir, tmp_path / "binary", poll_s=0.02)
    conv.start()
    n_steps = 6
    for k in range(n_steps):
        _write_dump(
            diag / f"e1-{k * 10:06d}.h5",
            "e1",
            rng.standard_normal(8),
            t=k * 0.5,
            it=k * 10,
            axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)],
        )
        time.sleep(0.03)

    completed = conv.finalize()
    assert "FLD/e1" in completed
    streamed = oio.load_series_nc(tmp_path / "binary" / "FLD" / "e1.nc")
    assert streamed.sizes["t"] == n_steps
    _assert_series_equiv(streamed, oio.load_series(diag))


def test_watcher_skips_raw_diagnostics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_field(run_dir, "e1", n_steps=3, nx=8)
    # A RAW diagnostic: no AXIS group, several per-particle datasets.
    raw = run_dir / "MS" / "RAW" / "species_1"
    raw.mkdir(parents=True)
    with h5py.File(raw / "RAW-species_1-000000.h5", "w") as f:
        f.attrs["TIME"] = np.array([0.0])
        f.attrs["ITER"] = np.array([0], dtype="int32")
        for q in ("p1", "x1"):
            f.create_dataset(q, data=np.arange(4, dtype="float32"))
        f.create_group("SIMULATION").attrs["NDIMS"] = np.array([1], dtype="int32")

    conv = ostream.StreamConverter(run_dir, tmp_path / "binary", poll_s=0.01)
    completed = conv.finalize()
    assert "FLD/e1" in completed
    assert not any("RAW" in c for c in completed)
    assert not (tmp_path / "binary" / "RAW" / "species_1.nc").exists()


# --- integration with save_run_datasets / post.collect -------------------


def test_save_run_datasets_reuses_streamed_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_field(run_dir, "e1", n_steps=4, nx=8)

    # Pretend the watcher already produced this during the run, and tag it so we
    # can tell a reuse (copy) from a rebuild.
    streamed_dir = tmp_path / "run" / "binary"
    ostream.convert_diagnostic_streaming(diag, streamed_dir / "FLD" / "e1.nc")
    with xr.open_dataset(streamed_dir / "FLD" / "e1.nc", engine="h5netcdf") as ds:
        ds = ds.load()
    ds.attrs["_marker"] = "from-watcher"
    ds.to_netcdf(streamed_dir / "FLD" / "e1.nc", engine="h5netcdf")

    out = tmp_path / "td" / "binary"
    written = oio.save_run_datasets(run_dir, out, stream=True, streamed_dir=streamed_dir)
    assert (out / "FLD" / "e1.nc") in written
    with xr.open_dataset(out / "FLD" / "e1.nc", engine="h5netcdf") as ds2:
        assert ds2.attrs.get("_marker") == "from-watcher"  # copied, not rebuilt


def test_save_run_datasets_stream_builds_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_field(run_dir, "e1", n_steps=5, nx=8)
    # stream=True but no streamed_dir -> build via the streaming writer.
    out = tmp_path / "binary"
    oio.save_run_datasets(run_dir, out, stream=True)
    _assert_series_equiv(oio.load_series_nc(out / "FLD" / "e1.nc"), oio.load_series(diag))


def test_run_osiris_streams_concurrently(tmp_path: Path) -> None:
    """run_osiris spawns the watcher alongside the (fake) OSIRIS subprocess.

    The fake binary is a shell script that drops pre-staged dumps into
    ``MS/FLD/e1`` one at a time with a pause between — exactly the cadence the
    watcher polls. No Python/h5py is needed in the subprocess.
    """
    n_steps = 5
    stage = tmp_path / "stage"
    stage.mkdir()
    rng = np.random.default_rng(1)
    ref = {}
    for k in range(n_steps):
        fn = f"e1-{k * 10:06d}.h5"
        data = rng.standard_normal(8)
        ref[fn] = data
        _write_dump(stage / fn, "e1", data, t=k * 0.5, it=k * 10,
                    axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)])

    lines = ["#!/bin/sh", "set -e", "mkdir -p MS/FLD/e1"]
    for k in range(n_steps):
        fn = f"e1-{k * 10:06d}.h5"
        lines.append(f"cp '{stage / fn}' MS/FLD/e1/{fn}")
        lines.append("sleep 0.1")
    fake = tmp_path / "fake-osiris.sh"
    fake.write_text("\n".join(lines) + "\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)

    result = orunner.run_osiris(
        "node_conf\n{\n}\n",
        binary=fake,
        mpi_ranks=1,
        run_root=tmp_path / "checkpoints",
        stream_convert=True,
        stream_poll_s=0.05,
    )

    assert result["exit_code"] == 0
    assert result["streamed_diagnostics"] == ["FLD/e1"]
    nc = Path(result["binary_dir"]) / "FLD" / "e1.nc"
    assert nc.exists()
    streamed = oio.load_series_nc(nc)
    assert streamed.sizes["t"] == n_steps
    _assert_series_equiv(streamed, oio.load_series(Path(result["run_dir"]) / "MS" / "FLD" / "e1"))


def test_collect_uses_streamed_binary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag = _write_field(run_dir, "e1", n_steps=3, nx=8)
    (run_dir / "os-stdin").write_text("node_conf\n{\n}\n")
    (run_dir / "stdout.log").write_text("ok\n")
    (run_dir / "stderr.log").write_text("")
    # Simulate runner having streamed into run_dir/binary during the run.
    binary_dir = run_dir / "binary"
    ostream.convert_diagnostic_streaming(diag, binary_dir / "FLD" / "e1.nc")

    run_output = {
        "solver result": {
            "run_dir": str(run_dir),
            "wall_time": 1.0,
            "exit_code": 0,
            "binary_dir": str(binary_dir),
            "streamed_diagnostics": ["FLD/e1"],
        }
    }
    cfg = {"osiris": {"deck": str(run_dir / "os-stdin"), "stream_convert": True}, "output": {}}

    td = tmp_path / "td"
    td.mkdir()
    opost.collect(run_output, cfg, str(td))

    assert not list(td.rglob("*.h5"))
    assert (td / "binary" / "FLD" / "e1.nc").exists()
    _assert_series_equiv(oio.load_series_nc(td / "binary" / "FLD" / "e1.nc"), oio.load_series(diag))
