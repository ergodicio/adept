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


def _write_raw_dump(path: Path, quantities: dict[str, np.ndarray], t: float, it: int) -> None:
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
    quants = {q: np.arange(npart, dtype="float64") for q in ("ene", "p1", "p2", "p3", "q", "x1")}
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


# --- regenerating plots from the saved NetCDFs (no rerun, no raw MS/ tree) ---


def _write_field(run_dir: Path, comp: str, n_steps: int, nx: int, seed: int) -> None:
    """Write an ``MS/FLD/<comp>`` 1-D field diagnostic over ``n_steps`` dumps."""
    rng = np.random.default_rng(seed)
    d = run_dir / "MS" / "FLD" / comp
    for k in range(n_steps):
        it = k * 10
        _write_dump(
            d / f"{comp}-{it:06d}.h5",
            comp,
            rng.standard_normal(nx),
            t=k * 0.5,
            it=it,
            axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)],
        )


def _write_phasespace(run_dir: Path, n_steps: int, nx: int, npmom: int) -> None:
    """Write an ``MS/PHA/p1x1/electrons`` 2-D phase-space series."""
    rng = np.random.default_rng(7)
    d = run_dir / "MS" / "PHA" / "p1x1" / "electrons"
    for k in range(n_steps):
        it = k * 10
        # data shape (p1, x1): the AXIS list is OSIRIS order (AXIS1 = x1).
        _write_dump(
            d / f"p1x1-electrons-{it:06d}.h5",
            "p1x1",
            rng.standard_normal((npmom, nx)),
            t=k * 0.5,
            it=it,
            axes=[
                ("x1", "x_1", r"c / \omega_p", 0.0, 10.0),
                ("p1", "p_1", r"m_e c", -1.0, 1.0),
            ],
        )


def _write_hist_energy(run_dir: Path, n_steps: int) -> None:
    """Write OSIRIS-style ``HIST/`` field + kinetic energy ASCII tables."""
    hist = run_dir / "HIST"
    hist.mkdir(parents=True, exist_ok=True)
    fld = ["! iter time e1 e2 e3 b1 b2 b3"]
    par = []
    for k in range(n_steps):
        t = k * 0.5
        # field falls, kinetic rises by the same amount -> total conserved.
        fld.append(f"{k * 10} {t} {1.0 - 0.1 * k} 0 0 0 0 0")
        par.append(f"{k * 10} {t} {0.1 * k}")
    (hist / "fld_ene").write_text("\n".join(fld) + "\n")
    (hist / "par01_ene").write_text("\n".join(par) + "\n")


def _make_full_run(tmp_path: Path, n_steps: int = 5, nx: int = 16) -> Path:
    """A run with fields, a density moment, a phase space, and HIST energy."""
    run_dir = _make_run(tmp_path, n_steps=n_steps, nx=nx)  # FLD/e1
    _write_field(run_dir, "e2", n_steps, nx, seed=2)
    _write_field(run_dir, "b3", n_steps, nx, seed=3)
    rng = np.random.default_rng(11)
    dens = run_dir / "MS" / "DENSITY" / "electrons" / "charge"
    for k in range(n_steps):
        it = k * 10
        _write_dump(
            dens / f"charge-{it:06d}.h5",
            "charge",
            rng.standard_normal(nx),
            t=k * 0.5,
            it=it,
            axes=[("x1", "x_1", r"c / \omega_p", 0.0, 10.0)],
        )
    _write_phasespace(run_dir, n_steps, nx, npmom=6)
    _write_hist_energy(run_dir, n_steps)
    return run_dir


def test_load_series_nc_roundtrips_grid_series(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, n_steps=4, nx=8)
    out = tmp_path / "binary"
    oio.save_run_datasets(run_dir, out)

    from_nc = oio.load_series_nc(out / "FLD" / "e1.nc")
    from_ms = oio.load_series(run_dir / "MS" / "FLD" / "e1")

    assert from_nc.dims == from_ms.dims
    assert from_nc.shape == from_ms.shape
    np.testing.assert_allclose(from_nc.values, from_ms.values)
    np.testing.assert_allclose(from_nc["t"].values, from_ms["t"].values)
    assert list(from_nc["iter"].values) == list(from_ms["iter"].values)
    # axis metadata is rebuilt into the dict attrs the plotters read.
    assert from_nc.attrs["axis_units"]["x1"] == from_ms.attrs["axis_units"]["x1"]
    # load_series dispatches to load_series_nc when handed a .nc file.
    np.testing.assert_allclose(oio.load_series(out / "FLD" / "e1.nc").values, from_ms.values)


def test_save_run_datasets_persists_hist_energy(tmp_path: Path) -> None:
    run_dir = _make_full_run(tmp_path)
    out = tmp_path / "binary"
    written = oio.save_run_datasets(run_dir, out)

    nc = out / "HIST" / "energy.nc"
    assert nc.exists()
    assert nc in written
    # load_hist_energy reads the saved NetCDF when there's no raw HIST/ ASCII.
    energy = oio.load_hist_energy(out)
    assert energy is not None
    assert "total" in energy
    assert energy.attrs.get("total_drift_frac") == 0.0  # conserved by construction


def test_save_canned_plots_regenerates_from_netcdf(tmp_path: Path) -> None:
    from adept.osiris import plots as oplots

    run_dir = _make_full_run(tmp_path)

    # Plots from the raw OSIRIS run...
    out_ms = tmp_path / "plots_ms"
    written_ms = oplots.save_canned_plots(run_dir, out_ms)

    # ...vs plots regenerated from the saved NetCDFs alone (no MS/ tree).
    binary = tmp_path / "binary"
    oio.save_run_datasets(run_dir, binary)
    assert not (binary / "MS").exists()
    out_nc = tmp_path / "plots_nc"
    written_nc = oplots.save_canned_plots(binary, out_nc)

    # The NetCDF-only regeneration reproduces the exact same plot set.
    assert set(written_nc) == set(written_ms)
    # ...spanning every family, incl. the energy traces that read outside MS/.
    for key in (
        "spacetime/e1",
        "omega_k/e1",
        "moments/electrons/charge",
        "profiles/electrons/density",
        "phasespace/electrons/p1x1",
        "phasespace_evolution/electrons/p1x1",
        "energy_vs_time",
        "energy_components_vs_time",
        "total_energy_vs_time",
    ):
        assert key in written_nc, f"missing {key}"
        assert written_nc[key].exists()


# --- OSIRIS autoscale: per-dump momentum bounds (if_ps_p_auto) ------------


def _write_phasespace_autoscaled(run_dir: Path, n_steps: int, nx: int, npmom: int) -> Path:
    """Phase space whose momentum bounds GROW each dump (if_ps_p_auto=.true.).

    The bin count (``npmom``) stays fixed while the AXIS min/max move — exactly
    what OSIRIS autoscale produces and what a shape-only series check misses.
    """
    rng = np.random.default_rng(7)
    d = run_dir / "MS" / "PHA" / "p1x1" / "electrons"
    for k in range(n_steps):
        it = k * 10
        p_hi = 1.0 + 0.5 * k  # bounds move dump-to-dump
        _write_dump(
            d / f"p1x1-electrons-{it:06d}.h5",
            "p1x1",
            rng.standard_normal((npmom, nx)),
            t=k * 0.5,
            it=it,
            axes=[
                ("x1", "x_1", r"c / \omega_p", 0.0, 10.0),
                ("p1", "p_1", r"m_e c", -p_hi, p_hi),
            ],
        )
    return d


def test_load_series_captures_autoscale_bounds(tmp_path: Path) -> None:
    d = _write_phasespace_autoscaled(tmp_path / "run", n_steps=4, nx=8, npmom=6)
    ser = oio.load_series(d)

    # p1 is autoscaled (bounds move); x1 is fixed (no bound coords).
    assert ser.attrs.get("autoscaled_dims") == ["p1"]
    assert "p1_min" in ser.coords and "p1_max" in ser.coords
    assert "x1_min" not in ser.coords and "x1_max" not in ser.coords
    np.testing.assert_allclose(ser["p1_max"].values, [1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(ser["p1_min"].values, [-1.0, -1.5, -2.0, -2.5])

    # physical_axis rebuilds each dump's true axis; the nominal dim coordinate
    # (first dump) is NOT reused for later steps.
    assert oio.physical_axis(ser, "p1", it=0)[-1] == 1.0
    assert oio.physical_axis(ser, "p1", it=-1)[-1] == 2.5
    # a fixed axis falls back to the stored coordinate.
    np.testing.assert_allclose(oio.physical_axis(ser, "x1"), ser["x1"].values)


def test_autoscale_bounds_survive_netcdf_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_phasespace_autoscaled(run_dir, n_steps=4, nx=8, npmom=6)
    out = tmp_path / "binary"
    oio.save_run_datasets(run_dir, out)

    from_nc = oio.load_series_nc(out / "PHA" / "p1x1" / "electrons.nc")
    assert from_nc.attrs.get("autoscaled_dims") == ["p1"]
    np.testing.assert_allclose(
        oio.physical_axis(from_nc, "p1", it=-1),
        np.linspace(-2.5, 2.5, from_nc.sizes["p1"]),
    )


def test_phasespace_plots_run_on_autoscaled_series(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    from adept.osiris import plots as oplots

    d = _write_phasespace_autoscaled(tmp_path / "run", n_steps=6, nx=8, npmom=6)
    ser = oio.load_series(d)

    # final-step heatmap is drawn on the LAST dump's autoscaled momentum range
    # (p_hi = 1 + 0.5*5 = 3.5), not the first dump's (1.0).
    final = ser.isel(t=-1)
    final.attrs["time"] = float(final["t"].values)
    ax = oplots.plot_phasespace(final)
    ylo, yhi = ax.get_ylim()
    assert yhi >= 3.0 and ylo <= -3.0
    plt.close(ax.figure)

    # faceted evolution renders one panel per sampled time without error.
    fig = oplots.plot_phasespace_evolution(ser, n_panels=4)
    assert len(fig.axes) >= 4  # panels (+ a shared colorbar)
    plt.close(fig)
