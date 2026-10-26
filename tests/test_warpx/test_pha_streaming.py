"""The ParticleHistogram2D conversion must never stack the cube in memory.

A production ``x1log_gamma_q1`` history is 11705 dumps of 1000x1024 =
47.9 GB float32, and the eager converter's result then went through
``series_to_dataset``'s deep copy — ~96 GB of peak, which is what SIGKILLed
all six srs-1d-ppc-scan postprocs in a 55 GB cgroup (job 57536430). The
streaming converter feeds the OSIRIS ``StreamWriter`` one slab at a time,
matching what the OSIRIS side has done since the drainer landed. These
tests pin the two properties that buys: the output is identical to the
eager path's, and the cube is never allocated.
"""

from __future__ import annotations

import contextlib
import tracemalloc
import types
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
import xarray as xr

from adept.osiris.io import load_series_nc, series_to_dataset
from adept.warpx import io as wio

FIXTURE = Path(__file__).parent / "fixtures" / "srs-mini-1d"
HISTOGRAMS = ("p1x1", "x1log_gamma_q1")

pytestmark = pytest.mark.skipif(
    not (FIXTURE / "diags").is_dir(),
    reason="srs-mini-1d fixture outputs not present",
)


@pytest.fixture(scope="module")
def units() -> wio.CodeUnits:
    return wio.code_units_for_run(FIXTURE)


@pytest.fixture(scope="module")
def deck() -> dict:
    return wio._deck_for_run(FIXTURE)


@contextlib.contextmanager
def _count_opens():
    """Count the source-dump opens ``adept.warpx.io`` performs in the block.

    Only the io module's *own* view of h5py is swapped — h5netcdf (and hence
    the StreamWriter's writes) keeps the real one, which an attribute patch
    on the shared module would break (`isinstance(path, h5py.File)`).
    """
    n = {"files": 0}

    def counting(name, mode="r", *a, **k):
        if str(mode).startswith("r"):
            n["files"] += 1
        return h5py.File(name, mode, *a, **k)

    shim = types.SimpleNamespace(File=counting, Dataset=h5py.Dataset, Group=h5py.Group)
    with mock.patch.object(wio, "h5py", shim):
        yield n


def _diag(name: str) -> Path:
    return FIXTURE / "diags" / "reducedfiles" / name


def _write_eager(diag_dir: Path, dest: Path, *, units, deck) -> Path:
    """What the converter did before streaming: whole cube, then to_netcdf."""
    ds = series_to_dataset(wio.load_particle_histogram2d(diag_dir, units=units, deck=deck))
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(
        dest,
        engine="h5netcdf",
        encoding={n: {"zlib": True, "complevel": 4, "shuffle": True} for n in ds.data_vars},
    )
    return dest


def _write_histogram_dir(
    root: Path,
    *,
    n_dumps: int,
    n_abs: int,
    n_ord: int,
    iters: list[int] | None = None,
    one_per_file: bool = True,
) -> Path:
    """A minimal openPMD file-based ParticleHistogram2D series.

    Mirrors what WarpX writes (``data/<iter>/meshes/data`` as
    ``(ord, abs)``), so the converter's real code path is exercised;
    ``iters`` lets a test lay the iterations out of filename order.
    """
    root.mkdir(parents=True, exist_ok=True)
    its = list(range(n_dumps)) if iters is None else list(iters)
    rng = np.random.default_rng(0)
    for j, it in enumerate(its):
        f_idx = j if one_per_file else 0
        mode = "a" if (not one_per_file and j) else "w"
        with h5py.File(root / f"openpmd_{f_idx:06d}.h5", mode) as f:
            f.attrs["meshesPath"] = np.bytes_(b"meshes/")
            grp = f.require_group(f"data/{it}")
            grp.attrs["time"] = float(it)
            grp.attrs["timeUnitSI"] = 1.0
            dset = grp.require_group("meshes").create_dataset("data", data=rng.random((n_ord, n_abs)), dtype="float64")
            dset.attrs["gridSpacing"] = np.array([1.0 / n_ord, 1.0 / n_abs])
            dset.attrs["gridGlobalOffset"] = np.array([0.0, 0.0])
            dset.attrs["gridUnitSI"] = 1.0
    return root


class TestStreamedMatchesEager:
    """The streamed NetCDF must be the eager one, modulo the OSIRIS bounds."""

    @pytest.mark.parametrize("name", HISTOGRAMS)
    def test_data_coords_and_attrs_identical(self, name, tmp_path, units, deck) -> None:
        d = _diag(name)
        eager = load_series_nc(_write_eager(d, tmp_path / "eager.nc", units=units, deck=deck))
        streamed = load_series_nc(wio.convert_histogram2d_streaming(d, tmp_path / "s.nc", units=units, deck=deck))

        assert streamed.dims == eager.dims
        np.testing.assert_array_equal(streamed.values, eager.values)
        for k in eager.coords:
            np.testing.assert_array_equal(streamed.coords[k].values, eager.coords[k].values)
        assert {k: str(v) for k, v in streamed.attrs.items()} == {k: str(v) for k, v in eager.attrs.items()}

    @pytest.mark.parametrize("name", HISTOGRAMS)
    def test_bound_coords_are_the_osiris_convention(self, name, tmp_path, units, deck) -> None:
        """StreamWriter records per-dump axis bounds for every dim, exactly as
        it does for OSIRIS diagnostics. A histogram's axes are fixed by the
        deck, so they must not vary — a varying bound would make
        ``load_series_nc`` declare the dim autoscaled and change plotting."""
        streamed = load_series_nc(
            wio.convert_histogram2d_streaming(_diag(name), tmp_path / "s.nc", units=units, deck=deck)
        )
        spatial = [d for d in streamed.dims if d != "t"]
        assert set(spatial) and all(f"{d}_min" in streamed.coords for d in spatial)
        for d in spatial:
            for b in ("min", "max"):
                v = streamed.coords[f"{d}_{b}"].values
                assert np.allclose(v, v[0]), f"{d}_{b} varies across dumps"
        assert "autoscaled_dims" not in streamed.attrs

    @pytest.mark.parametrize("name", HISTOGRAMS)
    def test_chunks_span_the_whole_slab(self, name, tmp_path, units, deck) -> None:
        """The batch path's default chunking (``(183, 16, 32)`` on the
        production cube) put one time slice across 2016 chunks — the 183x
        read amplification behind the 6.66 h collect_srs stage. A streamed
        file chunks whole slabs, so a per-dump read is chunk-aligned."""
        dest = wio.convert_histogram2d_streaming(_diag(name), tmp_path / "s.nc", units=units, deck=deck)
        with h5py.File(dest) as f:
            chunks = f[name].chunks
            shape = f[name].shape
        assert chunks[1:] == shape[1:], f"{name}: chunk {chunks} splits the {shape[1:]} slab"


class TestCubeIsNeverAllocated:
    """The eager cube lives only in ``load_particle_histogram2d``; nothing on
    the conversion path may reach it."""

    @pytest.mark.parametrize("name", HISTOGRAMS)
    def test_streaming_does_not_call_the_eager_loader(self, name, tmp_path, units, deck, monkeypatch) -> None:
        def _boom(*a, **k):
            raise AssertionError("the streaming path materialized the whole cube")

        monkeypatch.setattr(wio, "load_particle_histogram2d", _boom)
        dest = wio.convert_histogram2d_streaming(_diag(name), tmp_path / "s.nc", units=units, deck=deck)
        assert load_series_nc(dest).sizes["t"] == 11

    def test_save_run_datasets_streams_the_histograms(self, tmp_path, units, monkeypatch) -> None:
        def _boom(*a, **k):
            raise AssertionError("save_run_datasets materialized the whole cube")

        monkeypatch.setattr(wio, "load_particle_histogram2d", _boom)
        wio.save_run_datasets(FIXTURE, tmp_path, units=units, diagnostics=list(HISTOGRAMS))
        for name in HISTOGRAMS:
            assert (tmp_path / "PHA" / name / "electrons.nc").is_file()

    def test_peak_memory_is_a_slab_not_the_cube(self, tmp_path) -> None:
        """Peak allocation must scale with one slab, not with the history."""
        n_dumps, n_abs, n_ord = 400, 96, 128
        src = _write_histogram_dir(tmp_path / "diag", n_dumps=n_dumps, n_abs=n_abs, n_ord=n_ord)
        cube_bytes = n_dumps * n_abs * n_ord * 4
        slab_bytes = n_abs * n_ord * 4

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            wio.convert_histogram2d_streaming(src, tmp_path / "s.nc")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # One slab plus the writer's ~1 MiB batch buffer and h5py's own
        # buffers; a generous ceiling that the cube (19.7 MB) still fails.
        assert peak < min(cube_bytes / 4, 64 * slab_bytes + (1 << 21)), (
            f"peak {peak / 1e6:.1f} MB against a {cube_bytes / 1e6:.1f} MB cube"
        )
        assert load_series_nc(tmp_path / "s.nc").sizes["t"] == n_dumps


class TestOutOfOrderIterations:
    def test_falls_back_to_the_eager_path_and_still_sorts(self, tmp_path) -> None:
        """An append-only writer cannot reorder, so a series whose file order
        is not iteration order restarts on the eager path. WarpX does not
        produce one (``_openpmd_files`` sorts on the filename iteration
        index), but the result must be correct if it ever does."""
        iters = [7, 3, 5, 1]
        src = _write_histogram_dir(tmp_path / "diag", n_dumps=len(iters), n_abs=8, n_ord=6, iters=iters)

        streamed = load_series_nc(wio.convert_histogram2d_streaming(src, tmp_path / "s.nc"))
        eager = wio.load_particle_histogram2d(src)
        np.testing.assert_array_equal(streamed.coords["iter"].values, sorted(iters))
        np.testing.assert_array_equal(streamed.values, eager.values)
        # The eager path writes no per-dump axis bounds; their absence is the
        # observable signal that the fallback (not the writer) produced this.
        assert "x1_min" not in streamed.coords

    def test_in_order_series_streams(self, tmp_path) -> None:
        src = _write_histogram_dir(tmp_path / "diag", n_dumps=4, n_abs=8, n_ord=6)
        streamed = load_series_nc(wio.convert_histogram2d_streaming(src, tmp_path / "s.nc"))
        assert "x1_min" in streamed.coords  # StreamWriter, not the fallback

    def test_detection_is_inline_not_a_pre_scan(self, tmp_path) -> None:
        """The monotonicity check rides along the single walk — asking for it
        must not cost an extra pass."""
        src = _write_histogram_dir(tmp_path / "diag", n_dumps=6, n_abs=8, n_ord=6)
        geom = wio._histogram2d_geometry(src)
        with _count_opens() as n:
            list(wio._histogram2d_records(geom, require_monotonic=True))
        assert n["files"] == 6


class TestGeometryPass:
    """Deriving the axes must cost one file open, whatever the history."""

    @pytest.mark.parametrize("name", HISTOGRAMS)
    def test_geometry_matches_the_loaded_series(self, name, units, deck) -> None:
        geom = wio._histogram2d_geometry(_diag(name), units=units, deck=deck)
        da = wio.load_particle_histogram2d(_diag(name), units=units, deck=deck)
        assert (geom.n_abs, geom.n_ord) == da.shape[1:]
        assert (geom.abs_dim, geom.ord_dim) == da.dims[1:]
        np.testing.assert_array_equal(geom.x_axis, da.coords[geom.abs_dim].values)
        np.testing.assert_array_equal(geom.ord_axis, da.coords[geom.ord_dim].values)

    @pytest.mark.parametrize("n_dumps", [4, 40])
    def test_geometry_opens_exactly_one_file(self, n_dumps, tmp_path) -> None:
        src = _write_histogram_dir(tmp_path / f"d{n_dumps}", n_dumps=n_dumps, n_abs=8, n_ord=6)
        with _count_opens() as n:
            wio._histogram2d_geometry(src)
        assert n["files"] == 1


class TestSinglePass:
    """The conversion walks the tree once. At the 2-omega_0-Nyquist cadences
    (a dump every 7-9 steps) the ~6.5 ms per-file open on Lustre is what sets
    the conversion wall time, so a second walk is a second hour."""

    @pytest.mark.parametrize("n_dumps", [4, 40])
    def test_streaming_opens_each_file_once(self, n_dumps, tmp_path) -> None:
        src = _write_histogram_dir(tmp_path / f"d{n_dumps}", n_dumps=n_dumps, n_abs=8, n_ord=6)
        with _count_opens() as n:
            wio.convert_histogram2d_streaming(src, tmp_path / f"s{n_dumps}.nc")
        # 1 geometry open + one per dump; the writer's own dest is h5netcdf.
        assert n["files"] == n_dumps + 1

    @pytest.mark.parametrize("name", HISTOGRAMS)
    def test_streaming_opens_each_file_once_on_the_fixture(self, name, tmp_path, units, deck) -> None:
        with _count_opens() as n:
            wio.convert_histogram2d_streaming(_diag(name), tmp_path / "s.nc", units=units, deck=deck)
        assert n["files"] == 12  # 11 dumps + the geometry open


def test_empty_diag_dir_raises(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        wio.convert_histogram2d_streaming(tmp_path / "empty", tmp_path / "s.nc")


def test_streaming_rebuilds_a_partial_file(tmp_path, units, deck) -> None:
    """A killed postproc leaves a partial NetCDF; the next run must not
    resume into it (StreamWriter's append-mode resume is for the drainer)."""
    dest = tmp_path / "s.nc"
    d = _diag("p1x1")
    wio.convert_histogram2d_streaming(d, dest, units=units, deck=deck)
    n = load_series_nc(dest).sizes["t"]
    wio.convert_histogram2d_streaming(d, dest, units=units, deck=deck)
    assert load_series_nc(dest).sizes["t"] == n


def test_series_is_readable_by_the_osiris_consumers(tmp_path, units, deck) -> None:
    """The whole point of the contract: osiris_lpi's lazy readers work."""
    from adept.osiris.io import open_series

    dest = wio.convert_histogram2d_streaming(_diag("x1log_gamma_q1"), tmp_path / "s.nc", units=units, deck=deck)
    with open_series(dest) as ser:
        assert ser.dims == ("t", "x1", "gamma")
        assert isinstance(ser.isel(t=0).load(), xr.DataArray)
