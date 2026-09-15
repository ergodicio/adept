"""Field conversion must walk a diagnostic's dumps exactly once.

openPMD bundles every record of a dump into one file, but the converter used
to re-walk the whole tree once per record — 7 passes over a 7-record
diagnostic. At the field cadence a 2-omega_0 Nyquist needs (a dump every 7-9
steps at dt = 0.178/omega_0, so ~88k dumps over the ppc-scan run length) that
is 614k opens against 88k files, and at the ~6.5 ms/file measured on
Perlmutter's Lustre the redundant six-sevenths alone cost ~57 min per
simulation. It also loaded every record before consulting the whitelist, and
held the whole history as float64 slabs plus their stack.

These tests pin the fix: one open per dump, filtering before reading, and a
peak that does not grow with the history.
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

from adept.osiris.io import list_diagnostics_nc, load_series_nc
from adept.warpx import io as wio

FIXTURE = Path(__file__).parent / "fixtures" / "srs-mini-1d"
N_DUMPS = 11  # the fixture's diag1 history
RECORDS = 7  # Ex Ey Ez Bx By Bz rho_electrons

pytestmark = pytest.mark.skipif(
    not (FIXTURE / "diags").is_dir(),
    reason="srs-mini-1d fixture outputs not present",
)


@pytest.fixture(scope="module")
def units() -> wio.CodeUnits:
    return wio.code_units_for_run(FIXTURE)


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


def _write_openpmd(
    root: Path,
    *,
    labels: list[bytes],
    shape: tuple[int, ...],
    iters: list[int],
    records: tuple[str, ...] = ("E",),
    comps: tuple[str, ...] = ("x", "y", "z"),
) -> Path:
    """A minimal openPMD file-based field series (one iteration per file)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for j, it in enumerate(iters):
        with h5py.File(root / f"openpmd_{j:06d}.h5", "w") as f:
            f.attrs["meshesPath"] = np.bytes_(b"fields/")
            grp = f.create_group(f"data/{it}")
            grp.attrs["time"] = float(it)
            grp.attrs["timeUnitSI"] = 1.0
            fields = grp.create_group("fields")
            for rec in records:
                g = fields.create_group(rec)
                g.attrs["axisLabels"] = np.array(labels)
                g.attrs["gridSpacing"] = np.array([1e-8] * len(shape))
                g.attrs["gridGlobalOffset"] = np.array([0.0] * len(shape))
                g.attrs["gridUnitSI"] = 1.0
                for c in comps:
                    d = g.create_dataset(c, data=rng.random(shape))
                    d.attrs["unitSI"] = 1.0
                    d.attrs["position"] = np.array([0.0] * len(shape))
    return root


class TestOnePassPerDiagnostic:
    def test_plan_opens_exactly_one_file(self, units) -> None:
        with _count_opens() as n:
            _mp, plans = wio._field_plan(FIXTURE / "diags" / "diag1", units=units)
        assert n["files"] == 1
        assert len(plans) == RECORDS

    def test_conversion_cost_is_flat_in_record_count(self, tmp_path, units) -> None:
        """One record or six, the walk is the same. This is the whole change:
        the old loop was N_DUMPS opens *per record*."""
        counts = {}
        for k, diags in (("one", ["e1"]), ("six", ["e1", "e2", "e3", "b1", "b2", "b3"])):
            with _count_opens() as n:
                wio.save_run_datasets(FIXTURE, tmp_path / k, units=units, diagnostics=diags)
            counts[k] = n["files"]
        # One _field_plan open + one per dump + list_species' first-dump probe.
        assert counts["one"] == counts["six"] == N_DUMPS + 2
        assert counts["six"] < RECORDS * N_DUMPS  # the shape of the old cost
        assert {"FLD/e1", "FLD/e2", "FLD/b3"} <= set(list_diagnostics_nc(tmp_path / "six"))

    def test_whitelist_skips_records_before_reading(self, tmp_path, units) -> None:
        """A one-record whitelist must not cost the other six records' reads.

        The loop this replaced called ``load_field_series`` and only then
        checked ``want(rel)``, so ``diagnostics_to_log: [e1]`` still paid for
        all seven series.
        """
        with _count_opens() as n:
            wio.save_run_datasets(FIXTURE, tmp_path, units=units, diagnostics=["e1"])
        assert n["files"] == N_DUMPS + 2
        assert set(list_diagnostics_nc(tmp_path)) == {"FLD/e1"}

    def test_all_records_land_from_the_single_walk(self, tmp_path, units) -> None:
        wio.save_run_datasets(FIXTURE, tmp_path, units=units)
        keys = set(list_diagnostics_nc(tmp_path))
        assert {f"FLD/{c}" for c in ("e1", "e2", "e3", "b1", "b2", "b3")} <= keys
        assert "DENSITY/electrons/charge" in keys
        for k in ("FLD/e1", "FLD/b3", "DENSITY/electrons/charge"):
            assert load_series_nc(tmp_path / f"{k}.nc").sizes["t"] == N_DUMPS


class TestMatchesTheReferenceLoader:
    """``load_field_series`` stays the reference; the streamed files must
    reproduce it exactly, record for record."""

    @pytest.mark.parametrize(
        ("rel", "mesh", "comp"),
        [
            ("FLD/e1", "E", "z"),
            ("FLD/e2", "E", "x"),
            ("FLD/b3", "B", "y"),
            ("DENSITY/electrons/charge", "rho_electrons", None),
        ],
    )
    def test_record_identical(self, rel, mesh, comp, tmp_path, units) -> None:
        from adept.osiris.io import series_to_dataset

        wio.save_run_datasets(FIXTURE, tmp_path, units=units)
        streamed = load_series_nc(tmp_path / f"{rel}.nc")

        # Round-trip the reference through the batch write the streaming path
        # replaced, so the comparison is file-to-file: netCDF stores a
        # 1-element `sim.XMIN` as a scalar, which is a serialization detail,
        # not a difference between the two converters.
        ref_ds = series_to_dataset(wio.load_field_series(FIXTURE / "diags" / "diag1", mesh, comp, units=units))
        ref_p = tmp_path / "ref.nc"
        ref_ds.to_netcdf(
            ref_p,
            engine="h5netcdf",
            encoding={n: {"zlib": True, "complevel": 4, "shuffle": True} for n in ref_ds.data_vars},
        )
        ref = load_series_nc(ref_p)

        assert streamed.dims == ref.dims
        np.testing.assert_array_equal(streamed.values, ref.values)
        for c in ref.coords:
            np.testing.assert_array_equal(streamed.coords[c].values, ref.coords[c].values)
        assert {k: str(v) for k, v in streamed.attrs.items()} == {k: str(v) for k, v in ref.attrs.items()}

    def test_chunks_span_the_whole_slab(self, tmp_path, units) -> None:
        wio.save_run_datasets(FIXTURE, tmp_path, units=units, diagnostics=["e1"])
        with h5py.File(tmp_path / "FLD" / "e1.nc") as f:
            assert f["e1"].chunks[1:] == f["e1"].shape[1:]


class TestAxisOrder:
    def test_3d_permutes_into_osiris_storage_order(self, tmp_path) -> None:
        """openPMD ``(x, y, z)`` -> OSIRIS ``(t, x3, x2, x1)``: the only case
        where the per-dump transpose is not the identity."""
        nx, ny, nz = 3, 4, 5
        src = _write_openpmd(
            tmp_path / "diag", labels=[b"x", b"y", b"z"], shape=(nx, ny, nz), iters=[0, 10], comps=("z",)
        )
        _mp, plans = wio._field_plan(src)
        (plan,) = plans
        assert plan.dims == ("x3", "x2", "x1")
        assert plan.perm == (1, 0, 2)  # native (x2, x3, x1) -> (x3, x2, x1)

        ref = wio.load_field_series(src, "E", "z")
        assert ref.dims == ("t", "x3", "x2", "x1")
        assert ref.shape == (2, ny, nx, nz)
        with h5py.File(sorted(src.glob("*.h5"))[0]) as f:
            raw = f["data/0/fields/E/z"][...]
        np.testing.assert_array_equal(ref.isel(t=0).values, raw.transpose(1, 0, 2).astype("float32"))

    def test_2d_xz_needs_no_transpose(self, tmp_path) -> None:
        src = _write_openpmd(tmp_path / "diag", labels=[b"x", b"z"], shape=(4, 6), iters=[0], comps=("z",))
        (plan,) = wio._field_plan(src)[1]
        assert plan.dims == ("x2", "x1") and plan.perm == (0, 1)


class TestRobustness:
    def test_out_of_order_iterations_fall_back(self, tmp_path) -> None:
        """The append-only writer cannot reorder; the eager path can."""
        src = _write_openpmd(tmp_path / "diag", labels=[b"z"], shape=(8,), iters=[7, 3, 5, 1], comps=("z",))
        _mp, plans = wio._field_plan(src)
        out = tmp_path / "out"
        got = wio.convert_field_series_streaming(src, [("FLD/e1", plans[0])], out, meshes_path="fields")
        assert got == [out / "FLD" / "e1.nc"]
        streamed = load_series_nc(got[0])
        ref = wio.load_field_series(src, "E", "z")
        np.testing.assert_array_equal(streamed.coords["iter"].values, [1, 3, 5, 7])
        np.testing.assert_array_equal(streamed.values, ref.values)
        assert "x1_min" not in streamed.coords  # the eager fallback, not the writer

    def test_one_bad_record_does_not_abort_the_others(self, tmp_path, units, monkeypatch) -> None:
        """The per-record loop isolated failures; the single walk must too."""
        real = wio._field_slab

        def flaky(plan, node, *, scaled):
            if plan.rel == "FLD/e2":
                raise ValueError("synthetic bad record")
            return real(plan, node, scaled=scaled)

        monkeypatch.setattr(wio, "_field_slab", flaky)
        wio.save_run_datasets(FIXTURE, tmp_path, units=units, diagnostics=["e1", "e2", "e3"])
        keys = set(list_diagnostics_nc(tmp_path))
        assert "FLD/e2" not in keys
        assert {"FLD/e1", "FLD/e3"} <= keys
        assert load_series_nc(tmp_path / "FLD" / "e1.nc").sizes["t"] == N_DUMPS


class TestMemory:
    def test_peak_grows_far_slower_than_the_history(self, tmp_path) -> None:
        """``load_field_series`` accumulated float64 slabs then stacked them —
        peak proportional to the history, ~6.3 GB per component at 88k dumps
        of 3594 cells (and far worse in 2D). Streaming holds one slab per
        record plus their ~1 MiB write batches, so peak tracks the *slab*,
        not the series."""
        nz = 4096
        n_small, n_large = 100, 1600
        peaks = {}
        for n_dumps in (n_small, n_large):
            src = _write_openpmd(
                tmp_path / f"d{n_dumps}",
                labels=[b"z"],
                shape=(nz,),
                iters=list(range(n_dumps)),
                comps=("x", "y", "z"),
            )
            _mp, plans = wio._field_plan(src)
            targets = [(f"FLD/{p.name}", p) for p in plans]
            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                wio.convert_field_series_streaming(src, targets, tmp_path / f"o{n_dumps}", meshes_path="fields")
                _, peaks[n_dumps] = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

        def cube(n: int) -> int:
            return n * nz * 4 * 3  # the three components' stacked float32 histories

        # 16x the dumps must cost a small fraction of the 16x more data.
        grew = peaks[n_large] - peaks[n_small]
        assert grew < 0.10 * (cube(n_large) - cube(n_small)), (
            f"peak grew {grew / 1e6:.1f} MB against {(cube(n_large) - cube(n_small)) / 1e6:.1f} MB more data"
        )
        assert peaks[n_large] < cube(n_large) / 4
