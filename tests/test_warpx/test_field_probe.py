"""FieldProbe Line tables -> OSIRIS-style ``FLD/<comp>-line-…`` series.

FieldProbe writes one row per probe point per step (column layout pinned
against WarpX ``Source/Diagnostics/ReducedDiags/FieldProbe.cpp``: header of
un-``#``-prefixed ``[i]name(unit)`` tokens, rows ``step, time, part_x/y/z,
Ex..Bz, |S|`` sorted along the line). These tests exercise the reshape into
``(t, x2)`` / ``(t, x1)`` code-unit lineouts named by the osiris_lpi
line-report discovery contract (boundary line along x2 at a fixed x1 cell;
axial line along x1 at a fixed x2 cell), plus the ``save_run_datasets``
routing that keeps the long tables out of the generic ``REDUCED/`` parser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from adept.warpx import io as wio

_C = 299792458.0

N0_REF = 9.05e27  # 1/m^3
NX, NZ = 342, 3648
LO = (0.0, 0.0)  # (x, z)
HI = (3.0e-6, 3.2e-5)
DX = (HI[0] - LO[0]) / NX
DZ = (HI[1] - LO[1]) / NZ

HEADER = (
    "[0]step() [1]time(s) [2]part_x_lev0-(m) [3]part_y_lev0-(m) [4]part_z_lev0-(m) "
    "[5]part_Ex_lev0-(V/m) [6]part_Ey_lev0-(V/m) [7]part_Ez_lev0-(V/m) "
    "[8]part_Bx_lev0-(T) [9]part_By_lev0-(T) [10]part_Bz_lev0-(T) [11]part_S_lev0-(W/m^2)"
)

DECK = {
    "geometry.prob_lo": list(LO),
    "geometry.prob_hi": list(HI),
    "amr.n_cell": [NX, NZ],
}


def _row(step: int, time: float, x: float, z: float, i: int) -> str:
    # deterministic per-(step, point) field values so mapping is checkable:
    # Ex = step*1e3 + i, Ey = 2*that, Ez = 3*that, Bx..Bz = value/c
    v = step * 1.0e3 + i
    e = (v, 2 * v, 3 * v)
    b = tuple(c / _C for c in e)
    return (
        f"{step} {time:.14e} {x:.14e} 0.0 {z:.14e} "
        f"{e[0]:.14e} {e[1]:.14e} {e[2]:.14e} {b[0]:.14e} {b[1]:.14e} {b[2]:.14e} 0.0"
    )


def write_boundary_table(path: Path, *, z_fixed: float, steps=(10, 20, 30), n_pts=8) -> None:
    """A line along x (transverse) at fixed z — a boundary light monitor."""
    xs = LO[0] + (np.arange(n_pts) + 0.5) * (HI[0] - LO[0]) / n_pts
    lines = [HEADER]
    for s in steps:
        for i, x in enumerate(xs):
            lines.append(_row(s, s * 1.0e-17, x, z_fixed, i))
    path.write_text("\n".join(lines) + "\n")


def write_axial_table(path: Path, *, x_fixed: float, steps=(10, 20), n_pts=16) -> None:
    """A line along z (propagation) at fixed x — the (k, omega) source."""
    zs = LO[1] + (np.arange(n_pts) + 0.5) * (HI[1] - LO[1]) / n_pts
    lines = [HEADER]
    for s in steps:
        for i, z in enumerate(zs):
            lines.append(_row(s, s * 1.0e-17, x_fixed, z, i))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture()
def units() -> wio.CodeUnits:
    return wio.CodeUnits(n0=N0_REF)


class TestBoundaryLine:
    def test_names_cell_and_shape(self, tmp_path: Path, units: wio.CodeUnits) -> None:
        z_fixed = 24.5 * DZ  # inside cell 24
        p = tmp_path / "probe_in.txt"
        write_boundary_table(p, z_fixed=z_fixed)
        out = wio.load_field_probe_lines(p, units=units, deck=DECK)
        assert set(out) == {f"FLD/{c}-line-x2-0024" for c in ("e2", "b3", "e3", "b2")}
        da = out["FLD/e2-line-x2-0024"]
        assert da.dims == ("t", "x2")
        assert da.shape == (3, 8)
        assert da.name == "e2"

    def test_code_units_and_mapping(self, tmp_path: Path, units: wio.CodeUnits) -> None:
        p = tmp_path / "probe_in.txt"
        write_boundary_table(p, z_fixed=24.5 * DZ)
        out = wio.load_field_probe_lines(p, units=units, deck=DECK)
        # Ex -> e2 (cyclic relabel), scaled by 1/E0; By -> b3 by 1/B0
        e2 = out["FLD/e2-line-x2-0024"]
        b3 = out["FLD/b3-line-x2-0024"]
        v = 10 * 1.0e3 + 3  # step 10, point 3: Ex value
        np.testing.assert_allclose(float(e2.isel(t=0, x2=3)), v / units.E0, rtol=1e-6)
        np.testing.assert_allclose(float(b3.isel(t=0, x2=3)), (2 * v) / _C / units.B0, rtol=1e-6)
        # time and axis in code units
        np.testing.assert_allclose(e2.coords["t"].values, np.array([10, 20, 30]) * 1.0e-17 * units.wp0)
        assert e2.coords["x2"].values.max() < HI[0] / units.x0 + 1e-12
        np.testing.assert_array_equal(e2.coords["iter"].values, [10, 20, 30])

    def test_rows_shuffled_within_block_are_sorted(self, tmp_path: Path, units: wio.CodeUnits) -> None:
        p = tmp_path / "probe_in.txt"
        write_boundary_table(p, z_fixed=24.5 * DZ, steps=(10,))
        lines = p.read_text().strip().split("\n")
        rows = lines[1:]
        p.write_text("\n".join([lines[0]] + rows[::-1]) + "\n")
        out = wio.load_field_probe_lines(p, units=units, deck=DECK)
        e2 = out["FLD/e2-line-x2-0024"]
        # point i has Ex = step*1e3 + i: sorted-by-position means ascending i
        vals = e2.isel(t=0).values * units.E0
        np.testing.assert_allclose(np.diff(vals), 1.0, atol=5e-3)  # float32 storage
        assert np.all(np.diff(e2.coords["x2"].values) > 0)

    def test_incomplete_tail_block_dropped(self, tmp_path: Path, units: wio.CodeUnits) -> None:
        p = tmp_path / "probe_in.txt"
        write_boundary_table(p, z_fixed=24.5 * DZ, steps=(10, 20, 30))
        lines = p.read_text().strip().split("\n")
        # truncate mid-block: keep header + 2 full blocks + 3 rows of the third
        p.write_text("\n".join(lines[: 1 + 2 * 8 + 3]) + "\n")
        out = wio.load_field_probe_lines(p, units=units, deck=DECK)
        assert out["FLD/e2-line-x2-0024"].sizes["t"] == 2

    def test_restart_header_mid_file_ignored(self, tmp_path: Path, units: wio.CodeUnits) -> None:
        p = tmp_path / "probe_in.txt"
        write_boundary_table(p, z_fixed=24.5 * DZ, steps=(10, 20))
        lines = p.read_text().strip().split("\n")
        lines.insert(1 + 8, HEADER)  # a second header between the blocks
        p.write_text("\n".join(lines) + "\n")
        out = wio.load_field_probe_lines(p, units=units, deck=DECK)
        assert out["FLD/e2-line-x2-0024"].sizes["t"] == 2


class TestAxialLine:
    def test_names_and_components(self, tmp_path: Path, units: wio.CodeUnits) -> None:
        x_fixed = 171.5 * DX
        p = tmp_path / "probe_axis.txt"
        write_axial_table(p, x_fixed=x_fixed)
        out = wio.load_field_probe_lines(p, units=units, deck=DECK)
        # axial lines feed komega: e1 (EPW) and e2 (EM) only
        assert set(out) == {"FLD/e1-line-x1-0171", "FLD/e2-line-x1-0171"}
        e1 = out["FLD/e1-line-x1-0171"]
        assert e1.dims == ("t", "x1")
        assert e1.shape == (2, 16)
        # Ez -> e1
        v = 10 * 1.0e3 + 0
        np.testing.assert_allclose(float(e1.isel(t=0, x1=0)), (3 * v) / units.E0, rtol=1e-6)

    def test_si_passthrough_without_units(self, tmp_path: Path) -> None:
        p = tmp_path / "probe_axis.txt"
        write_axial_table(p, x_fixed=171.5 * DX)
        out = wio.load_field_probe_lines(p, units=None, deck=DECK)
        e1 = out["FLD/e1-line-x1-0171"]
        assert e1.attrs["units"] == "V/m"
        np.testing.assert_allclose(e1.coords["t"].values, np.array([10, 20]) * 1.0e-17)


class TestSaveRunDatasetsRouting:
    def _run_dir(self, tmp_path: Path) -> Path:
        rd = tmp_path / "run"
        red = rd / "diags" / "reducedfiles"
        red.mkdir(parents=True)
        write_boundary_table(red / "probe_in.txt", z_fixed=24.5 * DZ)
        write_boundary_table(red / "probe_out.txt", z_fixed=3625.5 * DZ)
        write_axial_table(red / "probe_axis.txt", x_fixed=171.5 * DX)
        (rd / "inputs").write_text(
            f"my_constants.n0 = {N0_REF}\n"
            "warpx.reduced_diags_names = probe_in probe_out probe_axis\n"
            "probe_in.type = FieldProbe\n"
            "probe_out.type = FieldProbe\n"
            "probe_axis.type = FieldProbe\n"
            f"geometry.prob_lo = {LO[0]} {LO[1]}\n"
            f"geometry.prob_hi = {HI[0]} {HI[1]}\n"
            f"amr.n_cell = {NX} {NZ}\n"
        )
        return rd

    def test_probe_tables_become_fld_lines(self, tmp_path: Path, units: wio.CodeUnits) -> None:
        rd = self._run_dir(tmp_path)
        out = tmp_path / "binary"
        wio.save_run_datasets(rd, out, units=units)
        # entrance + exit boundary pairs at distinct cells, one axial pair
        for rel in (
            "FLD/e2-line-x2-0024",
            "FLD/b3-line-x2-0024",
            "FLD/e2-line-x2-3625",
            "FLD/b3-line-x2-3625",
            "FLD/e1-line-x1-0171",
            "FLD/e2-line-x1-0171",
        ):
            assert (out / f"{rel}.nc").is_file(), rel
        # the long tables must NOT land in REDUCED/ (mis-stacked there)
        assert not (out / "REDUCED" / "probe_in.nc").exists()
        ds = xr.open_dataset(out / "FLD/e2-line-x2-0024.nc", engine="h5netcdf")
        assert "e2" in ds
        assert ds["e2"].dims == ("t", "x2")
