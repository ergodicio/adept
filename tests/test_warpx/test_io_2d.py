"""2D WarpX conversion: OSIRIS field/axis mapping, rho, derived s1 lineouts.

A synthetic 2D XZ openPMD diagnostic (a rightward vacuum plane wave with a
small longitudinal field and a uniform electron rho) exercises the cartesian
``(z, x, y) → (1, 2, 3)`` relabeling in multi-D, the ``(t, x2, x1)`` storage
order, the OSIRIS-ordered ``sim.XMIN/XMAX`` attrs, and the derived
``s1-line-x2-000{1,2}`` boundary lineouts (whose slab average of
``e2 b3`` is exactly ``a^2/2`` for the 4-cells-per-wavelength wave).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from adept.warpx import io as wio

_C = 299792458.0

NX, NZ = 12, 128
DX = DZ = 1.0e-8
X_OFF, Z_OFF = 1.0e-7, 5.0e-7
E0 = 3.0e9
N_E = 1.0e27  # electron density, 1/m^3
ITS = (0, 10, 20)
N0_REF = 9.05e27  # reference density for CodeUnits


def _mesh_attrs() -> dict:
    return {
        "axisLabels": np.array([b"x", b"z"]),
        "gridSpacing": np.array([DX, DZ]),
        "gridGlobalOffset": np.array([X_OFF, Z_OFF]),
        "gridUnitSI": 1.0,
    }


def write_run_2d(run_dir: Path) -> Path:
    """A tiny 2D XZ openPMD diag + inputs deck; returns the diag dir."""
    diag = run_dir / "diags" / "diag1"
    diag.mkdir(parents=True)
    # wavelength = 4 cells -> any 8-cell slab averages cos^2 to exactly 1/2
    k0 = 2.0 * np.pi / (4.0 * DZ)
    z = Z_OFF + np.arange(NZ) * DZ
    zz = np.broadcast_to(z, (NX, NZ))
    for i, it in enumerate(ITS):
        with h5py.File(diag / f"openpmd_{it:06d}.h5", "w") as f:
            f.attrs["meshesPath"] = np.bytes_("fields/")
            g = f.create_group(f"data/{it}")
            g.attrs["time"] = float(it) * 1.0e-17
            g.attrs["timeUnitSI"] = 1.0
            fields = g.create_group("fields")
            ex = E0 * np.cos(k0 * zz - 0.3 * i)
            ez = 0.1 * E0 * np.sin(k0 * zz)
            comps = {
                "E": {"x": ex, "y": 0.0 * ex, "z": ez},
                "B": {"x": 0.0 * ex, "y": ex / _C, "z": 0.0 * ex},
            }
            for mesh, cs in comps.items():
                mg = fields.create_group(mesh)
                for k, v in _mesh_attrs().items():
                    mg.attrs[k] = v
                for cname, arr in cs.items():
                    d = mg.create_dataset(cname, data=arr)
                    d.attrs["unitSI"] = 1.0
                    d.attrs["position"] = np.array([0.0, 0.0])
            rho = fields.create_dataset("rho_electrons", data=np.full((NX, NZ), -1.602176634e-19 * N_E))
            rho.attrs["unitSI"] = 1.0
            rho.attrs["position"] = np.array([0.0, 0.0])
            for k, v in _mesh_attrs().items():
                rho.attrs[k] = v
    # deck: n0 for units discovery + laser wavelength for the s1 slab width
    (run_dir / "inputs").write_text(
        f"my_constants.n0 = {N0_REF}\nlasers.names = laser1\nlaser1.wavelength = {4.0 * DZ}\n"
    )
    return diag


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    rd = tmp_path_factory.mktemp("warpx2d")
    write_run_2d(rd)
    return rd


@pytest.fixture(scope="module")
def units() -> wio.CodeUnits:
    return wio.CodeUnits(n0=N0_REF)


@pytest.fixture(scope="module")
def binary(run_dir: Path, tmp_path_factory, units: wio.CodeUnits) -> Path:
    out = tmp_path_factory.mktemp("binary2d")
    wio.save_run_datasets(run_dir, out, units=units)
    return out


class TestFieldMapping2D:
    def test_osiris_names_and_dims(self, run_dir: Path, units: wio.CodeUnits) -> None:
        da = wio.load_field_series(run_dir / "diags" / "diag1", "E", "z", units=units)
        assert da.name == "e1"
        assert tuple(da.dims) == ("t", "x2", "x1")
        assert da.sizes["x1"] == NZ and da.sizes["x2"] == NX
        # axes renamed and converted: x1 is the (longitudinal) z axis
        np.testing.assert_allclose(da.coords["x1"].values[0], Z_OFF / units.x0, rtol=1e-12)
        np.testing.assert_allclose(da.coords["x2"].values[0], X_OFF / units.x0, rtol=1e-12)

    def test_box_attrs_ordered_x1_first(self, run_dir: Path, units: wio.CodeUnits) -> None:
        da = wio.load_field_series(run_dir / "diags" / "diag1", "B", "y", units=units)
        assert da.name == "b3"
        xmin = np.asarray(da.attrs["sim.XMIN"], dtype=float)
        nx = np.asarray(da.attrs["sim.NX"], dtype=int)
        # entry 0 must be x1 (the z axis) — sim_box_bound indexes by dim digit
        np.testing.assert_allclose(xmin[0], Z_OFF / units.x0, rtol=1e-12)
        np.testing.assert_allclose(xmin[1], X_OFF / units.x0, rtol=1e-12)
        assert list(nx) == [NZ, NX]

    def test_all_components_map(self, run_dir: Path, units: wio.CodeUnits) -> None:
        for mesh, comp, name in [
            ("E", "x", "e2"),
            ("E", "y", "e3"),
            ("B", "x", "b2"),
            ("B", "z", "b1"),
        ]:
            da = wio.load_field_series(run_dir / "diags" / "diag1", mesh, comp, units=units)
            assert da.name == name, f"{mesh}/{comp} -> {da.name}, wanted {name}"


class TestSaveRunDatasets2D:
    def test_contract_keys(self, binary: Path) -> None:
        for rel in [
            "FLD/e1",
            "FLD/e2",
            "FLD/e3",
            "FLD/b1",
            "FLD/b2",
            "FLD/b3",
            "DENSITY/electrons/charge",
            "FLD/s1-line-x2-0001",
            "FLD/s1-line-x2-0002",
        ]:
            assert (binary / f"{rel}.nc").is_file(), f"missing {rel}"

    def test_rho_code_units(self, binary: Path, units: wio.CodeUnits) -> None:
        import xarray as xr

        ds = xr.load_dataset(binary / "DENSITY/electrons/charge.nc", engine="h5netcdf")
        da = ds["charge"]
        assert tuple(da.dims) == ("t", "x2", "x1")
        # rho = -e n_e -> charge density in units of e n0 is -n_e/n0
        np.testing.assert_allclose(float(da.values.mean()), -N_E / N0_REF, rtol=1e-6)

    def test_s1_lineouts_flux_value(self, binary: Path, units: wio.CodeUnits) -> None:
        import xarray as xr

        a_code = E0 / units.E0
        for tag in ("0001", "0002"):
            ds = xr.load_dataset(binary / f"FLD/s1-line-x2-{tag}.nc", engine="h5netcdf")
            da = ds["s1"]
            assert tuple(da.dims) == ("t", "x2")
            assert da.sizes["x2"] == NX
            # slab = 2 full wavelengths of the deck laser (8 cells from 2 lam0)
            start, stop = (int(v) for v in np.asarray(da.attrs["x1_slab_cells"]))
            assert stop - start == 8
            # vacuum traveling wave: <e2 b3> over full periods = a^2/2 exactly
            np.testing.assert_allclose(da.values, a_code**2 / 2.0, rtol=2e-4)
        # entrance slab starts at the guard, exit slab ends guard cells short
        ds1 = xr.load_dataset(binary / "FLD/s1-line-x2-0001.nc", engine="h5netcdf")
        assert int(np.asarray(ds1["s1"].attrs["x1_slab_cells"])[0]) == 2


def test_2d_step_estimate_uses_yee_cfl(tmp_path: Path) -> None:
    from adept.warpx.base import BaseWarpX

    deck = tmp_path / "inputs"
    deck.write_text(
        "stop_time = 1.0e-12\n"
        "amr.n_cell = 342 3648\n"
        "geometry.dims = 2\n"
        "geometry.prob_lo = 0.0 0.0\n"
        "geometry.prob_hi = 3.0020983112572184e-06 3.202256819802708e-05\n"
        "warpx.cfl = 0.9988192402161361\n"
    )
    derived = BaseWarpX({"warpx": {"deck": str(deck)}}).get_derived_quantities()
    # the srs-2d-follett-3um deck documents dt = 0.11098/omega0 -> 48,356
    # steps per ps; the 1D relation under-counted by sqrt(2)
    assert derived["num_steps_est"] == pytest.approx(48356, abs=2)
