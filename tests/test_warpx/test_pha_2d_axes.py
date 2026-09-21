"""P2 histogram-axes regression (dev_docs/p2_bugfix.md).

Two bugs surfaced on the first 2D run (exp 188989): a momentum abscissa
(``uz``) was hard-named ``x1`` and divided by ``x0`` — a fake spatial axis
spanning ±9e7 c/ω_p that the downstream box crop blanked — and
``sim.XMIN/XMAX`` were written in WarpX ``(x, z)`` deck order, so the
transverse extent cropped the z (x1) axis of p1x1 to 3 µm. These tests pin
the fixes with synthetic ParticleHistogram2D openPMD output: a ``(uz, ux)``
histogram must come out ``(t, p1, p2)`` with momentum coords and survive
``_crop_spatial_to_box`` unchanged; a 2D ``(z, uz)`` one must carry
x1-first box attrs.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from adept.warpx import io as wio

N0_REF = 9.05e27
LX, LZ = 3.0e-6, 3.2e-05  # 2D box, WarpX (x, z)
N_ABS, N_ORD = 8, 6
ITS = (0, 600)


def _write_hist2d(diag_dir: Path, abs_range, ord_range) -> None:
    """Minimal ParticleHistogram2D openPMD series (WarpX (ord, abs) layout)."""
    diag_dir.mkdir(parents=True)
    for it in ITS:
        with h5py.File(diag_dir / f"openpmd_{it:06d}.h5", "w") as f:
            f.attrs["meshesPath"] = np.bytes_("meshes/")
            g = f.create_group(f"data/{it}")
            g.attrs["time"] = float(it) * 1.0e-17
            g.attrs["timeUnitSI"] = 1.0
            meshes = g.create_group("meshes")
            d = meshes.create_dataset("data", data=np.full((N_ORD, N_ABS), 2.0))
            d.attrs["unitSI"] = 1.0
            d.attrs["gridSpacing"] = np.array(
                [
                    (ord_range[1] - ord_range[0]) / N_ORD,
                    (abs_range[1] - abs_range[0]) / N_ABS,
                ]
            )
            d.attrs["gridGlobalOffset"] = np.array([ord_range[0], abs_range[0]])
            d.attrs["gridUnitSI"] = 1.0


def _deck(name: str, fn_abs: str, fn_ord: str, abs_range, ord_range) -> dict:
    return {
        f"{name}.histogram_function_abs": fn_abs,
        f"{name}.histogram_function_ord": fn_ord,
        f"{name}.value_function": "w",
        f"{name}.species": "electrons",
        f"{name}.bin_number_abs": N_ABS,
        f"{name}.bin_number_ord": N_ORD,
        f"{name}.bin_min_abs": abs_range[0],
        f"{name}.bin_max_abs": abs_range[1],
        f"{name}.bin_min_ord": ord_range[0],
        f"{name}.bin_max_ord": ord_range[1],
        "electrons.charge": "-q_e",
        "geometry.prob_lo": [0.0, 0.0],
        "geometry.prob_hi": [LX, LZ],
        "amr.n_cell": [342, 3648],
    }


@pytest.fixture(scope="module")
def units() -> wio.CodeUnits:
    return wio.CodeUnits(n0=N0_REF)


class TestMomentumAbscissa:
    def _load(self, tmp_path: Path, units) -> object:
        diag = tmp_path / "p1p2"
        _write_hist2d(diag, (-5.0, 5.0), (-5.0, 5.0))
        deck = _deck("p1p2", "uz", "ux", (-5.0, 5.0), (-5.0, 5.0))
        return wio.load_particle_histogram2d(diag, units=units, deck=deck)

    def test_dims_and_momentum_coords(self, tmp_path: Path, units) -> None:
        da = self._load(tmp_path, units)
        assert da.dims == ("t", "p1", "p2")
        # momentum bins stay in m_e c — no x0 division (the ±9e7 bug)
        np.testing.assert_allclose(float(da.coords["p1"].min()), -5.0)
        np.testing.assert_allclose(float(da.coords["p1"].max()), 5.0)
        assert da.attrs["axis_units"]["p1"] == r"m_e c"
        assert da.attrs["axis_long_names"]["p1"] == "p_1"

    def test_survives_spatial_box_crop(self, tmp_path: Path, units) -> None:
        from adept.osiris.plots import _crop_spatial_to_box

        da = self._load(tmp_path, units)
        cropped = _crop_spatial_to_box(da)
        assert cropped.sizes == da.sizes  # momentum dims pass untouched
        np.testing.assert_array_equal(cropped.values, da.values)


class TestSpatialAbscissa2D:
    def _load(self, tmp_path: Path, units) -> object:
        diag = tmp_path / "p1x1"
        _write_hist2d(diag, (0.0, LZ), (-5.0, 5.0))
        deck = _deck("p1x1", "z", "uz", (0.0, LZ), (-5.0, 5.0))
        return wio.load_particle_histogram2d(diag, units=units, deck=deck)

    def test_dims_and_box_attrs_x1_first(self, tmp_path: Path, units) -> None:
        da = self._load(tmp_path, units)
        assert da.dims == ("t", "x1", "p1")
        assert da.attrs["sim.NDIMS"] == 2
        # x1-first: slot 0 is the z (propagation) extent, slot 1 transverse
        np.testing.assert_allclose(da.attrs["sim.XMAX"][0], LZ / units.x0, rtol=1e-12)
        np.testing.assert_allclose(da.attrs["sim.XMAX"][1], LX / units.x0, rtol=1e-12)
        # the spatial axis is converted and spans the full box
        np.testing.assert_allclose(float(da.coords["x1"].max()), LZ / units.x0, rtol=1e-12)

    def test_crop_keeps_full_z_extent(self, tmp_path: Path, units) -> None:
        from adept.osiris.plots import _crop_spatial_to_box

        da = self._load(tmp_path, units)
        cropped = _crop_spatial_to_box(da)
        # with x1-first attrs the crop window is the z box: nothing removed
        assert cropped.sizes["x1"] == da.sizes["x1"]
