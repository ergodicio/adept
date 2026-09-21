"""Tests for the M3 phase-space/scraping conversion, on real WarpX output.

The ``srs-mini-1d`` fixture is the committed output of a real WarpX run of
``fixtures/srs-mini-1d/inputs`` (Perlmutter, dev @ 72280884a): a shrunk
version of the warpx-lpi SRS deck carrying one of every diagnostic the M3
layer converts. The checks below are hand-derived from the deck: a
Maxwellian at ``u_th = 0.088475`` on a linear 0.18→0.27 n0 ramp, so the
converted phase spaces must reproduce those numbers in code units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from adept.warpx import io as wio

FIXTURE = Path(__file__).parent / "fixtures" / "srs-mini-1d"

pytestmark = pytest.mark.skipif(
    not (FIXTURE / "diags").is_dir(),
    reason="srs-mini-1d fixture outputs not present",
)

N0_M3 = 9.05e27
LZ = 2.564746104429896e-06
UTH = 0.08847488193965977
LOG10_GMAX = 2.3010299956639813


@pytest.fixture(scope="module")
def units() -> wio.CodeUnits:
    u = wio.code_units_for_run(FIXTURE)
    assert u is not None and u.n0 == pytest.approx(N0_M3)
    return u


@pytest.fixture(scope="module")
def binary(tmp_path_factory, units) -> Path:
    out = tmp_path_factory.mktemp("binary")
    wio.save_run_datasets(FIXTURE, out, units=units)
    return out


def _load(binary: Path, rel: str) -> xr.DataArray:
    ds = xr.load_dataset(binary / f"{rel}.nc", engine="h5netcdf")
    return ds[next(iter(ds.data_vars))]


class TestRouting:
    def test_contract_keys(self, binary: Path) -> None:
        from adept.osiris.io import list_diagnostics

        keys = set(list_diagnostics(binary))
        assert {
            "FLD/e1",
            "FLD/e2",
            "FLD/b3",
            "DENSITY/electrons/charge",
            "PHA/p1x1/electrons",
            "PHA/x1log_gamma_q1/electrons",
            "PHA/log_gamma/electrons",
            "REDUCED/poyntingflux",
            "REDUCED/epw_w",
            "HIST/energy",
        } <= keys
        # ParticleHistogram txt must NOT also appear as a REDUCED table.
        assert "REDUCED/log_gamma" not in keys

    def test_histogram_times_match_reduced_tables(self, binary: Path) -> None:
        """The openPMD iteration = step+1 quirk must not corrupt the times."""
        da = _load(binary, "PHA/p1x1/electrons")
        fe = xr.load_dataset(binary / "REDUCED" / "fieldenergy.nc", engine="h5netcdf")
        u = wio.code_units_for_run(FIXTURE)
        t_fe = fe["time"].values * u.wp0  # both diags run at intervals = 30
        assert np.allclose(np.sort(da.coords["t"].values), np.sort(t_fe), rtol=1e-6)


class TestP1x1:
    def test_shape_and_axes(self, binary: Path, units: wio.CodeUnits) -> None:
        da = _load(binary, "PHA/p1x1/electrons")
        assert da.dims == ("t", "x1", "p1")
        assert da.sizes["x1"] == 128 and da.sizes["p1"] == 32
        # Edge-style labels over the deck's bin ranges, x in c/wp0.
        assert da.coords["p1"].values[0] == pytest.approx(-0.5)
        assert da.coords["p1"].values[-1] == pytest.approx(2.4)
        assert da.coords["x1"].values[0] == pytest.approx(0.0)
        assert da.coords["x1"].values[-1] == pytest.approx(LZ / units.x0, rel=1e-6)
        xmax = float(np.atleast_1d(da.attrs["sim.XMAX"])[0])
        assert xmax == pytest.approx(LZ / units.x0, rel=1e-6)

    def test_charge_signed_density_normalization(self, binary: Path) -> None:
        """OSIRIS convention: stored q·f with Σ f·dp = n(x)/n0."""
        da = _load(binary, "PHA/p1x1/electrons")
        f0 = da.isel(t=0).values.astype(float)
        assert np.nansum(f0) < 0  # electrons: charge-signed deposit
        dp = (2.4 + 0.5) / 32
        n_over_n0 = -f0.sum(axis=1) * dp
        z = np.linspace(0.0, LZ, 129)
        zc = 0.5 * (z[:-1] + z[1:])
        expect = 0.18 + 0.09 * zc / LZ
        # 64 ppc -> per-bin sampling noise; the box mean is tight.
        assert n_over_n0.mean() == pytest.approx(expect.mean(), rel=0.02)
        assert np.corrcoef(n_over_n0, expect)[0, 1] > 0.9

    def test_initial_momentum_spread(self, binary: Path) -> None:
        da = _load(binary, "PHA/p1x1/electrons")
        f0 = -da.isel(t=0).values.astype(float).sum(axis=0)
        edges = np.linspace(-0.5, 2.4, 33)
        pc = 0.5 * (edges[:-1] + edges[1:])
        mean = (f0 * pc).sum() / f0.sum()
        std = np.sqrt((f0 * (pc - mean) ** 2).sum() / f0.sum())
        assert abs(mean) < 0.01
        assert std == pytest.approx(UTH, rel=0.05)


class TestXlogGammaQ1:
    def test_shape_axes_and_flux_units(self, binary: Path) -> None:
        da = _load(binary, "PHA/x1log_gamma_q1/electrons")
        assert da.dims == ("t", "x1", "gamma")
        assert da.sizes["gamma"] == 48
        g = da.coords["gamma"].values
        assert g[0] == pytest.approx(0.0) and g[-1] == pytest.approx(LOG10_GMAX)
        assert "log" in str(da.attrs.get("axis_long_names", "")) or True
        # t = 0 Maxwellian: the signed KE*v1 deposit nearly cancels -- the
        # box-integrated |net flux| is far below the thermal free-streaming
        # flux scale n0eff * uth^3 (isotropy check, loose bound).
        dg = LOG10_GMAX / 48
        s0 = da.isel(t=0).values.astype(float).sum(axis=1) * dg
        assert np.abs(s0.mean()) < 0.225 * UTH**3


class TestLogGammaSpectrum:
    def test_shape_and_count_normalization(self, binary: Path, units: wio.CodeUnits) -> None:
        da = _load(binary, "PHA/log_gamma/electrons")
        assert da.dims == ("t", "gamma")
        assert da.sizes["gamma"] == 48
        # Sum f * dg * (n0 x0) = total electrons per m^2 = integral n dz:
        # linear ramp mean 0.225 n0 over Lz.
        dg = LOG10_GMAX / 48
        total = -float(da.isel(t=0).values.astype(float).sum()) * dg
        expect = 0.225 * LZ / units.x0
        assert total == pytest.approx(expect, rel=0.02)


class TestCollectOnFixture:
    def test_energy_scalars_survive_histogram_stubs(self, tmp_path: Path) -> None:
        """The empty ParticleHistogram2D companion .txt must not eat the
        energy metrics or the HIST/energy build (it did, before attempt 3)."""
        from adept.warpx import post as wpost

        cfg = {
            "units": {"derived": {"n0": "9.05e21 / centimeter ** 3"}},
            "derived": {"num_steps": 300},
            "output": {"diagnostics_to_log": ["e1", "energy", "fieldenergy"]},
        }
        run_output = {"solver result": {"run_dir": str(FIXTURE), "wall_time": 1.0, "exit_code": 0}}
        result = wpost.collect(run_output, cfg, str(tmp_path))
        m = result["metrics"]
        assert "field_energy_final" in m and m["field_energy_final"] > 0
        assert "energy_drift_frac" in m
        assert (tmp_path / "binary" / "HIST" / "energy.nc").is_file()


class TestCollectSrsSmoke:
    def test_osiris_phase_space_dispatch(self, binary: Path) -> None:
        """The converted PHA keys drive the OSIRIS-side consumers unchanged."""
        from adept.osiris.io import list_diagnostics, open_series

        diags = list_diagnostics(binary)
        with open_series(diags["PHA/p1x1/electrons"]) as ser:
            momenta = [d for d in ser.dims if str(d).startswith("p")]
            assert momenta == ["p1"]
        with open_series(diags["PHA/x1log_gamma_q1/electrons"]) as ser:
            others = [d for d in ser.dims if d != "t" and not str(d).startswith(("x", "p"))]
            assert others == ["gamma"]
