"""Tests for ``adept.warpx.io`` — openPMD/reduced-diag loading and the
OSIRIS ``binary/`` NetCDF contract.

The fixtures under ``fixtures/smoke-1d/`` are the *real* output of the M1
live smoke run on Perlmutter (WarpX dev @ 72280884a, job 56960877): three
openPMD h5 dumps (steps 0/50/100) of E/B/j plus electrons, and the
FieldEnergy / ParticleEnergy reduced tables. The deck: 256 cells over
60 um, n0 = 9.05e27 m^-3 (critical density at 351 nm), a0 ~ 0.004 laser,
4 keV electrons (ux_std = 0.0885), 64 ppc.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from adept.warpx import io as wio

FIXTURE_RUN = Path(__file__).parent / "fixtures" / "smoke-1d"
DIAG_DIR = FIXTURE_RUN / "diags" / "diag1"

N0_CC = 9.05e21
UNITS = wio.CodeUnits.from_density_cc(N0_CC)

# Independent hand-computed scales for n0 = 9.05e21 /cc.
WP0 = 5.3668e15  # rad/s
X0_M = 55.86e-9  # c/wp0
E0 = 9.1478e12  # m_e c wp0 / e, V/m
U_AREA0 = 4.139e7  # m_e c^2 n0 x0, J/m^2


class TestCodeUnits:
    def test_derived_scales(self) -> None:
        assert UNITS.wp0 == pytest.approx(WP0, rel=1e-3)
        assert UNITS.x0 == pytest.approx(X0_M, rel=1e-3)
        assert UNITS.t0 == pytest.approx(1.0 / WP0, rel=1e-3)
        assert UNITS.E0 == pytest.approx(E0, rel=1e-3)
        assert UNITS.B0 == pytest.approx(E0 / 299792458.0, rel=1e-3)
        assert UNITS.j0 == pytest.approx(1.602176634e-19 * 9.05e27 * 299792458.0, rel=1e-9)
        assert UNITS.u_area0 == pytest.approx(U_AREA0, rel=1e-3)

    def test_from_cfg_pint_and_string_and_bare(self) -> None:
        from adept.normalization import UREG

        for n0 in (UREG.Quantity(9.05e21, "1/cc"), "9.05e21 / cc", 9.05e21):
            cu = wio.CodeUnits.from_cfg({"units": {"derived": {"n0": n0}}})
            assert cu is not None
            assert cu.wp0 == pytest.approx(WP0, rel=1e-3)
        assert wio.CodeUnits.from_cfg({}) is None

    def test_code_units_for_run_from_inputs_deck(self) -> None:
        # The fixture run dir has no units.yaml/config.yaml, so the rendered
        # deck's my_constants.n0 (SI m^-3) is the fallback.
        cu = wio.code_units_for_run(FIXTURE_RUN)
        assert cu is not None
        assert cu.n0 == pytest.approx(9.05e27, rel=1e-9)


class TestFieldSeries:
    def test_listing(self) -> None:
        recs = wio.list_field_records(DIAG_DIR)
        assert ("E", "x") in recs and ("B", "z") in recs and ("j", "y") in recs
        assert wio.list_species(DIAG_DIR) == ["electrons"]

    def test_osiris_mapping_and_code_units(self) -> None:
        # E/x is the laser polarization -> e2 under the (z,x,y)->(1,2,3) map.
        e2 = wio.load_field_series(DIAG_DIR, "E", "x", units=UNITS)
        assert e2.name == "e2"
        assert e2.dims == ("t", "x1")
        assert e2.shape == (3, 256)
        np.testing.assert_array_equal(e2.coords["iter"].values, [0, 50, 100])
        # Laser field in code units is O(a0) = 0.004.
        assert 0.001 < float(abs(e2).max()) < 0.1
        assert e2.attrs["units"] == r"m_e c \omega_p e^{-1}"
        assert e2.attrs["axis_units"] == {"x1": r"c / \omega_p"}
        assert e2.attrs["warpx_record"] == "E/x"

    def test_time_axis_in_wp_units(self) -> None:
        e1 = wio.load_field_series(DIAG_DIR, "E", "z", units=UNITS)
        assert e1.name == "e1"
        t = e1.coords["t"].values
        assert t[0] == 0.0
        # 100 steps at cfl=0.999: dt = 0.999 * dz / c, dz = 60um/256.
        dt_si = 0.999 * (60e-6 / 256) / 299792458.0
        assert t[2] == pytest.approx(100 * dt_si * UNITS.wp0, rel=1e-6)

    def test_spatial_axis_and_box(self) -> None:
        e2 = wio.load_field_series(DIAG_DIR, "E", "x", units=UNITS)
        x1 = e2.coords["x1"].values
        dz_code = (60e-6 / 256) / UNITS.x0
        # Cell-centered staggering: first node at dz/2.
        assert x1[0] == pytest.approx(0.5 * dz_code, rel=1e-6)
        assert float(np.diff(x1).mean()) == pytest.approx(dz_code, rel=1e-6)
        assert e2.attrs["sim.XMAX"][0] == pytest.approx(x1[-1], rel=1e-9)

    def test_si_passthrough_without_units(self) -> None:
        e2 = wio.load_field_series(DIAG_DIR, "E", "x", units=None)
        assert e2.attrs["units"] == "SI"
        assert e2.attrs["time_units"] == "s"
        # V/m-scale field, seconds-scale time, meters-scale axis.
        assert float(abs(e2).max()) > 1e8
        assert e2.coords["x1"].values[-1] == pytest.approx(60e-6, rel=1e-2)

    def test_conversion_scale_consistency(self) -> None:
        si = wio.load_field_series(DIAG_DIR, "B", "y", units=None)
        code = wio.load_field_series(DIAG_DIR, "B", "y", units=UNITS)
        assert code.name == "b3"
        np.testing.assert_allclose(code.values, si.values / np.float32(UNITS.B0), rtol=1e-5)


class TestParticles:
    def test_raw_long_form(self) -> None:
        raw = wio.load_particle_species(DIAG_DIR, "electrons", units=UNITS)
        assert raw.attrs["n_dumps"] == 3
        for v in ("x1", "p1", "p2", "p3", "ene", "w", "q"):
            assert v in raw.data_vars
        # Per-row time/iter coords identify each dump's particles.
        assert set(np.unique(raw.coords["iter"].values)) == {0, 50, 100}

    def test_thermal_spread_matches_deck(self) -> None:
        raw = wio.load_particle_species(DIAG_DIR, "electrons", units=UNITS)
        t0 = raw.where(raw.coords["iter"] == 0, drop=True)
        # Deck: maxwellian with ux/uy/uz_std = 0.0885 (4 keV).
        for p in ("p1", "p2", "p3"):
            assert float(t0[p].std()) == pytest.approx(0.0885, rel=0.02)

    def test_macroparticle_charge_convention(self) -> None:
        raw = wio.load_particle_species(DIAG_DIR, "electrons", units=UNITS)
        t0 = raw.where(raw.coords["iter"] == 0, drop=True)
        # Depositing q over the grid recovers the mean density in e*n0 units:
        # sum(q) / L_code = -<n>/n0 with n = 0.1 n0 exp(z/Ln) over 60 um,
        # Ln = 100 um -> <n>/n0 = 0.1 * (e^0.6 - 1)/0.6.
        box_code = 60e-6 / UNITS.x0
        mean_frac = 0.1 * (math.exp(0.6) - 1.0) / 0.6
        assert float(t0.q.sum()) / box_code == pytest.approx(-mean_frac, rel=0.01)

    def test_kinetic_energy_variable(self) -> None:
        raw = wio.load_particle_species(DIAG_DIR, "electrons", units=UNITS)
        ene = raw.ene.values
        assert (ene >= 0).all()
        # Thermal gamma - 1 ~ u^2/2 ~ 3 * 0.0885^2 / 2 ~ 0.012.
        assert float(np.mean(ene)) == pytest.approx(1.5 * 0.0885**2, rel=0.1)


class TestReducedDiags:
    def test_parse_header_and_columns(self) -> None:
        ds = wio.parse_reduced_diag(FIXTURE_RUN / "diags" / "reducedfiles" / "fieldenergy.txt")
        assert set(ds.data_vars) == {"time", "total_lev0", "E_lev0", "B_lev0"}
        assert ds["total_lev0"].attrs["units"] == "J"
        assert ds["time"].attrs["units"] == "s"
        np.testing.assert_array_equal(ds.coords["step"].values[:3], [0, 10, 20])
        # Row 1 of the fixture table.
        assert float(ds["total_lev0"].values[1]) == pytest.approx(6.03634926259876e05, rel=1e-12)

    def test_listing(self) -> None:
        assert set(wio.list_reduced_diags(FIXTURE_RUN)) == {"fieldenergy", "particleenergy"}

    def test_hist_energy_schema_and_units(self) -> None:
        hist = wio.hist_energy_from_reduced(FIXTURE_RUN, units=UNITS)
        assert hist is not None
        assert set(hist.data_vars) == {"field_energy", "kinetic_electrons", "kinetic_total", "total"}
        assert "total_drift_frac" in hist.attrs
        # J/m^2 -> code units via u_area0; check the step-10 field energy.
        assert float(hist.field_energy.values[1]) == pytest.approx(6.03634926259876e05 / UNITS.u_area0, rel=1e-3)
        # t axis in 1/wp: step 10 at dt = 0.999 dz / c.
        dt_si = 0.999 * (60e-6 / 256) / 299792458.0
        assert float(hist.t.values[1]) == pytest.approx(10 * dt_si * UNITS.wp0, rel=1e-6)


class TestSaveRunDatasets:
    @pytest.fixture(scope="class")
    def binary_dir(self, tmp_path_factory) -> Path:
        out = tmp_path_factory.mktemp("binary")
        wio.save_run_datasets(FIXTURE_RUN, out, units=UNITS)
        return out

    def test_contract_keys(self, binary_dir: Path) -> None:
        from adept.osiris import io as oio

        diags = oio.list_diagnostics(binary_dir)
        expected = {f"FLD/{c}{i}" for c in "ebj" for i in (1, 2, 3)}
        expected |= {"RAW/electrons", "HIST/energy", "REDUCED/fieldenergy", "REDUCED/particleenergy"}
        assert set(diags) == expected

    def test_osiris_load_series_roundtrip(self, binary_dir: Path) -> None:
        from adept.osiris import io as oio

        direct = wio.load_field_series(DIAG_DIR, "E", "x", units=UNITS)
        loaded = oio.load_series(binary_dir / "FLD" / "e2.nc")
        assert loaded.dims == ("t", "x1")
        np.testing.assert_allclose(loaded.values, direct.values, rtol=1e-6)
        np.testing.assert_allclose(loaded.coords["t"].values, direct.coords["t"].values)
        assert loaded.attrs["axis_units"] == {"x1": r"c / \omega_p"}

    def test_hist_energy_readable_by_osiris_loader(self, binary_dir: Path) -> None:
        from adept.osiris import io as oio

        energy = oio.load_hist_energy(binary_dir)
        assert energy is not None
        assert "total" in energy and "total_drift_frac" in energy.attrs

    def test_diagnostics_whitelist(self, tmp_path: Path) -> None:
        wio.save_run_datasets(FIXTURE_RUN, tmp_path, units=UNITS, diagnostics={"FLD/e1", "HIST/energy"})
        rels = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.nc")}
        assert rels == {"FLD/e1.nc", "HIST/energy.nc"}
