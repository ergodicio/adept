"""Tests for the M2 post-processing tail: canned plots, ``collect``, regen.

These run the *real* pipeline over the committed smoke-run fixtures — the
same code path the live Perlmutter run takes after WarpX exits — minus
MLflow (``collect`` is pure: it writes into a temp dir and returns
metrics).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adept.warpx import io as wio
from adept.warpx import post as wpost

FIXTURE_RUN = Path(__file__).parent / "fixtures" / "smoke-1d"

UNITS = wio.CodeUnits.from_density_cc(9.05e21)


@pytest.fixture(scope="module")
def collected(tmp_path_factory) -> tuple[Path, dict]:
    td = tmp_path_factory.mktemp("td")
    cfg = {
        "units": {"derived": {"n0": "9.05e21 / centimeter ** 3"}},
        "derived": {"num_steps": 100},
        "output": {},
    }
    run_output = {"solver result": {"run_dir": str(FIXTURE_RUN), "wall_time": 0.57, "exit_code": 0}}
    result = wpost.collect(run_output, cfg, str(td))
    return td, result["metrics"]


class TestCollect:
    def test_metrics(self, collected) -> None:
        _td, metrics = collected
        assert metrics["exit_code"] == 0.0
        assert metrics["final_step"] == 100.0
        assert metrics["completed_steps_frac"] == pytest.approx(1.0)
        # Final FieldEnergy total (6.46983968862424e5 J/m^2 at step 100) in
        # code energy units.
        assert metrics["field_energy_final"] == pytest.approx(6.46983968862424e5 / UNITS.u_area0, rel=1e-3)
        assert 0 < metrics["bfield_energy_final"] < metrics["efield_energy_final"]
        assert metrics["energy_drift_frac"] == pytest.approx(0.0038, rel=0.1)

    def test_artifacts(self, collected) -> None:
        td, _metrics = collected
        for fname in ("inputs", "warpx_used_inputs", "stdout.log"):
            assert (td / fname).is_file()
        assert (td / "reducedfiles" / "fieldenergy.txt").is_file()
        # The binary/ NetCDF contract.
        assert (td / "binary" / "FLD" / "e1.nc").is_file()
        assert (td / "binary" / "HIST" / "energy.nc").is_file()
        assert (td / "binary" / "RAW" / "electrons.nc").is_file()

    def test_plots_rendered(self, collected) -> None:
        td, _metrics = collected
        plots = td / "plots"
        for rel in (
            "spacetime/e1.png",
            "spacetime/e2.png",
            "omega_k/e2.png",
            "field_decomp/e2.png",
            "energy_vs_time.png",
            "total_energy_vs_time.png",
            "reduced/fieldenergy.png",
        ):
            assert (plots / rel).is_file(), f"missing {rel}"


class TestRegen:
    def test_regen_from_binary_dir(self, collected, tmp_path: Path) -> None:
        from adept.warpx import regen as wregen

        td, _metrics = collected
        written = wregen.regenerate(td / "binary", out_dir=tmp_path)
        assert "spacetime/e2" in written
        assert "reduced/fieldenergy" in written
        assert all(p.is_file() for p in written.values())

    def test_regen_from_raw_run_dir(self, tmp_path: Path) -> None:
        """A raw WarpX run dir (diags/, no binary/) converts then plots.

        The normalization comes off the rendered ``inputs`` deck
        (``my_constants.n0``), exercising :func:`io.code_units_for_run`.
        """
        import shutil

        from adept.warpx import regen as wregen

        run = tmp_path / "run"
        shutil.copytree(FIXTURE_RUN, run)
        written = wregen.regenerate(run, out_dir=tmp_path / "plots")
        assert (run / "binary" / "FLD" / "e1.nc").is_file()
        assert "spacetime/e1" in written
