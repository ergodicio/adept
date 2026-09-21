"""Tests for ``BaseWarpX.write_units`` — the WarpX ``units.yaml`` derivation.

WarpX decks are SI and carry no reference density, so the normalization
(time in ``1/wp0``, length in ``c/wp0``, velocity in ``c``) comes from the
manifest's ``warpx.reference_density``, falling back to the deck's
``my_constants.n0`` (SI). The reference numbers below match the OSIRIS
``test_units`` suite for n0 = 9.05e21 cm^-3, so a WarpX run and an OSIRIS
run at the same physical density log identical scales to MLflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adept.warpx.base import BaseWarpX

DECKS_DIR = Path(__file__).parent / "decks"
SMOKE_DECK = DECKS_DIR / "warpx-1d-smoke"


def _base(**warpx_extra) -> BaseWarpX:
    return BaseWarpX({"warpx": {"deck": str(SMOKE_DECK), **warpx_extra}})


def test_write_units_from_manifest_reference_density() -> None:
    quants = _base(reference_density="9.05e21 /cc").write_units()

    assert quants["n0"].to("1/cc").magnitude == pytest.approx(9.05e21, rel=1e-9)
    assert quants["wp0"].to("rad/s").magnitude == pytest.approx(5.3668e15, rel=1e-3)
    # Length unit is the skin depth c/wp0.
    assert quants["x0"].to("nm").magnitude == pytest.approx(55.86, rel=1e-3)
    assert quants["c_light"] == pytest.approx(1.0)
    assert quants["v0"].to("m/s").magnitude == pytest.approx(299792458.0, rel=1e-6)
    # Geometry/time are SI in the deck: 60 um box, 2 ps stop_time.
    assert quants["box_length"].to("micron").magnitude == pytest.approx(60.0, rel=1e-9)
    assert quants["sim_duration"].to("ps").magnitude == pytest.approx(2.0, rel=1e-9)


def test_write_units_bare_number_is_per_cc() -> None:
    quants = _base(reference_density=9.05e21).write_units()
    assert quants["n0"].to("1/cc").magnitude == pytest.approx(9.05e21, rel=1e-9)


def test_write_units_my_constants_fallback_is_si() -> None:
    # The smoke deck declares my_constants.n0 = 9.05e27 m^-3 = 9.05e21 cm^-3.
    quants = _base().write_units()
    assert quants["n0"].to("1/cc").magnitude == pytest.approx(9.05e21, rel=1e-6)
    assert quants["wp0"].to("rad/s").magnitude == pytest.approx(5.3668e15, rel=1e-3)


def test_write_units_laser_in_icf_units() -> None:
    # laser1: wavelength 351 nm, e_max 3.66e10 V/m -> a0 = 0.004 and
    # I = eps0 c E^2 / 2 = 1.78e14 W/cm^2 (I lambda^2 = 1.37e18 a0^2).
    quants = _base(reference_density="9.05e21 /cc").write_units()
    assert quants["laser_wavelength"].to("nm").magnitude == pytest.approx(351.0, rel=1e-9)
    assert quants["w_laser"].to("rad/s").magnitude == pytest.approx(5.3673e15, rel=1e-3)
    assert quants["laser_a0"] == pytest.approx(0.004, rel=1e-3)
    assert quants["laser_intensity"].to("W/cm^2").magnitude == pytest.approx(1.778e14, rel=1e-2)


def test_derived_quantities_grid_and_steps() -> None:
    base = _base(reference_density="9.05e21 /cc")
    base.write_units()
    derived = base.get_derived_quantities()

    # 256 cells over 60 um.
    assert derived["dx"] == [pytest.approx(60.0e-6 / 256, rel=1e-9)]
    assert derived["num_steps"] == 100
    # dt_est = cfl * dx / c; num_steps_est = stop_time / dt_est.
    dt_est = 0.999 * (60.0e-6 / 256) / 299792458.0
    assert derived["dt_est"] == pytest.approx(dt_est, rel=1e-9)
    assert derived["num_steps_est"] == int(2.0e-12 / dt_est)
    # Resolution in skin depths uses the units derived in write_units.
    assert derived["dx_over_skin_depth"][0] == pytest.approx((60.0e-6 / 256) / 55.86e-9, rel=1e-3)


def test_overrides_apply_before_logging() -> None:
    base = _base(overrides={"amr.n_cell": 512})
    assert base.cfg["deck"]["amr.n_cell"] == 512
