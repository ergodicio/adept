"""Tests for ``BaseOsiris.write_units`` — the OSIRIS ``units.yaml`` artifact.

OSIRIS normalizes time to ``1/wp0``, length to the skin depth ``c/wp0``, and
velocity to ``c``, all set by the reference density ``simulation.n0`` declared
in the deck. ``write_units`` derives the same canonical scales the other adept
solvers log so OSIRIS runs are comparable in MLflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adept.osiris import BaseOsiris

DECKS_DIR = Path(__file__).parent / "decks"
SRS_DECK = DECKS_DIR / "srs-1d_lpi"  # has simulation{n0=9.05e21}, xmax=1076.04, tmax=25000


def test_write_units_density_derived_scales() -> None:
    quants = BaseOsiris({"osiris": {"deck": str(SRS_DECK)}}).write_units()

    # wp0 and n0 depend only on the reference density and match the
    # corresponding kinetic-srs/adept run with the same n0.
    assert quants["n0"].to("1/cc").magnitude == pytest.approx(9.05e21, rel=1e-9)
    assert quants["wp0"].to("rad/s").magnitude == pytest.approx(5.3668e15, rel=1e-3)
    # OSIRIS length unit is the skin depth c/wp0, not the Debye length.
    assert quants["x0"].to("nm").magnitude == pytest.approx(55.86, rel=1e-3)
    # Velocity is normalized to c, so c_light and beta are both unity.
    assert quants["c_light"] == pytest.approx(1.0)
    assert quants["beta"] == pytest.approx(1.0)
    assert quants["v0"].to("m/s").magnitude == pytest.approx(299792458.0, rel=1e-6)
    # Geometry: box_length = (xmax - xmin) * skin depth, duration = tmax / wp0.
    assert quants["box_length"].to("micron").magnitude == pytest.approx(60.108, rel=1e-3)
    assert quants["sim_duration"].to("ps").magnitude == pytest.approx(4.6583, rel=1e-3)


def test_write_units_omits_temperature_keys() -> None:
    quants = BaseOsiris({"osiris": {"deck": str(SRS_DECK)}}).write_units()
    # OSIRIS has no single global temperature, so temperature-dependent
    # quantities are not defined and must not appear.
    for key in ("T0", "nuee", "logLambda_ee"):
        assert key not in quants


def test_write_units_from_reference_frequency(tmp_path: Path) -> None:
    # omega_p0 = the plasma frequency of n0 = 9.05e21 cm^-3, so the frequency
    # form must reproduce the density form's scales (and recover n0).
    deck = tmp_path / "omega_p0_deck"
    deck.write_text(
        "simulation { omega_p0 = 5.3668e15, }\n"
        "space { xmin(1) = 0.0, xmax(1) = 1076.04, }\n"
        "time { tmin = 0.0, tmax = 25000.0, }\n"
    )
    quants = BaseOsiris({"osiris": {"deck": str(deck)}}).write_units()
    assert quants["wp0"].to("rad/s").magnitude == pytest.approx(5.3668e15, rel=1e-4)
    assert quants["n0"].to("1/cc").magnitude == pytest.approx(9.05e21, rel=1e-3)
    assert quants["x0"].to("nm").magnitude == pytest.approx(55.86, rel=1e-3)
    assert quants["box_length"].to("micron").magnitude == pytest.approx(60.108, rel=1e-3)


def test_write_units_density_takes_precedence_over_frequency(tmp_path: Path) -> None:
    # OSIRIS uses n0 when both are present; omega_p0 here is deliberately wrong.
    deck = tmp_path / "both_deck"
    deck.write_text("simulation { n0 = 9.05e21, omega_p0 = 1.0e14, }\n")
    quants = BaseOsiris({"osiris": {"deck": str(deck)}}).write_units()
    assert quants["wp0"].to("rad/s").magnitude == pytest.approx(5.3668e15, rel=1e-3)


def test_write_units_returns_empty_without_reference_density(tmp_path: Path) -> None:
    deck = tmp_path / "no_n0"
    deck.write_text("grid { nx_p(1:1) = 64, }\n")
    assert BaseOsiris({"osiris": {"deck": str(deck)}}).write_units() == {}
