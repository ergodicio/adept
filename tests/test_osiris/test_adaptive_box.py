"""Tests for ``osiris.density`` adaptive box sizing.

When a manifest carries an ``osiris.density.gradient_scale_length``,
``BaseOsiris`` scales the simulation box so the deck's linear density ramp
realizes that gradient scale length at the reference density (default the
quarter-critical surface). This mirrors adept's ``_lpse2d`` / ``kinetic_srs``
grid sizing: ``ramp_span = L / n_ref * (nmax - nmin)``.

The reference deck ``srs-1d_lpi`` ramps n: 0.225 -> 0.275 (n_c units) across
x in (1e-5, 1076.03) c/wp0, box xmax = 1076.04, nx_p = 6000, with n0 = 9.05e21
(critical density for 351 nm, so the skin depth c/wp0 = 55.86 nm).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from adept.osiris import BaseOsiris
from adept.osiris import deck as osd
from adept.osiris import density as den

DECKS_DIR = Path(__file__).parent / "decks"
SRS_DECK = DECKS_DIR / "srs-1d_lpi"
TWO_D_DECK = DECKS_DIR / "F-Tsung_2d_lpi_deck"

DECK_DX = 1076.04 / 6000  # cell size of the reference deck (c/wp0)


def _value(sections, name, base):
    """First value in section ``name`` whose key base name is ``base``."""
    for sec_name, params in sections:
        if sec_name == name:
            for k, v in params.items():
                if k == base or k.split("(", 1)[0] == base:
                    return v
    raise KeyError(f"{name}.{base} not found")


def _scaled(L="300um", **extra):
    sections = osd.parse_deck_file(SRS_DECK)
    computed = den.apply_gradient_scale_length(sections, {"gradient_scale_length": L, **extra})
    return sections, computed


def test_inactive_without_density_block() -> None:
    sections = osd.parse_deck_file(SRS_DECK)
    before = copy.deepcopy(sections)
    assert den.apply_gradient_scale_length(sections, None) is None
    assert den.apply_gradient_scale_length(sections, {}) is None
    # no gradient scale length -> still a no-op
    assert den.apply_gradient_scale_length(sections, {"min": 0.2}) is None
    assert sections == before, "deck must be untouched when the feature is inactive"


def test_box_scaled_to_requested_scale_length() -> None:
    sections, computed = _scaled("300um")
    # ramp_span = L/0.25 * (0.275-0.225); L=300um at c/wp0=55.86nm -> ~5371 c/wp0
    assert _value(sections, "space", "xmax") == pytest.approx(1074.11, rel=1e-4)
    assert _value(sections, "space", "xmin") == pytest.approx(0.0)
    assert computed["box_norm"] == pytest.approx(1074.11, rel=1e-4)
    assert computed["density_min"] == pytest.approx(0.225)
    assert computed["density_max"] == pytest.approx(0.275)
    assert computed["reference_density"] == pytest.approx(0.25)


def test_constant_dx_nx_scales_and_divides_node_count() -> None:
    sections, computed = _scaled("300um")
    nx = _value(sections, "grid", "nx_p")
    box = _value(sections, "space", "xmax")
    # cell size is held fixed (constant dx) to within node-count rounding
    assert box / nx == pytest.approx(DECK_DX, rel=5e-3)
    # node_number(1) = 4 in the deck -> nx_p must split evenly across 4 nodes
    assert nx % 4 == 0
    assert computed["nx"] == nx


def test_diagnostic_window_and_profile_edges_track_box() -> None:
    sections, _ = _scaled("300um")
    box = _value(sections, "space", "xmax")
    # phase-space diagnostic window follows the box
    assert _value(sections, "diag_species", "ps_xmax") == pytest.approx(box)
    assert _value(sections, "diag_species", "ps_xmin") == pytest.approx(0.0)
    # profile control points (incl. the tiny vacuum edges) scale with the box
    prof_x = _value(sections, "profile", "x")
    assert prof_x[0] == pytest.approx(0.0)
    assert prof_x[-1] == pytest.approx(box)
    # density values are untouched (no min/max override given)
    assert _value(sections, "profile", "fx") == [0.0, 0.225, 0.275, 0.0]


def test_box_scales_linearly_with_scale_length() -> None:
    _, half = _scaled("150um")
    _, base = _scaled("300um")
    _, dbl = _scaled("600um")
    assert half["box_norm"] == pytest.approx(base["box_norm"] / 2, rel=1e-6)
    assert dbl["box_norm"] == pytest.approx(base["box_norm"] * 2, rel=1e-6)


def test_round_trips_the_deck_geometry() -> None:
    # The reference deck encodes L_n = 0.25 * 1076.03 / 0.05 c/wp0 = 300.54 um;
    # requesting that L must reproduce the deck's box.
    sections, computed = _scaled("300.54um")
    assert _value(sections, "space", "xmax") == pytest.approx(1076.04, rel=1e-4)
    assert computed["scale_factor"] == pytest.approx(1.0, rel=2e-3)


def test_numeric_scale_length_is_treated_as_normalized() -> None:
    # A bare number is taken to be already in c/wp0 units (no n0 conversion).
    _, computed = _scaled(500.0)
    assert computed["gradient_scale_length_norm"] == pytest.approx(500.0)
    # ramp_span = 500/0.25 * 0.05 = 100 c/wp0
    assert computed["ramp_span_norm"] == pytest.approx(100.0)


def test_min_max_override_widens_box_and_rewrites_fx() -> None:
    sections, computed = _scaled("300um", min=0.2, max=0.3)
    # (nmax-nmin) doubled from 0.05 to 0.1 -> ramp span (and box) doubles
    _, base = _scaled("300um")
    assert computed["box_norm"] == pytest.approx(base["box_norm"] * 2, rel=1e-6)
    # the new densities are written into the ramp endpoints
    assert _value(sections, "profile", "fx") == [0.0, 0.2, 0.3, 0.0]


def test_custom_reference_density() -> None:
    # ramp_span is inversely proportional to n_ref
    _, qc = _scaled("300um", reference_density=0.25)
    _, half = _scaled("300um", reference_density=0.125)
    assert half["box_norm"] == pytest.approx(qc["box_norm"] * 2, rel=1e-6)


def test_multidimensional_deck_raises() -> None:
    sections = osd.parse_deck_file(TWO_D_DECK)
    with pytest.raises(NotImplementedError, match="1D decks only"):
        den.apply_gradient_scale_length(sections, {"gradient_scale_length": "300um"})


def test_physical_length_without_reference_density_raises(tmp_path: Path) -> None:
    deck = tmp_path / "no_n0"
    deck.write_text(
        "grid { nx_p = 100, }\n"
        "space { xmin(1) = 0.0, xmax(1) = 100.0, }\n"
        "profile { num_x = 4, fx(1:4,1) = 0.0, 0.2, 0.3, 0.0, x(1:4,1) = 0, 1, 99, 100, }\n"
    )
    sections = osd.parse_deck_file(deck)
    with pytest.raises(ValueError, match="neither n0 nor"):
        den.apply_gradient_scale_length(sections, {"gradient_scale_length": "300um"})


def test_baseosiris_applies_box_and_records_derived() -> None:
    cfg = {"osiris": {"deck": str(SRS_DECK), "density": {"gradient_scale_length": "300um"}}}
    module = BaseOsiris(cfg)
    # box was scaled on the live sections that __call__ will render
    assert _value(module._sections, "space", "xmax") == pytest.approx(1074.11, rel=1e-4)
    # computed quantities are stashed back for MLflow provenance
    derived = cfg["osiris"]["density"]["derived"]
    assert derived["box_norm"] == pytest.approx(1074.11, rel=1e-4)
    assert derived["nx"] == _value(module._sections, "grid", "nx_p")
    # write_units sees the scaled box
    quants = module.write_units()
    assert quants["box_length"].to("micron").magnitude == pytest.approx(60.0, rel=1e-2)


def test_baseosiris_unchanged_without_density() -> None:
    cfg = {"osiris": {"deck": str(SRS_DECK)}}
    module = BaseOsiris(cfg)
    assert _value(module._sections, "space", "xmax") == pytest.approx(1076.04)
    assert "density" not in cfg["osiris"]
