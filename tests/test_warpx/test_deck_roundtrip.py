"""Tests for the WarpX ParmParse deck parser/renderer.

The invariant the wrapper relies on is dict-level round-trip:
``parse(render(parse(text))) == parse(text)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adept.warpx import deck as _deck

DECKS_DIR = Path(__file__).parent / "decks"
SMOKE_DECK = DECKS_DIR / "warpx-1d-smoke"


def test_roundtrip_smoke_deck() -> None:
    parsed = _deck.parse_deck_file(SMOKE_DECK)
    assert parsed == _deck.parse_deck(_deck.render_deck(parsed))
    # Spot-check types: ints stay ints, floats floats, expressions strings.
    assert parsed["max_step"] == 100
    assert parsed["geometry.prob_hi"] == 60.0e-6
    assert parsed["electrons.density_function(x,y,z)"] == "0.1*n0*exp(z/Ln)"
    assert parsed["electrons.charge"] == "-q_e"


def test_multiline_value_list() -> None:
    # AMReX tokenizes rather than parsing line-by-line: a value list runs
    # until the next `key =` definition, so it may span lines.
    parsed = _deck.parse_deck("amr.n_cell = 64\n    64\nmax_step = 10\n")
    assert parsed == {"amr.n_cell": [64, 64], "max_step": 10}


def test_quoted_expression_with_spaces_survives() -> None:
    text = 'e.density_function(x,y,z) = "n0 * exp(z / Ln)"\nmax_step = 1\n'
    parsed = _deck.parse_deck(text)
    assert parsed["e.density_function(x,y,z)"] == "n0 * exp(z / Ln)"
    assert parsed == _deck.parse_deck(_deck.render_deck(parsed))


def test_glued_key_value_and_comments() -> None:
    parsed = _deck.parse_deck("# header\nmax_step=42  # trailing comment\nwarpx.cfl =0.9\n")
    assert parsed == {"max_step": 42, "warpx.cfl": 0.9}


def test_repeated_key_last_wins_keeps_position() -> None:
    parsed = _deck.parse_deck("a.x = 1\nb.y = 2\na.x = 3\n")
    assert parsed == {"a.x": 3, "b.y": 2}
    assert list(parsed) == ["a.x", "b.y"]


def test_string_that_looks_numeric_is_quoted_on_render() -> None:
    deck = {"diag1.intervals": "100:200:10", "e.label": "123"}
    rendered = _deck.render_deck(deck)
    assert _deck.parse_deck(rendered) == deck


def test_parser_expression_with_double_equals_is_left_alone() -> None:
    # `x==0` inside an unquoted token must not be split as a glued definition.
    parsed = _deck.parse_deck("e.f(x,y,z) = if(x==0,1,0)\nmax_step = 1\n")
    assert parsed["e.f(x,y,z)"] == "if(x==0,1,0)"
    assert parsed == _deck.parse_deck(_deck.render_deck(parsed))


def test_merge_overrides_exact_base_and_new_keys() -> None:
    deck = _deck.parse_deck_file(SMOKE_DECK)
    _deck.merge_overrides(
        deck,
        {
            "amr.n_cell": 512,
            "electrons.density_function": "n0",  # base name resolves to the (x,y,z) key
            "warpx.use_filter": 1,  # brand-new key is appended
        },
    )
    assert deck["amr.n_cell"] == 512
    assert deck["electrons.density_function(x,y,z)"] == "n0"
    assert deck["warpx.use_filter"] == 1


def test_merge_overrides_ambiguous_base_raises() -> None:
    deck = {"e.f(x)": 1, "e.f(y)": 2}
    with pytest.raises(ValueError, match="Ambiguous"):
        _deck.merge_overrides(deck, {"e.f": 3})


def test_flat_dict_sanitizes_and_expands_lists() -> None:
    flat = _deck.deck_to_flat_dict({"electrons.density_function(x,y,z)": "n0", "laser1.position": [0.0, 0.0, 1.0e-6]})
    assert flat["electrons.density_function_x_y_z"] == "n0"
    assert flat["laser1.position.0"] == 0.0
    assert flat["laser1.position.2"] == 1.0e-6
