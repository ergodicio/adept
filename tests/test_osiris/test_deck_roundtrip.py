"""Round-trip tests for the OSIRIS namelist parser/renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from adept.osiris import deck as osd


REAL_DECKS = [
    "/home/phil/Desktop/pic/projects/twostream-pic/two-stream-1d",
    "/home/phil/Desktop/pic/projects/twostream-pic/example_deck",
    "/home/phil/Desktop/pic/osiris/decks/cuda/os-stdin-1d-therm",
    "/home/phil/Desktop/pic/osiris/decks/cuda/os-stdin-2d-therm",
]


@pytest.mark.parametrize("path", REAL_DECKS)
def test_roundtrip_identity(path: str) -> None:
    text = Path(path).read_text()
    parsed = osd.parse_deck(text)
    rendered = osd.render_deck(parsed)
    reparsed = osd.parse_deck(rendered)
    assert parsed == reparsed, f"Round-trip mismatch for {path}"


def test_parse_basic_section() -> None:
    text = """
    grid
    {
      nx_p(1:1) = 64,
      coordinates = "cartesian",
    }
    """
    sections = osd.parse_deck(text)
    assert sections == [("grid", {"nx_p(1:1)": 64, "coordinates": "cartesian"})]


def test_parse_repeated_sections_preserved_order() -> None:
    text = """
    species { name = "a", rqm = -1.0, }
    species { name = "b", rqm = +1.0, }
    """
    sections = osd.parse_deck(text)
    assert [(n, p["name"]) for n, p in sections] == [
        ("species", "a"),
        ("species", "b"),
    ]


def test_parse_empty_section() -> None:
    text = "current{}\n diag_current { }\n"
    sections = osd.parse_deck(text)
    assert sections == [("current", {}), ("diag_current", {})]


def test_parse_booleans_and_lists() -> None:
    text = """
    udist {
      uth(1:3) = 0.0707, 0.0707, 0.0707,
      ufl(1:3) = -1.0, 0.0, 0.0,
    }
    spe_bound { if_periodic = .true., }
    """
    sections = osd.parse_deck(text)
    assert sections[0][1]["uth(1:3)"] == [0.0707, 0.0707, 0.0707]
    assert sections[0][1]["ufl(1:3)"] == [-1.0, 0.0, 0.0]
    assert sections[1][1]["if_periodic"] is True


def test_comment_stripping_preserves_string_with_bang() -> None:
    text = 'foo { msg = "hello! world", x = 1, }'
    sections = osd.parse_deck(text)
    assert sections[0][1]["msg"] == "hello! world"
    assert sections[0][1]["x"] == 1


def test_merge_overrides_simple() -> None:
    text = "time { tmin = 0.0, tmax = 30.0, }"
    s = osd.parse_deck(text)
    osd.merge_overrides(s, {"time": {"tmax": 50.0}})
    assert s[0][1]["tmax"] == 50.0
    assert s[0][1]["tmin"] == 0.0


def test_merge_overrides_indexed_repeated_section() -> None:
    text = """
    species { name = "a", num_par_x(1:1) = 256, }
    species { name = "b", num_par_x(1:1) = 256, }
    """
    s = osd.parse_deck(text)
    osd.merge_overrides(s, {"species": {0: {"num_par_x": [512]}}})
    assert s[0][1]["num_par_x(1:1)"] == [512]
    assert s[1][1]["num_par_x(1:1)"] == 256


def test_merge_overrides_unknown_section_raises() -> None:
    s = osd.parse_deck("grid { nx_p(1:1) = 64, }")
    with pytest.raises(KeyError):
        osd.merge_overrides(s, {"nonexistent": {"foo": 1}})


def test_deck_to_flat_dict_indexes_repeated_sections() -> None:
    text = """
    species { name = "a", num_par_x(1:1) = 256, }
    species { name = "b", num_par_x(1:1) = 256, }
    grid { nx_p(1:1) = 64, }
    """
    s = osd.parse_deck(text)
    flat = osd.deck_to_flat_dict(s)
    assert flat["species_0.name"] == "a"
    assert flat["species_1.name"] == "b"
    assert flat["grid.nx_p_1:1"] == 64


def test_deck_to_flat_dict_expands_lists() -> None:
    s = osd.parse_deck("udist { uth(1:3) = 0.1, 0.2, 0.3, }")
    flat = osd.deck_to_flat_dict(s)
    assert flat["udist.uth_1:3.0"] == 0.1
    assert flat["udist.uth_1:3.1"] == 0.2
    assert flat["udist.uth_1:3.2"] == 0.3


def test_deck_to_flat_dict_keys_are_mlflow_safe() -> None:
    import re

    s = osd.parse_deck(
        Path(
            "/home/phil/Desktop/pic/projects/twostream-pic/two-stream-1d"
        ).read_text()
    )
    flat = osd.deck_to_flat_dict(s)
    allowed = re.compile(r"^[A-Za-z0-9_./:\- ]+$")
    for k in flat:
        assert allowed.match(k), f"Unsafe MLflow param key: {k!r}"
