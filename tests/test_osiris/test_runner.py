"""Smoke tests for the OSIRIS subprocess runner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from adept.osiris import runner


def test_discover_binary_explicit_wins(tmp_path: Path) -> None:
    fake = tmp_path / "fake-osiris"
    fake.write_text("")
    out = runner.discover_binary(str(fake))
    assert out == fake.resolve()


def test_discover_binary_env_fallback(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "fake-osiris-1d"
    fake.write_text("")
    monkeypatch.setenv("OSIRIS_BIN_1D", str(fake))
    out = runner.discover_binary(None, dim=1)
    assert out == fake.resolve()


def test_discover_binary_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        runner.discover_binary("/no/such/path/exists", dim=1)


def test_run_osiris_missing_binary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner.run_osiris(
            "node_conf {}",
            binary="/no/such/binary",
            mpi_ranks=1,
            run_root=tmp_path,
        )


@pytest.mark.skipif(
    not Path("/home/phil/Desktop/pic/osiris/bin/osiris-1D.e").exists(),
    reason="osiris-1D.e not built",
)
def test_run_osiris_invalid_deck_raises(tmp_path: Path) -> None:
    # A deck with a recognized section but garbage inside: OSIRIS exits 0
    # but writes 'Error reading ... / aborting...' to stderr. Our runner
    # turns that into a RuntimeError so it doesn't slip past silently.
    with pytest.raises(RuntimeError) as excinfo:
        runner.run_osiris(
            "node_conf { node_number(1:1) = junk_value, }",
            binary="/home/phil/Desktop/pic/osiris/bin/osiris-1D.e",
            mpi_ranks=1,
            run_root=tmp_path,
        )
    assert "OSIRIS" in str(excinfo.value)
