"""Smoke tests for the OSIRIS subprocess runner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from adept.osiris import runner


# Resolved from the same env vars the runner itself honors, so the live-binary
# smoke test below runs wherever OSIRIS is built and skips cleanly otherwise.
OSIRIS_BIN_1D = os.environ.get("OSIRIS_BIN_1D") or os.environ.get("OSIRIS_BIN")


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
    not (OSIRIS_BIN_1D and Path(OSIRIS_BIN_1D).exists()),
    reason="set OSIRIS_BIN_1D (or OSIRIS_BIN) to a built osiris-1D.e to run",
)
def test_run_osiris_invalid_deck_raises(tmp_path: Path) -> None:
    # A deck with a recognized section but garbage inside: OSIRIS exits 0
    # but writes 'Error reading ... / aborting...' to stderr. Our runner
    # turns that into a RuntimeError so it doesn't slip past silently.
    with pytest.raises(RuntimeError) as excinfo:
        runner.run_osiris(
            "node_conf { node_number(1:1) = junk_value, }",
            binary=OSIRIS_BIN_1D,
            mpi_ranks=1,
            run_root=tmp_path,
        )
    assert "OSIRIS" in str(excinfo.value)
