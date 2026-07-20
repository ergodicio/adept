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


def test_srun_chdir_warning_is_not_an_error() -> None:
    # srun's spurious launcher warning under /dev/shm staging must not be flagged
    # as an OSIRIS error (it aborts an otherwise-clean run); a real OSIRIS error
    # and a real abort still are.
    chdir = "[2026-06-30T23:35:54] error: couldn't chdir to `/dev/shm/x`: No such file or directory: going to /tmp instead"
    assert runner._looks_like_osiris_error(chdir) is False
    assert runner._looks_like_osiris_error("(*error*) Lindman not yet allowed with tiling") is True
    assert runner._looks_like_osiris_error("Error reading global simulation parameters, aborting...") is True


def test_run_osiris_missing_binary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner.run_osiris(
            "node_conf {}",
            binary="/no/such/binary",
            mpi_ranks=1,
            run_root=tmp_path,
        )


def _write_fake_binary(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)
    return path


def test_run_osiris_crash_with_output_is_salvaged(tmp_path: Path) -> None:
    # A binary that writes a dump then exits non-zero (e.g. OSIRIS segfaulting on
    # MPI/CUDA teardown *after* "Simulation completed") must NOT raise: the run
    # produced data, so the runner salvages it and lets the caller consolidate +
    # plot what was written.
    fake = _write_fake_binary(
        tmp_path / "fake-osiris",
        'mkdir -p MS/FLD/e1 && : > MS/FLD/e1/e1-000000.h5 && exit 139',
    )
    result = runner.run_osiris(
        "node_conf {}",
        binary=str(fake),
        mpi_ranks=1,
        run_root=tmp_path,
        stream_convert=False,
    )
    assert result["exit_code"] == 139
    assert result["crashed"] is True
    assert next((result["run_dir"] / "MS").rglob("*.h5"), None) is not None


def test_run_osiris_crash_no_output_raises(tmp_path: Path) -> None:
    # A binary that exits non-zero WITHOUT writing anything is a hard failure —
    # there is nothing to salvage, so the runner still raises.
    fake = _write_fake_binary(tmp_path / "fake-osiris", "exit 1")
    with pytest.raises(RuntimeError) as excinfo:
        runner.run_osiris(
            "node_conf {}",
            binary=str(fake),
            mpi_ranks=1,
            run_root=tmp_path,
            stream_convert=False,
        )
    assert "nothing to salvage" in str(excinfo.value)


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
