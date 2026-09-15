"""Smoke tests for the WarpX subprocess runner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from adept.warpx import runner

# Resolved from the same env vars the runner itself honors, so the live-binary
# smoke test below runs wherever WarpX is built and skips cleanly otherwise.
WARPX_BIN_1D = os.environ.get("WARPX_BIN_1D") or os.environ.get("WARPX_BIN")

DECKS_DIR = Path(__file__).parent / "decks"
SMOKE_DECK = DECKS_DIR / "warpx-1d-smoke"


def test_discover_binary_explicit_wins(tmp_path: Path) -> None:
    fake = tmp_path / "fake-warpx"
    fake.write_text("")
    out = runner.discover_binary(str(fake))
    assert out == fake.resolve()


def test_discover_binary_env_fallback(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "fake-warpx-1d"
    fake.write_text("")
    monkeypatch.setenv("WARPX_BIN_1D", str(fake))
    out = runner.discover_binary(None, dim=1)
    assert out == fake.resolve()


def test_discover_binary_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        runner.discover_binary("/no/such/path/exists", dim=1)


def test_run_warpx_missing_binary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner.run_warpx(
            "max_step = 1\n",
            binary="/no/such/binary",
            mpi_ranks=1,
            run_root=tmp_path,
        )


def _write_fake_binary(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)
    return path


def test_run_warpx_clean_exit(tmp_path: Path) -> None:
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        'test -f "$1" && echo "STEP 1 ends" && exit 0',
    )
    result = runner.run_warpx("max_step = 1\n", binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    assert result["exit_code"] == 0
    assert result["crashed"] is False
    # The rendered inputs file is the first positional argument, in the run dir.
    assert (result["run_dir"] / runner.INPUTS_FILENAME).read_text() == "max_step = 1\n"
    assert "STEP 1 ends" in (result["run_dir"] / "stdout.log").read_text()


# A deck declaring one reduced diagnostic (default path diags/reducedfiles)
# and one full diagnostic (default prefix diags/diag1), the way the fake
# binaries below write them.
_DECK_WITH_DIAGS = "max_step = 1\nwarpx.reduced_diags_names = fieldenergy\ndiagnostics.diags_names = diag1\n"

# What WarpX/AMReX leave behind when the deck fails validation: the archived
# inputs copy (written before ReadParameters validates anything) plus one
# Backtrace per rank from the abort handler. Never diagnostic output.
_STARTUP_ARTIFACTS = "echo 'warpx.foo = 1' > warpx_used_inputs && echo '=== backtrace ===' > Backtrace.0"


def test_run_warpx_crash_with_output_is_salvaged(tmp_path: Path) -> None:
    # A binary that writes a diagnostic then dies (e.g. an abort partway
    # through the run) must NOT raise: the run produced data, so the runner
    # salvages it and lets the caller post-process what was written.
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        "mkdir -p diags/reducedfiles && echo '#step time' > diags/reducedfiles/fieldenergy.txt"
        " && echo 'amrex::Abort::0::boom' >&2 && exit 6",
    )
    result = runner.run_warpx(_DECK_WITH_DIAGS, binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    assert result["exit_code"] == 6
    assert result["crashed"] is True
    assert (result["run_dir"] / "diags" / "reducedfiles" / "fieldenergy.txt").exists()


def test_run_warpx_crash_no_output_raises(tmp_path: Path) -> None:
    # A binary that exits non-zero WITHOUT writing anything is a hard failure —
    # there is nothing to salvage, so the runner raises with the AMReX detail.
    fake = _write_fake_binary(tmp_path / "fake-warpx", "echo 'amrex::Abort::0::bad inputs' >&2 && exit 1")
    with pytest.raises(RuntimeError) as excinfo:
        runner.run_warpx(_DECK_WITH_DIAGS, binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    assert "nothing to salvage" in str(excinfo.value)
    assert "amrex::Abort" in str(excinfo.value)


def test_run_warpx_startup_abort_is_not_salvaged(tmp_path: Path) -> None:
    # A deck that fails validation (e.g. a multi-valued override rendered as a
    # quoted string) aborts in ReadParameters after WarpX has already archived
    # warpx_used_inputs, and AMReX drops Backtrace.<rank>. Those are provenance
    # and crash artifacts, not output: the run must be reported as failed, not
    # recorded as finished with nothing to post-process.
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        f"{_STARTUP_ARTIFACTS} && echo '### ERROR   : amrex::Abort::0::ReadParameters' >&2 && exit 1",
    )
    with pytest.raises(RuntimeError) as excinfo:
        runner.run_warpx(_DECK_WITH_DIAGS, binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    assert "nothing to salvage" in str(excinfo.value)
    assert "ReadParameters" in str(excinfo.value)


def test_run_warpx_startup_artifacts_inside_diag_dirs_do_not_count(tmp_path: Path) -> None:
    # The exclusion is by identity, not location: if warpx.used_inputs_file
    # points into a diagnostic directory (and if a Backtrace lands there), they
    # still do not make the run salvageable. Empty files do not count either —
    # WarpX creates directories and touches files before it fills them.
    deck = _DECK_WITH_DIAGS + "warpx.used_inputs_file = diags/reducedfiles/used_inputs\n"
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        "mkdir -p diags/reducedfiles diags/diag1"
        " && echo 'warpx.foo = 1' > diags/reducedfiles/used_inputs"
        " && echo '=== backtrace ===' > diags/reducedfiles/Backtrace.0"
        " && : > diags/diag1/openpmd_000000.h5"
        " && echo 'amrex::Abort::0::boom' >&2 && exit 1",
    )
    with pytest.raises(RuntimeError, match="nothing to salvage"):
        runner.run_warpx(deck, binary=str(fake), mpi_ranks=1, run_root=tmp_path)


def test_run_warpx_undeclared_files_do_not_count(tmp_path: Path) -> None:
    # Only the diagnostics the deck declares can be output. A stray non-empty
    # file under diags/ that no diagnostic would have written is not salvage.
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        "mkdir -p diags/reducedfiles && echo '#step time' > diags/reducedfiles/fieldenergy.txt"
        " && echo 'amrex::Abort::0::boom' >&2 && exit 1",
    )
    with pytest.raises(RuntimeError, match="nothing to salvage"):
        runner.run_warpx("max_step = 1\n", binary=str(fake), mpi_ranks=1, run_root=tmp_path)


def test_run_warpx_salvage_honors_configured_paths(tmp_path: Path) -> None:
    # Diagnostic locations come from the deck, not from the default layout:
    # <diag>.file_prefix (used as a directory by openPMD and as a name prefix
    # by plotfiles), reduced_diags.path, and the per-diagnostic <rd>.path.
    deck = (
        "max_step = 1\n"
        "diagnostics.diags_names = fld chk\n"
        "fld.file_prefix = out/fields\n"
        "chk.file_prefix = out/chk_\n"
        "warpx.reduced_diags_names = fe pe\n"
        "reduced_diags.path = out/reduced\n"
        "pe.path = out/pe_only\n"
    )
    cases = {
        "openpmd-dir": "mkdir -p out/fields && echo x > out/fields/openpmd_000000.h5",
        "plotfile-prefix": "mkdir -p out/chk_00000/Level_0 && echo x > out/chk_00000/Level_0/Cell_D_00000",
        "reduced-default": "mkdir -p out/reduced && echo '#hdr' > out/reduced/fe.txt",
        "reduced-per-diag": "mkdir -p out/pe_only && echo '#hdr' > out/pe_only/pe.txt",
    }
    for label, writes in cases.items():
        fake = _write_fake_binary(tmp_path / f"fake-{label}", f"{_STARTUP_ARTIFACTS} && {writes} && exit 3")
        result = runner.run_warpx(deck, binary=str(fake), mpi_ranks=1, run_root=tmp_path / label)
        assert result["crashed"] is True, label
    # And the same deck with only the default-layout locations populated is a
    # failure: nothing was written where this deck says diagnostics go.
    fake = _write_fake_binary(
        tmp_path / "fake-default-layout",
        f"{_STARTUP_ARTIFACTS} && mkdir -p diags/reducedfiles && echo '#hdr' > diags/reducedfiles/fe.txt && exit 3",
    )
    with pytest.raises(RuntimeError, match="nothing to salvage"):
        runner.run_warpx(deck, binary=str(fake), mpi_ranks=1, run_root=tmp_path / "default-layout")


@pytest.mark.skipif(
    not (WARPX_BIN_1D and Path(WARPX_BIN_1D).exists()),
    reason="set WARPX_BIN_1D (or WARPX_BIN) to a built 1D WarpX executable to run",
)
def test_run_warpx_smoke_deck_live(tmp_path: Path) -> None:
    result = runner.run_warpx(
        SMOKE_DECK.read_text(),
        binary=WARPX_BIN_1D,
        mpi_ranks=1,
        run_root=tmp_path,
        launcher="mpirun",
    )
    assert result["exit_code"] == 0
    assert (result["run_dir"] / "warpx_used_inputs").exists()
