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

# A reduced table with its header and one sample, the way WarpX writes it
# (header at diagnostic init, first row at the step-0 flush).
_REDUCED_ROW = "printf '#[0]step() [1]time(s) [2]total(J)\\n0 0.0 1.5e3\\n'"


def test_run_warpx_crash_with_output_is_salvaged(tmp_path: Path) -> None:
    # A binary that writes a diagnostic then dies (e.g. an abort partway
    # through the run) must NOT raise: the run produced data, so the runner
    # salvages it and lets the caller post-process what was written.
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        f"mkdir -p diags/reducedfiles && {_REDUCED_ROW} > diags/reducedfiles/fieldenergy.txt"
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
    # lands on a reduced table's path (its content is non-comment lines, which
    # would read as data rows) and a Backtrace lands in a diagnostic
    # directory, they still do not make the run salvageable. Empty dumps do
    # not count either — files are created before they are filled.
    deck = _DECK_WITH_DIAGS + "warpx.used_inputs_file = diags/reducedfiles/fieldenergy.txt\n"
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        "mkdir -p diags/reducedfiles diags/diag1"
        " && echo 'warpx.foo = 1' > diags/reducedfiles/fieldenergy.txt"
        " && echo '=== backtrace ===' > diags/diag1/Backtrace.0"
        " && : > diags/diag1/openpmd_000000.h5"
        " && echo 'amrex::Abort::0::boom' >&2 && exit 1",
    )
    with pytest.raises(RuntimeError, match="nothing to salvage"):
        runner.run_warpx(deck, binary=str(fake), mpi_ranks=1, run_root=tmp_path)


def test_run_warpx_metadata_only_diagnostics_do_not_count(tmp_path: Path) -> None:
    # A run that aborts after diagnostics are initialized but before the first
    # sample leaves non-empty diagnostic files with no data in them: the
    # openPMD series pattern file (paraview.pmd), a reduced table holding only
    # its header, the empty companion table of a ParticleHistogram2D, and a
    # plotfile directory with only its headers. None of it is salvage.
    deck = (
        "max_step = 1\n"
        "diagnostics.diags_names = diag1 plt\n"
        "diag1.format = openpmd\n"
        "plt.format = plotfile\n"
        "warpx.reduced_diags_names = fieldenergy p1x1\n"
    )
    metadata_only = (
        f"{_STARTUP_ARTIFACTS}"
        " && mkdir -p diags/diag1 diags/plt00000/Level_0 diags/reducedfiles/p1x1"
        " && echo 'openpmd_%06T.h5' > diags/diag1/paraview.pmd"
        " && echo 'HyperCLaw-V1.1' > diags/plt00000/Header"
        " && echo 'hdr' > diags/plt00000/Level_0/Cell_H"
        " && printf '#[0]step() [1]time(s) [2]total(J)\\n' > diags/reducedfiles/fieldenergy.txt"
        " && : > diags/reducedfiles/p1x1.txt"
    )
    fake = _write_fake_binary(tmp_path / "fake-warpx", f"{metadata_only} && echo 'amrex::Abort::0::boom' >&2 && exit 1")
    with pytest.raises(RuntimeError, match="nothing to salvage"):
        runner.run_warpx(deck, binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    # ... and one sample in any of them is.
    payloads = {
        "openpmd-dump": "echo x > diags/diag1/openpmd_000000.h5",
        "plotfile-chunk": "echo x > diags/plt00000/Level_0/Cell_D_00000",
        "table-row": "echo '0 0.0 1.5e3' >> diags/reducedfiles/fieldenergy.txt",
        "histogram2d-dump": "echo x > diags/reducedfiles/p1x1/openpmd_000000.h5",
    }
    for label, write in payloads.items():
        fake_ok = _write_fake_binary(tmp_path / f"fake-{label}", f"{metadata_only} && {write} && exit 1")
        result = runner.run_warpx(deck, binary=str(fake_ok), mpi_ranks=1, run_root=tmp_path / label)
        assert result["crashed"] is True, label


def test_run_warpx_checkpoint_diagnostic_is_not_salvage(tmp_path: Path) -> None:
    # A checkpoint is restart state, not output post-processing can consume.
    deck = "max_step = 1\ndiagnostics.diags_names = chk\nchk.format = checkpoint\n"
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        "mkdir -p diags/chk00000/Level_0 && echo x > diags/chk00000/Level_0/Cell_D_00000 && exit 1",
    )
    with pytest.raises(RuntimeError, match="nothing to salvage"):
        runner.run_warpx(deck, binary=str(fake), mpi_ranks=1, run_root=tmp_path)


def test_run_warpx_undeclared_files_do_not_count(tmp_path: Path) -> None:
    # Only the diagnostics the deck declares can be output. A stray non-empty
    # file under diags/ that no diagnostic would have written is not salvage.
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        f"mkdir -p diags/reducedfiles && {_REDUCED_ROW} > diags/reducedfiles/fieldenergy.txt"
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
        "reduced_diags.extension = dat\n"
        "pe.path = out/pe_only\n"
        "pe.extension = csv\n"
    )
    cases = {
        "openpmd-dir": "mkdir -p out/fields && echo x > out/fields/openpmd_000000.h5",
        "plotfile-prefix": "mkdir -p out/chk_00000/Level_0 && echo x > out/chk_00000/Level_0/Cell_D_00000",
        "reduced-default": f"mkdir -p out/reduced && {_REDUCED_ROW} > out/reduced/fe.dat",
        "reduced-per-diag": f"mkdir -p out/pe_only && {_REDUCED_ROW} > out/pe_only/pe.csv",
    }
    for label, writes in cases.items():
        fake = _write_fake_binary(tmp_path / f"fake-{label}", f"{_STARTUP_ARTIFACTS} && {writes} && exit 3")
        result = runner.run_warpx(deck, binary=str(fake), mpi_ranks=1, run_root=tmp_path / label)
        assert result["crashed"] is True, label
    # And the same deck with only the default-layout locations populated is a
    # failure: nothing was written where this deck says diagnostics go.
    fake = _write_fake_binary(
        tmp_path / "fake-default-layout",
        f"{_STARTUP_ARTIFACTS} && mkdir -p diags/reducedfiles && {_REDUCED_ROW} > diags/reducedfiles/fe.txt && exit 3",
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
