"""Subprocess driver for WarpX.

WarpX takes its inputs file as the first positional argument. This module
sets up a per-run work directory, writes the rendered inputs there, invokes
the configured launcher (``srun`` by default — the team runs on
Perlmutter/Slurm; override with ``mpi_launcher: mpirun`` for a local MPI)
when ``mpi_ranks > 1`` — or runs the binary directly when ``mpi_ranks == 1``
— and captures stdout/stderr to files for later artifact upload.

Error handling differs from the OSIRIS runner on purpose: WarpX/AMReX fails
loudly (``amrex::Abort`` and failed assertions exit non-zero), so the exit
code is the primary signal and there is no exit-0 stderr fuzzing. The
salvage-if-output-exists behavior is kept: a crash after diagnostics were
written still lets post-processing run on the partial data. "Output" means
diagnostic payload at the locations the deck configures — a dump or a
table row, not the series metadata, table headers, archived
``warpx_used_inputs`` or AMReX ``Backtrace.<rank>`` that a run which never
reached its first sample leaves behind.
"""

from __future__ import annotations

import datetime as _dt
import os
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import deck as _deck

INPUTS_FILENAME = "inputs"
# WarpX's archived copy of the inputs it parsed (``warpx.used_inputs_file``).
# It is written before the deck is validated, so it exists for runs that
# aborted in ReadParameters.
USED_INPUTS_DEFAULT = "warpx_used_inputs"

# Tokens WarpX/AMReX print on aborts/assertions/signal handlers, matched
# lowercased. "### error" is WarpX's own WARPX_ABORT/ASSERT banner prefix
# ("### ERROR   : ", ablastr TextMsg); the rest come from AMReX. Only
# consulted for the failure message detail — the exit code decides
# success/failure.
_AMREX_ERR_TOKENS = ("### error", "amrex::abort", "amrex error", "assertion", "sigsegv", "sigfpe", "backtrace")


def _stream_to_file_and_buffer(stream, file_path: Path, tail: list[str], tail_max: int = 200) -> None:
    """Tee a subprocess stream to disk and a bounded in-memory tail."""
    with file_path.open("w") as fh:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            fh.write(line)
            fh.flush()
            tail.append(line)
            if len(tail) > tail_max:
                del tail[: len(tail) - tail_max]


def _make_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{stamp}_{uuid.uuid4().hex[:8]}"
    rd = run_root / name
    rd.mkdir()
    return rd


def _as_list(v: Any) -> list:
    return [] if v is None else (v if isinstance(v, list) else [v])


def _full_diag_prefixes(deck: _deck.Deck, run_dir: Path) -> list[Path]:
    """``<diag_name>.file_prefix`` (default ``diags/<diag_name>``) of every
    full diagnostic whose output post-processing can consume, resolved against
    ``run_dir``. Checkpoints are restart state, not output, and are skipped.

    ``file_prefix`` is a *name* prefix: the openPMD writer uses it as a
    directory (``diags/diag1/openpmd_000100.h5``) while the plotfile writer
    appends the step (``diags/diag100100/``), so callers match
    ``prefix.parent`` entries whose name starts with ``prefix.name``.
    """
    prefixes: list[Path] = []
    for name in _as_list(deck.get("diagnostics.diags_names")):
        if str(deck.get(f"{name}.format") or "plotfile").lower() == "checkpoint":
            continue
        prefixes.append(run_dir / str(deck.get(f"{name}.file_prefix") or f"diags/{name}"))
    return prefixes


def _reduced_diag_outputs(deck: _deck.Deck, run_dir: Path) -> list[tuple[Path, Path]]:
    """``(table, openpmd_dir)`` for every reduced diagnostic: the text table
    ``<path>/<name>.<extension>`` every reduced diagnostic opens, and the
    ``<path>/<name>/`` openPMD directory the ParticleHistogram2D type writes
    its histories to instead (its table stays empty). ``<name>.path`` /
    ``<name>.extension`` override ``reduced_diags.path`` / ``.extension``
    (defaults ``diags/reducedfiles`` / ``txt``).
    """
    default_path = str(deck.get("reduced_diags.path") or "diags/reducedfiles")
    default_ext = str(deck.get("reduced_diags.extension") or "txt")
    outputs: list[tuple[Path, Path]] = []
    for name in _as_list(deck.get("warpx.reduced_diags_names")):
        base = run_dir / str(deck.get(f"{name}.path") or default_path)
        ext = str(deck.get(f"{name}.extension") or default_ext)
        outputs.append((base / f"{name}.{ext}", base / name))
    return outputs


def _prefix_entries(prefix: Path) -> Iterator[Path]:
    """Directory entries next to ``prefix`` whose name starts with its name."""
    if prefix.parent.is_dir():
        yield from (e for e in prefix.parent.iterdir() if e.name.startswith(prefix.name))


def _files_under(entry: Path) -> Iterator[Path]:
    if entry.is_file():
        yield entry
    elif entry.is_dir():
        yield from (p for p in entry.rglob("*") if p.is_file())


def _is_openpmd_payload(p: Path, root: Path) -> bool:
    """A dump, not series metadata. openPMD-api writes ``paraview.pmd`` (the
    file pattern) when the series is created, before any iteration is
    flushed; the data is ``openpmd_<iter>.h5`` / ``openpmd.h5`` for the h5
    backend, ``openpmd_<iter>.bp{,4,5}/data.<rank>`` for ADIOS, or ``.json``.
    """
    if p.suffix in (".h5", ".json"):
        return True
    return any(part.endswith((".bp", ".bp4", ".bp5")) for part in p.relative_to(root).parts[:-1])


def _is_plotfile_payload(p: Path) -> bool:
    """An AMReX plotfile data chunk (``Level_<n>/Cell_D_<k>``, particle
    ``<species>/Level_<n>/DATA_<k>``) rather than the ``Header`` /
    ``Cell_H`` / ``warpx_job_info`` metadata written alongside it."""
    return p.parent.name.startswith("Level_") and p.name.startswith(("Cell_D", "DATA_"))


def _table_has_data_row(p: Path) -> bool:
    """A reduced-diagnostic table with at least one sample. WarpX writes the
    ``#[0]step() [1]time(s) ...`` header when the diagnostic is initialized,
    which is before the first sample."""
    with p.open(errors="replace") as fh:
        return any(ln.strip() and not ln.lstrip().startswith("#") for ln in fh)


def _run_produced_output(run_dir: Path, deck: _deck.Deck) -> bool:
    """True if the run wrote diagnostic *data* worth post-processing.

    A run that dies before its first sample still leaves diagnostic files:
    the openPMD series metadata, a reduced table's header row, empty tables,
    plotfile headers. Only payload counts — an openPMD dump or plotfile data
    chunk under a full diagnostic's configured prefix, a reduced table with
    a data row, or a ParticleHistogram2D dump under the reduced path.
    Provenance and crash artifacts are excluded by name wherever they land:
    the archived inputs copy (``warpx.used_inputs_file``), AMReX's per-rank
    ``Backtrace.<rank>`` dumps, and the files this runner wrote itself. A
    deck that aborts in ReadParameters produces exactly those and nothing
    else, and must be reported as a failure rather than salvaged.
    """
    used_inputs = Path(str(deck.get("warpx.used_inputs_file") or USED_INPUTS_DEFAULT)).name
    ours = {INPUTS_FILENAME, "stdout.log", "stderr.log", used_inputs}

    def candidate(p: Path) -> bool:
        return p.is_file() and p.name not in ours and not p.name.startswith("Backtrace.") and p.stat().st_size > 0

    for prefix in _full_diag_prefixes(deck, run_dir):
        for entry in _prefix_entries(prefix):
            root = entry if entry.is_dir() else entry.parent
            for p in _files_under(entry):
                if candidate(p) and (_is_openpmd_payload(p, root) or _is_plotfile_payload(p)):
                    return True
    for table, openpmd_dir in _reduced_diag_outputs(deck, run_dir):
        if candidate(table) and _table_has_data_row(table):
            return True
        for p in _files_under(openpmd_dir):
            if candidate(p) and _is_openpmd_payload(p, openpmd_dir):
                return True
    return False


def run_warpx(
    deck_text: str,
    *,
    binary: str | Path,
    mpi_ranks: int = 1,
    run_root: str | Path = "./checkpoints",
    env: dict[str, str] | None = None,
    launcher: str = "srun",
    extra_mpi_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run WarpX and return run metadata.

    Returns a dict with keys ``run_dir`` (Path), ``exit_code`` (int),
    ``crashed`` (bool), ``wall_time`` (float, seconds), and ``cmd``
    (list[str]).

    Raises ``RuntimeError`` on a non-zero exit code that left no diagnostic
    output behind (see :func:`_run_produced_output` — startup artifacts do
    not count); a crash *with* diagnostic output on disk is logged and
    salvaged so the caller still consolidates and plots what was written.
    """
    binary = Path(binary).expanduser().resolve()
    if not binary.exists():
        raise FileNotFoundError(f"WarpX binary not found: {binary}")

    run_dir = _make_run_dir(Path(run_root).expanduser().resolve())
    (run_dir / INPUTS_FILENAME).write_text(deck_text)

    if mpi_ranks > 1:
        cmd = [launcher, "-n", str(mpi_ranks)]
        if extra_mpi_args:
            cmd.extend(extra_mpi_args)
        cmd.append(str(binary))
    else:
        cmd = [str(binary)]
    cmd.append(INPUTS_FILENAME)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_tail: list[str] = []
    stderr_tail: list[str] = []

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=run_dir,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    t_out = threading.Thread(
        target=_stream_to_file_and_buffer,
        args=(proc.stdout, stdout_path, stdout_tail),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_stream_to_file_and_buffer,
        args=(proc.stderr, stderr_path, stderr_tail),
        daemon=True,
    )
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join()
    t_err.join()
    wall_time = time.time() - t0

    crashed = rc != 0
    if crashed:
        # Pull the AMReX abort/assert lines to the front of the message so the
        # cause is visible without opening the logs; fall back to the raw tail.
        err_lines = [ln for ln in (stdout_tail + stderr_tail) if any(tok in ln.lower() for tok in _AMREX_ERR_TOKENS)]
        detail = "".join(err_lines[-20:]) or "".join(stderr_tail[-50:]) or "(empty stderr)"
        failure_msg = f"WarpX exited with status {rc}.\n  cmd: {shlex.join(cmd)}\n  cwd: {run_dir}\n  detail:\n{detail}"
        if _run_produced_output(run_dir, _deck.parse_deck(deck_text)):
            print(f"[warpx] WARNING: {failure_msg}")
            print(
                "[warpx] the run produced diagnostic output despite the error above — "
                "consolidating and post-processing the (possibly partial) data "
                "anyway. Verify results carefully."
            )
        else:
            raise RuntimeError(f"{failure_msg}\n  (no diagnostic output was written — nothing to salvage.)")

    return {
        "run_dir": run_dir,
        "exit_code": rc,
        "crashed": crashed,
        "wall_time": wall_time,
        "cmd": cmd,
    }


def discover_binary(cfg_binary: str | None, *, dim: int | None = None) -> Path:
    """Resolve the WarpX binary path.

    Precedence: explicit ``cfg_binary`` > ``WARPX_BIN_<dim>D`` env var >
    ``WARPX_BIN`` env var. Returns an existing Path or raises.
    """
    candidates: list[str] = []
    if cfg_binary:
        candidates.append(cfg_binary)
    if dim is not None:
        env_key = f"WARPX_BIN_{dim}D"
        if env_key in os.environ:
            candidates.append(os.environ[env_key])
    if "WARPX_BIN" in os.environ:
        candidates.append(os.environ["WARPX_BIN"])

    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "No WarpX binary found. Set warpx.binary in the manifest or "
        "WARPX_BIN / WARPX_BIN_<dim>D in the environment. Tried: "
        f"{candidates}"
    )
