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
non-empty files under the diagnostic paths the deck configures — WarpX
archives ``warpx_used_inputs`` before it validates the deck and AMReX drops
``Backtrace.<rank>`` on an abort, so a run that never left startup leaves
files behind without leaving anything to post-process.
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


def _diagnostic_prefixes(deck: _deck.Deck, run_dir: Path) -> list[Path]:
    """Output prefixes the deck configures, resolved against ``run_dir``.

    Full diagnostics write under ``<diag_name>.file_prefix`` (default
    ``diags/<diag_name>``); reduced diagnostics under
    ``<reduced_diags_name>.path``, else ``reduced_diags.path`` (default
    ``diags/reducedfiles``). ``file_prefix`` is a *name* prefix: the openPMD
    writer uses it as a directory (``diags/diag1/openpmd_000100.h5``) while
    the plotfile writer appends the step (``diags/diag100100/``), so callers
    match ``prefix.parent`` entries whose name starts with ``prefix.name``.
    """
    prefixes: list[Path] = []
    for name in _as_list(deck.get("diagnostics.diags_names")):
        prefixes.append(run_dir / str(deck.get(f"{name}.file_prefix") or f"diags/{name}"))
    reduced_default = str(deck.get("reduced_diags.path") or "diags/reducedfiles")
    for name in _as_list(deck.get("warpx.reduced_diags_names")):
        prefixes.append(run_dir / str(deck.get(f"{name}.path") or reduced_default))
    return prefixes


def _diagnostic_files(run_dir: Path, deck: _deck.Deck) -> Iterator[Path]:
    """Regular files under the deck's diagnostic prefixes (see above)."""
    seen: set[Path] = set()
    for prefix in _diagnostic_prefixes(deck, run_dir):
        if not prefix.parent.is_dir():
            continue
        for entry in prefix.parent.iterdir():
            if not entry.name.startswith(prefix.name) or entry in seen:
                continue
            seen.add(entry)
            if entry.is_file():
                yield entry
            elif entry.is_dir():
                yield from (p for p in entry.rglob("*") if p.is_file())


def _run_produced_output(run_dir: Path, deck: _deck.Deck) -> bool:
    """True if the run wrote diagnostic data worth post-processing.

    Only non-empty files under the deck's configured diagnostic paths count.
    Provenance and crash artifacts are excluded wherever they land: the
    archived inputs copy (``warpx.used_inputs_file``), AMReX's per-rank
    ``Backtrace.<rank>`` dumps, and the files this runner wrote itself. A deck
    that aborts in ReadParameters produces exactly those and nothing else, and
    must be reported as a failure rather than salvaged.
    """
    used_inputs = Path(str(deck.get("warpx.used_inputs_file") or USED_INPUTS_DEFAULT)).name
    ours = {INPUTS_FILENAME, "stdout.log", "stderr.log", used_inputs}
    for p in _diagnostic_files(run_dir, deck):
        if p.name in ours or p.name.startswith("Backtrace."):
            continue
        if p.stat().st_size > 0:
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
