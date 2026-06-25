"""Subprocess driver for OSIRIS.

OSIRIS reads its deck from a file named ``os-stdin`` in the current
working directory by default. This module sets up a per-run work
directory, writes the rendered deck there, invokes the configured
launcher (``srun`` by default — the team runs on Perlmutter/Slurm; override
with ``mpi_launcher: mpirun`` for a local MPI) when ``mpi_ranks > 1`` — or
runs the binary directly when ``mpi_ranks == 1`` — and captures
stdout/stderr to files for later artifact upload.
"""

from __future__ import annotations

import datetime as _dt
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_OSIRIS_ERR_TOKENS = ("error", "aborting", "(*error*)")
# stderr noise emitted by X11 / mpirun that we should NOT treat as an error.
_OSIRIS_STDERR_NOISE = (
    "No protocol specified",
    "MPI_INIT",  # benign MPI banner noise on some setups
)


def _looks_like_osiris_error(line: str) -> bool:
    low = line.strip().lower()
    if not low:
        return False
    # OSIRIS prefixes diagnostics with (*warning*) / (*error*); a warning is
    # never a failure even when its text contains the word "error" (e.g. the
    # Sentoku-collisions "factor of 2 error" warning).
    if "(*warning*)" in low:
        return False
    if any(noise.lower() in low for noise in _OSIRIS_STDERR_NOISE):
        return False
    return any(tok in low for tok in _OSIRIS_ERR_TOKENS)


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


def run_osiris(
    deck_text: str,
    *,
    binary: str | Path,
    mpi_ranks: int = 1,
    run_root: str | Path = "./checkpoints",
    env: dict[str, str] | None = None,
    launcher: str = "srun",
    extra_mpi_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run OSIRIS and return run metadata.

    Returns a dict with keys ``run_dir`` (Path), ``exit_code`` (int),
    ``wall_time`` (float, seconds), and ``cmd`` (list[str]).

    Raises ``RuntimeError`` on non-zero exit code, with the last lines of
    stderr included in the message.
    """
    binary = Path(binary).expanduser().resolve()
    if not binary.exists():
        raise FileNotFoundError(f"OSIRIS binary not found: {binary}")

    run_dir = _make_run_dir(Path(run_root).expanduser().resolve())
    (run_dir / "os-stdin").write_text(deck_text)

    if mpi_ranks > 1:
        cmd = [launcher, "-n", str(mpi_ranks)]
        if extra_mpi_args:
            cmd.extend(extra_mpi_args)
        cmd.append(str(binary))
    else:
        cmd = [str(binary)]

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

    if rc != 0:
        tail = "".join(stderr_tail[-50:]) or "(empty stderr)"
        raise RuntimeError(
            f"OSIRIS exited with status {rc}.\n  cmd: {shlex.join(cmd)}\n  cwd: {run_dir}\n  stderr tail:\n{tail}"
        )

    # OSIRIS can exit 0 even on input-file errors: it prints something
    # like 'Error reading ... / aborting...' to stderr (and '(*error*)'
    # to stdout) before terminating. Detect both.
    err_lines = [ln for ln in stderr_tail if _looks_like_osiris_error(ln)]
    err_lines += [ln for ln in stdout_tail if "(*error*)" in ln]
    if err_lines:
        out_tail = "".join(stdout_tail[-20:])
        err_tail = "".join(stderr_tail[-20:])
        raise RuntimeError(
            "OSIRIS reported an error despite exit-code 0:\n"
            f"  cmd: {shlex.join(cmd)}\n"
            f"  cwd: {run_dir}\n"
            f"  stderr tail:\n{err_tail}\n"
            f"  stdout tail:\n{out_tail}"
        )

    return {
        "run_dir": run_dir,
        "exit_code": rc,
        "wall_time": wall_time,
        "cmd": cmd,
    }


def discover_binary(cfg_binary: str | None, *, dim: int | None = None) -> Path:
    """Resolve the OSIRIS binary path.

    Precedence: explicit ``cfg_binary`` > ``OSIRIS_BIN_<dim>D`` env var >
    ``OSIRIS_BIN`` env var. Returns an existing Path or raises.
    """
    candidates: list[str] = []
    if cfg_binary:
        candidates.append(cfg_binary)
    if dim is not None:
        env_key = f"OSIRIS_BIN_{dim}D"
        if env_key in os.environ:
            candidates.append(os.environ[env_key])
    if "OSIRIS_BIN" in os.environ:
        candidates.append(os.environ["OSIRIS_BIN"])

    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "No OSIRIS binary found. Set osiris.binary in the manifest or "
        "OSIRIS_BIN / OSIRIS_BIN_<dim>D in the environment. Tried: "
        f"{candidates}"
    )


def have_mpirun() -> bool:
    return shutil.which("mpirun") is not None
