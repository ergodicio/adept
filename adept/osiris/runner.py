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
    # srun launcher warning, not an OSIRIS error: when a task's working directory
    # (e.g. a per-node /dev/shm staging dir) is not resolvable at launch, srun
    # prints "error: couldn't chdir to `...`: ... going to /tmp instead". On
    # Perlmutter this fires spuriously even when the tasks do chdir correctly, so
    # it must not abort an otherwise-clean run. A genuine staging failure surfaces
    # downstream instead (no dumps to drain -> empty/short post-processing).
    "couldn't chdir to",
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


def _sync_tree(src: Path, dst: Path) -> None:
    """Copy every file under ``src`` to the same relative path under ``dst``,
    skipping any that already exist.

    Used at the end of a staged run to fold whatever the background drainer did
    not move (``HIST/``, ``RE/``, ``TIMINGS/``, any straggler ``MS/`` dump) from
    the ephemeral scratch into the durable run directory before the scratch is
    reclaimed. Existing targets (the streamed ``binary/`` NetCDFs, the dumps the
    drainer already mirrored) are left untouched.
    """
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        target = dst / p.relative_to(src)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)


def run_osiris(
    deck_text: str,
    *,
    binary: str | Path,
    mpi_ranks: int = 1,
    run_root: str | Path = "./checkpoints",
    env: dict[str, str] | None = None,
    launcher: str = "srun",
    extra_mpi_args: list[str] | None = None,
    stream_convert: bool = True,
    stream_poll_s: float = 10.0,
    stage_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run OSIRIS and return run metadata.

    Returns a dict with keys ``run_dir`` (Path), ``exit_code`` (int),
    ``wall_time`` (float, seconds), and ``cmd`` (list[str]). When
    ``stream_convert`` is set it also returns ``binary_dir`` (str, the directory
    the concurrent converter wrote into) and ``streamed_diagnostics``
    (sorted list[str] of relpaths fully converted while OSIRIS ran).

    With ``stream_convert=True`` a best-effort background thread
    (:class:`adept.osiris.stream.StreamConverter`) converts grid diagnostics to
    NetCDF *concurrently* with the run, polling every ``stream_poll_s`` seconds,
    so the conversion overlaps compute and dumps are read while still warm in
    cache. It is fully isolated: any failure there is logged and leaves the run
    (and the batch post-processing fallback) untouched.

    **Ramdisk staging (``stage_root``).** When ``stage_root`` is set (and
    ``stream_convert`` is on), OSIRIS runs with its working directory on that
    fast ephemeral filesystem (e.g. ``/dev/shm``) so its *synchronous* dump
    writes hit RAM instead of stalling on the parallel filesystem — effectively
    the asynchronous diagnostic I/O OSIRIS itself lacks. The background drainer
    mirrors each completed dump to the durable run directory under ``run_root``
    and deletes the scratch copy, keeping the ramdisk footprint bounded by the
    poll interval rather than the whole run. ``run_dir`` in the returned dict is
    always the durable directory, so post-processing is unchanged; the scratch
    is reclaimed before returning. Single-node only (``/dev/shm`` is per-node).

    Raises ``RuntimeError`` on non-zero exit code, with the last lines of
    stderr included in the message.
    """
    binary = Path(binary).expanduser().resolve()
    if not binary.exists():
        raise FileNotFoundError(f"OSIRIS binary not found: {binary}")

    # The durable run directory always lives under run_root. With staging, OSIRIS
    # works on a same-named scratch dir under stage_root and the drainer mirrors
    # back here; without it, OSIRIS works in run_dir directly (the original path).
    run_dir = _make_run_dir(Path(run_root).expanduser().resolve())
    staging = bool(stage_root) and stream_convert
    if stage_root and not stream_convert:
        print("[stream] stage_root ignored: ramdisk staging requires stream_convert=True")
    if staging:
        stage_dir: Path | None = Path(stage_root).expanduser().resolve() / run_dir.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        work_dir = stage_dir
        # Keep a durable copy of the deck alongside the mirrored outputs.
        (run_dir / "os-stdin").write_text(deck_text)
    else:
        stage_dir = None
        work_dir = run_dir
    (work_dir / "os-stdin").write_text(deck_text)

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

    # Best-effort concurrent H5 -> NetCDF converter, spawned alongside OSIRIS.
    # It reads OSIRIS's dumps from the working dir and writes NetCDFs into the
    # durable binary/ dir; in staging mode it also mirrors+reaps the scratch
    # into run_dir (persist_dir).
    converter = None
    binary_dir = run_dir / "binary"
    if stream_convert:
        try:
            from adept.osiris import stream as _stream

            converter = _stream.StreamConverter(
                work_dir,
                binary_dir,
                poll_s=stream_poll_s,
                persist_dir=run_dir if staging else None,
            )
        except Exception as e:  # never let the converter block the run
            print(f"[stream] disabled (init failed): {e}")
            converter = None

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
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
    if converter is not None:
        converter.start()
    rc = proc.wait()
    t_out.join()
    t_err.join()
    wall_time = time.time() - t0

    # Finalize the converter before any error handling below: the run is over,
    # so flush whatever the watcher had not yet reached and close the files.
    streamed: list[str] = []
    if converter is not None:
        try:
            streamed = sorted(converter.finalize())
        except Exception as e:
            print(f"[stream] finalize failed (batch fallback will cover it): {e}")

    # Staging: fold anything the drainer did not move (HIST/, RE/, straggler MS/
    # dumps) from the scratch into the durable run dir, then reclaim the ramdisk.
    # Done before error handling so a failed run's partial outputs still persist.
    if staging and stage_dir is not None:
        try:
            _sync_tree(stage_dir, run_dir)
        except Exception as e:
            print(f"[stream] final stage->persist sync issue (partial outputs may be lost): {e}")
        shutil.rmtree(stage_dir, ignore_errors=True)

    if rc != 0:
        tail = "".join(stderr_tail[-50:]) or "(empty stderr)"
        raise RuntimeError(
            f"OSIRIS exited with status {rc}.\n  cmd: {shlex.join(cmd)}\n  cwd: {work_dir}\n  stderr tail:\n{tail}"
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
            f"  cwd: {work_dir}\n"
            f"  stderr tail:\n{err_tail}\n"
            f"  stdout tail:\n{out_tail}"
        )

    result: dict[str, Any] = {
        "run_dir": run_dir,
        "exit_code": rc,
        "wall_time": wall_time,
        "cmd": cmd,
    }
    if stream_convert:
        result["binary_dir"] = str(binary_dir)
        result["streamed_diagnostics"] = streamed
    if staging:
        result["staged"] = True
    return result


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
