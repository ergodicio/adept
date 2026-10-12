"""Incremental / concurrent HDF5 -> NetCDF conversion for OSIRIS runs.

The batch converter in :mod:`adept.osiris.io` (:func:`io.save_run_datasets`)
reads a diagnostic's *entire* ``(t, ...)`` time history into memory and writes
it in one :func:`to_netcdf`. For the SRS field runs that is two problems: it
peaks at the whole stacked series in RAM, and it re-reads ~700k tiny HDF5 files
cold off Lustre at job end (see ``osiris-lpi/postproc-performance.md``).

This module rearranges the same per-dump primitives into a *read-one ->
write-one-slot* loop:

- **Stage A** (:func:`convert_diagnostic_streaming`): run the slot-writer at job
  end. Peak memory drops from the full series to a single dump. No concurrency,
  no write-completeness race.
- **Stage B** (:class:`StreamConverter`): a best-effort background thread that
  drains new dumps into the NetCDF *while OSIRIS is still running*, so the
  conversion overlaps the (hours-long) compute and each dump is read while still
  warm in the node page cache. A final authoritative sweep at job end fills
  whatever the watcher had not yet reached.

Both write the *same* on-disk schema as :func:`io.series_to_dataset`, so the
``binary/*.nc`` contract is unchanged and every downstream reader
(:func:`io.load_series_nc`, the plotting code) works without modification.

Design notes
------------
- **Unlimited ``t`` dimension, append in iteration order.** OSIRIS writes dumps
  sequentially and we process them sorted by iteration, so appending in that
  order is correct. The dimension is grown with ``resize_dimension`` so it ends
  up sized to *exactly* the dumps produced — no pre-sized guess from the deck
  (whose off-by-one is genuinely ambiguous) and no trailing fill slots to trim
  on early termination.
- **Batched, chunk-aligned writes.** :meth:`StreamWriter.append` buffers rows
  in memory and lands them in chunk-sized batches (one HDF5 slice write and one
  ``resize_dimension`` per batch). Appending row-by-row into a gzip chunk of
  ``chunk_t`` rows costs a decompress+recompress of the whole ~1 MiB chunk *per
  row* — ~40x write amplification at the every-3-step field cadence, which is
  what let the drainer fall behind OSIRIS and fill the staging ramdisk in job
  56005549 (see ``osiris-lpi/dev_docs/stream-drainer-recommendations.md``).
  The buffer is bounded by one chunk (~1 MiB); :meth:`StreamWriter.flush`
  (called once per watcher scan) spills any partial batch, so the on-disk file
  trails the consumed dumps by at most one poll interval. Threads are *not*
  used to parallelize conversion: h5py serializes every HDF5 call behind a
  process-global lock, so batching + cheap compression is where the in-process
  throughput is.
- **Parallel drain (``workers > 0``).** Batched writes fix write amplification
  but not the fixed per-dump *read* cost (file open, attribute walk, DataArray
  build, mirror/unlink), which caps a single drain loop at a few hundred tiny
  dumps/s — less than a fast small-box OSIRIS run produces (job 57235103).
  :class:`_WorkerPool` shards whole diagnostics across light subprocess
  workers (sticky assignment, so each NetCDF keeps exactly one writer); the
  parent keeps discovery, the floor-free check, and stats, and falls back to
  the in-thread core if the pool cannot spawn or dies. Enable via the
  ``workers`` argument / ``stream_workers`` cfg / ``ADEPT_STREAM_WORKERS``.
- **Iteration-based bookkeeping.** The writer tracks the last appended OSIRIS
  iteration (persisted in the ``iter`` coordinate, so restarts recover it) and
  the drain loops skip any dump at or below it. Unlike positional slot
  arithmetic this survives corrupt dumps being removed from the listing.
- **Corrupt dumps are quarantined, not fatal.** A dump that cannot be read
  (e.g. truncated by ENOSPC) is renamed to ``*.h5.bad`` — invisible to the
  ``*.h5`` listings — and the stream continues with the next dump. Previously a
  single bad dump dropped the whole writer and the watcher re-hit the same file
  every poll, forever.
- **Backlog spill valve (staging mode).** The ramdisk footprint is only
  "bounded by the poll interval" while conversion keeps up with production.
  When a diagnostic's pending backlog exceeds ``spill_backlog_files`` /
  ``spill_backlog_bytes`` (or the staging filesystem drops below
  ``floor_free_bytes``), the converter stops converting it and *mirrors* its
  dumps straight to the persist ``MS/`` tree instead (a plain copy is ~10x
  cheaper than convert+compress), so the ramdisk keeps draining no matter what.
  A spilled diagnostic stays mirror-only for the rest of the run (resuming the
  stream mid-run would leave an iteration gap in the NetCDF) and is caught up
  from the mirrored HDF5 during :meth:`StreamConverter.finalize`. OSIRIS must
  never see ENOSPC on a dump: in 56005549 that permanently corrupted its
  diagnostic output for the rest of the job.
- **Write-completeness race.** A dump is only safe to read once OSIRIS has moved
  on to the next one, so the watcher processes every dump *except the current
  highest-numbered one*; the final sweep (run is dead) picks up that last dump.
- **Per-dump axis bounds** (``{dim}_min`` / ``{dim}_max``) are always recorded,
  so OSIRIS autoscale (``if_ps_p_auto``) survives and a restart can recover them
  from the file. :func:`io.load_series_nc` only *flags* a dim as autoscaled when
  those bounds actually move, so the constant bounds carried for a fixed spatial
  axis are harmless.
- **Failure isolation.** Nothing here may abort the OSIRIS run: the watcher
  swallows and logs every exception (rate-limited — repeats of the same error
  log once per ``error_log_every`` occurrences), and the batch path in
  :func:`io.save_run_datasets` remains the safety net for anything not produced.
- **Observability.** Every ``stats_every_s`` the watcher logs one line with the
  interval's streamed/spilled/backlog/quarantined counts, so a smoke can assert
  the steady-state backlog is flat *while* OSIRIS runs. Checking the ramdisk
  after the job proves nothing — the drainer always catches up once production
  stops.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr

from adept.osiris import io as _io

# Target uncompressed bytes per (t-chunk x spatial) chunk. A modest t-chunk
# keeps compression/read performance close to the batch path; batched appends
# (see StreamWriter) amortize the write cost of a chunk over its rows. ~1 MiB
# is a reasonable HDF5 chunk size.
_TARGET_CHUNK_BYTES = 1 << 20

_STATS_KEYS = ("appended", "appended_bytes", "spilled", "spilled_bytes", "quarantined")


def _default_chunk_t(spatial_shape: tuple[int, ...]) -> int:
    """Pick a ``t``-chunk so each chunk is roughly :data:`_TARGET_CHUNK_BYTES`."""
    per_row = max(1, int(np.prod(spatial_shape))) * np.dtype(_io._DIAG_DTYPE).itemsize
    return int(max(1, min(512, _TARGET_CHUNK_BYTES // per_row)))


def _data_var_attrs(template: xr.DataArray, source_dir) -> dict:
    """Attrs for the diagnostic variable, matching :func:`io.series_to_dataset`.

    Starts from a single-dump :func:`io.load_grid_h5` DataArray and reproduces
    exactly the attrs the batch path leaves on the data variable: the per-dump
    ``time`` / ``iter`` / ``source`` are dropped, the dict-valued axis metadata
    is lifted onto the coordinate variables instead, ``source_dir`` is added, and
    everything left is coerced to a netCDF-writable scalar/array.
    """
    attrs = dict(template.attrs)
    for k in ("time", "iter", "source", "axis_units", "axis_long_names"):
        attrs.pop(k, None)
    attrs["source_dir"] = str(source_dir)
    return {k: _io._coerce_attr(v) for k, v in attrs.items() if v is not None}


class StreamWriter:
    """Append-only writer for one diagnostic's ``(t, ...)`` NetCDF time series.

    Created from the diagnostic's first dump (the schema template); each
    :meth:`append` buffers the next time slot, and slots are landed on disk in
    chunk-sized batches (or on :meth:`flush` / :meth:`close`), growing the
    unlimited ``t`` dimension once per batch. Reopening an existing file
    (``mode="a"``) resumes after the slots already on disk, which is what makes
    a restart idempotent; ``last_iter`` (recovered from the ``iter`` coordinate
    on resume) lets callers skip dumps already in the file. ``n_written``
    counts consumed dumps — on-disk slots plus the pending batch. The file is
    held open for the lifetime of the writer; call :meth:`close` to finalize.

    The default ``complevel`` is 1: at production dump cadences the drainer is
    compression-bound, and gzip-1 is 2-4x cheaper than the old gzip-4 for a
    modest size cost (offline postproc can always recompress).
    """

    def __init__(
        self, dest, template: xr.DataArray, *, source_dir=None, chunk_t: int | None = None, complevel: int = 1
    ):
        self.dest = Path(dest)
        self.name = str(template.name)
        self.spatial_dims = [str(d) for d in template.dims]
        self.spatial_shape = tuple(int(s) for s in template.shape)
        self._f: h5netcdf.File | None = None
        self._ct = int(chunk_t or _default_chunk_t(self.spatial_shape))
        # The pending batch: rows consumed by append() but not yet on disk.
        self._rows: list[np.ndarray] = []
        self._ts: list[float] = []
        self._its: list[int] = []
        self._bounds: dict[str, list[tuple[float, float]]] = {d: [] for d in self.spatial_dims}

        if self.dest.exists():
            # Resume: trust the existing schema, continue after written slots.
            self._f = h5netcdf.File(self.dest, "a")
            self._n_disk = int(self._f.dimensions["t"].size)
            self.last_iter = int(self._f.variables["iter"][self._n_disk - 1]) if self._n_disk else -1
        else:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            self._f = h5netcdf.File(self.dest, "w")
            self._create_schema(
                template, source_dir if source_dir is not None else self.dest.parent, self._ct, complevel
            )
            self._n_disk = 0
            self.last_iter = -1

    @property
    def n_written(self) -> int:
        return self._n_disk + len(self._rows)

    def _create_schema(self, template, source_dir, chunk_t, complevel) -> None:
        f = self._f
        axis_units = template.attrs.get("axis_units", {}) or {}
        axis_long = template.attrs.get("axis_long_names", {}) or {}

        f.dimensions["t"] = None  # unlimited; grown one batch per landing
        for d, sz in zip(self.spatial_dims, self.spatial_shape, strict=True):
            f.dimensions[d] = sz

        var = f.create_variable(
            self.name,
            ("t", *self.spatial_dims),
            dtype=np.dtype(_io._DIAG_DTYPE),
            chunks=(chunk_t, *self.spatial_shape),
            compression="gzip",
            compression_opts=complevel,
            shuffle=True,
        )
        for k, v in _data_var_attrs(template, source_dir).items():
            var.attrs[k] = v
        # List the non-dimension coordinate variables so xarray promotes them to
        # coordinates (not data variables) on read — the CF convention the batch
        # path's `to_netcdf` also emits.
        nondim = ["iter"] + [f"{d}_{b}" for d in self.spatial_dims for b in ("min", "max")]
        var.attrs["coordinates"] = " ".join(nondim)

        f.create_variable("t", ("t",), dtype="f8")
        f.create_variable("iter", ("t",), dtype="i8")
        for d in self.spatial_dims:
            cv = f.create_variable(d, (d,), dtype="f8")
            cv[:] = np.asarray(template.coords[d].values, dtype="f8")
            if axis_units.get(d):
                cv.attrs["units"] = axis_units[d]
            if axis_long.get(d):
                cv.attrs["long_name"] = axis_long[d]
            f.create_variable(f"{d}_min", ("t",), dtype="f8")
            f.create_variable(f"{d}_max", ("t",), dtype="f8")

    def append(self, da: xr.DataArray) -> None:
        """Buffer ``da`` (one dump) as the next time slot; lands in batches."""
        if da.shape != self.spatial_shape:
            raise ValueError(f"shape mismatch for {self.name}: {da.shape} != {self.spatial_shape}")
        self._rows.append(np.asarray(da.values, dtype=_io._DIAG_DTYPE))
        self._ts.append(float(da.attrs.get("time", np.nan)))
        it = int(da.attrs.get("iter", -1))
        self._its.append(it)
        self.last_iter = max(self.last_iter, it)
        for d in self.spatial_dims:
            cv = np.asarray(da.coords[d].values)
            self._bounds[d].append((float(cv[0]), float(cv[-1])) if cv.size else (np.nan, np.nan))
        if len(self._rows) >= self._ct:
            self._write_batch()

    def _write_batch(self) -> None:
        """Land the pending rows: one resize + one slice write per variable."""
        k = len(self._rows)
        if not k or self._f is None:
            return
        f = self._f
        i = self._n_disk
        f.resize_dimension("t", i + k)
        f.variables[self.name][i : i + k, ...] = np.stack(self._rows)
        f.variables["t"][i : i + k] = np.asarray(self._ts, dtype="f8")
        f.variables["iter"][i : i + k] = np.asarray(self._its, dtype="i8")
        for d in self.spatial_dims:
            b = np.asarray(self._bounds[d], dtype="f8")
            f.variables[f"{d}_min"][i : i + k] = b[:, 0]
            f.variables[f"{d}_max"][i : i + k] = b[:, 1]
        self._n_disk = i + k
        self._rows.clear()
        self._ts.clear()
        self._its.clear()
        for d in self.spatial_dims:
            self._bounds[d].clear()

    def flush(self) -> None:
        if self._f is not None:
            self._write_batch()
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._write_batch()
            self._f.close()
            self._f = None


def convert_diagnostic_streaming(diag_dir, dest, *, source_dir=None) -> Path:
    """Stage A: convert one grid diagnostic to NetCDF a dump at a time.

    A memory-bounded drop-in for ``series_to_dataset(load_series(diag_dir))``
    followed by ``to_netcdf`` — peak RAM is one dump plus one write batch
    (~1 MiB), not the whole stacked series. Rebuilds ``dest`` from scratch (any
    partial file is removed first).
    """
    diag_dir = Path(diag_dir)
    dest = Path(dest)
    dumps = _io._sort_dumps(diag_dir)
    if not dumps:
        raise FileNotFoundError(f"No .h5 dumps in {diag_dir}")
    if dest.exists():
        dest.unlink()

    template = _io.load_grid_h5(dumps[0])
    writer = StreamWriter(dest, template, source_dir=source_dir or diag_dir)
    try:
        writer.append(template)
        for p in dumps[1:]:
            writer.append(_io.load_grid_h5(p))
    finally:
        writer.close()
    return dest


class _DrainCore:
    """The per-diagnostic drain state machine, independent of any scheduling.

    Owns the writers, spill/quarantine state, and counters for a set of
    diagnostics, and drains one diagnostic per :meth:`drain_rel` call. Used
    in-thread by :class:`StreamConverter` (``workers=0``) and as the entire
    body of a drain worker process (``--drain-worker``), so it must not touch
    threads, discovery, or the poll loop — the caller decides *when* and *for
    which* diagnostics it runs.
    """

    def __init__(
        self,
        out_dir,
        *,
        persist_dir=None,
        discard_grid_h5: bool = False,
        spill_backlog_files: int = 20_000,
        spill_backlog_bytes: int = 4 << 30,
        error_log_every: int = 200,
        logger=print,
    ):
        self.out_dir = Path(out_dir)
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self.persist_ms = (self.persist_dir / "MS") if self.persist_dir is not None else None
        self.discard_grid_h5 = bool(discard_grid_h5)
        self.spill_backlog_files = int(spill_backlog_files or 0)
        self.spill_backlog_bytes = int(spill_backlog_bytes or 0)
        self.error_log_every = max(1, int(error_log_every))
        self._log = logger
        self._writers: dict[str, StreamWriter] = {}
        self._completed: set[str] = set()  # grid relpaths fully converted to .nc
        self._raw_completed: set[str] = set()  # raw relpaths fully mirrored to persist
        self._is_raw: dict[str, bool] = {}  # rel -> raw? (cached; avoids re-peeking h5)
        self._spilled: set[str] = set()  # grid rels in sticky mirror-only mode
        self._bad: set[Path] = set()  # unreadable dumps that could not be renamed away
        self._err_counts: dict[str, int] = {}
        self._stats = {"appended": 0, "appended_bytes": 0, "spilled": 0, "spilled_bytes": 0, "quarantined": 0}

    # --- logging ----------------------------------------------------------

    def _log_every(self, key: str, msg: str) -> None:
        """Log ``msg`` on the first occurrence of ``key`` and every Nth after."""
        n = self._err_counts.get(key, 0) + 1
        self._err_counts[key] = n
        if n == 1 or n % self.error_log_every == 0:
            self._log(msg + (f" [seen x{n}]" if n > 1 else ""))

    # --- entry point ------------------------------------------------------

    def drain_rel(self, rel: str, d: Path, *, final: bool, force_spill: bool) -> tuple[int, int]:
        """Drain one diagnostic dir; returns its (files, bytes) pending at entry.

        Never raises: any error is logged and the diagnostic falls back to the
        batch pass (its half-written writer is dropped so the file is rebuilt
        rather than reused).
        """
        if rel in self._completed or rel in self._raw_completed:
            return 0, 0
        try:
            if self._diag_is_raw(rel, d):
                # RAW: nothing to do unless staging (then mirror + reap the
                # HDF5 so the scratch stays bounded and the batch path finds
                # the dumps on persist).
                if self.persist_dir is not None:
                    self._drain_raw(rel, d, final=final)
                return 0, 0
            return self._drain_grid(rel, d, final=final, force_spill=force_spill)
        except Exception as e:
            self._log_every(f"drain:{rel}", f"[stream] {rel}: {e} (will fall back to batch)")
            # Drop the writer so a half-written file is rebuilt by the batch
            # pass rather than reused.
            w = self._writers.pop(rel, None)
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
            return 0, 0

    def close_all(self) -> set[str]:
        """Close every writer; returns the grid rels fully converted."""
        for rel, w in self._writers.items():
            try:
                w.close()
            except Exception:
                pass
            self._completed.add(rel)
        self._writers.clear()
        return set(self._completed)

    def _diag_is_raw(self, rel: str, d: Path) -> bool:
        cached = self._is_raw.get(rel)
        if cached is None:
            cached = _io._diag_is_raw(rel, d)
            self._is_raw[rel] = cached
        return cached

    # --- draining ---------------------------------------------------------

    def _drain_grid(self, rel: str, d: Path, *, final: bool, force_spill: bool) -> tuple[int, int]:
        """Drain one grid diagnostic; returns its (files, bytes) pending at entry.

        Non-staged: stream to NetCDF in place — no mirror, no reap. Staged:
        stream + reap (mirroring unless ``discard_grid_h5``), or mirror-only
        when spilling. The final sweep additionally catches a spilled
        diagnostic up from its persist mirror before closing.
        """
        staged = self.persist_dir is not None
        dumps = [p for p in _io._sort_dumps(d) if p not in self._bad]
        # Write-completeness: a dump is safe only once the next one exists
        # (OSIRIS has moved on). The final sweep runs after OSIRIS is dead, so
        # the last dump is safe too.
        safe = dumps if final else dumps[:-1]
        n_pending = len(safe)
        pending_bytes = 0
        if staged:
            for p in safe:
                try:
                    pending_bytes += p.stat().st_size
                except OSError:
                    pass

        if staged and not final:
            spill = (
                force_spill
                or rel in self._spilled
                or (self.spill_backlog_files and n_pending > self.spill_backlog_files)
                or (self.spill_backlog_bytes and pending_bytes > self.spill_backlog_bytes)
            )
            if spill:
                if rel not in self._spilled:
                    self._spilled.add(rel)
                    self._log(
                        f"[stream] {rel}: backlog {n_pending} dumps / {pending_bytes >> 20} MiB "
                        "over spill threshold -- mirror-only until finalize"
                    )
                for p in safe:
                    # Always mirror when spilling: these bytes exist nowhere else.
                    self._reap(rel, p, mirror=True)
                self._stats["spilled"] += n_pending
                self._stats["spilled_bytes"] += pending_bytes
                return n_pending, pending_bytes

        candidates = list(safe)
        if final and staged and rel in self._spilled:
            # Catch up from the persist mirror first; merge by iteration so the
            # append order stays monotonic.
            spill_dir = self.persist_ms / rel
            if spill_dir.is_dir():
                candidates = sorted(
                    [p for p in _io._sort_dumps(spill_dir) if p not in self._bad] + candidates,
                    key=_io._iter_from_name,
                )
        if not candidates:
            return n_pending, pending_bytes

        w = self._writers.get(rel)
        if w is None:
            try:
                w, candidates = self._open_writer(rel, d, candidates)
            except Exception:
                dest = self.out_dir / f"{rel}.nc"
                if not (staged and not self.discard_grid_h5 and dest.exists()):
                    raise  # nothing safe to do here; drain_rel falls back to batch
                # Resuming a previous owner's file failed — e.g. that drain
                # worker was killed mid-write and left the HDF5 unreadable.
                # In mirror mode every appended dump also exists under
                # persist/MS, so sideline the broken file and go mirror-only;
                # the finalize catch-up rebuilds the complete series fresh.
                os.replace(dest, dest.with_name(dest.name + ".corrupt"))
                self._spilled.add(rel)
                self._log(
                    f"[stream] {rel}: existing NetCDF unreadable on resume -- sidelined to "
                    f"{dest.name}.corrupt, mirror-only until the finalize rebuild"
                )
                if final:
                    # Redo this drain: with the rel now spilled it merges the
                    # persist mirror into the candidates and opens a fresh file.
                    return self._drain_grid(rel, d, final=True, force_spill=force_spill)
                for p in candidates:
                    self._reap(rel, p, mirror=True)
                self._stats["spilled"] += n_pending
                self._stats["spilled_bytes"] += pending_bytes
                return n_pending, pending_bytes
            if w is None:
                return n_pending, pending_bytes

        for p in candidates:
            if _io._iter_from_name(p) <= w.last_iter:
                # Already in the file (resume / idempotency): just tidy up.
                self._cleanup_consumed(rel, p)
                continue
            loaded = self._load_dump(rel, p)
            if loaded is None:
                continue
            da, src = loaded
            w.append(da)
            self._stats["appended"] += 1
            self._stats["appended_bytes"] += int(da.values.nbytes)
            self._cleanup_consumed(rel, src)
        w.flush()

        if final:
            w.close()
            self._completed.add(rel)
            if staged and self.discard_grid_h5:
                self._prune_empty(self.persist_ms / rel)
        return n_pending, pending_bytes

    def _open_writer(self, rel: str, d: Path, candidates: list[Path]) -> tuple[StreamWriter | None, list[Path]]:
        """Create (or reopen) the writer from the first readable candidate."""
        while candidates:
            p0 = candidates[0]
            loaded = self._load_dump(rel, p0)
            if loaded is None:
                candidates = candidates[1:]
                continue
            template, src = loaded
            w = StreamWriter(self.out_dir / f"{rel}.nc", template, source_dir=d)
            self._writers[rel] = w
            if w.n_written == 0:
                w.append(template)  # reuse the template we just loaded as slot 0
                self._stats["appended"] += 1
                self._stats["appended_bytes"] += int(template.values.nbytes)
                self._cleanup_consumed(rel, src)
                candidates = candidates[1:]
            # On resume (n_written > 0) p0 stays; the iteration filter decides.
            return w, candidates
        return None, []

    def _dump_sources(self, rel: str, p: Path) -> list[Path]:
        """Every place a listed dump might be readable, most likely first.

        A staged dump can be mirrored to the persist tree and reaped off the
        scratch between the directory listing and the read — a *move*, not
        corruption — so the mirrored twin is a legitimate second source.
        """
        srcs = [p]
        if self.persist_ms is not None and self.persist_ms not in p.parents:
            srcs.append(self.persist_ms / rel / p.name)
        return srcs

    def _load_dump(self, rel: str, p: Path) -> tuple[xr.DataArray, Path] | None:
        """Load one dump as ``(data, the path it came from)``, following a move.

        Tries the listed path, then the persist mirror: a dump that vanished is
        one another sweep already mirrored, and must not be quarantined —
        quarantining it is the second half of the ``FLD/e2`` truncation in job
        57235103, where the finalize sweep hit ``ENOENT`` on the first such dump
        and dropped every iteration from there to the end of the run. A dump
        that is *present but unreadable* is still corruption and is quarantined
        as before. Returns ``None`` (one rate-limited log line) when no source
        yields data.
        """
        for src in self._dump_sources(rel, p):
            try:
                return _io.load_grid_h5(src), src
            except Exception as e:
                if src.exists():  # present but unreadable: genuine corruption
                    self._quarantine(rel, src, e)
                    return None
        self._log_every(f"gone:{rel}", f"[stream] {rel}: dump {p.name} vanished before it could be read")
        return None

    def _cleanup_consumed(self, rel: str, p: Path) -> None:
        """Dispose of one dump whose contents are (now) in the NetCDF."""
        if self.persist_dir is None:
            return  # non-staged: the in-place tree is the durable copy
        if self.persist_ms is not None and self.persist_ms in p.parents:
            # A spilled dump drained from the persist mirror during finalize:
            # in discard mode the .nc is the durable artifact, drop the HDF5.
            if self.discard_grid_h5:
                try:
                    p.unlink()
                except OSError:
                    pass
            return
        self._reap(rel, p, mirror=not self.discard_grid_h5)

    def _quarantine(self, rel: str, p: Path, err: Exception) -> None:
        """Sideline an unreadable dump (``*.h5.bad``) and keep streaming.

        Truncated dumps are exactly what an ENOSPC-choked run leaves behind;
        one of them must not stall the whole diagnostic (previously the writer
        was dropped and the watcher re-hit the same file every poll)."""
        self._stats["quarantined"] += 1
        self._log_every(f"bad:{rel}", f"[stream] {rel}: quarantining unreadable dump {p.name}: {err}")
        try:
            p.rename(p.with_name(p.name + ".bad"))  # ".bad": invisible to *.h5 listings
        except OSError:
            self._bad.add(p)  # cannot rename: remember and skip it from now on

    def _prune_empty(self, dirpath: Path) -> None:
        """Remove now-empty persist-mirror directories after a discard catch-up."""
        if self.persist_ms is None:
            return
        cur = dirpath
        try:
            while (
                (cur == self.persist_ms or self.persist_ms in cur.parents) and cur.is_dir() and not any(cur.iterdir())
            ):
                cur.rmdir()
                cur = cur.parent
        except OSError:
            pass

    def _drain_raw(self, rel: str, d: Path, *, final: bool) -> None:
        """Mirror a RAW diagnostic's completed dumps to persist + reap scratch."""
        dumps = _io._sort_dumps(d)
        if not dumps:
            return
        safe = dumps if final else dumps[:-1]
        for p in safe:
            self._reap(rel, p)
        if final:
            self._raw_completed.add(rel)

    def _reap(self, rel: str, p: Path, *, mirror: bool = True) -> None:
        """Delete one scratch dump, first mirroring it to ``persist_dir/MS``
        unless ``mirror`` is False (``discard_grid_h5`` mode: the dump is
        already in the streamed NetCDF and the raw HDF5 is not wanted on the
        persist filesystem). Best-effort: on any failure the scratch file is
        left in place for the runner's final sync to catch, so data is never
        lost and the watcher never crashes."""
        if self.persist_dir is None:
            return
        try:
            if mirror:
                dest = self.persist_ms / rel / p.name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + ".part")
                    shutil.copyfile(p, tmp)
                    os.replace(tmp, dest)  # atomic publish on the persist filesystem
            p.unlink()
        except Exception as e:
            self._log_every(f"reap:{rel}", f"[stream] reap {rel}/{p.name} deferred: {e}")


class StreamConverter:
    """Stage B: a background thread that drains diagnostics during the run.

    Spawned by :func:`adept.osiris.runner.run_osiris` alongside the OSIRIS
    subprocess, it polls ``run_dir/MS`` and drains each grid diagnostic's
    completed dumps into ``out_dir/<relpath>.nc``. RAW (particle) diagnostics do
    not fit the fixed-slot model (variable particle count per dump) and are left
    to the batch path. Every operation is best-effort: any error is logged
    (rate-limited) and the run is never affected.

    **Staging mode (``persist_dir`` set).** When OSIRIS writes its ``MS/`` dumps
    to a fast ephemeral scratch (e.g. a ``/dev/shm`` ramdisk passed as
    ``run_dir``), the converter also *drains* the scratch: after each completed
    dump is handled it is mirrored to ``persist_dir/MS/<relpath>/`` and
    **deleted from the scratch**, so the ramdisk high-water mark is bounded by
    the poll interval rather than the whole run. Grid diagnostics are mirrored
    *and* streamed to ``out_dir`` (which the caller points at durable storage);
    RAW diagnostics, which cannot be slot-streamed, are mirrored as HDF5 so the
    batch post-processing path finds them on ``persist_dir``. The net durable
    layout under ``persist_dir`` is identical to a non-staged run, so
    post-processing reads it unchanged. With ``persist_dir=None`` nothing is
    mirrored or deleted — the converter only streams grid diagnostics in place,
    exactly as before.

    **Backlog spill valve (staging mode).** The bound above only holds while
    conversion keeps up with OSIRIS. When a diagnostic's pending backlog
    exceeds ``spill_backlog_files`` or ``spill_backlog_bytes`` — or the staging
    filesystem's free space drops below ``floor_free_bytes`` — that diagnostic
    switches to mirror-only draining (plain copy to ``persist_dir/MS/``, ~10x
    cheaper than convert+compress), so the ramdisk keeps draining no matter
    what. A spilled diagnostic stays mirror-only for the rest of the run
    (resuming the stream mid-run would leave an iteration gap in the NetCDF)
    and its NetCDF is caught up from the mirrored HDF5 in :meth:`finalize`; in
    ``discard_grid_h5`` mode the mirrored copies are deleted again once
    appended.
    """

    def __init__(
        self,
        run_dir,
        out_dir,
        *,
        poll_s: float = 10.0,
        logger=print,
        persist_dir=None,
        discard_grid_h5: bool = False,
        spill_backlog_files: int = 20_000,
        spill_backlog_bytes: int = 4 << 30,
        floor_free_bytes: int = 8 << 30,
        stats_every_s: float = 60.0,
        rediscover_every: int = 12,
        error_log_every: int = 200,
        workers: int | None = None,
        finalize_join_s: float = 1800.0,
    ):
        self.ms = Path(run_dir) / "MS"
        self.out_dir = Path(out_dir)
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self.persist_ms = (self.persist_dir / "MS") if self.persist_dir is not None else None
        # Staging only: when set, grid dumps are deleted from the scratch after
        # being appended to the NetCDF *without* being mirrored to persist_dir/MS
        # (the .nc is the durable artifact). Saves one inode+copy per dump on the
        # persist filesystem. RAW dumps are always mirrored (the batch path needs
        # the HDF5), and so are spilled grid dumps (their bytes exist nowhere
        # else until the finalize catch-up appends them).
        self.discard_grid_h5 = bool(discard_grid_h5)
        self.poll_s = float(poll_s)
        self.spill_backlog_files = int(spill_backlog_files or 0)
        self.spill_backlog_bytes = int(spill_backlog_bytes or 0)
        self.floor_free_bytes = int(floor_free_bytes or 0)
        self.stats_every_s = float(stats_every_s)
        self.rediscover_every = max(1, int(rediscover_every))
        self.error_log_every = max(1, int(error_log_every))
        # How long :meth:`finalize` waits for the watcher to finish its current
        # scan. Must outlast one scan over the deepest backlog the run reaches
        # (minutes on a fast small-box run), not one poll interval — see
        # :meth:`finalize` for what a premature timeout costs.
        self.finalize_join_s = float(finalize_join_s)
        self._log = logger
        self._multibase_skipped: set[str] = set()  # multi-report dirs left to the batch pass
        self._err_counts: dict[str, int] = {}
        self._known: dict[str, Path] = {}  # cached diagnostic discovery
        self._scan_i = 0
        self._stats_snapshot = dict.fromkeys(_STATS_KEYS, 0)
        self._stats_t = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if workers is None:
            workers = int(os.environ.get("ADEPT_STREAM_WORKERS", "0") or 0)
        self.workers = max(0, int(workers))
        # In-thread drain core: the only path when workers == 0, and the
        # fallback when the worker pool cannot be spawned or dies mid-run.
        self._core = _DrainCore(
            self.out_dir,
            persist_dir=self.persist_dir,
            discard_grid_h5=self.discard_grid_h5,
            spill_backlog_files=self.spill_backlog_files,
            spill_backlog_bytes=self.spill_backlog_bytes,
            error_log_every=self.error_log_every,
            logger=logger,
        )
        # Worker-process config: same knobs, stringly typed for the wire.
        # Paths absolute so workers don't depend on inheriting the parent cwd.
        self._core_cfg = {
            "out_dir": str(self.out_dir.absolute()),
            "persist_dir": str(self.persist_dir.absolute()) if self.persist_dir is not None else None,
            "discard_grid_h5": self.discard_grid_h5,
            "spill_backlog_files": self.spill_backlog_files,
            "spill_backlog_bytes": self.spill_backlog_bytes,
            "error_log_every": self.error_log_every,
        }
        self._pool: _WorkerPool | None = None
        self._assign: dict[str, int] = {}  # sticky rel -> worker id (writer ownership)
        # Union of every core's spilled rels. Shipped with each scan so a rel
        # that moves to a new owner (worker death) stays mirror-only — resuming
        # its stream mid-run would leave an iteration gap in the NetCDF.
        self._spilled_global: set[str] = set()

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._start_pool()
        self._thread = threading.Thread(target=self._run, name="osiris-stream-convert", daemon=True)
        self._thread.start()

    def _start_pool(self) -> None:
        if self.workers > 0 and self._pool is None:
            try:
                self._pool = _WorkerPool(self.workers, self._core_cfg, logger=self._log)
            except Exception as e:  # pool is an optimization, never a requirement
                self._log(f"[stream] drain worker pool unavailable, draining in-thread: {e}")
                self._pool = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once(final=False)
            except Exception as e:  # a watcher must never crash the run
                self._log_every("scan", f"[stream] scan error (continuing): {e}")
            self._stop.wait(self.poll_s)

    def finalize(self) -> set[str]:
        """Stop the thread, run the authoritative final sweep, close all writers.

        After OSIRIS has exited every dump is safe to read (no writer can still
        be racing), so the final sweep processes the last dump each diagnostic
        had been holding back, catches spilled diagnostics up from their persist
        mirror, and (in staging mode) drains everything off the scratch before
        closing the files. Returns the set of *grid* diagnostic relpaths fully
        converted by the stream (so the caller can skip rebuilding them in the
        batch pass); RAW relpaths are mirrored but not returned, since the batch
        pass still builds their NetCDFs from the mirrored HDF5.

        The watcher must be **finished**, not merely asked to stop, before the
        sweep starts: both drain the same diagnostics through the same writers,
        so two concurrent sweeps put two writers on one NetCDF — each resizing
        ``t`` against its own stale ``_n_disk`` (duplicated / out-of-order rows,
        then ``Can't broadcast``) and each mirroring+reaping dumps the other is
        about to read. That is what corrupted ``FLD/e2`` in job 57235103: the
        old join timed out after one *poll interval* while a scan over a deep
        backlog was still running for minutes. ``finalize_join_s`` has to
        outlast one scan instead; if the watcher is somehow still going after
        it, the sweep is skipped rather than run alongside — the batch pass in
        :func:`io.save_run_datasets` rebuilds whatever is missing.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.finalize_join_s)
        swept = self._thread is None or not self._thread.is_alive()
        if swept:
            try:
                self._scan_once(final=True)
            except Exception as e:
                self._log(f"[stream] finalize sweep error (batch fallback covers it): {e}")
        else:
            self._log(
                f"[stream] CRITICAL: watcher still draining after {self.finalize_join_s:.0f}s -- "
                "skipping the final sweep rather than running a second concurrent writer "
                "(the batch pass rebuilds what is missing)"
            )
        completed: set[str] = set()
        if self._pool is not None:
            try:
                completed |= self._pool.close()
            except Exception as e:
                self._log(f"[stream] worker pool close error (batch fallback covers it): {e}")
        completed |= self._core.close_all()
        # Without the authoritative sweep no NetCDF can be claimed complete (the
        # watcher may still have been appending to any of them), so hand the
        # batch pass everything rather than a half-drained set.
        return completed if swept else set()

    # --- logging / stats --------------------------------------------------

    def _log_every(self, key: str, msg: str) -> None:
        """Log ``msg`` on the first occurrence of ``key`` and every Nth after.

        The ENOSPC failure mode repeats the *same* error for every dump on
        every poll; unthrottled, that was megabytes of log spam per hour.
        """
        n = self._err_counts.get(key, 0) + 1
        self._err_counts[key] = n
        if n == 1 or n % self.error_log_every == 0:
            self._log(msg + (f" [seen x{n}]" if n > 1 else ""))

    @property
    def _spilled(self) -> set[str]:
        """Union of spilled rels across the in-thread core and all workers
        (introspection/debugging view; the authoritative copies live in the
        cores, kept coherent via ``_spilled_global``)."""
        return set(self._core._spilled) | self._spilled_global

    def _totals(self) -> dict[str, int]:
        """Cumulative drain counters across the in-thread core and all workers."""
        totals = dict(self._core._stats)
        if self._pool is not None:
            for k, v in self._pool.totals().items():
                totals[k] += v
        return totals

    def _maybe_stats(self, totals: dict[str, int], backlog_files: int, backlog_bytes: int, *, final: bool) -> None:
        """One drain-vs-production line per ``stats_every_s`` (and at finalize)."""
        now = time.monotonic()
        if not final and (now - self._stats_t) < self.stats_every_s:
            return
        delta = {k: totals[k] - self._stats_snapshot[k] for k in totals}
        if final or any(delta.values()) or backlog_files:
            extra = ""
            if self.persist_dir is not None:
                try:
                    extra = f", stage free {shutil.disk_usage(self.ms).free >> 30} GiB"
                except OSError:
                    pass
            self._log(
                f"[stream] stats({now - self._stats_t:.0f}s): streamed {delta['appended']} dumps "
                f"({delta['appended_bytes'] >> 20} MiB), spilled {delta['spilled']} "
                f"({delta['spilled_bytes'] >> 20} MiB), backlog {backlog_files} dumps "
                f"({backlog_bytes >> 20} MiB), quarantined {delta['quarantined']}{extra}"
            )
        self._stats_t = now
        self._stats_snapshot = dict(totals)

    # --- scanning ---------------------------------------------------------

    def _diags(self) -> dict[str, Path]:
        """Every diagnostic dir under ``MS/`` (grid and RAW), keyed by relpath."""
        out: dict[str, Path] = {}
        if not self.ms.is_dir():
            return out
        for d in self.ms.rglob("*"):
            if not d.is_dir():
                continue
            try:
                groups = _io._series_dumps(d)
            except OSError:
                continue
            if not groups:
                continue
            rel = str(d.relative_to(self.ms))
            if len(groups) > 1:
                # A directory holding several report series (e.g. two s1 line
                # lineouts) can't be streamed as one series; leave it to the
                # batch pass (save_run_datasets), which writes one NetCDF per
                # series via the per-report handles from list_diagnostics.
                if rel not in self._multibase_skipped:
                    self._multibase_skipped.add(rel)
                    self._log(f"[stream] {rel}: {len(groups)} report series -> batch pass")
                continue
            out[rel] = d
        return out

    def _discover(self, *, final: bool) -> dict[str, Path]:
        """Cached discovery. The full ``rglob`` walk touches every backlogged
        file, so its cost grows exactly when the drainer is behind; new
        diagnostic *directories* appear rarely, so the walk runs only every
        ``rediscover_every`` polls and the known dirs are re-listed directly in
        between."""
        if final or not self._known or (self._scan_i % self.rediscover_every == 1):
            self._known = self._diags()
        return self._known

    def _scan_once(self, *, final: bool) -> None:
        self._scan_i += 1
        force_spill = False
        if self.persist_dir is not None and not final and self.floor_free_bytes:
            try:
                if shutil.disk_usage(self.ms).free < self.floor_free_bytes:
                    force_spill = True
                    self._log_every(
                        "floor",
                        f"[stream] CRITICAL: staging filesystem under {self.floor_free_bytes >> 30} GiB free"
                        " -- spilling all grid dumps to persist (OSIRIS must never see ENOSPC)",
                    )
            except OSError:
                pass
        rel_map = self._discover(final=final)
        if self._pool is not None and self._pool.alive_ids():
            backlog_files, backlog_bytes = self._scan_pool(rel_map, final=final, force_spill=force_spill)
        else:
            # In-thread drain (workers == 0, pool never came up, or pool died).
            self._core._spilled.update(self._spilled_global)
            backlog_files = 0
            backlog_bytes = 0
            for rel, d in rel_map.items():
                nf, nb = self._core.drain_rel(rel, d, final=final, force_spill=force_spill)
                backlog_files += nf
                backlog_bytes += nb
            self._spilled_global |= self._core._spilled
        self._maybe_stats(self._totals(), backlog_files, backlog_bytes, final=final)

    def _partition(self, rel_map: dict[str, Path]) -> dict[int, dict[str, str]]:
        """Sticky, balanced rel -> worker assignment.

        A rel keeps its worker for the run (the worker holds the open writer);
        new rels go to the least-loaded live worker, and rels owned by a dead
        worker are reassigned once (the new owner's writer reopens the NetCDF
        in append mode and resumes from its ``iter`` coordinate).
        """
        alive = self._pool.alive_ids()
        counts = dict.fromkeys(alive, 0)
        for i in self._assign.values():
            if i in counts:
                counts[i] += 1
        for rel in sorted(rel_map):
            if self._assign.get(rel) not in counts:
                target = min(counts, key=counts.get)
                self._assign[rel] = target
                counts[target] += 1
        buckets: dict[int, dict[str, str]] = {i: {} for i in alive}
        for rel, d in rel_map.items():
            buckets[self._assign[rel]][rel] = str(d)
        return buckets

    def _scan_pool(self, rel_map: dict[str, Path], *, final: bool, force_spill: bool) -> tuple[int, int]:
        buckets = self._partition(rel_map)
        backlog_files, backlog_bytes = self._pool.scan(
            buckets, final=final, force_spill=force_spill, spilled=self._spilled_global
        )
        if not self._pool.alive_ids():
            self._log("[stream] CRITICAL: all drain workers died -- draining in-thread from next scan")
        return backlog_files, backlog_bytes


class _WorkerPool:
    """N drain-worker subprocesses, each owning a disjoint set of diagnostics.

    Workers are separate *processes* because h5py serializes every HDF5 call
    behind one process-global lock — in-process threads cannot parallelize the
    drain, and the fixed per-dump cost (open, attribute walk, DataArray build,
    mirror/unlink) is where the throughput ceiling is (job 57235103: the
    nx=1797 boxes out-produced a single drain loop from the first scan).

    Each worker is ``python -m adept.osiris.stream --drain-worker <socket>``;
    ``adept``'s package ``__init__`` resolves solvers lazily, so the worker
    imports only the light stack (h5py/numpy/xarray, ~1 s — no jax), and runs a
    plain :class:`_DrainCore` fed over an AF_UNIX :mod:`multiprocessing.connection`.
    Message protocol (pickled tuples), parent -> worker::

        ("init", core_cfg)                                  -> ("ready",)
        ("scan", {rel: dir}, final, force_spill, [spilled]) -> ("done", reply)
        ("close",)                                          -> ("closed", [rels])

    with ``("log", msg)`` messages interleaved worker -> parent at any time.
    Everything is best-effort: a worker that dies is dropped (its rels are
    reassigned by the parent's sticky partition), and the parent falls back to
    the in-thread core when none are left.
    """

    def __init__(self, n: int, core_cfg: dict, *, logger=print, connect_timeout_s: float = 120.0):
        self._log = logger
        self._tmp = tempfile.mkdtemp(prefix="adept-stream-")  # node-local, short AF_UNIX path
        self._authkey = secrets.token_bytes(16)
        address = os.path.join(self._tmp, "sock")
        self._listener = Listener(address, family="AF_UNIX", authkey=self._authkey)
        env = dict(
            os.environ,
            ADEPT_STREAM_AUTHKEY=self._authkey.hex(),
            OMP_NUM_THREADS="1",  # conversion is IO/gzip-bound; don't oversubscribe cores
        )
        self._procs: list[subprocess.Popen | None] = []
        self._conns: list = []
        self._cum: list[dict[str, int]] = []  # per-worker cumulative counters (last seen)
        self._dead: dict[str, int] = dict.fromkeys(_STATS_KEYS, 0)  # retained from dead workers
        for i in range(int(n)):
            # stdout/stderr inherit the run's log so worker tracebacks are visible.
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "adept.osiris.stream", "--drain-worker", address],
                env=env,
            )
            conn = self._accept(connect_timeout_s, proc)
            ok = False
            if conn is not None:
                try:
                    conn.send(("init", core_cfg))
                    ok = self._await(conn, proc, ("ready",)) is not None
                except Exception:
                    ok = False
            if not ok:
                self._log(f"[stream] drain worker {i} failed to start")
                if conn is not None:
                    conn.close()
                proc.kill()
                proc.wait()
                self._procs.append(None)
                self._conns.append(None)
            else:
                self._procs.append(proc)
                self._conns.append(conn)
            self._cum.append(dict.fromkeys(_STATS_KEYS, 0))
        alive = self.alive_ids()
        if not alive:
            self.close()
            raise RuntimeError(f"none of {n} drain workers came up")
        self._log(f"[stream] drain pool: {len(alive)}/{n} workers up")

    def _accept(self, timeout_s: float, proc: subprocess.Popen):
        """Accept one worker connection, bounded by ``timeout_s`` (a crashed
        child would otherwise block ``Listener.accept`` forever)."""
        out: list = []

        def _do() -> None:
            try:
                out.append(self._listener.accept())
            except OSError:
                pass

        th = threading.Thread(target=_do, daemon=True)
        th.start()
        deadline = time.monotonic() + timeout_s
        while th.is_alive() and time.monotonic() < deadline:
            if proc.poll() is not None:
                return None  # child died before connecting
            th.join(timeout=0.5)
        return out[0] if out else None

    def _await(self, conn, proc, kinds: tuple[str, ...]):
        """Receive until a message of ``kinds`` arrives, forwarding logs.

        Returns None when the worker is dead (EOF, or exited with nothing left
        in flight). No overall deadline: a legitimately busy scan over a deep
        backlog can take many minutes."""
        while True:
            try:
                if conn.poll(1.0):
                    msg = conn.recv()
                elif proc is not None and proc.poll() is not None:
                    if not conn.poll(0.5):  # drain anything already in flight
                        return None
                    msg = conn.recv()
                else:
                    continue
            except (EOFError, OSError):
                return None
            if msg[0] == "log":
                self._log(msg[1])
            elif msg[0] in kinds:
                return msg

    def alive_ids(self) -> list[int]:
        return [i for i, c in enumerate(self._conns) if c is not None]

    def totals(self) -> dict[str, int]:
        totals = dict(self._dead)
        for i in self.alive_ids():
            for k in _STATS_KEYS:
                totals[k] += self._cum[i][k]
        return totals

    def _mark_dead(self, i: int) -> None:
        conn, proc = self._conns[i], self._procs[i]
        self._conns[i] = None
        self._procs[i] = None
        for k in _STATS_KEYS:
            self._dead[k] += self._cum[i][k]
        self._cum[i] = dict.fromkeys(_STATS_KEYS, 0)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._log(f"[stream] CRITICAL: drain worker {i} died; its diagnostics will be reassigned")

    def scan(
        self, buckets: dict[int, dict[str, str]], *, final: bool, force_spill: bool, spilled: set[str]
    ) -> tuple[int, int]:
        """One drain round: dispatch each worker's bucket, gather replies.

        ``spilled`` is the parent's global spilled-rel set; it is sent to every
        worker (sticky mirror-only survives reassignment) and updated in place
        with rels the workers newly spilled this round."""
        dispatched: list[int] = []
        spilled_wire = sorted(spilled)
        for i, rel_map in buckets.items():
            conn = self._conns[i]
            if conn is None:
                continue
            try:
                conn.send(("scan", rel_map, final, force_spill, spilled_wire))
                dispatched.append(i)
            except (OSError, ValueError):
                self._mark_dead(i)
        backlog_files = 0
        backlog_bytes = 0
        for i in dispatched:
            reply = self._await(self._conns[i], self._procs[i], ("done",))
            if reply is None:
                self._mark_dead(i)
                continue
            r = reply[1]
            self._cum[i] = {k: int(r["stats"].get(k, 0)) for k in _STATS_KEYS}
            backlog_files += int(r["backlog_files"])
            backlog_bytes += int(r["backlog_bytes"])
            spilled.update(r["spilled_rels"])
        return backlog_files, backlog_bytes

    def close(self) -> set[str]:
        """Close every worker's writers, reap the processes, clean up the socket.

        Returns the union of grid rels the workers fully converted."""
        completed: set[str] = set()
        for i in self.alive_ids():
            conn, proc = self._conns[i], self._procs[i]
            try:
                conn.send(("close",))
                reply = self._await(conn, proc, ("closed",))
                if reply is not None:
                    completed.update(reply[1])
            except (OSError, ValueError):
                pass
            # Retire the worker's counters so totals() survives the close.
            for k in _STATS_KEYS:
                self._dead[k] += self._cum[i][k]
            self._cum[i] = dict.fromkeys(_STATS_KEYS, 0)
            self._conns[i] = None
            try:
                conn.close()
            except OSError:
                pass
            if proc is not None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                self._procs[i] = None
        try:
            self._listener.close()
        except OSError:
            pass
        shutil.rmtree(self._tmp, ignore_errors=True)
        return completed


def _drain_worker_main(address: str) -> int:
    """Entry point of one drain worker process (``--drain-worker``).

    A thin RPC shell around :class:`_DrainCore`: the parent decides when to
    scan and which diagnostics this worker owns; everything else (writers,
    spill state, quarantine) lives here for the lifetime of the run. Logs are
    shipped to the parent rather than printed so they land in the run's
    ordered log."""
    authkey = bytes.fromhex(os.environ.get("ADEPT_STREAM_AUTHKEY", ""))
    conn = Client(address, family="AF_UNIX", authkey=authkey)

    def _send_log(msg) -> None:
        try:
            conn.send(("log", str(msg)))
        except OSError:
            pass  # parent gone; nothing useful left to do with the message

    core: _DrainCore | None = None
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return 0  # parent gone: exit quietly, the final sync covers the rest
        kind = msg[0]
        if kind == "init":
            cfg = dict(msg[1])
            core = _DrainCore(
                cfg["out_dir"],
                persist_dir=cfg["persist_dir"],
                discard_grid_h5=cfg["discard_grid_h5"],
                spill_backlog_files=cfg["spill_backlog_files"],
                spill_backlog_bytes=cfg["spill_backlog_bytes"],
                error_log_every=cfg["error_log_every"],
                logger=_send_log,
            )
            conn.send(("ready",))
        elif kind == "scan":
            _, rel_map, final, force_spill, spilled = msg
            core._spilled.update(spilled)
            backlog_files = 0
            backlog_bytes = 0
            for rel, dstr in rel_map.items():
                nf, nb = core.drain_rel(rel, Path(dstr), final=final, force_spill=force_spill)
                backlog_files += nf
                backlog_bytes += nb
            conn.send(
                (
                    "done",
                    {
                        "stats": dict(core._stats),
                        "backlog_files": backlog_files,
                        "backlog_bytes": backlog_bytes,
                        "spilled_rels": sorted(core._spilled),
                    },
                )
            )
        elif kind == "close":
            completed = sorted(core.close_all()) if core is not None else []
            try:
                conn.send(("closed", completed))
            finally:
                conn.close()
            return 0


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="adept OSIRIS stream-drain worker (internal)")
    _ap.add_argument("--drain-worker", metavar="SOCKET", required=True, help="AF_UNIX socket of the parent pool")
    _args = _ap.parse_args()
    sys.exit(_drain_worker_main(_args.drain_worker))
