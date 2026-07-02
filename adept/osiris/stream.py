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
  sequentially and we process them sorted by iteration, so a plain append (slot
  ``i`` = the ``i``-th dump) is correct and needs no stride arithmetic. The
  dimension is grown one slot at a time with ``resize_dimension`` so it ends up
  sized to *exactly* the dumps produced — no pre-sized guess from the deck
  (whose off-by-one is genuinely ambiguous) and no trailing fill slots to trim
  on early termination. This deviates from the fixed-dim sketch in the perf
  doc; it is simpler and equally correct given in-order processing.
- **Write-completeness race.** A dump is only safe to read once OSIRIS has moved
  on to the next one, so the watcher processes every dump *except the current
  highest-numbered one*; the final sweep (run is dead) picks up that last dump.
- **Per-dump axis bounds** (``{dim}_min`` / ``{dim}_max``) are always recorded,
  so OSIRIS autoscale (``if_ps_p_auto``) survives and a restart can recover them
  from the file. :func:`io.load_series_nc` only *flags* a dim as autoscaled when
  those bounds actually move, so the constant bounds carried for a fixed spatial
  axis are harmless.
- **Failure isolation.** Nothing here may abort the OSIRIS run: the watcher
  swallows and logs every exception, and the batch path in
  :func:`io.save_run_datasets` remains the safety net for anything not produced.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr

from adept.osiris import io as _io

# Target uncompressed bytes per (t-chunk x spatial) chunk. A modest t-chunk
# keeps compression/read performance close to the batch path while bounding the
# read-modify-write cost of slot writes (which is hidden behind OSIRIS compute
# anyway). ~1 MiB is a reasonable HDF5 chunk size.
_TARGET_CHUNK_BYTES = 1 << 20


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
    :meth:`append` writes the next time slot, growing the unlimited ``t``
    dimension by one. Reopening an existing file (``mode="a"``) resumes after the
    slots already on disk, which is what makes a restart idempotent. The file is
    held open for the lifetime of the writer; call :meth:`close` to finalize.
    """

    def __init__(self, dest, template: xr.DataArray, *, source_dir=None, chunk_t: int | None = None, complevel: int = 4):
        self.dest = Path(dest)
        self.name = str(template.name)
        self.spatial_dims = [str(d) for d in template.dims]
        self.spatial_shape = tuple(int(s) for s in template.shape)
        self._f: h5netcdf.File | None = None

        if self.dest.exists():
            # Resume: trust the existing schema, continue after written slots.
            self._f = h5netcdf.File(self.dest, "a")
            self._n = int(self._f.dimensions["t"].size)
        else:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            self._f = h5netcdf.File(self.dest, "w")
            self._create_schema(template, source_dir if source_dir is not None else self.dest.parent, chunk_t, complevel)
            self._n = 0

    @property
    def n_written(self) -> int:
        return self._n

    def _create_schema(self, template, source_dir, chunk_t, complevel) -> None:
        f = self._f
        axis_units = template.attrs.get("axis_units", {}) or {}
        axis_long = template.attrs.get("axis_long_names", {}) or {}

        f.dimensions["t"] = None  # unlimited; grown one slot per append
        for d, sz in zip(self.spatial_dims, self.spatial_shape):
            f.dimensions[d] = sz

        ct = chunk_t or _default_chunk_t(self.spatial_shape)
        var = f.create_variable(
            self.name,
            ("t", *self.spatial_dims),
            dtype=np.dtype(_io._DIAG_DTYPE),
            chunks=(ct, *self.spatial_shape),
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
        """Write ``da`` (one dump) into the next time slot."""
        if da.shape != self.spatial_shape:
            raise ValueError(f"shape mismatch for {self.name}: {da.shape} != {self.spatial_shape}")
        f = self._f
        i = self._n
        f.resize_dimension("t", i + 1)
        f.variables[self.name][i, ...] = np.asarray(da.values, dtype=_io._DIAG_DTYPE)
        f.variables["t"][i] = float(da.attrs.get("time", np.nan))
        f.variables["iter"][i] = int(da.attrs.get("iter", -1))
        for d in self.spatial_dims:
            cv = np.asarray(da.coords[d].values)
            lo, hi = (float(cv[0]), float(cv[-1])) if cv.size else (np.nan, np.nan)
            f.variables[f"{d}_min"][i] = lo
            f.variables[f"{d}_max"][i] = hi
        self._n = i + 1

    def flush(self) -> None:
        if self._f is not None:
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def convert_diagnostic_streaming(diag_dir, dest, *, source_dir=None) -> Path:
    """Stage A: convert one grid diagnostic to NetCDF a dump at a time.

    A memory-bounded drop-in for ``series_to_dataset(load_series(diag_dir))``
    followed by ``to_netcdf`` — peak RAM is one dump, not the whole stacked
    series. Rebuilds ``dest`` from scratch (any partial file is removed first).
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


class StreamConverter:
    """Stage B: a background thread that drains diagnostics during the run.

    Spawned by :func:`adept.osiris.runner.run_osiris` alongside the OSIRIS
    subprocess, it polls ``run_dir/MS`` and drains each grid diagnostic's
    completed dumps into ``out_dir/<relpath>.nc``. RAW (particle) diagnostics do
    not fit the fixed-slot model (variable particle count per dump) and are left
    to the batch path. Every operation is best-effort: any error is logged and
    the run is never affected.

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
    """

    def __init__(self, run_dir, out_dir, *, poll_s: float = 10.0, logger=print, persist_dir=None):
        self.ms = Path(run_dir) / "MS"
        self.out_dir = Path(out_dir)
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self.persist_ms = (self.persist_dir / "MS") if self.persist_dir is not None else None
        self.poll_s = float(poll_s)
        self._log = logger
        self._writers: dict[str, StreamWriter] = {}
        self._completed: set[str] = set()  # grid relpaths fully converted to .nc
        self._raw_completed: set[str] = set()  # raw relpaths fully mirrored to persist
        self._is_raw: dict[str, bool] = {}  # rel -> raw? (cached; avoids re-peeking h5)
        self._multibase_skipped: set[str] = set()  # multi-report dirs left to the batch pass
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="osiris-stream-convert", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once(final=False)
            except Exception as e:  # a watcher must never crash the run
                self._log(f"[stream] scan error (continuing): {e}")
            self._stop.wait(self.poll_s)

    def finalize(self) -> set[str]:
        """Stop the thread, run the authoritative final sweep, close all writers.

        After OSIRIS has exited every dump is safe to read (no writer can still
        be racing), so the final sweep processes the last dump each diagnostic
        had been holding back (and, in staging mode, drains it off the scratch)
        before closing the files. Returns the set of *grid* diagnostic relpaths
        fully converted by the stream (so the caller can skip rebuilding them in
        the batch pass); RAW relpaths are mirrored but not returned, since the
        batch pass still builds their NetCDFs from the mirrored HDF5.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.poll_s, 30.0) + 30.0)
        try:
            self._scan_once(final=True)
        except Exception as e:
            self._log(f"[stream] finalize sweep error (batch fallback covers it): {e}")
        for rel, w in self._writers.items():
            try:
                w.close()
            except Exception:
                pass
            self._completed.add(rel)
        return set(self._completed)

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

    def _diag_is_raw(self, rel: str, d: Path) -> bool:
        cached = self._is_raw.get(rel)
        if cached is None:
            cached = _io._diag_is_raw(rel, d)
            self._is_raw[rel] = cached
        return cached

    def _scan_once(self, *, final: bool) -> None:
        for rel, d in self._diags().items():
            if rel in self._completed or rel in self._raw_completed:
                continue
            try:
                if self._diag_is_raw(rel, d):
                    # RAW: nothing to do unless staging (then mirror + reap the
                    # HDF5 so the scratch stays bounded and the batch path finds
                    # the dumps on persist).
                    if self.persist_dir is not None:
                        self._drain_raw(rel, d, final=final)
                elif self.persist_dir is None:
                    self._drain_grid_inplace(rel, d, final=final)
                else:
                    self._drain_grid_staged(rel, d, final=final)
            except Exception as e:
                self._log(f"[stream] {rel}: {e} (will fall back to batch)")
                # Drop the writer so a half-written file is rebuilt by the batch
                # pass rather than reused.
                w = self._writers.pop(rel, None)
                if w is not None:
                    try:
                        w.close()
                    except Exception:
                        pass

    # --- draining ---------------------------------------------------------

    def _drain_grid_inplace(self, rel: str, d: Path, *, final: bool) -> None:
        """Stream a grid diagnostic to NetCDF in place — no mirror, no reap.

        The original non-staged behavior: dumps accumulate in ``run_dir`` and
        slot ``i`` is the ``i``-th dump, indexed positionally against the full
        (never-pruned) dump list.
        """
        dumps = _io._sort_dumps(d)
        if not dumps:
            return
        # Write-completeness: a dump is safe only once the next one exists (OSIRIS
        # has moved on). The final sweep runs after OSIRIS is dead, so the last
        # dump is safe too.
        safe = dumps if final else dumps[:-1]
        if not safe:
            return  # nothing safe to write yet (only the in-progress dump exists)

        w = self._writers.get(rel)
        if w is None:
            template = _io.load_grid_h5(dumps[0])
            w = StreamWriter(self.out_dir / f"{rel}.nc", template, source_dir=d)
            self._writers[rel] = w
            if w.n_written == 0:
                w.append(template)  # reuse the template we just loaded as slot 0

        for p in safe[w.n_written :]:
            w.append(_io.load_grid_h5(p))
        w.flush()

        if final:
            w.close()
            self._completed.add(rel)

    def _drain_grid_staged(self, rel: str, d: Path, *, final: bool) -> None:
        """Stream a grid diagnostic *and* drain its dumps off the scratch.

        Because every handled dump is deleted from the scratch (:meth:`_reap`),
        the directory only ever holds dumps not yet processed (plus the
        in-progress last one), so every ``safe`` dump is new — no positional
        bookkeeping against on-disk slots is needed.
        """
        dumps = _io._sort_dumps(d)
        if not dumps:
            return
        safe = dumps if final else dumps[:-1]
        if not safe:
            return

        w = self._writers.get(rel)
        if w is None:
            w = StreamWriter(self.out_dir / f"{rel}.nc", _io.load_grid_h5(dumps[0]), source_dir=d)
            self._writers[rel] = w
        for p in safe:
            w.append(_io.load_grid_h5(p))
            self._reap(rel, p)
        w.flush()

        if final:
            w.close()
            self._completed.add(rel)

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

    def _reap(self, rel: str, p: Path) -> None:
        """Mirror one scratch dump to ``persist_dir/MS`` then delete the scratch
        copy. Best-effort: on any failure the scratch file is left in place for
        the runner's final sync to catch, so data is never lost and the watcher
        never crashes."""
        if self.persist_dir is None:
            return
        try:
            dest = self.persist_ms / rel / p.name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                shutil.copyfile(p, tmp)
                os.replace(tmp, dest)  # atomic publish on the persist filesystem
            p.unlink()
        except Exception as e:
            self._log(f"[stream] reap {rel}/{p.name} deferred: {e}")
