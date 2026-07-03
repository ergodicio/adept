"""Lightweight loaders for OSIRIS HDF5 output.

OSIRIS dump files have a tiny, well-defined structure:

- one dataset at the root, named after the diagnostic (``e1``, ``charge``,
  ``x1p1``, ...);
- an ``AXIS`` group with ``AXIS1``, ``AXIS2``, ... datasets each holding
  ``[min, max]`` plus ``LONG_NAME``, ``NAME``, ``UNITS`` attrs;
- a ``SIMULATION`` group with run-wide metadata as attrs (``DT``, ``NX``,
  ``XMIN``, ``XMAX``, ``NDIMS``);
- root attrs ``TIME``, ``ITER``, ``NAME``, ``LABEL``, ``UNITS``.

OSIRIS writes datasets in Fortran-axis order, so the *first* HDF5 axis
becomes the *last* numpy axis. This loader reverses the AXIS list when
constructing coordinates so the returned ``xarray.DataArray`` has dims
in numpy order (rows first), with each dim's ``name`` matching the
OSIRIS axis label.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

_ITER_RE = re.compile(r"-(\d+)\.h5$")

# Precision for diagnostic *data* (field/phase-space grids and RAW particle
# quantities) in the saved NetCDF artifacts. OSIRIS writes its dumps in single
# precision, so float32 matches the native precision and halves artifact size
# vs float64. Coordinates (time, spatial axes) stay float64 for axis precision
# (e.g. the omega-k FFT relies on the float64 time axis).
_DIAG_DTYPE = "float32"


def _decode(v) -> str:
    """OSIRIS stores attrs as ``S256`` bytes inside length-1 arrays."""
    if isinstance(v, np.ndarray):
        v = v.flat[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").rstrip()
    return str(v)


def _axis_metadata(h5f: h5py.File) -> list[dict]:
    """Return per-axis dicts in *OSIRIS* order (axis 1 first).

    Each dict has ``name``, ``long_name``, ``units``, ``min``, ``max``.
    """
    axes: list[dict] = []
    if "AXIS" not in h5f:
        return axes
    grp = h5f["AXIS"]
    for ax_name in sorted(grp.keys()):  # AXIS1, AXIS2, ...
        ax = grp[ax_name]
        vals = ax[:]
        axes.append(
            {
                "name": _decode(ax.attrs.get("NAME", ax_name)),
                "long_name": _decode(ax.attrs.get("LONG_NAME", ax_name)),
                "units": _decode(ax.attrs.get("UNITS", "")),
                "min": float(vals[0]),
                "max": float(vals[-1]),
            }
        )
    return axes


def _iter_from_name(path: Path) -> int:
    m = _ITER_RE.search(path.name)
    return int(m.group(1)) if m else -1


def load_grid_h5(path: str | Path) -> xr.DataArray:
    """Load one OSIRIS field/charge/current dump into an xarray DataArray.

    The returned array has dims in numpy order (e.g. ``("x2", "x1")`` for
    a 2D dump, ``("x1",)`` for 1D). Use ``.transpose(...)`` if you prefer
    the OSIRIS convention.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        # Identify the data dataset (everything that's not AXIS / SIMULATION).
        data_keys = [k for k in f.keys() if k not in ("AXIS", "SIMULATION")]
        if len(data_keys) != 1:
            raise ValueError(f"Expected exactly one data dataset in {path}; got {data_keys}")
        name = data_keys[0]
        arr = f[name][...].astype(_DIAG_DTYPE)
        axes_osiris = _axis_metadata(f)
        # Reverse to numpy order.
        axes_numpy = list(reversed(axes_osiris))

        coords = {}
        dims = []
        for i, ax in enumerate(axes_numpy):
            n_i = arr.shape[i]
            coords[ax["name"]] = np.linspace(ax["min"], ax["max"], n_i)
            dims.append(ax["name"])

        attrs = {
            "time": float(f.attrs["TIME"][0]) if "TIME" in f.attrs else float("nan"),
            "iter": int(f.attrs["ITER"][0]) if "ITER" in f.attrs else _iter_from_name(path),
            "long_name": _decode(f.attrs.get("LABEL", name)),
            "units": _decode(f.attrs.get("UNITS", "")),
            "time_units": _decode(f.attrs.get("TIME UNITS", "")),
            "source": str(path),
            "axis_units": {ax["name"]: ax["units"] for ax in axes_numpy},
            "axis_long_names": {ax["name"]: ax["long_name"] for ax in axes_numpy},
        }
        if "SIMULATION" in f:
            sim = f["SIMULATION"].attrs
            for key in ("DT", "NDIMS", "XMIN", "XMAX", "NX", "PERIODIC"):
                if key in sim:
                    val = sim[key]
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    attrs[f"sim.{key}"] = val

    return xr.DataArray(arr, coords=coords, dims=dims, name=name, attrs=attrs)


def load_phasespace_h5(path: str | Path) -> xr.DataArray:
    """Phase-space dumps have the exact same on-disk structure as grid
    dumps, so this is just a semantic alias."""
    return load_grid_h5(path)


def _is_raw_h5(f: h5py.File) -> bool:
    """Heuristic: is this an OSIRIS RAW (particle) dump rather than a grid dump?

    RAW dumps hold several 1-D per-particle datasets and have no ``AXIS``
    group, whereas grid / phase-space dumps hold a single gridded dataset plus
    an ``AXIS`` group. Treat anything with more than one data dataset, or no
    ``AXIS`` group, as non-grid.
    """
    data_keys = [k for k in f.keys() if k not in ("AXIS", "SIMULATION")]
    return len(data_keys) != 1 or "AXIS" not in f


def load_raw_h5(path: str | Path) -> xr.Dataset:
    """Load one OSIRIS RAW (particle) dump into an ``xarray.Dataset``.

    Unlike field / charge / phase-space *grid* dumps (one gridded dataset plus
    an ``AXIS`` group), a RAW dump holds several 1-D per-particle datasets
    (``x1``, ``p1``, ``p2``, ``p3``, ``ene``, ``q``, ...), a ``SIMULATION``
    attrs group, and root attrs ``TIME`` / ``ITER`` — but typically no
    ``AXIS`` group, because the data is not gridded.

    Dataset names are discovered dynamically (different decks dump different
    quantities); each becomes a data variable indexed by a single particle
    dimension ``"pidx"``. Per-quantity ``UNITS`` / ``LONG_NAME`` attrs, when
    present, ride on the matching variable; ``TIME`` / ``ITER`` ride on the
    Dataset.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        data_keys = sorted(k for k in f.keys() if k not in ("AXIS", "SIMULATION") and isinstance(f[k], h5py.Dataset))
        data_vars: dict[str, tuple] = {}
        for name in data_keys:
            dset = f[name]
            arr = dset[...].astype(_DIAG_DTYPE).reshape(-1)
            var_attrs = {}
            if "UNITS" in dset.attrs:
                var_attrs["units"] = _decode(dset.attrs["UNITS"])
            if "LONG_NAME" in dset.attrs:
                var_attrs["long_name"] = _decode(dset.attrs["LONG_NAME"])
            data_vars[name] = ("pidx", arr, var_attrs)

        npart = max((v[1].shape[0] for v in data_vars.values()), default=0)

        attrs = {
            "time": float(f.attrs["TIME"][0]) if "TIME" in f.attrs else float("nan"),
            "iter": int(f.attrs["ITER"][0]) if "ITER" in f.attrs else _iter_from_name(path),
            "long_name": _decode(f.attrs.get("LABEL", path.stem)),
            "time_units": _decode(f.attrs.get("TIME UNITS", "")),
            "source": str(path),
            "n_particles": int(npart),
        }
        if "SIMULATION" in f:
            sim = f["SIMULATION"].attrs
            for key in ("DT", "NDIMS", "XMIN", "XMAX", "NX", "PERIODIC"):
                if key in sim:
                    val = sim[key]
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    attrs[f"sim.{key}"] = val

    return xr.Dataset(data_vars, attrs=attrs)


def load_raw_series(directory: str | Path, *, drop_initial: bool = False) -> xr.Dataset:
    """Concatenate every RAW (particle) dump in ``directory`` long-form.

    RAW dumps have a *variable* particle count per timestep (OSIRIS samples a
    ``raw_fraction`` of particles each dump), so they cannot be stacked into a
    rectangular ``(t, particle)`` array the way grid dumps are. Instead every
    dump is concatenated along the particle dimension ``"pidx"`` with a per-row
    ``t`` / ``iter`` coordinate identifying which dump each particle came from.
    The union of quantities across dumps is preserved (missing quantities fill
    with NaN for that dump's rows).
    """
    directory = Path(directory)
    dumps = _sort_dumps(directory)
    if not dumps:
        raise FileNotFoundError(f"No .h5 dumps in {directory}")

    # Optionally drop the t=0 (initial-condition) RAW dump: OSIRIS dumps RAW
    # periodically from n=0, but at full raw_fraction that IC snapshot is the
    # thermal start state and just bloats the artifact. Filter by filename so the
    # (large) n=0 dump is never loaded. Kept if it is the sole dump.
    if drop_initial and len(dumps) > 1:
        dumps = [p for p in dumps if _iter_from_name(p) != 0] or dumps

    per_dump: list[xr.Dataset] = []
    times: list[np.ndarray] = []
    iters: list[np.ndarray] = []
    for p in dumps:
        ds = load_raw_h5(p)
        n = ds.sizes.get("pidx", 0)
        per_dump.append(ds)
        times.append(np.full(n, ds.attrs["time"], dtype="float64"))
        iters.append(np.full(n, ds.attrs["iter"], dtype="int64"))

    combined = xr.concat(per_dump, dim="pidx", data_vars="all", coords="minimal")
    combined = combined.assign_coords(
        t=("pidx", np.concatenate(times) if times else np.empty(0)),
        iter=(
            "pidx",
            np.concatenate(iters) if iters else np.empty(0, dtype="int64"),
        ),
    )
    attrs = dict(per_dump[0].attrs)
    for k in ("time", "iter", "source", "n_particles"):
        attrs.pop(k, None)
    attrs["source_dir"] = str(directory)
    attrs["n_dumps"] = len(dumps)
    combined.attrs = attrs
    return combined


def _diag_is_raw(relpath: str, directory: str | Path) -> bool:
    """Detect a RAW (particle) diagnostic.

    Primary signal: the diagnostic relpath starts with ``"RAW/"``. As a
    defensive fallback, peek at the first dump and treat it as RAW when it
    fails the grid heuristic (more than one data dataset, or no ``AXIS``).
    """
    if relpath.startswith("RAW/") or Path(relpath).parts[0] == "RAW":
        return True
    dumps = _sort_dumps(Path(directory))
    if not dumps:
        return False
    try:
        with h5py.File(dumps[0], "r") as f:
            return _is_raw_h5(f)
    except Exception:
        return False


_BASE_RE = re.compile(r"-\d+\.h5$")


def _dump_base(path: Path) -> str:
    """A dump's series base = its filename minus the trailing ``-<iter>.h5``.

    ``e1-savg-000123.h5`` -> ``e1-savg``. A directory usually holds one base, but
    multiple line/report diagnostics of the same field share a directory and
    differ only by an index in the base — e.g. two ``s1,...,line`` reports produce
    ``s1-tavg-line-x2-01`` and ``s1-tavg-line-x2-02``. Those are distinct time
    series and must not be stacked into one.
    """
    return _BASE_RE.sub("", path.name)


def _series_dumps(directory: Path) -> dict[str, list[Path]]:
    """Group ``directory``'s ``.h5`` dumps into time series keyed by base name."""
    groups: dict[str, list[Path]] = {}
    for p in directory.iterdir():
        if p.is_file() and p.suffix == ".h5":
            groups.setdefault(_dump_base(p), []).append(p)
    for dumps in groups.values():
        dumps.sort(key=_iter_from_name)
    return groups


def _resolve_series(handle: Path) -> tuple[Path, str | None]:
    """Resolve a diagnostic handle to ``(directory, base)``.

    A handle from :func:`list_diagnostics` is either a real dump directory (one
    series, ``base`` is ``None``) or a synthetic ``<dir>/<base>`` path naming one
    report series inside a directory that holds several.
    """
    handle = Path(handle)
    if handle.is_dir():
        return handle, None
    return handle.parent, handle.name


def _sort_dumps(handle: Path, base: str | None = None) -> list[Path]:
    """Sorted ``.h5`` dumps for one diagnostic series.

    ``handle`` may be a dump directory or a synthetic per-report handle
    (``<dir>/<base>``); ``base`` selects one series inside a multi-report
    directory. A bare directory holding more than one series is ambiguous and
    raises — load one via its per-report handle from :func:`list_diagnostics`.
    """
    directory, hbase = _resolve_series(handle)
    if base is None:
        base = hbase
    groups = _series_dumps(directory)
    if base is not None:
        return groups.get(base, [])
    if len(groups) > 1:
        raise ValueError(
            f"{directory} holds multiple report series {sorted(groups)}; "
            "load one via its per-report handle from list_diagnostics()."
        )
    return next(iter(groups.values()), [])


def load_series(directory: str | Path) -> xr.DataArray:
    """Stack every dump in ``directory`` into a ``(t, ...)`` DataArray.

    All files must share the same diagnostic name, the same spatial /
    phase-space shape, and the same axis bounds — this is the standard
    OSIRIS convention for a single diagnostic's time history.

    For regenerating plots from saved artifacts, ``directory`` may instead be a
    single ``.nc`` file written by :func:`save_run_datasets`; it is then loaded
    via :func:`load_series_nc`, returning the same stacked ``(t, ...)`` array.
    """
    directory = Path(directory)
    if directory.is_file() and directory.suffix == ".nc":
        return load_series_nc(directory)
    dumps = _sort_dumps(directory)
    if not dumps:
        raise FileNotFoundError(f"No .h5 dumps in {directory}")
    first = load_grid_h5(dumps[0])

    n = len(dumps)
    times = np.empty(n, dtype="float64")
    iters = np.empty(n, dtype="int64")
    data = np.empty((n, *first.shape), dtype=first.dtype)
    # Per-dump axis bounds (min, max) for every non-time dim. OSIRIS autoscale
    # (deck ``if_ps_p_auto`` / ``if_ps_gamma_auto``) re-picks a phase space's
    # momentum / gamma bounds *every dump*, so an axis can move dump-to-dump
    # while its bin count stays fixed — a shape-only check misses it.
    bounds = {d: np.empty((n, 2), dtype="float64") for d in first.dims}

    def _record(i: int, da: xr.DataArray) -> None:
        data[i] = da.values
        times[i] = da.attrs["time"]
        iters[i] = da.attrs["iter"]
        for d in first.dims:
            cv = np.asarray(da.coords[d].values)
            bounds[d][i] = (cv[0], cv[-1]) if cv.size else (np.nan, np.nan)

    _record(0, first)
    for i, p in enumerate(dumps[1:], start=1):
        da = load_grid_h5(p)
        if da.shape != first.shape:
            raise ValueError(f"Shape mismatch in series: {p} has {da.shape}, expected {first.shape}")
        _record(i, da)

    coords = {"t": times, "iter": ("t", iters)}
    coords.update({d: first.coords[d] for d in first.dims})
    # An axis whose per-dump bounds move (beyond fp noise) is autoscaled: keep
    # the first dump's axis as the nominal dimension coordinate, but carry the
    # true per-dump bounds along ``t`` so consumers can reconstruct the physical
    # axis for any timestep (see :func:`physical_axis`) and so the bounds
    # survive the NetCDF round-trip.
    autoscaled: list[str] = []
    for d in first.dims:
        b = bounds[d]
        if not (np.allclose(b[:, 0], b[0, 0]) and np.allclose(b[:, 1], b[0, 1])):
            autoscaled.append(str(d))
            coords[f"{d}_min"] = ("t", b[:, 0])
            coords[f"{d}_max"] = ("t", b[:, 1])
    dims = ("t", *first.dims)
    attrs = dict(first.attrs)
    attrs.pop("time", None)
    attrs.pop("iter", None)
    attrs.pop("source", None)
    attrs["source_dir"] = str(directory)
    if autoscaled:
        attrs["autoscaled_dims"] = autoscaled
    return xr.DataArray(data, coords=coords, dims=dims, name=first.name, attrs=attrs)


def physical_axis(da: xr.DataArray, dim: str, it: int = -1) -> np.ndarray:
    """Physical coordinate values for ``dim``, honoring OSIRIS autoscale.

    When ``dim`` was autoscaled (its per-dump bounds move; see
    :func:`load_series`), the series carries ``{dim}_min`` / ``{dim}_max`` along
    ``t`` while the nominal dimension coordinate is only the first dump's axis.
    This rebuilds ``linspace(min, max, n)`` for time index ``it`` (default the
    last dump). Works on a still-stacked ``(t, …)`` series (the bounds are
    vectors along ``t``) and on an already time-sliced array (the bounds are
    scalar coordinates). Falls back to the dimension coordinate when no per-dump
    bounds are present (non-autoscaled axes, or arrays not built by
    :func:`load_series`).
    """
    n = int(da.sizes[dim])
    lo_c, hi_c = f"{dim}_min", f"{dim}_max"
    if lo_c in da.coords and hi_c in da.coords:
        lo, hi = da.coords[lo_c], da.coords[hi_c]
        if "t" in getattr(lo, "dims", ()):  # still a (t, …) series
            lo, hi = lo.isel(t=it), hi.isel(t=it)
        return np.linspace(float(lo), float(hi), n)
    return np.asarray(da.coords[dim].values)


def list_diagnostics(run_dir: str | Path) -> dict[str, Path]:
    """Map every diagnostic name to its data source.

    Two layouts are supported transparently:

    - a raw OSIRIS run directory with an ``MS/`` tree: any directory that
      directly contains ``.h5`` dumps is a diagnostic, keyed by its relative
      path under ``MS/`` (e.g. ``"FLD/e1"``, ``"PHA/x1p1/beam_pos"``);
    - the ``binary/`` directory of saved NetCDFs from :func:`save_run_datasets`
      (no ``MS/``): each ``.nc`` is a diagnostic, keyed by its relative path
      without the suffix (same keys as above).

    The returned handles (dump dirs or ``.nc`` files) are both accepted by
    :func:`load_series`, so callers regenerate plots from either source without
    caring which. Raises ``FileNotFoundError`` if neither layout is found.
    """
    run_dir = Path(run_dir)
    ms = run_dir / "MS"
    if ms.is_dir():
        out: dict[str, Path] = {}
        for d in ms.rglob("*"):
            if not d.is_dir():
                continue
            groups = _series_dumps(d)
            if not groups:
                continue
            rel = str(d.relative_to(ms))
            if len(groups) == 1:
                out[rel] = d
            else:
                # A directory with several report series (e.g. two s1 line
                # lineouts at different perpendicular cells) exposes each as its
                # own diagnostic via a synthetic <dir>/<base> handle, so they are
                # converted to separate NetCDFs instead of merged into one.
                for base in sorted(groups):
                    out[f"{rel}/{base}"] = d / base
        # stage_discard_h5 layout: grid dumps were streamed to run_dir/binary
        # and never mirrored to MS/ (which then holds only RAW). Merge the
        # streamed NetCDFs for any diagnostic with no raw dumps on disk so
        # post-processing and plots see the full set; load_series accepts the
        # .nc handles directly.
        bdir = run_dir / "binary"
        if bdir.is_dir():
            for rel, p in list_diagnostics_nc(bdir).items():
                out.setdefault(rel, p)
        return out
    # No MS/ tree — fall back to the saved-NetCDF layout (a run dir's binary/
    # subdir first, else run_dir itself *being* a binary dir).
    bdir = run_dir / "binary"
    if bdir.is_dir():
        nc = list_diagnostics_nc(bdir)
        if nc:
            return nc
    nc = list_diagnostics_nc(run_dir)
    if nc:
        return nc
    raise FileNotFoundError(f"No MS/ directory or saved NetCDFs under {run_dir}")


def _coerce_attr(v):
    """Make an attr value writable by the netCDF backends.

    ``load_series`` stores some attrs as dicts (per-axis units/long-names),
    which netCDF cannot serialize; those are handled separately by
    :func:`series_to_dataset`. Here we JSON-encode any stray dict and turn
    lists/tuples into arrays.
    """
    if isinstance(v, dict):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return np.asarray(v)
    return v


def series_to_dataset(da: xr.DataArray) -> xr.Dataset:
    """Wrap a :func:`load_series` DataArray into a netCDF-serializable Dataset.

    The dict-valued ``axis_units`` / ``axis_long_names`` attrs are lifted onto
    the matching coordinate variables (the CF-idiomatic place for them), and
    any remaining non-scalar attrs are coerced so ``to_netcdf`` succeeds.
    """
    da = da.copy()
    # The autoscaled-dim list is reconstructed on load from the {dim}_min/_max
    # coordinates (which ride along as coords), so it need not survive as an
    # attr — and dropping it avoids serializing a string array.
    da.attrs.pop("autoscaled_dims", None)
    axis_units = da.attrs.pop("axis_units", {}) or {}
    axis_long = da.attrs.pop("axis_long_names", {}) or {}
    da.attrs = {k: _coerce_attr(v) for k, v in da.attrs.items() if v is not None}
    ds = da.to_dataset(name=da.name)
    for dim, units in axis_units.items():
        if dim in ds.coords:
            ds[dim].attrs["units"] = units
    for dim, long_name in axis_long.items():
        if dim in ds.coords:
            ds[dim].attrs["long_name"] = long_name
    return ds


def load_series_nc(path: str | Path) -> xr.DataArray:
    """Load a grid/phase-space series NetCDF back into a plotter-ready DataArray.

    This is the inverse of :func:`series_to_dataset` (the form
    :func:`save_run_datasets` writes to disk): it reopens a single-diagnostic
    ``.nc`` file and rebuilds the ``axis_units`` / ``axis_long_names`` dict
    attrs from the per-coordinate metadata, so the returned ``DataArray`` is
    indistinguishable (for plotting purposes) from one produced by
    :func:`load_series` off the raw HDF5 tree. The whole point is that the
    canned plots can be regenerated from the saved NetCDFs alone.

    The file is read fully into memory and closed before returning.
    """
    ds = xr.load_dataset(path, engine="h5netcdf")
    names = list(ds.data_vars)
    if not names:
        raise ValueError(f"No data variable in {path}")
    # series_to_dataset writes exactly one data variable (the diagnostic);
    # `iter` rides along as a coordinate, not a data var.
    name = names[0]
    da = ds[name]

    axis_units: dict[str, str] = {}
    axis_long: dict[str, str] = {}
    for dim in da.dims:
        if dim == "t" or dim not in ds.coords:
            continue
        cu = ds[dim].attrs.get("units")
        cl = ds[dim].attrs.get("long_name")
        if cu:
            axis_units[str(dim)] = cu
        if cl:
            axis_long[str(dim)] = cl
    da.attrs["axis_units"] = axis_units
    da.attrs["axis_long_names"] = axis_long
    da.attrs.setdefault("time_units", da.attrs.get("time_units", r"1/\omega_p"))
    # Rebuild the autoscaled-dim list from the per-dump bound coords, so a
    # round-tripped series is indistinguishable (for plotting) from a fresh
    # :func:`load_series`. A dim is autoscaled only when its bounds actually
    # *move*: the batch path writes {dim}_min/_max solely for moving dims, but
    # the streaming writer (:mod:`adept.osiris.stream`) records them for every
    # dim, so mere presence is not enough — check that they vary.
    autoscaled: list[str] = []
    for d in da.dims:
        lo_c, hi_c = f"{d}_min", f"{d}_max"
        if lo_c not in ds.coords or hi_c not in ds.coords:
            continue
        lo = np.asarray(ds[lo_c].values)
        hi = np.asarray(ds[hi_c].values)
        if lo.size and not (np.allclose(lo, lo[0]) and np.allclose(hi, hi[0])):
            autoscaled.append(str(d))
    if autoscaled:
        da.attrs["autoscaled_dims"] = autoscaled
    return da


def list_diagnostics_nc(binary_dir: str | Path) -> dict[str, Path]:
    """Map every saved-NetCDF diagnostic to its ``.nc`` file.

    The inverse layout of :func:`save_run_datasets`: walks ``binary_dir`` for
    ``*.nc`` files and keys each by its relative path with the suffix dropped
    (e.g. ``"FLD/e1"``, ``"PHA/x1p1/beam_pos"``, ``"HIST/energy"``), mirroring
    :func:`list_diagnostics` over the raw ``MS/`` tree.
    """
    binary_dir = Path(binary_dir)
    out: dict[str, Path] = {}
    for p in sorted(binary_dir.rglob("*.nc")):
        rel = str(p.relative_to(binary_dir).with_suffix(""))
        out[rel] = p
    return out


def _compression_encoding(ds: xr.Dataset) -> dict:
    """Per-variable zlib settings for ``to_netcdf``.

    Field / phase-space grids are smooth and compress several-fold; ``shuffle``
    helps the float32 byte pattern. RAW (per-particle) data is noise-like and
    barely compresses, but the setting does no harm.
    """
    return {name: {"zlib": True, "complevel": 4, "shuffle": True} for name in ds.data_vars}


def save_run_datasets(
    run_dir: str | Path,
    out_dir: str | Path,
    diagnostics: list[str] | set[str] | None = None,
    *,
    raw_drop_initial: bool = False,
    stream: bool = False,
    streamed_dir: str | Path | None = None,
) -> list[Path]:
    """Convert each diagnostic's full time history to a netCDF file.

    One file per diagnostic is written under ``out_dir``, mirroring the OSIRIS
    ``MS/`` layout (e.g. ``out_dir/FLD/e1.nc``,
    ``out_dir/PHA/x1p1/beam_pos.nc``). Each file holds the stacked ``(t, ...)``
    series — every time slice OSIRIS dumped for that diagnostic.

    ``diagnostics``, when given, whitelists which diagnostics to convert,
    matched against either the relative path (``"FLD/e1"``) or the leaf name
    (``"e1"``). Returns the list of written paths.

    Two opt-in paths bound conversion memory to a single dump (see
    :mod:`adept.osiris.stream`):

    - ``streamed_dir`` — a directory where the concurrent converter already
      wrote ``<relpath>.nc`` *during the run*. Such a grid diagnostic is copied
      into ``out_dir`` rather than rebuilt, so it is never re-read off disk.
    - ``stream`` — for any grid diagnostic not already in ``streamed_dir``,
      build the NetCDF a dump at a time instead of stacking the whole series in
      memory. RAW diagnostics always use the (in-memory) concat path.
    """
    out_dir = Path(out_dir)
    streamed_dir = Path(streamed_dir) if streamed_dir is not None else None
    diags = list_diagnostics(run_dir)
    written: list[Path] = []
    for relpath in sorted(diags):
        if diagnostics is not None and (relpath not in diagnostics and Path(relpath).name not in diagnostics):
            continue
        dest = out_dir / f"{relpath}.nc"
        try:
            is_raw = _diag_is_raw(relpath, diags[relpath])
            pre = (streamed_dir / f"{relpath}.nc") if streamed_dir is not None else None
            if pre is not None and not is_raw and pre.is_file():
                # The concurrent converter already produced this during the run.
                dest.parent.mkdir(parents=True, exist_ok=True)
                if pre.resolve() != dest.resolve():
                    shutil.copy(pre, dest)
            elif is_raw:
                # RAW (particle) dumps: per-particle datasets, no grid/AXIS.
                ds: xr.Dataset = load_raw_series(diags[relpath], drop_initial=raw_drop_initial)
                dest.parent.mkdir(parents=True, exist_ok=True)
                ds.to_netcdf(dest, engine="h5netcdf", encoding=_compression_encoding(ds))
            elif stream:
                from adept.osiris import stream as _stream  # lazy: pulls in the writer only when needed

                _stream.convert_diagnostic_streaming(diags[relpath], dest, source_dir=diags[relpath])
            else:
                ds = series_to_dataset(load_series(diags[relpath]))
                dest.parent.mkdir(parents=True, exist_ok=True)
                ds.to_netcdf(dest, engine="h5netcdf", encoding=_compression_encoding(ds))
            written.append(dest)
        except Exception as e:  # one bad diagnostic must not abort the rest
            print(f"[post] skipping diagnostic {relpath}: {e}")
            continue

    # Persist the HIST scalar-energy history (field + per-species kinetic) so
    # the energy-conservation plot can be regenerated from the saved NetCDFs
    # alone — it is the one plot input that lives outside the MS/ dump tree.
    try:
        energy = load_hist_energy(run_dir)
        if energy is not None:
            dest = out_dir / "HIST" / "energy.nc"
            dest.parent.mkdir(parents=True, exist_ok=True)
            energy.to_netcdf(dest, engine="h5netcdf", encoding=_compression_encoding(energy))
            written.append(dest)
    except Exception as e:
        print(f"[post] skipping HIST energy: {e}")

    return written


def _parse_hist_table(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a whitespace-delimited OSIRIS ``HIST`` table.

    Skips blank and ``!``/``#`` comment lines and any non-numeric header row.
    Assumes the OSIRIS column convention ``iteration  time  <values...>`` and
    returns ``(t, values)`` where ``t`` is shape ``(n,)`` and ``values`` is
    ``(n, ncols - 2)``. Returns ``None`` if nothing numeric is found.
    """
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "!#":
                continue
            try:
                rows.append([float(tok) for tok in s.split()])
            except ValueError:
                continue  # header row of column names
    if not rows:
        return None
    width = min(len(r) for r in rows)
    if width < 2:
        return None
    arr = np.array([r[:width] for r in rows], dtype="float64")
    return arr[:, 1], arr[:, 2:]


def _interp_onto(src_t: np.ndarray, src_v: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
    """Linear-interpolate ``src_v(src_t)`` onto ``ref_t`` (identity if aligned)."""
    if src_t.shape == ref_t.shape and np.allclose(src_t, ref_t):
        return src_v
    return np.interp(ref_t, src_t, src_v)


def load_hist_energy(run_dir: str | Path) -> xr.Dataset | None:
    """Parse OSIRIS ``HIST/`` scalar energy time-history files, if present.

    OSIRIS writes whitespace-delimited ASCII time-history tables under
    ``HIST/`` when energy diagnostics are enabled in the deck. Recognized:

    - ``*fld_ene*`` — per-component field energy; value columns are summed into
      a single ``field_energy`` series.
    - ``*par*_ene*`` — per-species particle (kinetic) energy; each file's value
      columns are summed into ``kinetic_<stem>``.

    Returns an ``xr.Dataset`` on a shared ``t`` axis containing whatever was
    found — ``field_energy``, ``kinetic_<stem>``, ``kinetic_total``, and
    ``total`` (= field + kinetic, with ``attrs['total_drift_frac']`` the
    fractional spread of the total) when both halves are present — or ``None``
    if there is no ``HIST/`` directory or nothing parseable in it. Per-file time
    axes that disagree are interpolated onto the field-energy time axis.

    Note: the column convention assumed is the documented OSIRIS
    ``iteration time <values...>`` layout; validate against a real run with
    energy diagnostics enabled before relying on absolute magnitudes.

    When given the ``binary/`` directory of saved NetCDFs (no raw ``HIST/``
    ASCII), this instead reloads the pre-parsed ``HIST/energy.nc`` written by
    :func:`save_run_datasets`, so the energy-conservation plot is reproducible
    from the saved artifacts alone.
    """
    saved = Path(run_dir) / "HIST" / "energy.nc"
    if saved.is_file():
        return xr.load_dataset(saved, engine="h5netcdf")
    hist = Path(run_dir) / "HIST"
    if not hist.is_dir():
        return None

    field: tuple[np.ndarray, np.ndarray] | None = None
    kinetic: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in sorted(hist.iterdir()):
        if not f.is_file() or "ene" not in f.name.lower():
            continue
        parsed = _parse_hist_table(f)
        if parsed is None or parsed[1].size == 0:
            continue
        t, values = parsed
        if "fld" in f.name.lower():
            # fld_ene columns are the per-component field energies
            # (B1 B2 B3 E1 E2 E3); their sum is the total EM field energy.
            field = (t, values.sum(axis=1))
        elif "par" in f.name.lower():
            # par<NN>_ene columns are "Total Par." (particle COUNT) then
            # "Kin. Energy" — take the energy (last) column. Summing would fold
            # the ~1e6 particle count into the energy and swamp it.
            kinetic[f.stem] = (t, values[:, -1])

    if field is None and not kinetic:
        return None

    ref_t = field[0] if field is not None else next(iter(kinetic.values()))[0]
    data_vars: dict[str, tuple] = {}
    if field is not None:
        data_vars["field_energy"] = ("t", _interp_onto(field[0], field[1], ref_t))

    kin_total: np.ndarray | None = None
    for stem, (t, e) in kinetic.items():
        ei = _interp_onto(t, e, ref_t)
        data_vars[f"kinetic_{stem}"] = ("t", ei)
        kin_total = ei if kin_total is None else kin_total + ei
    if kin_total is not None:
        data_vars["kinetic_total"] = ("t", kin_total)

    attrs: dict[str, float] = {}
    if field is not None and kin_total is not None:
        total = data_vars["field_energy"][1] + kin_total
        data_vars["total"] = ("t", total)
        denom = float(np.max(np.abs(total))) or 1.0
        attrs["total_drift_frac"] = float((total.max() - total.min()) / denom)

    ds = xr.Dataset(data_vars, coords={"t": ref_t}, attrs=attrs)
    ds["t"].attrs.update(long_name="time", units=r"1/\omega_p")
    return ds
