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
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

_ITER_RE = re.compile(r"-(\d+)\.h5$")


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
            raise ValueError(
                f"Expected exactly one data dataset in {path}; got {data_keys}"
            )
        name = data_keys[0]
        arr = f[name][...].astype("float64")
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


def _sort_dumps(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix == ".h5"]
    return sorted(files, key=_iter_from_name)


def load_series(directory: str | Path) -> xr.DataArray:
    """Stack every dump in ``directory`` into a ``(t, ...)`` DataArray.

    All files must share the same diagnostic name, the same spatial /
    phase-space shape, and the same axis bounds — this is the standard
    OSIRIS convention for a single diagnostic's time history.
    """
    directory = Path(directory)
    dumps = _sort_dumps(directory)
    if not dumps:
        raise FileNotFoundError(f"No .h5 dumps in {directory}")
    first = load_grid_h5(dumps[0])

    times = np.empty(len(dumps), dtype="float64")
    iters = np.empty(len(dumps), dtype="int64")
    data = np.empty((len(dumps), *first.shape), dtype=first.dtype)
    data[0] = first.values
    times[0] = first.attrs["time"]
    iters[0] = first.attrs["iter"]
    for i, p in enumerate(dumps[1:], start=1):
        da = load_grid_h5(p)
        if da.shape != first.shape:
            raise ValueError(
                f"Shape mismatch in series: {p} has {da.shape}, "
                f"expected {first.shape}"
            )
        data[i] = da.values
        times[i] = da.attrs["time"]
        iters[i] = da.attrs["iter"]

    coords = {"t": times, "iter": ("t", iters)}
    coords.update({d: first.coords[d] for d in first.dims})
    dims = ("t", *first.dims)
    attrs = dict(first.attrs)
    attrs.pop("time", None)
    attrs.pop("iter", None)
    attrs.pop("source", None)
    attrs["source_dir"] = str(directory)
    return xr.DataArray(
        data, coords=coords, dims=dims, name=first.name, attrs=attrs
    )


def list_diagnostics(run_dir: str | Path) -> dict[str, Path]:
    """Map every diagnostic name to its directory under ``run_dir/MS/``.

    Discovery rule: any directory that directly contains ``.h5`` dumps is
    a diagnostic. The returned key is the directory's relative path under
    ``MS/`` (e.g. ``"FLD/e1"``, ``"PHA/x1p1/beam_pos"``).
    """
    run_dir = Path(run_dir)
    ms = run_dir / "MS"
    if not ms.is_dir():
        raise FileNotFoundError(f"No MS/ directory under {run_dir}")
    out: dict[str, Path] = {}
    for d in ms.rglob("*"):
        if not d.is_dir():
            continue
        if any(c.suffix == ".h5" for c in d.iterdir()):
            out[str(d.relative_to(ms))] = d
    return out


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


def save_run_datasets(
    run_dir: str | Path,
    out_dir: str | Path,
    diagnostics: list[str] | set[str] | None = None,
) -> list[Path]:
    """Convert each diagnostic's full time history to a netCDF file.

    One file per diagnostic is written under ``out_dir``, mirroring the OSIRIS
    ``MS/`` layout (e.g. ``out_dir/FLD/e1.nc``,
    ``out_dir/PHA/x1p1/beam_pos.nc``). Each file holds the stacked ``(t, ...)``
    series — every time slice OSIRIS dumped for that diagnostic.

    ``diagnostics``, when given, whitelists which diagnostics to convert,
    matched against either the relative path (``"FLD/e1"``) or the leaf name
    (``"e1"``). Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    diags = list_diagnostics(run_dir)
    written: list[Path] = []
    for relpath in sorted(diags):
        if diagnostics is not None and (
            relpath not in diagnostics and Path(relpath).name not in diagnostics
        ):
            continue
        ds = series_to_dataset(load_series(diags[relpath]))
        dest = out_dir / f"{relpath}.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(dest, engine="h5netcdf")
        written.append(dest)
    return written
