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
        data_keys = sorted(
            k
            for k in f.keys()
            if k not in ("AXIS", "SIMULATION") and isinstance(f[k], h5py.Dataset)
        )
        data_vars: dict[str, tuple] = {}
        for name in data_keys:
            dset = f[name]
            arr = dset[...].astype("float64").reshape(-1)
            var_attrs = {}
            if "UNITS" in dset.attrs:
                var_attrs["units"] = _decode(dset.attrs["UNITS"])
            if "LONG_NAME" in dset.attrs:
                var_attrs["long_name"] = _decode(dset.attrs["LONG_NAME"])
            data_vars[name] = ("pidx", arr, var_attrs)

        npart = max((v[1].shape[0] for v in data_vars.values()), default=0)

        attrs = {
            "time": float(f.attrs["TIME"][0]) if "TIME" in f.attrs else float("nan"),
            "iter": int(f.attrs["ITER"][0])
            if "ITER" in f.attrs
            else _iter_from_name(path),
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


def load_raw_series(directory: str | Path) -> xr.Dataset:
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


def _sort_dumps(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix == ".h5"]
    return sorted(files, key=_iter_from_name)


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
            if any(c.suffix == ".h5" for c in d.iterdir()):
                out[str(d.relative_to(ms))] = d
        return out
    # No MS/ tree — fall back to the saved-NetCDF layout.
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
        try:
            if _diag_is_raw(relpath, diags[relpath]):
                # RAW (particle) dumps: per-particle datasets, no grid/AXIS.
                ds: xr.Dataset = load_raw_series(diags[relpath])
            else:
                ds = series_to_dataset(load_series(diags[relpath]))
            dest = out_dir / f"{relpath}.nc"
            dest.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(dest, engine="h5netcdf")
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
            energy.to_netcdf(dest, engine="h5netcdf")
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
        total_per_row = values.sum(axis=1)
        if "fld" in f.name.lower():
            field = (t, total_per_row)
        elif "par" in f.name.lower():
            kinetic[f.stem] = (t, total_per_row)

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
