r"""Loaders and converters for WarpX output.

WarpX writes SI everywhere: full diagnostics as openPMD (here always the
HDF5 backend, file-based encoding — ``diags/<diag>/openpmd_%06T.h5``) and
reduced diagnostics as whitespace-delimited text tables under
``diags/reducedfiles/``. This module reads both with plain ``h5py`` /
line parsing (no openpmd-api dependency) and converts them into the same
per-diagnostic NetCDF contract the OSIRIS wrapper emits under ``binary/``:

- one ``.nc`` per diagnostic, keyed like the OSIRIS ``MS/`` tree
  (``FLD/e1``, ``DENSITY/electrons/charge``, ``RAW/electrons``,
  ``PHA/<name>/<species>``, ``HIST/energy``), holding the stacked
  ``(t, x1)`` time history;
- data in **code units** (time in ``1/ω_p``, length in ``c/ω_p``, fields
  in ``m_e c ω_p / e``, …) fixed by the manifest reference density, with
  the same dim names / attrs (``axis_units``, ``sim.XMIN`` …) OSIRIS
  series carry.

Because the contract matches, :func:`adept.osiris.io.list_diagnostics` /
``load_series`` read these files unchanged, and the OSIRIS canned-plot set
(:mod:`adept.osiris.plots`) renders WarpX runs as-is — that is the whole
point (see ``dev_docs/warpx-wrapper-plan.md``, M2).

Axis convention (cartesian 1D/2D/3D): the WarpX propagation axis is ``z``;
the OSIRIS-comparison mapping is the cyclic relabeling
``(z, x, y) → (1, 2, 3)`` for both field components and axes:

    ``E_z → e1``  ``E_x → e2``  ``E_y → e3``
    ``B_z → b1``  ``B_x → b2``  ``B_y → b3``   (likewise ``j``)
    axes  ``z → x1``  ``x → x2``  ``y → x3``

``(z, x, y)`` is a right-handed triple, so OSIRIS sign conventions —
including the left/right-going Riemann pairs ``(e2, b3)`` / ``(e3, b2)``
and the Poynting component ``s1 = e2 b3 − e3 b2`` — carry over verbatim.
In 2D (WarpX XZ geometry) the simulated plane maps to OSIRIS ``(x1, x2)``
with ``y``/``3`` out of plane, matching the OSIRIS 2D convention. Series
are stored ``(t, x2, x1)`` (2D) exactly like OSIRIS diagnostics. RZ
output (axis labels outside ``{x, y, z}``) keeps WarpX-native names and
axes — no OSIRIS mapping is defined for it here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

# SI constants (CODATA, matching scipy.constants to the digits WarpX uses).
_C = 299792458.0
_QE = 1.602176634e-19
_ME = 9.1093837015e-31
_EPS0 = 8.8541878128e-12

# Diagnostic *data* is stored float32 in the saved NetCDFs (same convention
# as the OSIRIS wrapper: halves artifact size; coordinates stay float64).
_DIAG_DTYPE = "float32"

_OPENPMD_H5_RE = re.compile(r"^openpmd_(\d+)\.h5$")


# --- code units -------------------------------------------------------------


@dataclass(frozen=True)
class CodeUnits:
    """The OSIRIS-style normalization, fully determined by a reference density.

    ``n0`` is the reference electron density in 1/m^3; everything else
    derives from it: ``wp0 = sqrt(n0 e^2 / (eps0 m_e))``, lengths in the
    skin depth ``x0 = c/wp0``, fields in ``E0 = m_e c wp0 / e`` (and
    ``B0 = E0/c``), current in ``j0 = e n0 c``, charge density in
    ``rho0 = e n0``. The 1D energy-per-transverse-area unit is
    ``u_area0 = m_e c^2 n0 x0`` (equivalently ``eps0 E0^2 x0``), chosen so
    ``U_code = 0.5 * ∫ E_code^2 dx_code`` matches the OSIRIS field-energy
    convention.
    """

    n0: float  # 1/m^3
    wp0: float = field(init=False)  # rad/s
    x0: float = field(init=False)  # m
    t0: float = field(init=False)  # s
    E0: float = field(init=False)  # V/m
    B0: float = field(init=False)  # T
    j0: float = field(init=False)  # A/m^2
    rho0: float = field(init=False)  # C/m^3
    u_area0: float = field(init=False)  # J/m^2

    def __post_init__(self) -> None:
        wp0 = math.sqrt(self.n0 * _QE**2 / (_EPS0 * _ME))
        object.__setattr__(self, "wp0", wp0)
        object.__setattr__(self, "x0", _C / wp0)
        object.__setattr__(self, "t0", 1.0 / wp0)
        object.__setattr__(self, "E0", _ME * _C * wp0 / _QE)
        object.__setattr__(self, "B0", _ME * wp0 / _QE)
        object.__setattr__(self, "j0", _QE * self.n0 * _C)
        object.__setattr__(self, "rho0", _QE * self.n0)
        object.__setattr__(self, "u_area0", _ME * _C**2 * self.n0 * (_C / wp0))

    @classmethod
    def from_density_cc(cls, n0_cc: float) -> CodeUnits:
        return cls(n0=float(n0_cc) * 1.0e6)

    @classmethod
    def from_cfg(cls, cfg: dict) -> CodeUnits | None:
        """Rebuild the normalization from a run config's derived units.

        Reads ``cfg["units"]["derived"]["n0"]`` — a pint Quantity in the live
        post-processing path, or its string form when the config came back off
        a ``units.yaml`` artifact. Returns ``None`` when no units were derived
        (no reference density anywhere), in which case conversion falls back
        to SI passthrough.
        """
        n0 = ((cfg.get("units") or {}).get("derived") or {}).get("n0")
        if n0 is None:
            return None
        if hasattr(n0, "to"):  # pint Quantity
            return cls(n0=float(n0.to("1/m^3").magnitude))
        if isinstance(n0, str):
            from adept.normalization import UREG

            return cls(n0=float(UREG.Quantity(n0).to("1/m^3").magnitude))
        return cls(n0=float(n0) * 1.0e6)  # bare number: cm^-3 convention


def code_units_for_run(src: str | Path) -> CodeUnits | None:
    """Best-effort normalization for an already-finished run's artifacts.

    For offline regeneration: looks for ``units.yaml`` (the adept artifact
    dumping ``write_units``'s output) at ``src`` or its parent, then for
    ``config.yaml`` with ``warpx.reference_density``, then for the run's
    rendered ``inputs`` deck with ``my_constants.n0`` (SI m^-3). Returns
    ``None`` when nothing usable is found.
    """
    import yaml

    from adept.normalization import UREG

    src = Path(src)
    dirs = (src, src.parent)
    for d in dirs:
        p = d / "units.yaml"
        if p.is_file():
            try:
                units = yaml.safe_load(p.read_text()) or {}
                n0 = units.get("derived", units).get("n0")
                if n0 is not None:
                    return CodeUnits(n0=float(UREG.Quantity(str(n0)).to("1/m^3").magnitude))
            except Exception:
                pass
    for d in dirs:
        p = d / "config.yaml"
        if p.is_file():
            try:
                cfg = yaml.safe_load(p.read_text()) or {}
                n0 = (cfg.get("warpx") or {}).get("reference_density")
                if n0 is not None:
                    q = UREG.Quantity(n0) if isinstance(n0, str) else UREG.Quantity(float(n0), "1/cc")
                    if q.dimensionless:
                        q = q.magnitude * UREG.cc**-1
                    return CodeUnits(n0=float(q.to("1/m^3").magnitude))
            except Exception:
                pass
    for d in dirs:
        p = d / "inputs"
        if p.is_file():
            try:
                from adept.warpx import deck as _deck

                n0 = _deck.parse_deck_file(p).get("my_constants.n0")
                if n0 is not None:
                    return CodeUnits(n0=float(n0))
            except Exception:
                pass
    return None


# --- OSIRIS-comparison naming ----------------------------------------------

# Cartesian cyclic relabeling (z, x, y) -> (1, 2, 3); see module docstring.
# Applies to field components and, via _AXIS_TO_OSIRIS, to axis labels.
_COMPONENT_TO_OSIRIS = {"z": "1", "x": "2", "y": "3"}
_AXIS_TO_OSIRIS = {"z": "x1", "x": "x2", "y": "x3"}

# Per-record conversion + labels. ``scale`` names the CodeUnits attribute
# that divides the SI values; ``units`` is the OSIRIS-style TeX unit label.
_MESH_KINDS = {
    "E": {"osiris": "e", "scale": "E0", "units": r"m_e c \omega_p e^{-1}", "long": "E"},
    "B": {"osiris": "b", "scale": "B0", "units": r"m_e \omega_p e^{-1}", "long": "B"},
    "j": {"osiris": "j", "scale": "j0", "units": r"e n_0 c", "long": "j"},
}


def _mesh_diag_key(mesh: str, comp: str | None, cartesian: bool = True) -> tuple[str, dict[str, Any]] | None:
    """Map an openPMD mesh (+ component) to an OSIRIS-contract diagnostic key.

    Returns ``(relpath, info)`` where ``info`` carries the conversion scale
    attribute name and label strings, or ``None`` for records this layer does
    not convert. ``cartesian`` says the record's axes are a subset of
    ``{x, y, z}`` (1D z / 2D XZ / 3D), where the ``(z, x, y) → (1, 2, 3)``
    relabeling is defined; RZ records keep native names via the fallback in
    the caller.
    """
    if mesh in _MESH_KINDS and comp is not None:
        kind = _MESH_KINDS[mesh]
        if cartesian and comp in _COMPONENT_TO_OSIRIS:
            name = f"{kind['osiris']}{_COMPONENT_TO_OSIRIS[comp]}"
            long_name = f"{kind['long']}_{_COMPONENT_TO_OSIRIS[comp]}"
        else:
            name = f"{mesh}{comp}"
            long_name = f"{mesh}_{comp}"
        return f"FLD/{name}", {"scale": kind["scale"], "units": kind["units"], "long_name": long_name, "name": name}
    if mesh == "rho" and comp is None:
        return "DENSITY/total/charge", {
            "scale": "rho0",
            "units": r"e n_0",
            "long_name": r"\rho",
            "name": "charge",
        }
    if mesh.startswith("rho_") and comp is None:
        species = mesh[len("rho_") :]
        return f"DENSITY/{species}/charge", {
            "scale": "rho0",
            "units": r"e n_0",
            "long_name": r"\rho_{" + species + "}",
            "name": "charge",
        }
    return None


# --- openPMD (h5) reading ---------------------------------------------------


def _openpmd_files(diag_dir: Path) -> list[Path]:
    return sorted(
        (p for p in diag_dir.iterdir() if p.is_file() and _OPENPMD_H5_RE.match(p.name)),
        key=lambda p: int(_OPENPMD_H5_RE.match(p.name).group(1)),
    )


def _decode(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.flat[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _iteration_groups(f: h5py.File) -> list[tuple[int, h5py.Group]]:
    """``(iteration, group)`` pairs in a file, sorted (handles group-based too)."""
    data = f.get("data")
    if data is None:
        return []
    out = []
    for k in data.keys():
        try:
            out.append((int(k), data[k]))
        except ValueError:
            continue
    return sorted(out, key=lambda kv: kv[0])


def _record_components(rec: h5py.Group | h5py.Dataset) -> dict[str | None, h5py.Dataset | h5py.Group]:
    """Map component name -> dataset/constant-group for an openPMD record.

    A scalar record (``rho``) is the dataset itself (key ``None``); a vector
    record (``E``) is a group of components. Constant components are groups
    carrying ``value``/``shape`` attrs instead of data.
    """
    if isinstance(rec, h5py.Dataset):
        return {None: rec}
    out: dict[str | None, h5py.Dataset | h5py.Group] = {}
    for k in rec.keys():
        out[k] = rec[k]
    return out if out else {None: rec}


def _read_component(node: h5py.Dataset | h5py.Group) -> tuple[np.ndarray, float]:
    """Values (native shape) and unitSI for a (possibly constant) component."""
    unit_si = float(node.attrs.get("unitSI", 1.0))
    if isinstance(node, h5py.Dataset):
        return node[...], unit_si
    # Constant record component: attrs value + shape.
    shape = tuple(int(s) for s in np.atleast_1d(node.attrs.get("shape", [1])))
    value = node.attrs.get("value", 0.0)
    return np.full(shape, value), unit_si


def _mesh_axes(mesh: h5py.Group | h5py.Dataset, n_cells: tuple[int, ...]) -> list[dict]:
    """Per-axis dicts (label, SI coordinates) for a mesh record.

    Coordinates are node positions ``(offset + (i + stagger) * spacing) *
    gridUnitSI`` in the order of ``axisLabels`` (matching the C-order data
    layout WarpX writes).
    """
    attrs = mesh.attrs
    labels = [_decode(x) for x in np.atleast_1d(attrs.get("axisLabels", [b"z"]))]
    spacing = np.atleast_1d(attrs.get("gridSpacing", [1.0])).astype(float)
    offset = np.atleast_1d(attrs.get("gridGlobalOffset", [0.0])).astype(float)
    grid_unit = float(attrs.get("gridUnitSI", 1.0))
    axes = []
    for d, label in enumerate(labels):
        n = n_cells[d] if d < len(n_cells) else 1
        axes.append(
            {
                "label": label,
                "coords": (offset[d] + (np.arange(n) + 0.0) * spacing[d]) * grid_unit,
                "spacing": spacing[d] * grid_unit,
            }
        )
    return axes


def _component_stagger(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    return np.atleast_1d(node.attrs.get("position", [0.0])).astype(float)


class _NonMonotonicIterations(Exception):
    """Raised mid-walk when a series' file order is not iteration order.

    An append-only writer cannot reorder, so the streaming converters catch
    this and restart on the eager path. No openPMD file-based series WarpX
    produces triggers it (``_openpmd_files`` sorts on the filename iteration
    index and ``_iteration_groups`` sorts within a file), but the eager
    result is correct either way.
    """


def list_field_records(diag_dir: str | Path) -> list[tuple[str, str | None]]:
    """``(mesh, component)`` pairs present in a full diagnostic's first dump."""
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        return []
    out: list[tuple[str, str | None]] = []
    with h5py.File(files[0], "r") as f:
        meshes_path = _decode(f.attrs.get("meshesPath", "fields/")).strip("/")
        for _it, grp in _iteration_groups(f)[:1]:
            meshes = grp.get(meshes_path)
            if meshes is None:
                continue
            for mesh_name in meshes.keys():
                for comp in _record_components(meshes[mesh_name]):
                    out.append((mesh_name, comp))
    return out


@dataclass
class _FieldRecordPlan:
    """One mesh component's OSIRIS-side identity, from the first dump alone.

    The axes, code-unit scaling, ``binary/`` key and storage-order transpose
    are all fixed by the first file's grid metadata plus the deck-independent
    ``_mesh_diag_key`` mapping — no pixel data and no walk of the history.
    That is what lets :func:`convert_field_series_streaming` decide *what to
    write and where* before opening a single dump, so the whitelist can skip
    records instead of loading and discarding them.

    ``perm`` reorders an openPMD-order slab into ``dims`` (OSIRIS stores
    multi-D as ``(t, …, x2, x1)``); it is the identity in 1D and 2D XZ.
    """

    mesh: str
    comp: str | None
    rel: str
    name: str
    dims: tuple[str, ...]
    perm: tuple[int, ...]
    coords: dict[str, np.ndarray]
    attrs: dict[str, Any]
    scale: float


def _field_plan(
    diag_dir: str | Path,
    *,
    units: CodeUnits | None = None,
) -> tuple[str, list[_FieldRecordPlan]]:
    """``(meshes_path, plans)`` for every record in a full diagnostic.

    Costs **one** file open regardless of history length.
    """
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        raise FileNotFoundError(f"No openpmd_*.h5 files in {diag_dir}")

    plans: list[_FieldRecordPlan] = []
    with h5py.File(files[0], "r") as f:
        meshes_path = _decode(f.attrs.get("meshesPath", "fields/")).strip("/")
        groups = _iteration_groups(f)[:1]
        if not groups:
            raise FileNotFoundError(f"No iteration groups in {files[0]}")
        meshes = groups[0][1].get(meshes_path)
        if meshes is None:
            raise FileNotFoundError(f"No meshes under {meshes_path!r} in {files[0]}")
        for mesh in meshes.keys():
            rec = meshes[mesh]
            for comp, node in _record_components(rec).items():
                if isinstance(node, h5py.Dataset):
                    shape = tuple(int(s) for s in node.shape)
                else:  # constant record component
                    shape = tuple(int(s) for s in np.atleast_1d(node.attrs.get("shape", [1])))
                unit_si = float(node.attrs.get("unitSI", 1.0))

                axes = _mesh_axes(rec, shape)
                stag = _component_stagger(node)
                for d, ax in enumerate(axes):
                    s = stag[d] if d < stag.size else 0.0
                    ax["coords"] = ax["coords"] + s * ax["spacing"]
                cartesian = all(ax["label"] in _AXIS_TO_OSIRIS for ax in axes)

                key_info = _mesh_diag_key(mesh, comp, cartesian)
                if key_info is not None:
                    rel, info = key_info
                    scale = getattr(units, info["scale"]) if units is not None else 1.0
                    name, long_name, val_units = info["name"], info["long_name"], info["units"]
                else:
                    scale = 1.0
                    name = mesh if comp is None else f"{mesh}{comp}"
                    long_name = name
                    val_units = "SI"
                    rel = f"FLD/{name}"
                if units is None:
                    val_units = "SI"

                dims_native: list[str] = []
                coords: dict[str, np.ndarray] = {}
                axis_units: dict[str, str] = {}
                axis_long: dict[str, str] = {}
                extents: list[tuple[str, float, float, int]] = []
                for ax in axes:
                    if cartesian:
                        dim = _AXIS_TO_OSIRIS[ax["label"]]
                        axis_long[dim] = f"x_{dim[1:]}"
                    else:
                        dim = ax["label"]
                        axis_long[dim] = ax["label"]
                    cv = np.asarray(ax["coords"], dtype="float64")
                    if units is not None:
                        cv = cv / units.x0
                        axis_units[dim] = r"c / \omega_p"
                    else:
                        axis_units[dim] = "m"
                    coords[dim] = cv
                    dims_native.append(dim)
                    extents.append((dim, float(cv[0]), float(cv[-1]), int(cv.size)))
                # sim.XMIN/XMAX/NX are indexed by the OSIRIS axis number
                # (x1 -> entry 0), so order them x1, x2, … regardless of the
                # openPMD axis order.
                if cartesian:
                    extents.sort(key=lambda e: e[0])

                # OSIRIS multi-D storage order: (t, …, x2, x1).
                dims = sorted(dims_native, reverse=True) if cartesian and len(dims_native) > 1 else list(dims_native)
                perm = tuple(dims_native.index(d) for d in dims)

                attrs = {
                    "long_name": long_name,
                    "units": val_units,
                    "time_units": r"1/\omega_p" if units is not None else "s",
                    "axis_units": axis_units,
                    "axis_long_names": axis_long,
                    "source_dir": str(diag_dir),
                    "warpx_record": mesh if comp is None else f"{mesh}/{comp}",
                    "unit_si": unit_si * (float(np.asarray(scale)) if units is not None else 1.0),
                    "sim.NDIMS": len(axes),
                    "sim.XMIN": [e[1] for e in extents],
                    "sim.XMAX": [e[2] for e in extents],
                    "sim.NX": [e[3] for e in extents],
                }
                plans.append(
                    _FieldRecordPlan(
                        mesh=mesh,
                        comp=comp,
                        rel=rel,
                        name=name,
                        dims=tuple(dims),
                        perm=perm,
                        coords=coords,
                        attrs=attrs,
                        scale=float(np.asarray(scale)) if units is not None else 1.0,
                    )
                )
    return meshes_path, plans


def _field_slab(plan: _FieldRecordPlan, node, *, scaled: bool) -> np.ndarray:
    """One dump of one record, in OSIRIS storage order and float32."""
    arr, _unit_si = _read_component(node)
    slab = np.asarray(arr, dtype=_DIAG_DTYPE)
    if scaled and plan.scale != 1.0:
        slab = slab / np.asarray(plan.scale, dtype=_DIAG_DTYPE)
    if plan.perm != tuple(range(slab.ndim)):
        slab = slab.transpose(plan.perm)
    return np.ascontiguousarray(slab)


def _field_records(
    diag_dir: Path,
    meshes_path: str,
    plans: list[_FieldRecordPlan],
    *,
    units: CodeUnits | None,
    require_monotonic: bool = False,
    skip: set[str] | None = None,
    on_error: Any = None,
) -> Iterator[tuple[int, float, dict[str, np.ndarray]]]:
    """Walk the files **once**, yielding ``(iteration, time, {plan.rel: slab})``.

    Every requested record comes out of the *same* file open. The per-record
    loop it replaces re-walked the whole tree once per component — 7 passes
    over 87,780 files at a 2-omega_0-Nyquist field cadence, ~67 min of pure
    open overhead at the ~6.5 ms/file measured on Perlmutter's Lustre.

    ``skip`` (a live set of ``plan.rel``) is consulted every dump and lets a
    caller retire a record mid-walk; with ``on_error`` a failed record is
    reported and retired instead of aborting the walk, which is how the
    single pass keeps the per-record failure isolation of the loop it
    replaces. Without ``on_error`` a read failure propagates.
    """
    t_scale = units.wp0 if units is not None else 1.0
    last_it = -(1 << 62)
    for path in _openpmd_files(diag_dir):
        with h5py.File(path, "r") as f:
            for it, grp in _iteration_groups(f):
                meshes = grp.get(meshes_path)
                if meshes is None:
                    continue
                if require_monotonic and it <= last_it:
                    raise _NonMonotonicIterations(f"{diag_dir}: iteration {it} follows {last_it}")
                last_it = it
                t_si = float(grp.attrs.get("time", 0.0)) * float(grp.attrs.get("timeUnitSI", 1.0))
                slabs: dict[str, np.ndarray] = {}
                for plan in plans:
                    if skip is not None and plan.rel in skip:
                        continue
                    rec = meshes.get(plan.mesh)
                    if rec is None:
                        continue
                    comps = _record_components(rec)
                    if plan.comp not in comps:
                        continue
                    try:
                        slabs[plan.rel] = _field_slab(plan, comps[plan.comp], scaled=units is not None)
                    except Exception as e:
                        if on_error is None:
                            raise
                        on_error(plan.rel, int(it), e)
                        if skip is not None:
                            skip.add(plan.rel)
                if slabs:
                    yield int(it), t_si * t_scale, slabs


def _field_slab_da(plan: _FieldRecordPlan, it: int, t: float, slab: np.ndarray) -> xr.DataArray:
    """One dump of one record as a DataArray in the StreamWriter's contract."""
    return xr.DataArray(
        slab,
        coords={d: plan.coords[d] for d in plan.dims},
        dims=plan.dims,
        name=plan.name,
        attrs={**plan.attrs, "time": float(t), "iter": int(it)},
    )


def convert_field_series_streaming(
    diag_dir: str | Path,
    targets: list[tuple[str, _FieldRecordPlan]],
    out_dir: str | Path,
    *,
    units: CodeUnits | None = None,
    meshes_path: str = "fields",
    complevel: int = 4,
) -> list[Path]:
    """Convert several field records to NetCDF in **one** walk of the dumps.

    ``targets`` pairs each record's ``binary/`` key (already disambiguated
    against keys other diagnostics took) with its
    :class:`_FieldRecordPlan`. One :class:`~adept.osiris.stream.StreamWriter`
    is held open per target and fed a slab at a time, so:

    - each file is opened once instead of once per component;
    - peak memory is one slab per target plus their ~1 MiB write batches,
      against :func:`load_field_series`'s list-of-float64-slabs then
      ``np.stack`` (~6.3 GB per component at 87,780 dumps of 3594 cells,
      and far worse in 2D);
    - the caller's whitelist skips records *before* they are read.

    A record whose read fails is dropped (logged once) and the rest continue,
    preserving the per-record isolation of the loop this replaces. An
    out-of-order iteration falls back to the per-record eager path.
    Returns the written paths.
    """
    from adept.osiris.stream import StreamWriter

    diag_dir = Path(diag_dir)
    out_dir = Path(out_dir)
    if not targets:
        return []

    def _dest(rel: str) -> Path:
        p = out_dir / f"{rel}.nc"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _eager() -> list[Path]:
        from adept.osiris.io import series_to_dataset

        done: list[Path] = []
        for rel, plan in targets:
            try:
                da = load_field_series(diag_dir, plan.mesh, plan.comp, units=units)
                ds = series_to_dataset(da)
                dest = _dest(rel)
                ds.to_netcdf(
                    dest,
                    engine="h5netcdf",
                    encoding={n: {"zlib": True, "complevel": complevel, "shuffle": True} for n in ds.data_vars},
                )
                done.append(dest)
            except Exception as e:
                print(f"[post] skipping {rel} from {diag_dir.name}: {e}")
        return done

    for rel, _plan in targets:
        # Never resume into a partial file left by a killed postproc.
        _dest(rel).unlink(missing_ok=True)

    writers: dict[str, Any] = {}
    by_plan = {plan.rel: (rel, plan) for rel, plan in targets}
    dropped: set[str] = set()

    def _drop(plan_rel: str, it: int, exc: Exception) -> None:
        rel, _plan = by_plan[plan_rel]
        print(f"[post] dropping {rel} from {diag_dir.name} at iteration {it}: {exc}")
        w = writers.pop(rel, None)
        if w is not None:
            w.close()
        _dest(rel).unlink(missing_ok=True)

    try:
        for it, t, slabs in _field_records(
            diag_dir,
            meshes_path,
            [p for _r, p in targets],
            units=units,
            require_monotonic=True,
            skip=dropped,
            on_error=_drop,
        ):
            for plan_rel, slab in slabs.items():
                rel, plan = by_plan[plan_rel]
                try:
                    da = _field_slab_da(plan, it, t, slab)
                    if rel not in writers:
                        writers[rel] = StreamWriter(_dest(rel), da, source_dir=diag_dir, complevel=complevel)
                    writers[rel].append(da)
                except Exception as e:  # one bad record must not abort the rest
                    _drop(plan_rel, it, e)
                    dropped.add(plan_rel)
    except _NonMonotonicIterations:
        for w in writers.values():
            w.close()
        writers.clear()
        for rel, _plan in targets:
            _dest(rel).unlink(missing_ok=True)
        return _eager()
    finally:
        for w in writers.values():
            w.close()
    return [_dest(rel) for rel, plan in targets if plan.rel not in dropped and _dest(rel).exists()]


def load_field_series(
    diag_dir: str | Path,
    mesh: str,
    comp: str | None = None,
    *,
    units: CodeUnits | None = None,
) -> xr.DataArray:
    """Stack one mesh component's time history into a ``(t, …)`` DataArray.

    Values and axes are converted to code units when ``units`` is given
    (fields via the record's scale, coordinates via ``x0``, time via
    ``wp0``); otherwise everything stays SI and the attrs say so. The
    returned array follows the OSIRIS series conventions — dims ``(t, x1)``
    in 1D and ``(t, x2, x1)`` in 2D XZ (axes relabeled ``z → x1``,
    ``x → x2``, ``y → x3``), coords ``t``/``iter``, attrs ``axis_units`` /
    ``sim.XMIN`` (ordered ``x1, x2, …``) — so downstream plotting treats it
    like any OSIRIS diagnostic. RZ axes keep their native labels.

    Stacks the whole history in memory, one record per call. The converter
    uses :func:`convert_field_series_streaming` instead, which writes every
    record of a diagnostic in a single walk; this stays the reference
    implementation and the interactive entry point.
    """
    diag_dir = Path(diag_dir)
    meshes_path, plans = _field_plan(diag_dir, units=units)
    plan = next((p for p in plans if p.mesh == mesh and p.comp == comp), None)
    if plan is None:
        present = sorted((p.mesh, p.comp) for p in plans)
        raise KeyError(f"component {comp!r} not in mesh {mesh!r} ({present})")

    slabs: list[np.ndarray] = []
    times: list[float] = []
    iters: list[int] = []
    for it, t, got in _field_records(diag_dir, meshes_path, [plan], units=units):
        slabs.append(got[plan.rel])
        times.append(t)
        iters.append(it)
    if not slabs:
        raise FileNotFoundError(f"mesh {mesh!r} (comp {comp!r}) not found in {diag_dir}")

    order = np.argsort(iters, kind="stable")
    data = np.stack([slabs[i] for i in order])
    coords: dict[str, Any] = {
        "t": np.asarray(times, dtype="float64")[order],
        "iter": ("t", np.asarray(iters, dtype="int64")[order]),
    }
    coords.update({d: plan.coords[d] for d in plan.dims})
    return xr.DataArray(data, coords=coords, dims=("t", *plan.dims), name=plan.name, attrs=dict(plan.attrs))


def load_particle_species(
    diag_dir: str | Path,
    species: str,
    *,
    units: CodeUnits | None = None,
) -> xr.Dataset:
    """Concatenate one species' particle dumps long-form (the RAW analog).

    Every dump's particles are concatenated along ``pidx`` with per-row
    ``t`` / ``iter`` coordinates (particle counts vary dump-to-dump, so a
    rectangular stack is impossible — same layout as
    :func:`adept.osiris.io.load_raw_series`). Variables, in code units when
    ``units`` is given:

    - ``x1`` (1D): position + positionOffset along z, in ``c/ω_p``;
    - ``p1``/``p2``/``p3``: momentum (z, x, y — the cyclic OSIRIS mapping)
      in units of ``m_s c`` for the species mass ``m_s``;
    - ``ene``: ``γ - 1`` (kinetic energy in ``m_s c^2``);
    - ``q``: signed macroparticle charge ``(q_s/e) · w / (n0 · x0)``,
      dimensionless — depositing ``q`` over cells of size ``Δx1`` recovers
      the charge density in ``e n_0`` units;
    - ``w``: the raw openPMD weighting (physical particles per macroparticle,
      per m^2 of transverse area in 1D).

    Without ``units``, positions stay in m, momenta in kg·m/s, and ``q`` is
    omitted.
    """
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        raise FileNotFoundError(f"No openpmd_*.h5 files in {diag_dir}")

    rows: dict[str, list[np.ndarray]] = {}
    times: list[np.ndarray] = []
    iters: list[np.ndarray] = []
    n_dumps = 0
    for path in files:
        with h5py.File(path, "r") as f:
            ppath = _decode(f.attrs.get("particlesPath", "particles/")).strip("/")
            for it, grp in _iteration_groups(f):
                parts = grp.get(ppath)
                if parts is None or species not in parts:
                    continue
                sp = parts[species]
                rec = _species_records(sp, units)
                if rec is None:
                    continue
                n = next(len(v) for v in rec.values())
                n_dumps += 1
                for k, v in rec.items():
                    rows.setdefault(k, []).append(v)
                t_si = float(grp.attrs.get("time", 0.0)) * float(grp.attrs.get("timeUnitSI", 1.0))
                times.append(np.full(n, t_si * (units.wp0 if units is not None else 1.0), dtype="float64"))
                iters.append(np.full(n, int(it), dtype="int64"))
    if n_dumps == 0:
        raise FileNotFoundError(f"species {species!r} not found in {diag_dir}")

    data_vars = {k: ("pidx", np.concatenate(v).astype(_DIAG_DTYPE)) for k, v in rows.items()}
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(t=("pidx", np.concatenate(times)), iter=("pidx", np.concatenate(iters)))
    ds.attrs.update(
        long_name=f"{species} particles",
        time_units=r"1/\omega_p" if units is not None else "s",
        source_dir=str(diag_dir),
        n_dumps=n_dumps,
    )
    return ds


def _species_records(sp: h5py.Group, units: CodeUnits | None) -> dict[str, np.ndarray] | None:
    """Extract one dump's per-particle arrays for a species (SI or code units)."""
    if "position" not in sp:
        return None
    pos = _record_components(sp["position"])
    off = _record_components(sp["positionOffset"]) if "positionOffset" in sp else {}
    out: dict[str, np.ndarray] = {}

    # Number of particles: from any non-constant dataset, else constant shape.
    n = None
    for node in list(pos.values()) + list(off.values()):
        arr, _u = _read_component(node)
        n = max(n or 0, arr.size)
    if not n:
        return None

    def full(node) -> np.ndarray:
        arr, u = _read_component(node)
        arr = np.asarray(arr, dtype="float64") * u
        return np.broadcast_to(arr, (n,)) if arr.size != n else arr.reshape(n)

    # 1D: z is the only stored axis; the cyclic mapping puts it on x1.
    for axis, name in (("z", "x1"), ("x", "x2"), ("y", "x3")):
        if axis in pos and isinstance(pos[axis], h5py.Dataset):
            x = full(pos[axis])
            if axis in off:
                x = x + full(off[axis])
            out[name] = x / units.x0 if units is not None else x

    mass = None
    if "mass" in sp:
        m_arr, m_u = _read_component(_record_components(sp["mass"])[None])
        mass = float(np.asarray(m_arr).flat[0]) * m_u
    if "momentum" in sp:
        mom = _record_components(sp["momentum"])
        p_scale = (mass or _ME) * _C
        usq = None
        for axis, name in (("z", "p1"), ("x", "p2"), ("y", "p3")):
            if axis in mom:
                p = full(mom[axis])
                pc = p / p_scale
                out[name] = pc if units is not None else p
                usq = pc**2 if usq is None else usq + pc**2
        if usq is not None:
            out["ene"] = np.sqrt(1.0 + usq) - 1.0

    if "weighting" in sp:
        w = full(_record_components(sp["weighting"])[None])
        out["w"] = w
        if units is not None and "charge" in sp:
            q_arr, q_u = _read_component(_record_components(sp["charge"])[None])
            q_over_e = float(np.asarray(q_arr).flat[0]) * q_u / _QE
            out["q"] = q_over_e * w / (units.n0 * units.x0)

    # BoundaryScraping extras: when the particles come from a scraping buffer
    # the per-particle crossing step/time ride along (code time units when a
    # normalization is given, so `t_scraped` pairs with the `t` coordinate).
    t_scale = units.wp0 if units is not None else 1.0
    for extra, name, scale in (
        ("stepScraped", "step_scraped", 1.0),
        ("timeScraped", "t_scraped", t_scale),
        ("deltaTimeScraped", "dt_scraped", t_scale),
    ):
        if extra in sp:
            out[name] = full(_record_components(sp[extra])[None]) * scale
    return out


def list_species(diag_dir: str | Path) -> list[str]:
    """Species names present in a full diagnostic's first dump."""
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        return []
    with h5py.File(files[0], "r") as f:
        ppath = _decode(f.attrs.get("particlesPath", "particles/")).strip("/")
        for _it, grp in _iteration_groups(f)[:1]:
            parts = grp.get(ppath)
            if parts is not None:
                return sorted(parts.keys())
    return []


# --- reduced diagnostics ----------------------------------------------------

_COLUMN_RE = re.compile(r"\[(\d+)\]([^([\]]+)\(([^)]*)\)")


def parse_reduced_diag(path: str | Path) -> xr.Dataset:
    """Parse one reduced-diagnostic text table into an SI ``xr.Dataset``.

    WarpX reduced diags are whitespace-separated with a one-line header of
    ``[i]name(unit)`` tokens — ``#``-prefixed for every type except
    FieldProbe. Column 0 is the step and column 1 the time in seconds; they
    become the ``step`` coordinate and ``time`` variable, with every further
    column a data variable named from the header (its unit in
    ``attrs["units"]``). Values are left in SI — converters downstream pick
    the columns they understand.
    """
    path = Path(path)
    header: list[tuple[int, str, str]] = []
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            cols = _COLUMN_RE.findall(s)
            if cols and not header:
                header = [(int(i), name.strip(), unit) for i, name, unit in cols]
                continue
            try:
                rows.append([float(tok) for tok in s.split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype="float64")
    names = {i: (name, unit) for i, name, unit in header if i < width}

    step = arr[:, 0].astype("int64")
    ds = xr.Dataset(coords={"step": step})
    for i in range(1, width):
        name, unit = names.get(i, (f"col{i}", ""))
        var = "time" if i == 1 else name
        ds[var] = ("step", arr[:, i])
        ds[var].attrs["units"] = unit
    ds.attrs["source"] = str(path)
    ds.attrs["diag_name"] = path.stem
    return ds


def list_reduced_diags(run_dir: str | Path) -> dict[str, Path]:
    """Map reduced-diagnostic names to their ``.txt`` tables for a run."""
    reduced = Path(run_dir) / "diags" / "reducedfiles"
    if not reduced.is_dir():
        return {}
    return {p.stem: p for p in sorted(reduced.glob("*.txt"))}


def hist_energy_from_reduced(run_dir: str | Path, units: CodeUnits | None = None) -> xr.Dataset | None:
    """Build the OSIRIS ``HIST/energy`` dataset from WarpX reduced diags.

    Combines the ``FieldEnergy`` total with the per-species ``ParticleEnergy``
    columns into the exact schema :func:`adept.osiris.io.load_hist_energy`
    returns (``field_energy``, ``kinetic_<species>``, ``kinetic_total``,
    ``total`` + ``attrs["total_drift_frac"]``, on a shared code-units ``t``
    axis), so the OSIRIS energy-conservation plots and the drift metric work
    on WarpX runs unchanged. Values are converted to the code energy unit
    (``m_e c^2 n_0 x_0`` per transverse area in 1D) when ``units`` is given,
    else left in the native J/m^2. Returns ``None`` when neither diagnostic
    was enabled.
    """
    # Per-table best-effort: ParticleHistogram2D leaves an *empty* companion
    # .txt next to its openPMD dir, which must not abort the energy build.
    tables = {}
    for name, p in list_reduced_diags(run_dir).items():
        try:
            tables[name] = parse_reduced_diag(p)
        except Exception:
            continue
    fld = next((ds for ds in tables.values() if any(v.startswith("total_lev") for v in ds.data_vars)), None)
    par = next((ds for ds in tables.values() if "total" in ds.data_vars and "time" in ds.data_vars), None)
    if fld is None and par is None:
        return None

    e_scale = units.u_area0 if units is not None else 1.0
    t_scale = units.wp0 if units is not None else 1.0

    ref = fld if fld is not None else par
    ref_t = ref["time"].values * t_scale

    def onto(ds: xr.Dataset, col: str) -> np.ndarray:
        src_t = ds["time"].values * t_scale
        v = ds[col].values / e_scale
        if src_t.shape == ref_t.shape and np.allclose(src_t, ref_t):
            return v
        return np.interp(ref_t, src_t, v)

    data_vars: dict[str, tuple] = {}
    if fld is not None:
        data_vars["field_energy"] = ("t", onto(fld, "total_lev0"))

    kin_total = None
    if par is not None:
        species_cols = [v for v in par.data_vars if v not in ("time", "total") and not v.endswith("_mean")]
        for sp in species_cols:
            data_vars[f"kinetic_{sp}"] = ("t", onto(par, sp))
        kin_total = onto(par, "total")
        data_vars["kinetic_total"] = ("t", kin_total)

    attrs: dict[str, float] = {}
    if fld is not None and kin_total is not None:
        total = data_vars["field_energy"][1] + kin_total
        data_vars["total"] = ("t", total)
        denom = float(np.max(np.abs(total))) or 1.0
        attrs["total_drift_frac"] = float((total.max() - total.min()) / denom)

    ds = xr.Dataset(data_vars, coords={"t": ref_t}, attrs=attrs)
    ds["t"].attrs.update(long_name="time", units=r"1/\omega_p" if units is not None else "s")
    return ds


# --- particle phase-space histograms (the PHA analog) ------------------------
#
# WarpX's ParticleHistogram2D reduced diagnostic deposits a per-particle
# ``value_function`` onto a fixed 2-D (abscissa, ordinate) grid and writes the
# history as openPMD meshes under ``diags/reducedfiles/<name>/``; the 1-D
# ParticleHistogram writes one text column per bin. Both are the WarpX-native
# analog of the OSIRIS species phase spaces, so they are converted into
# ``PHA/<name>/<species>.nc`` with the OSIRIS conventions the SRS analyses
# (osiris_lpi) expect:
#
# - axes labeled *edge-style* (``linspace(min, max, n)`` over the full bin
#   range — the adept-OSIRIS convention; consumers convert to bin centers);
# - a count deposit (``value_function = "w"``) stored as the charge-signed
#   phase-space density ``q/|q| · f`` with the OSIRIS cartesian normalization
#   ``sum_bins f · d(axis) = n(x)/n0`` — i.e. the raw per-bin weight sum
#   divided by ``Δx_bin · n0 · d(axis)``;
# - any other deposit treated as a *flux* density: the value function must be
#   the ``m_e c^3``-reduced form (e.g. ``w*(g-1)*(uz/g)`` for the OSIRIS
#   ``x1gl_q1`` energy-flux deposit ``KE·v1``), and the stored field is the
#   per-bin sum divided by ``Δx_bin · n0 · d(axis)`` so that integrating over
#   the ordinate axis recovers the flux in code units (``n0 m_e c^3``, the
#   same unit as the laser flux ``I0 = (a0 ω0)^2/2``).
#
# The ordinate dim is named from the deck's ``histogram_function_ord``:
# ``log10(...)`` → ``gamma`` (the OSIRIS log-gamma axis), a bare ``uz``/``ux``/
# ``uy`` → ``p1``/``p2``/``p3`` (the cyclic mapping), anything else → ``ord``.
#
# Conversion runs in two passes over the openPMD files: :func:`_histogram2d_plan`
# indexes the iterations and derives every axis/attr while holding no pixel
# data, then :func:`_histogram2d_slabs` yields one scaled float32 slab per
# dump. :func:`convert_histogram2d_streaming` feeds those slabs straight to the
# OSIRIS ``StreamWriter``, so a history is never stacked in memory;
# :func:`load_particle_histogram2d` fills the whole cube and is for callers
# that want the array.


def _deck_for_run(run_dir: str | Path) -> dict | None:
    """The parsed (flat) inputs deck rendered into a run directory, if any."""
    p = Path(run_dir) / "inputs"
    if not p.is_file():
        return None
    try:
        from adept.warpx import deck as _deck

        return _deck.parse_deck_file(p)
    except Exception:
        return None


def _deck_get(deck: dict | None, key: str, default=None):
    """Deck lookup tolerant of parser-argument specs on the stored key."""
    if deck is None:
        return default
    if key in deck:
        return deck[key]
    candidates = [k for k in deck if k.split("(", 1)[0] == key]
    if len(candidates) == 1:
        return deck[candidates[0]]
    return default


def _deck_species_of(deck: dict | None, diag_name: str) -> str:
    sp = _deck_get(deck, f"{diag_name}.species")
    if isinstance(sp, list):
        sp = sp[0] if sp else None
    return str(sp) if sp is not None else "particles"


def _species_charge_sign(deck: dict | None, species: str) -> float:
    """Sign of a species' charge from the deck (``-q_e`` → -1); default -1.

    Only the *sign* matters: OSIRIS phase spaces store the charge density
    ``q·f`` and every consumer divides the (self-detected) sign back out, so
    a wrong default is cosmetic. Electrons are the overwhelmingly common
    histogram subject, hence -1.
    """
    q = _deck_get(deck, f"{species}.charge")
    if isinstance(q, str):
        return -1.0 if q.strip().startswith("-") else 1.0
    if isinstance(q, (int, float)):
        return -1.0 if q < 0 else 1.0
    return -1.0


def _ordinate_info(fn_ord: str | None) -> tuple[str, str, str]:
    """``(dim, long_name, kind)`` for a histogram axis (bin) function.

    Classifies momentum (``uz/ux/uy`` → ``p1/p2/p3``, in m_e c), spatial
    (``z/x/y`` → ``x1/x2/x3`` via the cyclic relabeling, meters → c/ω_p)
    and ``log10(...)`` → ``gamma`` axes. Used for both the ordinate and —
    since the p1p2 mislabeling fix (dev_docs/p2_bugfix.md) — the abscissa:
    a momentum abscissa must NOT be named ``x1``/converted by ``x0``, or
    the spatial box crop downstream blanks the phase space.
    """
    s = (fn_ord or "").replace(" ", "")
    if "log10" in s:
        return "gamma", r"log_{10}\gamma", "log_gamma"
    for u, i in (("uz", "1"), ("ux", "2"), ("uy", "3")):
        if s == u:
            return f"p{i}", f"p_{i}", "momentum"
    for u, i in (("z", "1"), ("x", "2"), ("y", "3")):
        if s == u:
            return f"x{i}", f"x_{i}", "spatial"
    return "ord", fn_ord or "ord", "other"


def _edge_style_axis(lo: float, hi: float, n: int) -> np.ndarray:
    """The adept-OSIRIS edge-style axis label: ``linspace(lo, hi, n)``."""
    return np.linspace(float(lo), float(hi), int(n))


def _box_attrs(deck: dict | None, units: CodeUnits | None) -> dict[str, Any]:
    """``sim.XMIN/XMAX/NDIMS`` attrs (code units) from the deck geometry.

    Ordered **x1-first** like the field converter (``sim.XMIN`` entry 0 is
    the OSIRIS x1 = WarpX z axis): the deck's ``prob_lo/hi`` are WarpX
    ``(x[, y], z)``-ordered, which is what let the transverse extent land in
    slot 0 and crop the p1x1 phase space to 3 µm (dev_docs/p2_bugfix.md,
    root cause 2 — the histogram-path twin of commit ``36a22ae``).
    """
    lo = _deck_get(deck, "geometry.prob_lo")
    hi = _deck_get(deck, "geometry.prob_hi")
    if lo is None or hi is None:
        return {}
    lo = lo if isinstance(lo, list) else [lo]
    hi = hi if isinstance(hi, list) else [hi]
    order = {1: [0], 2: [1, 0], 3: [2, 0, 1]}.get(len(lo))
    if order is None:
        return {}
    x0 = units.x0 if units is not None else 1.0
    try:
        return {
            "sim.NDIMS": len(lo),
            "sim.XMIN": [float(lo[i]) / x0 for i in order],
            "sim.XMAX": [float(hi[i]) / x0 for i in order],
        }
    except (TypeError, ValueError):
        return {}


def list_histogram2d_diags(run_dir: str | Path) -> dict[str, Path]:
    """ParticleHistogram2D output dirs under ``diags/reducedfiles/``."""
    reduced = Path(run_dir) / "diags" / "reducedfiles"
    if not reduced.is_dir():
        return {}
    return {d.name: d for d in sorted(reduced.iterdir()) if d.is_dir() and _openpmd_files(d)}


@dataclass
class _Histogram2DGeometry:
    """A ParticleHistogram2D's axes, attrs and scaling, from its first dump.

    Everything needed to convert the series is fixed by the deck and the
    first file's grid metadata, so this costs **one** file open regardless of
    history length. ``_histogram2d_records`` then walks the files once to
    produce the slabs — the whole conversion is a single pass over the tree.
    """

    diag_dir: Path
    name: str
    n_abs: int
    n_ord: int
    ord_first: bool
    scale: float | None
    t_scale: float
    abs_dim: str
    ord_dim: str
    x_axis: np.ndarray
    ord_axis: np.ndarray
    attrs: dict[str, Any]


def _histogram2d_meshes(files: list[Path]) -> Iterator[tuple[int, Any, Any]]:
    """Yield ``(iteration, group, mesh record)`` across all files, in order."""
    for path in files:
        with h5py.File(path, "r") as f:
            meshes_path = _decode(f.attrs.get("meshesPath", "meshes/")).strip("/")
            for it, grp in _iteration_groups(f):
                meshes = grp.get(meshes_path)
                if meshes is None or not len(meshes.keys()):
                    continue
                mesh_name = "data" if "data" in meshes else next(iter(meshes.keys()))
                yield int(it), grp, meshes[mesh_name]


def _histogram2d_geometry(
    diag_dir: str | Path,
    *,
    units: CodeUnits | None = None,
    deck: dict | None = None,
) -> _Histogram2DGeometry:
    """Derive a ParticleHistogram2D's axes/attrs/scaling from its first dump."""
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        raise FileNotFoundError(f"No openpmd_*.h5 files in {diag_dir}")
    name = diag_dir.name

    fn_ord = _deck_get(deck, f"{name}.histogram_function_ord")
    fn_abs = _deck_get(deck, f"{name}.histogram_function_abs")
    fn_val = _deck_get(deck, f"{name}.value_function")
    species = _deck_species_of(deck, name)

    n_abs_deck = _deck_get(deck, f"{name}.bin_number_abs")
    n_ord_deck = _deck_get(deck, f"{name}.bin_number_ord")
    ranges = {
        ax: (_deck_get(deck, f"{name}.bin_min_{ax}"), _deck_get(deck, f"{name}.bin_max_{ax}")) for ax in ("abs", "ord")
    }

    shape: tuple[int, ...] | None = None
    grid_lo: np.ndarray | None = None
    grid_hi: np.ndarray | None = None
    for _it, _grp, rec in _histogram2d_meshes(files[:1]):
        node = next(iter(_record_components(rec).values()))
        if isinstance(node, h5py.Dataset):
            shape = tuple(int(s) for s in node.shape)
        else:  # constant record component: shape lives in the attrs
            shape = tuple(int(s) for s in np.atleast_1d(node.attrs.get("shape", [1])))
        spacing = np.atleast_1d(rec.attrs.get("gridSpacing", [1.0] * len(shape))).astype(float)
        offset = np.atleast_1d(rec.attrs.get("gridGlobalOffset", [0.0] * len(shape))).astype(float)
        gu = float(rec.attrs.get("gridUnitSI", 1.0))
        grid_lo = offset * gu
        grid_hi = (offset + spacing * np.asarray(shape, dtype=float)) * gu
        break
    if shape is None:
        raise FileNotFoundError(f"No histogram meshes found in {diag_dir}")

    # WarpX writes the dataset as (bin_number_ord, bin_number_abs); confirm
    # against the deck's bin counts when they are available (and transpose if
    # a future layout flips them), then reorder to (abs, ord).
    if len(shape) != 2:
        raise ValueError(f"{diag_dir}: expected a 2-D histogram mesh, got shape {shape}")
    ord_first = True
    if n_abs_deck is not None and n_ord_deck is not None:
        na, no = int(n_abs_deck), int(n_ord_deck)
        if shape == (na, no) and shape != (no, na):
            ord_first = False
    n_ord, n_abs = shape if ord_first else shape[::-1]
    axis_index = {"abs": 1 if ord_first else 0, "ord": 0 if ord_first else 1}

    def _range(ax: str, n: int) -> tuple[float, float]:
        lo, hi = ranges[ax]
        if lo is not None and hi is not None:
            return float(lo), float(hi)
        d = axis_index[ax]
        if grid_lo is not None and d < grid_lo.size:
            return float(grid_lo[d]), float(grid_hi[d])
        return 0.0, float(n)

    abs_lo, abs_hi = _range("abs", n_abs)
    ord_lo, ord_hi = _range("ord", n_ord)

    ord_dim, ord_long, ord_kind = _ordinate_info(str(fn_ord) if fn_ord is not None else None)
    # The abscissa goes through the same classifier (p2_bugfix.md root cause
    # 1: a `uz` abscissa hard-named x1 and divided by x0 produced a fake
    # spatial axis the box crop then blanked). No/unrecognized deck function
    # keeps the legacy spatial-x1 reading (all 1D campaigns).
    abs_dim, abs_long, abs_kind = _ordinate_info(str(fn_abs)) if fn_abs is not None else ("x1", "x_1", "spatial")
    if abs_kind == "other":
        abs_dim, abs_long, abs_kind = "x1", "x_1", "spatial"
    is_count = fn_val is not None and str(fn_val).strip() == "w"
    if fn_val is None:
        is_count = True  # no deck: assume a plain weight histogram

    # bin widths in their native units (m for spatial, m_e c for momentum) —
    # the count -> density normalization divides by the native-unit bin area
    d_abs = (abs_hi - abs_lo) / max(n_abs, 1)
    d_ord = (ord_hi - ord_lo) / max(n_ord, 1)
    q_sign = _species_charge_sign(deck, species)
    x_axis = _edge_style_axis(abs_lo, abs_hi, n_abs)
    axis_kind_units = {
        "spatial": (r"c / \omega_p", "m"),
        "momentum": (r"m_e c", r"m_e c"),
        "log_gamma": ("", ""),
    }
    scale: float | None = None
    t_scale = 1.0
    if units is not None:
        denom = d_abs * units.n0 * d_ord
        scale = (q_sign if is_count else 1.0) / denom if denom > 0 else 1.0
        t_scale = units.wp0
        if abs_kind == "spatial":
            x_axis = x_axis / units.x0
        x_units = axis_kind_units.get(abs_kind, ("", ""))[0]
        val_units = r"n_0\,\Delta_{bins}^{-1}" if is_count else r"n_0 m_e c^3\,\Delta_{bins}^{-1}"
    else:
        x_units = axis_kind_units.get(abs_kind, ("", ""))[1]
        val_units = "SI (raw per-bin sum)"

    ord_axis = _edge_style_axis(ord_lo, ord_hi, n_ord)
    ord_units = {"momentum": r"m_e c", "log_gamma": ""}.get(ord_kind, "")

    attrs: dict[str, Any] = {
        "long_name": name,
        "units": val_units,
        "time_units": r"1/\omega_p" if units is not None else "s",
        "axis_units": {abs_dim: x_units, ord_dim: ord_units},
        "axis_long_names": {abs_dim: abs_long, ord_dim: ord_long},
        "source_dir": str(diag_dir),
        "warpx_species": species,
        "sim.NDIMS": 1,  # geometry-aware override from _box_attrs below
    }
    for key, fn in (("warpx_function_abs", fn_abs), ("warpx_function_ord", fn_ord), ("warpx_value_function", fn_val)):
        if fn is not None:
            attrs[key] = str(fn)
    attrs.update(_box_attrs(deck, units))

    return _Histogram2DGeometry(
        diag_dir=diag_dir,
        name=name,
        n_abs=int(n_abs),
        n_ord=int(n_ord),
        ord_first=ord_first,
        scale=scale,
        t_scale=t_scale,
        abs_dim=abs_dim,
        ord_dim=ord_dim,
        x_axis=x_axis,
        ord_axis=ord_axis,
        attrs=attrs,
    )


def _histogram2d_records(
    geom: _Histogram2DGeometry, *, require_monotonic: bool = False
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Walk the files **once**, yielding ``(iteration, time, slab)`` per dump.

    The slab comes out ``(abs, ord)``, scaled, in the float32 diagnostic
    dtype — one dump resident at a time (4.1 MB for a 1000x1024 histogram,
    ~12 MB transient while h5py's float64 read is still alive). With
    ``require_monotonic`` an out-of-order iteration raises
    :class:`_NonMonotonicIterations` instead of being silently mis-ordered.
    """
    scale = None if geom.scale is None else np.asarray(geom.scale, dtype=_DIAG_DTYPE)
    last_it = -(1 << 62)
    for it, grp, rec in _histogram2d_meshes(_openpmd_files(geom.diag_dir)):
        if require_monotonic and it <= last_it:
            raise _NonMonotonicIterations(f"{geom.diag_dir}: iteration {it} follows {last_it}")
        last_it = it
        t_si = float(grp.attrs.get("time", 0.0)) * float(grp.attrs.get("timeUnitSI", 1.0))
        arr, _unit_si = _read_component(next(iter(_record_components(rec).values())))
        slab = np.asarray(arr, dtype=_DIAG_DTYPE)
        slab = slab.T if geom.ord_first else slab
        if scale is not None:
            slab = slab * scale
        yield it, t_si * geom.t_scale, np.ascontiguousarray(slab)


def _histogram2d_slab_da(geom: _Histogram2DGeometry, it: int, t: float, slab: np.ndarray) -> xr.DataArray:
    """One dump as a ``(abs, ord)`` DataArray in the StreamWriter's contract."""
    return xr.DataArray(
        slab,
        coords={geom.abs_dim: geom.x_axis, geom.ord_dim: geom.ord_axis},
        dims=(geom.abs_dim, geom.ord_dim),
        name=geom.name,
        attrs={**geom.attrs, "time": float(t), "iter": int(it)},
    )


def load_particle_histogram2d(
    diag_dir: str | Path,
    *,
    units: CodeUnits | None = None,
    deck: dict | None = None,
) -> xr.DataArray:
    """Stack a ParticleHistogram2D history into an OSIRIS-style phase space.

    Returns a ``(t, <abs>, <ord>)`` DataArray in code units (see the section
    comment above for the normalization) — both axes classified from the
    deck's bin functions (``z → x1`` spatial, ``uz/ux → p1/p2`` momentum,
    …), so a ``(uz, ux)`` histogram comes out ``(t, p1, p2)``. The deck (the run's rendered
    ``inputs``) supplies the abscissa/ordinate/value functions, the species
    and the bin ranges; without it the ranges fall back to the openPMD grid
    attrs and the deposit is treated as a count. Without ``units`` the raw
    per-bin SI sums are returned unscaled.

    This materializes the **whole** ``(t, abs, ord)`` cube (47.9 GB for a
    production ``x1log_gamma_q1``) and walks the files twice — once to size
    it, once to fill it. Converters should use
    :func:`convert_histogram2d_streaming`, which writes the same NetCDF in a
    single pass with one slab resident; this eager form is for interactive
    use and for the histories small enough to hold.
    """
    geom = _histogram2d_geometry(diag_dir, units=units, deck=deck)
    # Sizing pass: iteration numbers only, no pixel data (so its cost is set
    # by the dump count, not the slab size).
    index: list[tuple[int, float]] = []
    for it, grp, _rec in _histogram2d_meshes(_openpmd_files(geom.diag_dir)):
        index.append((int(it), float(grp.attrs.get("time", 0.0)) * float(grp.attrs.get("timeUnitSI", 1.0))))
    if not index:
        raise FileNotFoundError(f"No histogram meshes found in {geom.diag_dir}")

    order = np.argsort([it for it, _ in index], kind="stable")
    rows = np.empty(len(index), dtype=np.int64)
    rows[order] = np.arange(len(index))
    t = np.asarray([ts for _, ts in index], dtype="float64")[order] * geom.t_scale
    its = np.asarray([it for it, _ in index], dtype="int64")[order]

    data = np.empty((len(index), geom.n_abs, geom.n_ord), dtype=_DIAG_DTYPE)
    for j, (_it, _t, slab) in enumerate(_histogram2d_records(geom)):
        data[rows[j]] = slab

    coords: dict[str, Any] = {
        "t": t,
        "iter": ("t", its),
        geom.abs_dim: geom.x_axis,
        geom.ord_dim: geom.ord_axis,
    }
    return xr.DataArray(
        data, coords=coords, dims=("t", geom.abs_dim, geom.ord_dim), name=geom.name, attrs=dict(geom.attrs)
    )


def convert_histogram2d_streaming(
    diag_dir: str | Path,
    dest: str | Path,
    *,
    units: CodeUnits | None = None,
    deck: dict | None = None,
    complevel: int = 4,
) -> Path:
    """Convert a ParticleHistogram2D history to NetCDF in a single pass.

    The WarpX twin of :func:`adept.osiris.stream.convert_diagnostic_streaming`
    (and it reuses that module's ``StreamWriter``), so the two engines'
    phase-space conversions have the same memory profile: peak resident is
    one slab plus one write batch, not the whole stacked cube. For the
    production ``x1log_gamma_q1`` (11705 dumps x 1000 x 1024) that is ~30 MB
    against a 47.9 GB cube — and, because
    :func:`load_particle_histogram2d`'s result then went through
    ``series_to_dataset``'s deep copy, ~96 GB of true peak. That was the
    srs-1d-ppc-scan postproc OOM (all 6 postprocs SIGKILLed in a 55 GB
    cgroup, job 57536430).

    Each file is opened **once**: the axes, attrs and scaling all come from
    the first dump (:func:`_histogram2d_geometry`), and the ``t`` dimension
    is unlimited, so nothing needs the history length up front. At the
    2-omega_0-Nyquist field cadences (a dump every 7-9 steps) the per-file
    open cost — ~6.5 ms measured on Perlmutter's Lustre — is what sets the
    conversion wall time, so halving the walks halves the stage.

    ``StreamWriter`` also chunks the variable ``(chunk_t, n_abs, n_ord)``
    with ``chunk_t`` sized to ~1 MiB — 1 dump per chunk at production bin
    counts — so a per-dump read on the result is chunk-aligned. The batch
    path's default h5netcdf chunking (``(183, 16, 32)`` on the production
    cube) spread one time slice over 2016 chunks, the 183x read
    amplification behind the 6.66 h ``collect_srs`` stage.

    Restarts on the eager path if the walk hits an out-of-order iteration
    (see :class:`_NonMonotonicIterations`). Rebuilds ``dest`` from scratch.
    Returns ``dest``.
    """
    from adept.osiris.stream import StreamWriter

    dest = Path(dest)
    geom = _histogram2d_geometry(diag_dir, units=units, deck=deck)

    def _eager() -> Path:
        from adept.osiris.io import series_to_dataset

        ds = series_to_dataset(load_particle_histogram2d(diag_dir, units=units, deck=deck))
        dest.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(
            dest,
            engine="h5netcdf",
            encoding={n: {"zlib": True, "complevel": complevel, "shuffle": True} for n in ds.data_vars},
        )
        return dest

    if dest.exists():
        # Never resume into a partial file: StreamWriter's append-mode resume
        # is for the drainer, which re-reads the same dumps. Here a leftover
        # from a killed postproc would silently truncate the history.
        dest.unlink()

    writer = None
    try:
        for it, t, slab in _histogram2d_records(geom, require_monotonic=True):
            da = _histogram2d_slab_da(geom, it, t, slab)
            if writer is None:
                # The first dump doubles as the schema template.
                writer = StreamWriter(dest, da, source_dir=geom.diag_dir, complevel=complevel)
            writer.append(da)
    except _NonMonotonicIterations:
        if writer is not None:
            writer.close()
            writer = None
        dest.unlink(missing_ok=True)
        return _eager()
    finally:
        if writer is not None:
            writer.close()
    if not dest.exists():  # pragma: no cover - _histogram2d_geometry already raised
        raise FileNotFoundError(f"No histogram meshes found in {geom.diag_dir}")
    return dest


_BIN_HEADER_RE = re.compile(r"^bin(\d+)=(.+)$")


def is_particle_histogram_table(path: str | Path, deck: dict | None = None) -> bool:
    """Whether a reduced text table is a 1-D ParticleHistogram.

    Decided from the deck's ``<name>.type`` when available, else from the
    header column names (``bin<j>=<center>``).
    """
    name = Path(path).stem
    dtype = _deck_get(deck, f"{name}.type")
    if dtype is not None:
        return str(dtype) == "ParticleHistogram"
    try:
        ds = parse_reduced_diag(path)
    except Exception:
        return False
    return any(_BIN_HEADER_RE.match(str(v)) for v in ds.data_vars)


def load_particle_histogram_txt(
    path: str | Path,
    *,
    units: CodeUnits | None = None,
    deck: dict | None = None,
) -> xr.DataArray:
    """Parse a 1-D ParticleHistogram text table into a ``(t, <axis>)`` series.

    The header carries the bin *centers* (``bin<j>=<center>``); the returned
    axis is the OSIRIS edge-style label over the full bin range. With a
    normalization the per-bin weight sums become the charge-signed count
    density ``q/|q| · Σw / (n0 · x0 · d_axis)`` (the whole-box analog of the
    phase-space normalization above); the axis dim is named from the deck's
    ``histogram_function`` (``log10(...)`` → ``gamma``).
    """
    path = Path(path)
    ds = parse_reduced_diag(path)
    name = path.stem

    bins: list[tuple[int, float, str]] = []
    for v in ds.data_vars:
        m = _BIN_HEADER_RE.match(str(v))
        if m:
            bins.append((int(m.group(1)), float(m.group(2)), str(v)))
    if len(bins) < 2:
        raise ValueError(f"{path}: no bin<j>=<center> histogram columns found")
    bins.sort(key=lambda b: b[0])
    centers = np.asarray([b[1] for b in bins], dtype="float64")
    d = float(np.mean(np.diff(centers)))
    axis = _edge_style_axis(centers[0] - d / 2.0, centers[-1] + d / 2.0, centers.size)

    data = np.stack([ds[v].values for _j, _c, v in bins], axis=1).astype("float64")
    t = ds["time"].values.astype("float64")
    its = ds["step"].values.astype("int64")

    fn = _deck_get(deck, f"{name}.histogram_function")
    dim, long_name, _kind = _ordinate_info(str(fn) if fn is not None else "log10")
    species = _deck_species_of(deck, name)
    q_sign = _species_charge_sign(deck, species)
    if units is not None:
        data = data * (q_sign / (units.n0 * units.x0 * d))
        t = t * units.wp0
        val_units = r"n_0 x_0\,\Delta_{bins}^{-1}"
    else:
        val_units = "SI (raw per-bin sum)"

    da = xr.DataArray(
        data.astype(_DIAG_DTYPE),
        coords={"t": t, "iter": ("t", its), dim: axis},
        dims=("t", dim),
        name=name,
        attrs={
            "long_name": name,
            "units": val_units,
            "time_units": r"1/\omega_p" if units is not None else "s",
            "axis_units": {dim: ""},
            "axis_long_names": {dim: long_name},
            "source": str(path),
            "warpx_species": species,
        },
    )
    if fn is not None:
        da.attrs["warpx_histogram_function"] = str(fn)
    return da


def list_scraping_dirs(run_dir: str | Path) -> dict[str, Path]:
    """``{edge: dir}`` for BoundaryScraping outputs (``particles_at_<edge>``)."""
    diags = Path(run_dir) / "diags"
    if not diags.is_dir():
        return {}
    out: dict[str, Path] = {}
    for d in sorted(diags.iterdir()):
        if not d.is_dir():
            continue
        for sub in sorted(d.glob("particles_at_*")):
            if sub.is_dir() and _openpmd_files(sub):
                out[sub.name[len("particles_at_") :]] = sub
    return out


# --- batch conversion (the binary/ contract) --------------------------------


def _laser_wavelength_code(deck: dict | None, units: CodeUnits | None) -> float | None:
    """The first laser's wavelength in code units (c/ω_p), when known."""
    if deck is None or units is None:
        return None
    names = _deck_get(deck, "lasers.names")
    name = (names[0] if names else None) if isinstance(names, list) else names
    if name is None:
        return None
    lam = _deck_get(deck, f"{name}.wavelength")
    try:
        return float(lam) / units.x0
    except (TypeError, ValueError):
        return None


def save_s1_lineouts(
    out_dir: str | Path,
    *,
    deck: dict | None = None,
    units: CodeUnits | None = None,
    guard_cells: int = 2,
    window_wavelengths: float = 2.0,
) -> list[Path]:
    r"""Derive OSIRIS-style ``s1`` boundary lineouts from converted 2D fields.

    OSIRIS 2D SRS decks dump the net longitudinal Poynting flux ``s1`` as
    time-averaged lineouts along ``x2`` just inside the laser entrance and
    exit, and the whole R/T budget chain downstream (``osiris_lpi``'s F5 /
    ``s1_transmission_budget``) keys off those two diagnostics. WarpX has no
    equivalent output, so this synthesizes them from the already-converted
    2D field series in ``out_dir``: ``s1 = e2 b3 − e3 b2`` (code units, in
    which the pump intensity is ``I0 = (a0 ω0)^2 / 2``), averaged over a
    thin ``x1`` slab at each end of the box.

    The slab average replaces the OSIRIS ``tavg``: the dumps are
    instantaneous, and a traveling wave's ``2 k0`` flux oscillation averages
    out over a slab an integer number of half-wavelengths wide. The slab
    spans ``window_wavelengths`` pump wavelengths (wavelength from the deck
    laser when known, else 16 cells), starting ``guard_cells`` inside each
    ``x1`` edge — downstream of the hard-source antenna the SRS decks park
    half a cell off the lo-z wall, so the entrance slab sees incident minus
    reflected flux exactly like the OSIRIS entrance lineout.

    The Yee-staggered components are multiplied positionally (same cell
    index) rather than through coordinate alignment; the half-cell offset is
    irrelevant at the slab/budget level. Writes
    ``FLD/s1-line-x2-0001.nc`` (entrance) and ``FLD/s1-line-x2-0002.nc``
    (exit) — names matched by ``osiris_lpi``'s s1-lineout discovery — and
    returns the written paths (empty when the converted fields are not 2D or
    the transverse pair is missing).
    """
    from adept.osiris.io import series_to_dataset

    out_dir = Path(out_dir)

    def _open(comp: str) -> xr.DataArray | None:
        p = out_dir / "FLD" / f"{comp}.nc"
        if not p.is_file():
            return None
        ds = xr.open_dataset(p, engine="h5netcdf")
        return ds[comp] if comp in ds else None

    e2, b3 = _open("e2"), _open("b3")
    if e2 is None or b3 is None:
        return []
    if not {"t", "x1", "x2"} <= set(map(str, e2.dims)) or e2.ndim != 3:
        return []
    e3, b2 = _open("e3"), _open("b2")
    pairs = [(e2, b3, 1.0)]
    if e3 is not None and b2 is not None and e3.ndim == 3:
        pairs.append((e3, b2, -1.0))

    n_x1 = int(e2.sizes["x1"])
    x1v = np.asarray(e2.coords["x1"].values, dtype=float)
    dx1 = float(x1v[1] - x1v[0]) if n_x1 > 1 else 1.0
    lam = _laser_wavelength_code(deck, units)
    win = round(window_wavelengths * lam / dx1) if lam and dx1 > 0 else 16
    win = max(4, min(win, max(4, n_x1 // 8)))
    guard = max(0, int(guard_cells))
    if 2 * (guard + win) > n_x1:
        return []
    slabs = {
        "0001": slice(guard, guard + win),
        "0002": slice(n_x1 - guard - win, n_x1 - guard),
    }

    nt = min(int(da.sizes["t"]) for pair in pairs for da in pair[:2])
    written: list[Path] = []
    for tag, sl in slabs.items():
        s1 = None
        for e, b, sign in pairs:
            ev = e.isel(t=slice(nt), x1=sl).transpose("t", "x2", "x1").values
            bv = b.isel(t=slice(nt), x1=sl).transpose("t", "x2", "x1").values
            term = sign * np.mean(ev.astype("float64") * bv, axis=-1)
            s1 = term if s1 is None else s1 + term
        ref = e2.isel(t=slice(nt))
        da = xr.DataArray(
            s1.astype(_DIAG_DTYPE),
            coords={
                "t": ref.coords["t"].values,
                "iter": ("t", ref.coords["iter"].values),
                "x2": e2.coords["x2"].values,
            },
            dims=["t", "x2"],
            name="s1",
            attrs={
                "long_name": "s_1",
                "units": r"m_e c^3 n_0" if units is not None else "SI",
                "time_units": e2.attrs.get("time_units", r"1/\omega_p"),
                "axis_units": {"x2": e2.coords["x2"].attrs.get("units", "m" if units is None else r"c / \omega_p")},
                "axis_long_names": {"x2": "x_2"},
                "derived_from": "e2*b3 - e3*b2" if len(pairs) == 2 else "e2*b3",
                "x1_slab_cells": [int(sl.start), int(sl.stop)],
                "x1_slab_window": [float(x1v[sl.start]), float(x1v[sl.stop - 1])],
                "guard_cells": guard,
            },
        )
        rel = f"FLD/s1-line-x2-{tag}"
        dest = out_dir / f"{rel}.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        series_to_dataset(da).to_netcdf(dest, engine="h5netcdf")
        written.append(dest)
    return written


# --- FieldProbe line reports (the F11/F12 high-cadence inputs) ---------------

# FieldProbe table columns -> the OSIRIS component the cyclic (z, x, y) ->
# (1, 2, 3) relabeling assigns them (module docstring).
_PROBE_COMP_TO_OSIRIS = {"Ez": "e1", "Ex": "e2", "Ey": "e3", "Bz": "b1", "Bx": "b2", "By": "b3"}
_PROBE_COL_RE = re.compile(r"part_(x|y|z|Ex|Ey|Ez|Bx|By|Bz|S)_lev0")

# Components each line orientation feeds downstream (osiris_lpi bundle2d):
# boundary lines (along x2) carry the Riemann pairs for the omega
# spectrograms; the axial line (along x1) feeds the (k, omega) maps.
_PROBE_LINE_COMPONENTS = {"x1": ("e1", "e2"), "x2": ("e2", "b3", "e3", "b2")}


def is_field_probe_table(path: str | Path, deck: dict | None = None, name: str | None = None) -> bool:
    """True when a reduced-diag table is FieldProbe output (long format)."""
    if deck is not None and name is not None:
        if str(_deck_get(deck, f"{name}.type")) == "FieldProbe":
            return True
    try:
        with open(path) as fh:
            return "part_x_lev" in fh.readline()
    except OSError:
        return False


def _probe_cell_index(fixed_si: float, axis_fixed: str, deck: dict | None, fallback_d: float) -> int:
    """Grid cell index of a probe line's fixed coordinate.

    ``axis_fixed`` is the WarpX axis (``"x"`` or ``"z"``) held constant along
    the line; the index comes from the deck geometry (2D ``prob_lo/hi`` and
    ``n_cell`` are ``[x, z]``-ordered). Without a deck the cell-size proxy is
    ``fallback_d`` (the probe spacing) — ordering, which is all the
    downstream entrance/exit convention needs, is preserved either way.
    """
    lo = _deck_get(deck, "geometry.prob_lo")
    hi = _deck_get(deck, "geometry.prob_hi")
    nc = _deck_get(deck, "amr.n_cell")
    i = {"x": 0, "z": 1}[axis_fixed]
    try:
        lo_i, hi_i, n_i = float(lo[i]), float(hi[i]), int(nc[i])
        d = (hi_i - lo_i) / n_i
        return int(np.clip(np.floor((fixed_si - lo_i) / d), 0, n_i - 1))
    except (TypeError, ValueError, IndexError):
        d = abs(fallback_d) if fallback_d else 1.0
        return max(0, int(np.floor(fixed_si / d)))


def load_field_probe_lines(
    path: str | Path,
    *,
    units: CodeUnits | None = None,
    deck: dict | None = None,
    components: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, xr.DataArray]:
    r"""Convert one FieldProbe Line table into OSIRIS-style field lineouts.

    FieldProbe writes **one row per probe point per step** (``step, time,
    part_x/y/z, Ex..Bz, |S|`` — sorted by probe id, i.e. along the line), so
    the generic :func:`parse_reduced_diag` one-row-per-step model does not
    apply. This reshapes the long table into ``(t, x2)`` / ``(t, x1)``
    series in code units, named by the ``osiris_lpi`` line-report contract:
    ``FLD/<comp>-line-x2-<cell>`` for a line along the transverse axis at
    fixed x1 cell ``<cell>`` (boundary light monitor — smallest cell =
    entrance, largest = exit) and ``FLD/<comp>-line-x1-<cell>`` for the
    axial line at fixed x2 cell. Orientation is inferred from the probe
    positions themselves (which WarpX axis varies along the line); the cell
    index of the fixed coordinate comes from the deck geometry.

    ``components`` maps line orientation (``"x1"``/``"x2"``) to the OSIRIS
    components to emit (default ``_PROBE_LINE_COMPONENTS`` — the ones the
    bundle reductions read). Returns ``{rel: DataArray}``; empty when the
    table has no complete step blocks.
    """
    path = Path(path)
    with open(path) as fh:
        header_line = fh.readline()
    col_of: dict[str, int] = {}
    for idx, cname, _unit in _COLUMN_RE.findall(header_line):
        m = _PROBE_COL_RE.search(cname)
        if m:
            col_of[m.group(1)] = int(idx)
        elif cname.strip().startswith("step"):
            col_of["step"] = int(idx)
        elif cname.strip().startswith("time"):
            col_of["time"] = int(idx)
    needed = {"step", "time", "x", "z"}
    if not needed <= set(col_of):
        raise ValueError(f"{path.name}: not a FieldProbe table (columns {sorted(col_of)})")

    # loadtxt (C fast path) covers the normal single-header file; a restart
    # can append another header mid-file, which only genfromtxt tolerates.
    try:
        arr = np.loadtxt(path, skiprows=1)
    except ValueError:
        with np.errstate(invalid="ignore"):
            arr = np.genfromtxt(path, skip_header=1, invalid_raise=False)
    if arr.ndim == 1:
        arr = arr[None, :]
    ok = np.isfinite(arr[:, [col_of["step"], col_of["time"], col_of["x"], col_of["z"]]]).all(axis=1)
    arr = arr[ok]
    if arr.shape[0] == 0:
        return {}

    steps = arr[:, col_of["step"]].astype(np.int64)
    bounds = np.flatnonzero(np.diff(steps) != 0) + 1
    blocks = np.split(np.arange(steps.size), bounds)
    # one block per step, last occurrence wins (restart overlap); uniform
    # probe count only — drop incomplete blocks (crash-truncated tail)
    by_step: dict[int, np.ndarray] = {int(steps[b[0]]): b for b in blocks}
    sizes = [b.size for b in by_step.values()]
    counts = np.bincount(sizes)
    n_pts = int(np.flatnonzero(counts == counts.max()).max())  # ties -> larger block
    if n_pts < 2:
        return {}
    dropped = sum(1 for s in sizes if s != n_pts)
    if dropped:
        print(f"[post] {path.name}: dropped {dropped} incomplete probe blocks")
    order = sorted(s for s, b in by_step.items() if b.size == n_pts)
    if not order:
        return {}

    first = arr[by_step[order[0]]]
    px, pz = first[:, col_of["x"]], first[:, col_of["z"]]
    along_z = np.ptp(pz) > np.ptp(px)
    vary_col, fixed_col = (col_of["z"], col_of["x"]) if along_z else (col_of["x"], col_of["z"])
    axis = "x1" if along_z else "x2"  # OSIRIS axis the line runs along
    fixed_axis = "x" if along_z else "z"
    pos = np.sort(first[:, vary_col])
    d_vary = float(np.median(np.diff(pos))) if n_pts > 1 else 1.0
    fixed_si = float(np.median(first[:, fixed_col]))
    cell = _probe_cell_index(fixed_si, fixed_axis, deck, d_vary)

    nt = len(order)
    t_si = np.empty(nt)
    comps = (components or _PROBE_LINE_COMPONENTS)[axis]
    osiris_to_probe = {v: k for k, v in _PROBE_COMP_TO_OSIRIS.items()}
    data = {c: np.empty((nt, n_pts), dtype=_DIAG_DTYPE) for c in comps if osiris_to_probe[c] in col_of}
    for it, s in enumerate(order):
        block = arr[by_step[s]]
        sortix = np.argsort(block[:, vary_col])
        t_si[it] = float(block[0, col_of["time"]])
        for c, out in data.items():
            out[it] = block[sortix, col_of[osiris_to_probe[c]]]
    if not data:
        return {}

    if units is not None:
        t = t_si * units.wp0
        x = pos / units.x0
        scale = {"e": 1.0 / units.E0, "b": 1.0 / units.B0}
        t_units, x_units = r"1/\omega_p", r"c / \omega_p"
        f_units = {"e": r"m_e c \omega_p / e", "b": r"m_e \omega_p / e"}
    else:
        t, x = t_si, pos
        scale = {"e": 1.0, "b": 1.0}
        t_units, x_units = "s", "m"
        f_units = {"e": "V/m", "b": "T"}

    out: dict[str, xr.DataArray] = {}
    for c, vals in data.items():
        da = xr.DataArray(
            (vals * scale[c[0]]).astype(_DIAG_DTYPE),
            coords={"t": t, "iter": ("t", np.asarray(order, dtype=np.int64)), axis: x},
            dims=["t", axis],
            name=c,
            attrs={
                "long_name": f"{c[0]}_{c[1]}",
                "units": f_units[c[0]],
                "time_units": t_units,
                "axis_units": {axis: x_units},
                "axis_long_names": {axis: f"x_{axis[-1]}"},
                "probe_fixed_cell": int(cell),
                "probe_fixed_si": fixed_si,
                "source": path.name,
            },
        )
        out[f"FLD/{c}-line-{axis}-{cell:04d}"] = da
    return out


def _full_diag_dirs(run_dir: Path) -> list[Path]:
    """Full-diagnostic openPMD directories under ``diags/`` (skips reducedfiles)."""
    diags = run_dir / "diags"
    if not diags.is_dir():
        return []
    out = []
    for d in sorted(diags.iterdir()):
        if d.is_dir() and d.name != "reducedfiles" and _openpmd_files(d):
            out.append(d)
    return out


def save_run_datasets(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    units: CodeUnits | None = None,
    diagnostics: list[str] | set[str] | None = None,
) -> list[Path]:
    """Convert a WarpX run's outputs into the OSIRIS ``binary/`` NetCDF tree.

    Walks every full openPMD diagnostic under ``run_dir/diags/`` and every
    reduced-diagnostic text table, writing one NetCDF per diagnostic under
    ``out_dir``:

    - ``FLD/e1.nc`` … (fields, 1D OSIRIS naming), ``DENSITY/<sp>/charge.nc``;
    - ``RAW/<species>.nc`` (long-form particle dumps);
    - ``PHA/<name>/<species>.nc`` (ParticleHistogram2D phase spaces and 1-D
      ParticleHistogram spectra, in the OSIRIS phase-space conventions —
      see the phase-space section above);
    - ``SCRAPED/<species>/<edge>.nc`` (BoundaryScraping buffers, long-form
      with per-particle ``t_scraped``);
    - ``REDUCED/<name>.nc`` (native SI reduced tables);
    - ``HIST/energy.nc`` (the OSIRIS energy-history schema, from
      FieldEnergy + ParticleEnergy when present);
    - ``FLD/s1-line-x2-000{1,2}.nc`` (2D runs: derived entrance/exit
      Poynting-flux lineouts, see :func:`save_s1_lineouts`);
    - ``FLD/<comp>-line-x{1,2}-<cell>.nc`` (FieldProbe Line reduced diags,
      reshaped into OSIRIS-style high-cadence field lineouts — the F11/F12
      inputs; see :func:`load_field_probe_lines`).

    The run's rendered ``inputs`` deck (when present) drives the routing:
    histogram functions, species names, bin ranges and charge signs come
    from it. When several full diagnostics dump the same record, later ones
    get a ``-<diagname>`` suffix on the key. ``diagnostics``, when given,
    whitelists keys (matched on the relative path or any of its
    components). Each diagnostic is best-effort: a failure logs and skips
    rather than aborting the rest. Returns the written paths.
    """
    from adept.osiris.io import series_to_dataset

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    deck = _deck_for_run(run_dir)
    written: list[Path] = []
    taken: set[str] = set()

    def want(rel: str) -> bool:
        if diagnostics is None or rel in diagnostics:
            return True
        return any(part in diagnostics for part in rel.split("/"))

    def write(ds: xr.Dataset, rel: str) -> None:
        dest = out_dir / f"{rel}.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        enc = {name: {"zlib": True, "complevel": 4, "shuffle": True} for name in ds.data_vars}
        ds.to_netcdf(dest, engine="h5netcdf", encoding=enc)
        written.append(dest)

    for diag_dir in _full_diag_dirs(run_dir):
        # All of a diagnostic's records are written from ONE walk of its
        # dumps (convert_field_series_streaming). The keys are known from the
        # first file, so `want` filters before anything is read — the
        # per-record loop this replaces loaded each series and then discarded
        # it if the whitelist said no.
        try:
            meshes_path, plans = _field_plan(diag_dir, units=units)
        except Exception as e:
            print(f"[post] skipping fields from {diag_dir.name}: {e}")
            meshes_path, plans = "fields", []
        targets: list[tuple[str, _FieldRecordPlan]] = []
        for plan in plans:
            rel = plan.rel
            if rel in taken:
                rel = f"{rel}-{diag_dir.name}"
            if not want(rel):
                continue
            targets.append((rel, plan))
        if targets:
            try:
                got = convert_field_series_streaming(diag_dir, targets, out_dir, units=units, meshes_path=meshes_path)
                written += got
                done = {p for p in got}
                taken.update(rel for rel, _p in targets if (out_dir / f"{rel}.nc") in done)
            except Exception as e:
                print(f"[post] skipping fields from {diag_dir.name}: {e}")
        for species in list_species(diag_dir):
            rel = f"RAW/{species}"
            if rel in taken:
                rel = f"{rel}-{diag_dir.name}"
            if not want(rel):
                continue
            try:
                ds = load_particle_species(diag_dir, species, units=units)
                write(ds, rel)
                taken.add(rel)
            except Exception as e:
                print(f"[post] skipping {rel} from {diag_dir.name}: {e}")

    # ParticleHistogram2D phase spaces (openPMD dirs under reducedfiles/).
    # Streamed a dump at a time (see convert_histogram2d_streaming): these
    # are the only diagnostics whose stacked cube is too big to hold, and
    # the streaming writer also gives them t-aligned chunking.
    for name, diag_dir in list_histogram2d_diags(run_dir).items():
        rel = f"PHA/{name}/{_deck_species_of(deck, name)}"
        if not want(rel):
            continue
        dest = out_dir / f"{rel}.nc"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            convert_histogram2d_streaming(diag_dir, dest, units=units, deck=deck)
            written.append(dest)
            taken.add(rel)
        except Exception as e:
            print(f"[post] skipping {rel}: {e}")

    # BoundaryScraping buffers (particles_at_<edge> dirs inside full diags).
    for edge, sub in list_scraping_dirs(run_dir).items():
        for species in list_species(sub):
            rel = f"SCRAPED/{species}/{edge}"
            if not want(rel):
                continue
            try:
                ds = load_particle_species(sub, species, units=units)
                write(ds, rel)
                taken.add(rel)
            except Exception as e:
                print(f"[post] skipping {rel}: {e}")

    # Reduced text tables: 1-D ParticleHistograms become PHA spectra, the
    # rest stay native SI tables under REDUCED/. ParticleHistogram2D leaves
    # an empty companion .txt next to its openPMD dir (observed at dev
    # 72280884a) — its data was converted above, so the stub is skipped.
    h2d_names = set(list_histogram2d_diags(run_dir))
    for name, path in list_reduced_diags(run_dir).items():
        if name in h2d_names or str(_deck_get(deck, f"{name}.type")) == "ParticleHistogram2D":
            continue
        # FieldProbe Line tables become OSIRIS-style FLD/<comp>-line-… series
        # (long format; the generic REDUCED/ parser would mis-stack them).
        if is_field_probe_table(path, deck=deck, name=name):
            try:
                for rel, da in load_field_probe_lines(path, units=units, deck=deck).items():
                    if rel in taken:
                        rel = f"{rel}-{name}"
                    if not want(rel):
                        continue
                    write(series_to_dataset(da), rel)
                    taken.add(rel)
            except Exception as e:
                print(f"[post] skipping FieldProbe lines from {name}: {e}")
            continue
        if is_particle_histogram_table(path, deck):
            rel = f"PHA/{name}/{_deck_species_of(deck, name)}"
            if not want(rel):
                continue
            try:
                da = load_particle_histogram_txt(path, units=units, deck=deck)
                write(series_to_dataset(da), rel)
                taken.add(rel)
            except Exception as e:
                print(f"[post] skipping {rel}: {e}")
            continue
        rel = f"REDUCED/{name}"
        if not want(rel):
            continue
        try:
            write(parse_reduced_diag(path), rel)
        except Exception as e:
            print(f"[post] skipping {rel}: {e}")

    try:
        energy = hist_energy_from_reduced(run_dir, units=units)
        if energy is not None and want("HIST/energy"):
            write(energy, "HIST/energy")
    except Exception as e:
        print(f"[post] skipping HIST energy: {e}")

    # Derived s1 entrance/exit lineouts (2D runs only; see save_s1_lineouts).
    if want("FLD/s1-line-x2-0001") or want("FLD/s1-line-x2-0002"):
        try:
            written += save_s1_lineouts(out_dir, deck=deck, units=units)
        except Exception as e:
            print(f"[post] skipping derived s1 lineouts: {e}")

    return written
