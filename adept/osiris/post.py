"""Post-processing for OSIRIS runs.

After ``BaseOsiris.__call__`` finishes, ``collect`` walks the ``MS/``
output tree, converts each diagnostic's full time history into an xarray
netCDF file in the adept temp dir (so MLflow uploads it instead of the raw
HDF5 dumps), copies the deck / stdout / stderr, and returns scalar metrics.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from adept.osiris import io as _io

# OSIRIS dump filenames look like ``e1-000600.h5`` or
# ``x1p1-beam_pos-000060.h5``. The iteration number is always a
# zero-padded integer right before ``.h5``.
_ITER_RE = re.compile(r"-(\d+)\.h5$")


def _iter_h5_files(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and p.suffix == ".h5"]


def _latest_h5(d: Path) -> Path | None:
    files = _iter_h5_files(d)
    if not files:
        return None

    def keyfn(p: Path) -> int:
        m = _ITER_RE.search(p.name)
        return int(m.group(1)) if m else -1

    return max(files, key=keyfn)


def _field_energy_from_dump(h5_path: Path) -> float:
    """Integrate ``field^2 * cell_volume`` for a single dump file.

    OSIRIS stores one component per file (e1, e2, ..., b3) so this returns
    the contribution of *that one component*.
    """
    import h5py  # local import keeps adept import light

    with h5py.File(h5_path, "r") as f:
        # The dataset name matches the diagnostic name (e.g. "e1").
        name = h5_path.stem.rsplit("-", 1)[0]
        if name not in f:
            return float("nan")
        arr = f[name][...]
        # Reconstruct cell volume from SIMULATION attrs.
        sim = f["SIMULATION"]
        ndims = int(sim.attrs["NDIMS"][0])
        xmin = sim.attrs["XMIN"]
        xmax = sim.attrs["XMAX"]
        nx = sim.attrs["NX"]
        dvol = 1.0
        for d in range(ndims):
            dvol *= (float(xmax[d]) - float(xmin[d])) / int(nx[d])
        return 0.5 * float((arr.astype("float64") ** 2).sum()) * dvol


def _total_field_energy(ms: Path) -> float:
    """Sum |E|^2/2 and |B|^2/2 across components present in MS/FLD/.

    Returns NaN if no field dumps are present.
    """
    fld = ms / "FLD"
    if not fld.is_dir():
        return float("nan")
    total = 0.0
    found = False
    for comp_dir in fld.iterdir():
        if not comp_dir.is_dir():
            continue
        last = _latest_h5(comp_dir)
        if last is None:
            continue
        e = _field_energy_from_dump(last)
        if e == e:  # not NaN
            total += e
            found = True
    return total if found else float("nan")


def _final_iter(ms: Path) -> int:
    """Largest iteration number seen across all FLD dumps; -1 if none."""
    fld = ms / "FLD"
    if not fld.is_dir():
        return -1
    best = -1
    for comp_dir in fld.iterdir():
        for p in _iter_h5_files(comp_dir):
            m = _ITER_RE.search(p.name)
            if m:
                best = max(best, int(m.group(1)))
    return best


def collect(run_output: dict, cfg: dict, td: str) -> dict[str, Any]:
    """Post-process a finished OSIRIS run.

    Side effects: converts each diagnostic's full time history into a netCDF
    file under ``td/binary/`` (mirroring the ``MS/`` layout) so MLflow gets
    combined xarray datasets rather than the raw HDF5 dumps; copies the
    rendered ``os-stdin`` plus ``stdout.log`` / ``stderr.log``.
    Returns ``{"metrics": {...}}`` for adept to log to MLflow.
    """
    solver = run_output["solver result"]
    run_dir: Path = Path(solver["run_dir"])
    ms = run_dir / "MS"
    td = Path(td)
    whitelist = (cfg.get("output") or {}).get("diagnostics_to_log") or None
    raw_drop_initial = bool((cfg.get("output") or {}).get("raw_drop_initial", False))

    metrics: dict[str, float] = {
        "wall_time_s": float(solver["wall_time"]),
        "exit_code": float(solver["exit_code"]),
        "field_energy_final": _total_field_energy(ms),
        "final_iter": float(_final_iter(ms)),
    }

    # Convert each diagnostic's time history to an xarray netCDF.
    if ms.is_dir():
        _io.save_run_datasets(
            run_dir, td / "binary", diagnostics=whitelist,
            raw_drop_initial=raw_drop_initial,
        )

    # plots imports matplotlib; do it lazily to keep `import adept.osiris` light.
    from adept.osiris import plots as _plots

    # E/B-field split + energy-conservation metrics (best effort).
    try:
        comps = _plots.field_energy_components(run_dir)
        metrics["efield_energy_final"] = float(comps["E_energy"].values[-1])
        metrics["bfield_energy_final"] = float(comps["B_energy"].values[-1])
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[post] field-energy components unavailable: {e}")
    try:
        energy = _io.load_hist_energy(run_dir)
        if energy is not None and "total_drift_frac" in energy.attrs:
            metrics["energy_drift_frac"] = float(energy.attrs["total_drift_frac"])
    except Exception as e:
        print(f"[post] HIST energy unavailable: {e}")

    # Drive amplitude/frequency from the deck → laser energy budget.
    deck = cfg.get("deck") or {}
    a0, omega0 = deck.get("antenna.a0"), deck.get("antenna.omega0")

    # Render the standard plot set into td/plots so MLflow logs them as
    # artifacts. Never let a plotting failure abort metric/artifact logging.
    try:
        kwargs = _plots.canned_plot_kwargs(cfg.get("output"))
        if a0 is not None and omega0 is not None:
            kwargs.update(a0=float(a0), omega0=float(omega0))
        _plots.save_canned_plots(run_dir, td / "plots", **kwargs)
    except Exception as e:
        print(f"[post] plotting failed: {e}")

    # Reflected/transmitted/absorbed laser-energy fractions as scalar metrics.
    if a0 is not None and omega0 is not None:
        try:
            budget = _plots.laser_energy_budget(run_dir, a0=float(a0), omega0=float(omega0))
            metrics["laser_reflectivity"] = budget["R"]
            metrics["laser_transmissivity"] = budget["T"]
            metrics["laser_absorbed_frac"] = budget["absorbed"]
        except Exception as e:
            print(f"[post] laser energy budget unavailable: {e}")

    # Always include the rendered deck + stdout/stderr.
    for fname in ("os-stdin", "stdout.log", "stderr.log"):
        src = run_dir / fname
        if src.exists():
            shutil.copy(src, td / fname)

    return {"metrics": metrics}
