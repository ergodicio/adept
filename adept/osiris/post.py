"""Post-processing for OSIRIS runs.

After ``BaseOsiris.__call__`` finishes, ``collect`` walks the ``MS/``
output tree, copies the final-step HDF5 file from each diagnostic to the
adept temp dir (so MLflow uploads it), optionally tarballs the whole
tree, and returns scalar metrics.
"""

from __future__ import annotations

import re
import shutil
import tarfile
from pathlib import Path
from typing import Any


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


def _walk_diag_dirs(ms: Path) -> list[Path]:
    """Return every directory under ``ms`` that contains ``.h5`` files."""
    out: list[Path] = []
    for p in ms.rglob("*"):
        if p.is_dir() and any(c.suffix == ".h5" for c in p.iterdir()):
            out.append(p)
    return out


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


def _diagnostic_name(diag_dir: Path) -> str:
    """Short tag for a diagnostic directory, used in metric / artifact names."""
    return diag_dir.name


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

    Side effects: copies the final-step HDF5 file for each diagnostic into
    ``td``; copies ``stdout.log`` / ``stderr.log``; copies the rendered
    ``os-stdin``; optionally writes ``ms.tar.gz``.
    Returns ``{"metrics": {...}}`` for adept to log to MLflow.
    """
    solver = run_output["solver result"]
    run_dir: Path = Path(solver["run_dir"])
    ms = run_dir / "MS"
    td = Path(td)
    upload_all = bool(cfg.get("output", {}).get("upload_all", False))
    whitelist = cfg.get("output", {}).get("diagnostics_to_log") or None

    metrics: dict[str, float] = {
        "wall_time_s": float(solver["wall_time"]),
        "exit_code": float(solver["exit_code"]),
        "field_energy_final": _total_field_energy(ms),
        "final_iter": float(_final_iter(ms)),
    }

    # Copy final-step HDF5 per diagnostic.
    if ms.is_dir():
        for diag_dir in _walk_diag_dirs(ms):
            tag = _diagnostic_name(diag_dir)
            if whitelist is not None and tag not in whitelist:
                continue
            last = _latest_h5(diag_dir)
            if last is None:
                continue
            rel = diag_dir.relative_to(ms)
            dest_dir = td / "final_step" / rel
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(last, dest_dir / last.name)

    # Always include the rendered deck + stdout/stderr.
    for fname in ("os-stdin", "stdout.log", "stderr.log"):
        src = run_dir / fname
        if src.exists():
            shutil.copy(src, td / fname)

    # Optional full archive.
    if upload_all and ms.is_dir():
        archive = td / "ms.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(ms, arcname="MS")

    return {"metrics": metrics}
