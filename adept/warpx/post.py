"""Post-processing for WarpX runs.

After ``BaseWarpX.__call__`` finishes, ``collect``:

- converts the run's openPMD + reduced-diagnostic output into the OSIRIS
  ``binary/<diag>.nc`` NetCDF contract under the adept temp dir (code
  units, fixed by the manifest reference density — see
  :mod:`adept.warpx.io`), so MLflow uploads combined xarray datasets
  rather than raw dumps;
- renders the canned plot set into ``td/plots`` (:mod:`adept.warpx.plots`);
- copies the provenance files (rendered ``inputs``, ``warpx_used_inputs``,
  stdout/stderr, the reduced ``.txt`` tables);
- returns scalar metrics, including the final/planned step cross-check —
  WarpX exits 0 on several early-termination paths (``break_signals``,
  silently-ignored typo'd params), so ``final_step`` vs the plan is the
  cheap tripwire the M0 verification called for.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from adept.warpx import io as _io

# WarpX per-step progress lines look like "STEP 100 ends. TIME = ..." — the
# last one gives the final completed step even if the run died early.
_STEP_RE = re.compile(r"^STEP (\d+) ends", re.MULTILINE)


def _final_step(stdout_log: Path) -> int:
    if not stdout_log.is_file():
        return -1
    best = -1
    for m in _STEP_RE.finditer(stdout_log.read_text(errors="replace")):
        best = max(best, int(m.group(1)))
    return best


def _planned_steps(cfg: dict) -> int | None:
    """Steps the deck asked for: ``max_step``, else the CFL-based estimate.

    Both are derived by ``BaseWarpX.get_derived_quantities``. A run
    terminated by ``stop_time`` completes at the *estimate* only
    approximately (WarpX picks dt internally), so callers comparing against
    ``num_steps_est`` should leave slack.
    """
    derived = cfg.get("derived") or {}
    if derived.get("num_steps") is not None:
        return int(derived["num_steps"])
    if derived.get("num_steps_est") is not None:
        return int(derived["num_steps_est"])
    return None


def collect(run_output: dict, cfg: dict, td: str) -> dict[str, Any]:
    """Post-process a finished WarpX run; returns ``{"metrics": {...}}``."""
    solver = run_output["solver result"]
    run_dir = Path(solver["run_dir"])
    td = Path(td)

    metrics: dict[str, float] = {
        "wall_time_s": float(solver["wall_time"]),
        "exit_code": float(solver["exit_code"]),
        "final_step": float(_final_step(run_dir / "stdout.log")),
    }

    # Exit-0 early-termination tripwire: compare completed steps to the plan.
    planned = _planned_steps(cfg)
    if planned:
        frac = max(metrics["final_step"], 0.0) / planned
        metrics["completed_steps_frac"] = frac
        # stop_time-terminated runs land near (not at) the estimate; only a
        # clearly short run is suspicious.
        if frac < 0.9 and (cfg.get("derived") or {}).get("num_steps") is not None:
            print(
                f"[post] WARNING: run completed step {metrics['final_step']:.0f} of {planned} "
                "(exit-0 early termination? check break_signals / warnings in stdout.log)"
            )

    units = _io.CodeUnits.from_cfg(cfg)
    if units is None:
        print("[post] no reference density derived — datasets stay in SI units")

    # Convert openPMD + reduced output into the binary/ NetCDF contract.
    whitelist = (cfg.get("output") or {}).get("diagnostics_to_log") or None
    try:
        _io.save_run_datasets(run_dir, td / "binary", units=units, diagnostics=whitelist)
    except Exception as e:
        print(f"[post] dataset conversion failed: {e}")

    # Energy metrics off the (step-cadence) reduced diagnostics: final field
    # energy split and the conservation drift. Code energy units when the
    # normalization is fixed, else native J/m^2 (1D).
    try:
        e_scale = units.u_area0 if units is not None else 1.0
        # Per-table best-effort: ParticleHistogram2D leaves an empty companion
        # .txt that must not abort the energy scalars.
        tables = {}
        for name, p in _io.list_reduced_diags(run_dir).items():
            try:
                tables[name] = _io.parse_reduced_diag(p)
            except Exception:
                continue
        fld = next((ds for ds in tables.values() if "total_lev0" in ds.data_vars), None)
        if fld is not None:
            metrics["field_energy_final"] = float(fld["total_lev0"].values[-1]) / e_scale
            if "E_lev0" in fld:
                metrics["efield_energy_final"] = float(fld["E_lev0"].values[-1]) / e_scale
            if "B_lev0" in fld:
                metrics["bfield_energy_final"] = float(fld["B_lev0"].values[-1]) / e_scale
        energy = _io.hist_energy_from_reduced(run_dir, units=units)
        if energy is not None and "total_drift_frac" in energy.attrs:
            metrics["energy_drift_frac"] = float(energy.attrs["total_drift_frac"])
    except Exception as e:
        print(f"[post] reduced-diag metrics unavailable: {e}")

    # Canned plots (never let a plotting failure abort metric logging).
    # plots imports matplotlib; do it lazily to keep `import adept.warpx` light.
    try:
        from adept.warpx import plots as _plots

        kwargs = _plots.canned_plot_kwargs(cfg.get("output"))
        _plots.save_canned_plots(td / "binary", td / "plots", **kwargs)
    except Exception as e:
        print(f"[post] plotting failed: {e}")

    for fname in ("inputs", "warpx_used_inputs", "stdout.log", "stderr.log"):
        src = run_dir / fname
        if src.exists():
            shutil.copy(src, td / fname)

    reduced = run_dir / "diags" / "reducedfiles"
    if reduced.is_dir():
        out = td / "reducedfiles"
        out.mkdir(exist_ok=True)
        for p in sorted(reduced.glob("*.txt")):
            shutil.copy(p, out / p.name)

    return {"metrics": metrics}
