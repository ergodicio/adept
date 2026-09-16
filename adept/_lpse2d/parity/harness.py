"""Run a translated LPSE deck with adept and compare it with the LPSE reference run.

This is the 2026-09-14 cross-check tooling (``original-lpse/runs/tools/{run_adept_deck,
log_lpse_reference, compare_m1, compare_m5}.py``) made importable, with every machine path
an argument. The comparison metrics are computed exactly as ``log_lpse_reference.py`` did so
values logged from here are comparable with the M1–M5 baseline on MLflow ``lpse-parity``:

- ``<field>_energy_growth_{adept,lpse}_<lo>_<hi>ps`` — slope of ``log(energy)`` over the
  window (least squares, samples with ``energy > 0``; ``nan`` with fewer than four samples);
- ``<field>_energy_growth_ratio_<lo>_<hi>ps`` — adept / LPSE;
- ``epw_half_max_time_{adept,lpse}_ps`` — first time the EPW energy exceeds half its maximum;
- ``iaw_max_{adept,lpse}`` — when both codes track the IAW density.

``run_deck`` goes through ``ergoExo`` (so the run is logged to MLflow like any other) and
returns the post-processed datasets; ``compare`` and ``log_reference`` are pure functions of
files and ids and never need the solver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adept._lpse2d.lpse_deck import read_metrics

logger = logging.getLogger(__name__)

# Fit windows (ps) used for the 09-14 baseline, from ``runs/tools/rerun_all_server.sh``.
DEFAULT_WINDOWS: dict[str, tuple[tuple[float, float], ...]] = {
    "hom_srs_020": ((0.8, 1.2), (1.0, 1.4)),
    "hom_tpd_023": ((0.5, 0.8), (0.9, 1.2)),
    "test_006": ((2.5, 3.5), (3.0, 4.0)),
    "test_025": ((0.2, 0.6),),
    "test_029": ((0.2, 0.6), (0.6, 1.0)),
    "test_010": ((0.3, 0.9), (0.9, 1.5)),
    "test_032": ((0.5, 1.5), (1.5, 2.5)),
    "test_013": ((0.8, 1.4), (1.6, 2.2)),
    "test_022": ((0.8, 1.4), (1.6, 2.2)),
    "test_036": ((0.1, 0.3),),
    "test_037": ((0.3, 0.8), (0.9, 1.4)),
}
# Config overrides the 09-14 reruns applied per deck (``rerun_all_server.sh``).
DEFAULT_OVERRIDES: dict[str, dict] = {"test_036": {"terms": {"hpe": {"v_min": 0.05}}}}

# (LPSE metrics column, adept series variable, metric prefix)
ENERGY_PAIRS = (("EPW_energy", "epw_energy", "epw"), ("E1_energy", "e1_sq", "raman"))
IAW_PAIR = ("Nelf_max", "iaw_density_abs_max")
TIME_KEY = "t (ps)"


def parse_windows(spec: str) -> tuple[tuple[float, float], ...]:
    """``"0.3-0.9,0.9-1.5"`` -> ``((0.3, 0.9), (0.9, 1.5))``."""
    out = []
    for w in spec.split(","):
        lo, hi = (float(v) for v in w.split("-"))
        out.append((lo, hi))
    return tuple(out)


def window_tag(lo: float, hi: float) -> str:
    return f"{lo:g}_{hi:g}ps"


def growth_rate(t: np.ndarray, energy: np.ndarray, lo: float, hi: float) -> float:
    """Exponential growth rate (1/ps) of ``energy`` over ``lo <= t <= hi``: the slope of a
    linear fit to ``log(energy)`` on samples with positive energy; ``nan`` below four."""
    t = np.asarray(t, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    m = (t >= lo) & (t <= hi) & (energy > 0)
    if m.sum() <= 3:
        return float("nan")
    return float(np.polyfit(t[m], np.log(energy[m]), 1)[0])


def _series_dict(series) -> dict[str, np.ndarray]:
    """An xarray Dataset, a mapping of arrays, or a path to a ``series.xr``/``series.nc``."""
    if isinstance(series, (str, Path)):
        import xarray as xr

        series = xr.open_dataset(series, engine="h5netcdf")
    if hasattr(series, "data_vars"):
        out = {k: np.asarray(series[k].values) for k in series.data_vars}
        out[TIME_KEY] = np.asarray(series[TIME_KEY].values)  # the series' time coordinate
        return out
    return {k: np.asarray(v) for k, v in dict(series).items()}


def _metrics_dict(lpse_metrics) -> dict[str, np.ndarray]:
    if isinstance(lpse_metrics, (str, Path)):
        return read_metrics(lpse_metrics)
    return {k: np.asarray(v) for k, v in dict(lpse_metrics).items()}


def compare(series, lpse_metrics, windows) -> dict[str, float]:
    """Growth-rate and timing comparison of an adept series against an LPSE metrics table.

    ``series``: the post-processed adept series (Dataset, mapping, or file path) with
    ``t (ps)``, ``epw_energy`` and optionally ``e1_sq`` / ``iaw_density_abs_max``;
    ``lpse_metrics``: ``lpse.metrics`` (path or ``read_metrics`` dict); ``windows``: fit
    windows in ps as ``((lo, hi), ...)`` or ``"lo-hi,lo-hi"``. Non-finite values are kept
    (as ``nan``) so the caller decides what to log."""
    s = _series_dict(series)
    d = _metrics_dict(lpse_metrics)
    if isinstance(windows, str):
        windows = parse_windows(windows)
    t, tl = s[TIME_KEY], d["time"]
    pairs = [p for p in ENERGY_PAIRS if p[0] in d and p[1] in s]
    metrics: dict[str, float] = {}
    for lo, hi in windows:
        tag = window_tag(lo, hi)
        for lcol, acol, name in pairs:
            ga = growth_rate(t, s[acol], lo, hi)
            gl = growth_rate(tl, d[lcol], lo, hi)
            metrics[f"{name}_energy_growth_adept_{tag}"] = ga
            metrics[f"{name}_energy_growth_lpse_{tag}"] = gl
            if np.isfinite(ga) and np.isfinite(gl) and gl != 0:
                metrics[f"{name}_energy_growth_ratio_{tag}"] = ga / gl
    epw, el = s["epw_energy"], d["EPW_energy"]
    metrics["epw_half_max_time_adept_ps"] = float(t[np.argmax(epw > 0.5 * epw.max())])
    metrics["epw_half_max_time_lpse_ps"] = float(tl[np.argmax(el > 0.5 * el.max())])
    if IAW_PAIR[0] in d and IAW_PAIR[1] in s:
        metrics["iaw_max_adept"] = float(np.max(s[IAW_PAIR[1]]))
        metrics["iaw_max_lpse"] = float(np.max(d[IAW_PAIR[0]]))
    return metrics


def energy_floor_crossings(series, lpse_metrics, floor_window=(0.3, 1.0), factors=(10, 1000)) -> dict[str, float]:
    """Times at which the EPW energy first exceeds ``factor`` x its noise-floor median
    (``compare_m1.py``'s second table)."""
    s, d = _series_dict(series), _metrics_dict(lpse_metrics)
    t, tl = s[TIME_KEY], d["time"]
    epw, el = s["epw_energy"], d["EPW_energy"]
    lo, hi = floor_window
    fa = np.median(epw[(t > lo) & (t < hi)])
    fl = np.median(el[(tl > lo) & (tl < hi)])
    out = {}
    for f in factors:
        out[f"epw_{f}x_floor_time_adept_ps"] = float(t[np.argmax(epw > f * fa)])
        out[f"epw_{f}x_floor_time_lpse_ps"] = float(tl[np.argmax(el > f * fl)])
    return out


def read_flux(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """LPSE ``lpse.flux``: ``(time, flux[nt, n_metric, 6])`` in W/cm^2, faces ordered
    ``+x -x +y -y +z -z``."""
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    table = np.asarray(rows, dtype=np.float64)
    t = table[:, 0]
    n_metric = int(table[0, 1])
    return t, table[:, 2:].reshape(len(t), n_metric, 6)


LPSE_FLUX_FACE = {"+x": 0, "-x": 1, "+y": 2, "-y": 3, "+z": 4, "-z": 5}
ADEPT_WALL_FACE = {"left": "-x", "right": "+x", "bottom": "-y", "top": "+y"}
KEV_TO_ERG = 1.602e-9
_NC_PER_UM2 = 1.1148e21  # critical density (1/cm^3) times wavelength^2 (um^2)


def hpe_wall_power(series, cfg: dict, window: tuple[float, float]) -> dict[tuple[str, int], float]:
    """adept HPE wall power (W/cm^2) per ``(wall, bin)`` from the cumulative
    ``hpe_wall_energy_<wall>_bin<i>`` series (keV per real electron): the slope over
    ``window`` times the electrons per cm of depth (uniform box, ``density.max``), per unit
    wall length. Mirrors ``compare_m5.py`` (which fitted ``0.12 < t <= 0.30`` ps)."""
    s = _series_dict(series)
    t = s[TIME_KEY]
    wavelength_um = float(str(cfg["units"]["laser_wavelength"]).replace("um", ""))
    n_e = float(cfg["density"]["max"]) * _NC_PER_UM2 / wavelength_um**2
    lx = float(str(cfg["grid"]["xmax"]).replace("um", "")) * 1e-4
    ly = 2.0 * float(str(cfg["grid"]["ymax"]).replace("um", "")) * 1e-4
    n_total = n_e * lx * ly
    lengths = {"left": ly, "right": ly, "bottom": lx, "top": lx}
    lo, hi = window
    m = (t > lo) & (t <= hi)
    bins = cfg["terms"]["hpe"]["flux_bins"]
    out = {}
    for i in range(len(bins) - 1):
        for wall, length in lengths.items():
            key = f"hpe_wall_energy_{wall}_bin{i}"
            if key not in s:
                continue
            rate = np.polyfit(t[m], s[key][m], 1)[0]  # keV * fraction / ps
            out[(wall, i)] = float(rate * n_total * KEV_TO_ERG * 1e-7 * 1e12 / length)
    return out


def compare_hpe_flux(series, cfg: dict, flux_path: str | Path, window: tuple[float, float]) -> dict[str, float]:
    """adept / LPSE wall-flux ratios per ``(wall, bin)`` as ``hpe_flux_ratio_<wall>_bin<i>``,
    with the two absolute values alongside (``compare_m5.py``)."""
    _, flux = read_flux(flux_path)
    adept_power = hpe_wall_power(series, cfg, window)
    out = {}
    for (wall, i), power in adept_power.items():
        if i >= flux.shape[1]:
            continue
        lpse = float(np.mean(flux[:, i, LPSE_FLUX_FACE[ADEPT_WALL_FACE[wall]]]))
        out[f"hpe_flux_adept_{wall}_bin{i}"] = power
        out[f"hpe_flux_lpse_{wall}_bin{i}"] = lpse
        if lpse:
            out[f"hpe_flux_ratio_{wall}_bin{i}"] = power / lpse
    return out


def merge_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Recursive in-place update of ``cfg`` by ``overrides``; returns ``cfg``."""
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merge_overrides(cfg[k], v)
        else:
            cfg[k] = v
    return cfg


def translate_deck(
    parms: str | Path, overrides: dict | None = None, run: str | None = None, experiment: str = "lpse-parity"
):
    """Translate an ``lpse.parms`` deck and apply ``overrides``; returns ``(cfg, report)``."""
    from adept._lpse2d.lpse_deck import parse_parms, translate_parms

    parms = Path(parms)
    run = run or parms.resolve().parent.name
    cfg, report = translate_parms(parse_parms(parms), experiment=experiment, run=run)
    return merge_overrides(cfg, overrides), report


@dataclass
class DeckRun:
    """What ``run_deck`` hands back: the config actually run, the translator report, the
    MLflow run id and the post-processing datasets."""

    deck: str
    cfg: dict
    report: dict
    run_id: str | None
    series: object
    fields: object
    metrics: dict
    out_dir: Path | None = None
    extra: dict = field(default_factory=dict)


def run_deck(
    parms: str | Path,
    overrides: dict | None = None,
    out_dir: str | Path | None = None,
    run: str | None = None,
    experiment: str = "lpse-parity",
) -> DeckRun:
    """Run a translated deck through ``ergoExo``; optionally save ``config.yaml``,
    ``series.nc``, ``fields.nc`` and ``metrics.npy`` under ``out_dir``."""
    import yaml

    from adept import ergoExo

    cfg, report = translate_deck(parms, overrides, run=run, experiment=experiment)
    deck = cfg["mlflow"]["run"]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as fo:
            yaml.safe_dump(cfg, fo, sort_keys=False)
    exo = ergoExo()
    modules = exo.setup(cfg)
    _, ppo, run_id = exo(modules)
    if out_dir is not None:
        ppo["series"].to_netcdf(out_dir / "series.nc", engine="h5netcdf", invalid_netcdf=True)
        ppo["x"].to_netcdf(out_dir / "fields.nc", engine="h5netcdf", invalid_netcdf=True)
        np.save(out_dir / "metrics.npy", ppo["metrics"], allow_pickle=True)
    return DeckRun(deck, cfg, report, run_id, ppo["series"], ppo["x"], ppo["metrics"], out_dir)


REFERENCE_FILES = ("data/lpse.metrics", "lpse.parms", "laser_include.txt")


def log_reference(
    run_id: str, lpse_dir: str | Path, metrics: dict[str, float], deck: str | None = None, full: bool = False
):
    """Attach the LPSE reference to an adept MLflow run: the comparison ``metrics`` (finite
    values only), the tags ``lpse_reference_deck`` / ``comparison``, and the reference files
    under ``lpse_reference/`` — ``lpse.metrics``, ``lpse.parms`` and ``laser_include.txt`` by
    default, the whole ``lpse_dir`` tree with ``full`` (tag ``lpse_reference_complete``)."""
    from adept import patched_mlflow as mlflow

    lpse_dir = Path(lpse_dir)
    deck = deck or lpse_dir.name
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
        mlflow.set_tags({"lpse_reference_deck": deck, "comparison": "adept-vs-lpse"})
        if full:
            mlflow.log_artifacts(str(lpse_dir), artifact_path="lpse_reference")
            mlflow.set_tag("lpse_reference_complete", "true")
        else:
            for rel in REFERENCE_FILES:
                path = lpse_dir / rel
                if path.is_file():
                    mlflow.log_artifact(str(path), artifact_path="lpse_reference")
    logger.info(
        "logged to %s: %s", run_id, {k: round(v, 3) for k, v in metrics.items() if "ratio" in k or "half_max" in k}
    )
