"""Absolute-threshold bisection driver for the envelope-2d solver (LPSE ``AbsoluteThreshold``).

Runs the same configuration at a sequence of pump intensities and bisects on a growth
criterion evaluated from the run's post-processed metrics. Each run is an ordinary
``ergoExo`` run (its own MLflow run); the bisection itself is recorded as the parent run's
parameters and metrics.

Usage::

    from adept._lpse2d.threshold import find_threshold

    result = find_threshold(cfg, intensity_lo="1e14W/cm^2", intensity_hi="1e15W/cm^2", n_iter=6)
    result["threshold"]  # W/cm^2, midpoint of the final bracket

The default criterion is LPSE's: the run is "unstable" when the fitted EPW energy growth
rate exceeds ``growth_min`` (1/ps) and the fit is measurable; ``criterion`` can be any
callable of the metrics dict returning a bool.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np

from adept._lpse2d.helpers import _Q


def _intensity_w_cm2(value) -> float:
    return float(_Q(value).to("W/cm^2").value) if isinstance(value, str) else float(value)


def default_criterion(metrics: dict, growth_min: float = 0.0) -> bool:
    """Unstable when the EPW energy grows: a measurable fit with a positive rate above ``growth_min``."""
    measurable = float(metrics.get("epw_growth_measurable", 0.0)) > 0.5
    rate = float(metrics.get("epw_growth_rate_per_ps", 0.0))
    return measurable and rate > growth_min


def run_at_intensity(cfg: dict, intensity_w_cm2: float, run_name: str | None = None) -> dict:
    """One ergoExo run of ``cfg`` with ``units['laser intensity']`` replaced; returns the metrics."""
    from adept import ergoExo

    run_cfg = deepcopy(cfg)
    run_cfg["units"]["laser intensity"] = f"{intensity_w_cm2:.6g}W/cm^2"
    if run_name is not None:
        run_cfg.setdefault("mlflow", {})["run"] = run_name
    exo = ergoExo()
    modules = exo.setup(run_cfg)
    _, ppo, _ = exo(modules)
    return dict(ppo["metrics"])


def find_threshold(
    cfg: dict,
    intensity_lo,
    intensity_hi,
    n_iter: int = 6,
    criterion: Callable[[dict], bool] | None = None,
    growth_min: float = 0.0,
    runner: Callable[[dict, float, str | None], dict] = run_at_intensity,
    log_mlflow: bool = True,
) -> dict:
    """Bisect the pump intensity between ``intensity_lo`` (expected stable) and ``intensity_hi``
    (expected unstable) with ``n_iter`` midpoint runs. The bracket endpoints are run first and
    must straddle the criterion, mirroring LPSE's ``AbsoluteThreshold`` bracket check.

    Returns ``{"threshold", "bracket", "history": [(intensity, unstable, metrics), ...]}``."""
    crit = criterion or (lambda m: default_criterion(m, growth_min))
    lo, hi = _intensity_w_cm2(intensity_lo), _intensity_w_cm2(intensity_hi)
    if not lo < hi:
        raise ValueError("intensity_lo must be below intensity_hi")
    base_name = cfg.get("mlflow", {}).get("run", "threshold")
    history = []

    def evaluate(intensity: float, tag: str) -> bool:
        metrics = runner(cfg, intensity, f"{base_name}-{tag}")
        unstable = bool(crit(metrics))
        history.append((intensity, unstable, metrics))
        return unstable

    if evaluate(lo, "lo"):
        raise ValueError(f"the lower bracket {lo:.3e} W/cm^2 is already unstable; lower intensity_lo")
    if not evaluate(hi, "hi"):
        raise ValueError(f"the upper bracket {hi:.3e} W/cm^2 is still stable; raise intensity_hi")
    for i in range(n_iter):
        mid = float(np.sqrt(lo * hi))  # geometric midpoint: thresholds scale multiplicatively
        if evaluate(mid, f"iter{i}"):
            hi = mid
        else:
            lo = mid
    result = {"threshold": float(np.sqrt(lo * hi)), "bracket": (lo, hi), "history": history}
    if log_mlflow:
        try:
            import mlflow

            with mlflow.start_run(run_name=f"{base_name}-bisection"):
                mlflow.log_params(
                    {
                        "intensity_lo": _intensity_w_cm2(intensity_lo),
                        "intensity_hi": _intensity_w_cm2(intensity_hi),
                        "n_iter": n_iter,
                    }
                )
                mlflow.log_metrics({"threshold_W_cm2": result["threshold"], "bracket_lo": lo, "bracket_hi": hi})
                for step, (intensity, unstable, _) in enumerate(history):
                    mlflow.log_metrics({"intensity": intensity, "unstable": float(unstable)}, step=step)
        except Exception as exc:  # logging must never break the bisection
            print(f"threshold: MLflow logging skipped ({exc})")
    return result


# ---------------------------------------------------------------------------------------
# LPSE's own search (AbsoluteThreshold.cpp; deck absoluteThreshold.*) -- plan 2 O.8
# ---------------------------------------------------------------------------------------


def gain_criterion(series: dict, gain: float = 23.026, noise_time_range=(0.01, 0.1)) -> bool:
    """LPSE ``AbsoluteThreshold::checkIsAboveThreshold``: the run is above threshold when
    ``log(A_end / A_noise) > gain`` with ``A`` the peak EPW amplitude (``max_phi`` of the
    default series; LPSE's ``max |rho_k|``) and ``A_noise`` the running mean of ``A`` over the
    samples before ``noise_time_range[1]`` (LPSE reads only the upper bound)."""
    t = np.asarray(series["t (ps)"], dtype=np.float64)
    amp = np.asarray(series["max_phi"], dtype=np.float64)
    noise = amp[t < float(noise_time_range[1])]
    if noise.size == 0 or not np.all(np.isfinite(noise)) or noise.mean() <= 0.0:
        raise ValueError("gain_criterion: no finite samples before noise_time_range[1] to set the noise level")
    return bool(np.log(amp[-1] / noise.mean()) > gain)


def run_series_at_intensity(cfg: dict, intensity_w_cm2: float, run_name: str | None = None) -> dict:
    """One ``ergoExo`` run of ``cfg`` at ``intensity_w_cm2``; returns the default series as a
    dict of numpy arrays (``t (ps)``, ``max_phi``, ...)."""
    from adept import ergoExo

    run_cfg = deepcopy(cfg)
    run_cfg["units"]["laser intensity"] = f"{intensity_w_cm2:.6g}W/cm^2"
    if run_name is not None:
        run_cfg.setdefault("mlflow", {})["run"] = run_name
    exo = ergoExo()
    modules = exo.setup(run_cfg)
    _, ppo, _ = exo(modules)
    series = ppo["series"]
    return {k: np.asarray(series[k].values) for k in series.data_vars} | {"t (ps)": np.asarray(series["t (ps)"].values)}


def find_threshold_lpse(
    cfg: dict,
    intensity=None,
    dI_fract: float = 0.3333,
    gain: float = 23.026,
    n_iter: int = 5,
    noise_time_range=(0.01, 0.1),
    runner: Callable[[dict, float, str | None], dict] = run_series_at_intensity,
    max_stage_one_steps: int = 100,
) -> dict:
    """LPSE's ``absoluteThreshold`` search (``AbsoluteThreshold::updateLaserIntensity``): from
    ``intensity`` (default ``units['laser intensity']``) step by ``dI_fract * intensity`` away
    from the first result's side until the gain criterion flips (stage 1), then ``n_iter``
    steps of halving size towards the flip (stage 2). Defaults are the deck defaults
    (``gain`` 23.026 = ln 1e10, ``dI_fract`` 0.3333, ``numIterations`` 5,
    ``noiseTimeRange`` 0.01-0.1 ps). ``cfg['threshold']`` (the translator's block from
    ``absoluteThreshold.*``) overrides the defaults.

    Returns ``{"threshold", "history": [(intensity, above, series), ...]}``."""
    block = cfg.get("threshold") or {}
    dI_fract = float(block.get("dI_fract", dI_fract))
    gain = float(block.get("gain", gain))
    n_iter = int(block.get("n_iter", n_iter))
    noise_time_range = tuple(block.get("noise_time_range", noise_time_range))
    intensity = _intensity_w_cm2(intensity if intensity is not None else cfg["units"]["laser intensity"])
    base_name = cfg.get("mlflow", {}).get("run", "threshold")
    history = []

    def above(value: float, tag: str) -> bool:
        series = runner(cfg, value, f"{base_name}-{tag}")
        result = gain_criterion(series, gain, noise_time_range)
        history.append((value, result, series))
        return result

    step = initial_step = dI_fract * intensity
    first_above = above(intensity, "stage0")
    direction = -1.0 if first_above else 1.0
    intensity += direction * step
    # stage 1: march until the criterion flips
    for n in range(max_stage_one_steps):
        result = above(intensity, f"stage1-{n}")
        if result != first_above:
            break
        intensity += direction * step
        if intensity <= initial_step / 1000.0:
            intensity = step / 2.0
            break
    else:
        raise ValueError("find_threshold_lpse: stage 1 did not bracket the threshold in max_stage_one_steps")
    # stage 2: halving steps back towards the flip (LPSE halves on entering stage 2 too)
    step /= 2.0
    intensity -= direction * step
    for n in range(n_iter):
        result = above(intensity, f"stage2-{n}")
        step /= 2.0
        intensity += (-step) if result else step
        if intensity <= 0.0:
            step /= 2.0
            intensity = step
    return {"threshold": float(intensity), "history": history}
