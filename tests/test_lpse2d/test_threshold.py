"""Absolute-threshold bisection driver (LPSE AbsoluteThreshold), with a fake runner."""

import numpy as np
import pytest


def test_bisection_converges_on_a_known_threshold():
    from adept._lpse2d.threshold import find_threshold

    true_threshold = 3.0e14

    def fake_runner(cfg, intensity, run_name):
        rate = 2.0 * (intensity / true_threshold - 1.0)
        return {"epw_growth_measurable": 1.0, "epw_growth_rate_per_ps": rate}

    cfg = {"units": {"laser intensity": "1e15W/cm^2"}, "mlflow": {"run": "t"}}
    result = find_threshold(cfg, "1e14W/cm^2", "1e15W/cm^2", n_iter=10, runner=fake_runner, log_mlflow=False)
    lo, hi = result["bracket"]
    assert lo < true_threshold < hi
    assert abs(result["threshold"] - true_threshold) / true_threshold < 0.01
    assert len(result["history"]) == 12  # two bracket runs + 10 bisections
    assert all(h[1] == (h[0] > true_threshold) for h in result["history"])


def test_bad_brackets_are_rejected():
    from adept._lpse2d.threshold import find_threshold

    def always(cfg, intensity, name):
        return {"epw_growth_measurable": 1.0, "epw_growth_rate_per_ps": 5.0}

    def never(cfg, intensity, name):
        return {"epw_growth_measurable": 0.0, "epw_growth_rate_per_ps": 0.0}

    cfg = {"units": {"laser intensity": "1e15W/cm^2"}}
    with pytest.raises(ValueError, match="already unstable"):
        find_threshold(cfg, 1e14, 1e15, n_iter=1, runner=always, log_mlflow=False)
    with pytest.raises(ValueError, match="still stable"):
        find_threshold(cfg, 1e14, 1e15, n_iter=1, runner=never, log_mlflow=False)
    assert np.isfinite(1.0)
