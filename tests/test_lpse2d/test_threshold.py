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


def test_lpse_search_converges_with_the_gain_criterion():
    """LPSE's stage-1 march and stage-2 halving with the ln(A_end / A_noise) > gain criterion,
    on a fake runner whose peak amplitude grows as exp(g(I) t)."""
    from adept._lpse2d.threshold import find_threshold_lpse, gain_criterion

    i_0, gain, t_end = 3.0e14, 23.026, 2.0

    def fake_runner(cfg, intensity, run_name):
        t = np.linspace(0.0, t_end, 401)
        # exp(g t) from 0.1 ps with g = (gain / t_end) I / I_0: the gain is reached at t_end
        # for I = I_0 t_end / (t_end - 0.1)
        rate = (gain / t_end) * (intensity / i_0)
        return {"t (ps)": t, "max_phi": 1e-10 * np.exp(rate * np.where(t > 0.1, t - 0.1, 0.0))}

    true_threshold = i_0 * t_end / (t_end - 0.1)
    cfg = {"units": {"laser intensity": "1e15W/cm^2"}, "mlflow": {"run": "t"}}
    result = find_threshold_lpse(cfg, n_iter=8, runner=fake_runner)
    assert abs(result["threshold"] - true_threshold) / true_threshold < 0.02
    assert all(h[1] == gain_criterion(h[2], gain, (0.01, 0.1)) for h in result["history"])
    # a start below the threshold marches upward first
    cfg["units"]["laser intensity"] = "1e14W/cm^2"
    result = find_threshold_lpse(cfg, n_iter=8, runner=fake_runner)
    assert result["history"][1][0] > result["history"][0][0]
    assert abs(result["threshold"] - true_threshold) / true_threshold < 0.02
    # the translator's block overrides the defaults
    cfg["threshold"] = {"gain": gain, "dI_fract": 0.5, "n_iter": 8, "noise_time_range": [0.01, 0.1]}
    result = find_threshold_lpse(cfg, runner=fake_runner)
    assert abs(result["history"][1][0] - 1.5e14) < 1.0
