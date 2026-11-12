"""The ω-k FFT assumes uniform time sampling; gapped series must be refused.

A WarpX diagnostic whose ``intervals`` is a union of burst slices converts to
one series with gaps. ``plot_omega_k`` takes ``dt = t[1] - t[0]``, so without
the guard such a series would be transformed as if it were uniformly sampled.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from adept.osiris import plots as oplt


def _series(t: np.ndarray) -> xr.DataArray:
    x = np.linspace(0.0, 20.0, 64, endpoint=False)
    data = np.sin(0.5 * t[:, None] - 1.2 * x[None, :]).astype(np.float32)
    return xr.DataArray(data, dims=("t", "x1"), coords={"t": t, "x1": x})


UNIFORM = np.arange(200) * 0.534
GAPPED = np.concatenate([np.arange(100) * 0.534, 500.0 + np.arange(100) * 0.534])


def test_uniform_series_is_accepted() -> None:
    ser = _series(UNIFORM)
    _, ax = plt.subplots()
    oplt.plot_omega_k(ser, ax=ax)
    fig = oplt.plot_omega_k_figure(ser)
    assert oplt._omega_k_zoom_window(ser, requested=2.0) == 2.0
    plt.close("all")
    assert fig is not None


def test_gapped_series_is_refused() -> None:
    ser = _series(GAPPED)
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="uniformly sampled"):
        oplt.plot_omega_k(ser, ax=ax)
    plt.close("all")
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="one diagnostic per burst"):
        oplt.plot_omega_k_figure(ser)
    assert set(plt.get_fignums()) == before  # no Figure leaked
    assert oplt._omega_k_zoom_window(ser, requested=2.0) is None


def test_float_rounding_in_dump_times_is_tolerated() -> None:
    # Dump times are step * dt in floating point: tiny jitter, still uniform.
    t = np.arange(300) * (3 * 0.178) * (1 + 1e-12 * np.random.default_rng(0).standard_normal(300))
    _, ax = plt.subplots()
    oplt.plot_omega_k(_series(t), ax=ax)
    plt.close("all")
