#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Structured split-integrator gates for issue #343."""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pytest

from adept._hermite_legendre_1d.modules import BaseHermiteLegendre1D
from adept._hermite_legendre_1d.storage import get_save_quantities


def _run_split(tmax: float = 2.0, dt: float = 0.1):
    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": 4.0 * np.pi,
            "alpha": np.sqrt(2.0),
            "u": 0.0,
            "v_a": -2.5,
            "v_b": 2.5,
            "gamma": 0.5,
            "nu_H": 0.0,
            "nu_L": 0.2,
            "enforce_conservation": True,
            "field": True,
        },
        "grid": {
            "Nx": 16,
            "Nh": 12,
            "Nl": 16,
            "tmax": tmax,
            "dt": dt,
            "integrator": "split",
            "split_field_iters": 3,
        },
        "initialization": {"type": "two-stream", "eps": 0.02, "mode": 1},
        "save": {"default": {"t": {"nt": round(tmax / dt) + 1}}},
        "units": {},
    }
    module = BaseHermiteLegendre1D(cfg)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module(trainable_modules={})["solver result"].ys["default"]


def test_split_step_is_finite_conserving_and_reports_diagnostics():
    data = _run_split()

    for name in (
        "energy",
        "boundary_df_a_max",
        "boundary_df_b_max",
        "boundary_df_max",
        "high_legendre_fraction",
        "step_residual",
        "conservation_correction",
    ):
        assert np.all(np.isfinite(np.asarray(data[name]))), f"{name} went non-finite"

    for name in ("mass", "momentum", "energy"):
        values = np.asarray(data[name])
        scale = max(abs(values[0]), 1.0)
        assert np.max(np.abs(values - values[0])) / scale < 2e-11, f"{name} was not corrected"

    assert np.all(np.asarray(data["boundary_df_max"]) >= 0.0)
    high_fraction = np.asarray(data["high_legendre_fraction"])
    assert np.all((0.0 <= high_fraction) & (high_fraction <= 1.0))
    assert np.max(np.asarray(data["step_residual"])) < 1.0e-2
    assert np.max(np.asarray(data["conservation_correction"])) > 0.0


def test_default_diagnostics_are_saved_at_every_step():
    cfg = {
        "grid": {"nt": 3, "tmax": 0.3},
        "physics": {"alpha": np.sqrt(2.0), "u": 0.0, "v_a": 2.0, "v_b": 18.0, "Lx": 2.0 * np.pi},
        "save": {},
    }
    axis = get_save_quantities(cfg)["save"]["default"]["t"]["ax"]
    np.testing.assert_allclose(axis, [0.0, 0.1, 0.2, 0.3])


def test_retired_implicit_integrator_is_rejected():
    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": 2.0 * np.pi,
            "alpha": np.sqrt(2.0),
            "v_a": -2.5,
            "v_b": 2.5,
        },
        "grid": {"Nx": 8, "Nh": 6, "Nl": 6, "tmax": 0.1, "dt": 0.1, "integrator": "implicit"},
        "initialization": {"type": "linear-advection"},
        "save": {},
        "units": {},
    }
    module = BaseHermiteLegendre1D(cfg)
    module.write_units()
    module.get_derived_quantities()
    module.get_solver_quantities()

    with pytest.raises(ValueError, match="supported integrators are 'split', 'lawson', and 'imex'"):
        module.init_diffeqsolve()
