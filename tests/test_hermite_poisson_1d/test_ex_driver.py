#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""External longitudinal (Ex) field driver for Hermite-Poisson 1D.

Parity with adept._vlasov1d's LongitudinalElectricFieldDriver: a prescribed Ex
is added to the velocity-space force (Poisson field + ponderomotive), driving the
plasma directly. The field follows the vlasov1d amplitude convention

    E_drive(x, t) = Σ_pulses env(x,t) · (w0+dw0) · a0 · sin(k0 x − (w0+dw0) t),

reads the flat cfg["drivers"]["ex"] pulse dict (same format as the ey driver), and
is exposed as the "de" diagnostic field.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D
from adept._hermite_poisson_1d.vector_field import LongitudinalElectricFieldDriver


def test_driver_field_formula():
    """Single pulse with effectively-unity envelopes reproduces the analytic
    (w0+dw0)·a0·sin(k0 x − (w0+dw0) t) field exactly."""
    Lx, Nx = 10.0, 64
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    # Huge widths (default 1e10) -> envelopes are 1.0 to float64 precision in the
    # interior; finite rise avoids a 0/0 in get_envelope.
    pulse = {"k0": 0.7, "w0": 1.3, "dw0": 0.1, "a0": 0.05, "t_rise": 1.0, "x_rise": 1.0}
    driver = LongitudinalElectricFieldDriver(x, {"0": pulse})

    t = 2.0
    got = np.asarray(driver(t, {}))
    w_total = 1.3 + 0.1
    expected = w_total * 0.05 * np.sin(0.7 * np.asarray(x) - w_total * t)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_driver_sums_multiple_pulses():
    """The driver is the linear superposition of its pulses."""
    Lx, Nx = 8.0, 48
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    p0 = {"k0": 0.6, "w0": 1.0, "a0": 0.03, "t_rise": 1.0, "x_rise": 1.0}
    p1 = {"k0": -0.6, "w0": 1.2, "a0": 0.02, "t_rise": 1.0, "x_rise": 1.0}

    both = LongitudinalElectricFieldDriver(x, {"0": p0, "1": p1})
    only0 = LongitudinalElectricFieldDriver(x, {"0": p0})
    only1 = LongitudinalElectricFieldDriver(x, {"1": p1})

    t = 1.5
    np.testing.assert_allclose(np.asarray(both(t, {})), np.asarray(only0(t, {})) + np.asarray(only1(t, {})), atol=1e-12)


def test_no_ex_pulses_returns_zeros():
    """An empty (or missing) ex config yields an identically-zero field."""
    Lx, Nx = 10.0, 32
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    for cfg in ({}, None):
        driver = LongitudinalElectricFieldDriver(x, cfg)
        out = np.asarray(driver(3.0, {}))
        assert out.shape == (Nx,)
        assert np.max(np.abs(out)) == 0.0


def _build_module(ex_enabled: bool, a0: float = 1.0e-3):
    cfg = {
        "physics": {
            "Lx": 10.0,
            "alpha_e": 1.0,
            "alpha_i": 1.0,
            "nu": 0.0,
            "static_ions": True,
            "c_light": 1.0,
        },
        "grid": {"Nn": 16, "Ni": 2, "Nx": 32, "tmax": 5.0, "dt": 0.1},
        "drivers": {},
        "save": {},
    }
    if ex_enabled:
        cfg["drivers"]["ex"] = {
            "0": {
                "k0": 2.0 * np.pi / 10.0,  # mode 1 on the Lx=10 grid
                "w0": 1.0,
                "a0": a0,
                "t_rise": 1.0,
                "t_width": 1.0e10,
            }
        }
    module = BaseHermitePoisson1D(cfg)
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module


def test_ex_driver_drives_kinetic_response():
    """With the Ex driver on, a quiet Maxwellian develops density/field structure
    (nonzero Poisson energy); with it off the run stays exactly quiet."""
    out = {}
    for enabled in (False, True):
        module = _build_module(enabled)
        sol = module(None)["solver result"]
        scal = sol.ys["default"]
        out[enabled] = float(np.nanmax(np.asarray(scal["e_energy"])))

    assert out[False] == 0.0
    assert out[True] > 0.0


def test_de_diagnostic_matches_driver():
    """The 'de' state field records the driver evaluated at the step start."""
    module = _build_module(True)
    vf = module.diffeqsolve_quants["terms"].vector_field
    y0 = module.state

    t0 = 7 * vf.dt
    new_state = vf(t0, y0, {})
    expected = np.asarray(vf.ex_driver(t0, {}))

    assert "de" in new_state
    np.testing.assert_allclose(np.asarray(new_state["de"]), expected, atol=1e-12)
    assert np.max(np.abs(expected)) > 0.0  # driver actually firing at t0
