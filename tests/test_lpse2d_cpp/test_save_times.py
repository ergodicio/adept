"""Every lpse2d save falls on a solver step time n * grid.dt.

Between steps the solver's dense output is the linear interpolation (1 - s) y_n + s y_{n+1} of the
state, so a save time off the step grid scales the |phi|^2 of an envelope mode rotating by theta per
step by 1 - 2 s (1 - s)(1 - cos theta). The old default series axis linspace(0, tmax, nt) put sample k
at the fractional step k / (nt - 1), and the old field-save axis linspace(tmin, tmax, ...) was off the
step grid unless (tmax - tmin) / dt was a whole number.

Tolerances (fixed before the tests were first run):

* save times: rtol 1e-12 against n * grid.dt (the solver accumulates t in float64 over <= 100 steps);
* a free single EPW mode (no drivers, no damping, periodic box) keeps |phi_k|^2, hence e_sq, at every
  step to round-off: rtol 1e-10 over 76 steps of FFT pairs (~1e-13 expected);
* the check is sensitive only if the mode rotates: 1 - cos(theta) >= 0.1, measured from the saved
  k-fields, so the old series axis would have shrunk e_sq by >= 5 % at mid-run.
"""

import math

import numpy as np
import pytest
import yaml

GRID_DT_FS = 6.0
TMAX_FS = 450.0  # nt = int(450 / 6 + 1) = 76: the old series spacing was 456 / 75 = 6.08 fs
MODE_INDEX = 40  # a kx index inside the low-pass band; it sets the rotation per step
SAVE_TIME_RTOL = 1.0e-12
CONSERVATION_RTOL = 1.0e-10
MIN_ONE_MINUS_COS = 0.1


def _epw_cfg() -> dict:
    with open("tests/test_lpse2d_cpp/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["drivers"] = {}
    cfg["grid"].update({"ymax": "0.02um", "ymin": "-0.02um", "dt": f"{GRID_DT_FS}fs", "tmax": f"{TMAX_FS}fs"})
    cfg["save"]["fields"]["t"] = {"tmin": "0fs", "tmax": f"{TMAX_FS}fs", "dt": f"{GRID_DT_FS}fs"}
    cfg["mlflow"]["run"] = "save-times-free-mode"
    return cfg


def _prepared(cfg: dict) -> dict:
    """The setup steps up to the save functions (no MLflow, no run)."""
    from adept._lpse2d_cpp.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_save_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return get_save_quantities(cfg)


def test_step_save_times_round_the_cadence_to_whole_steps():
    from adept._lpse2d_cpp.helpers import step_save_times

    dt = 0.006
    # 15 fs = 2.5 steps rounds half away from zero, as LPSE's roundf: every 3 steps from the first
    # step at or after 10 fs (12 fs) to the last step at or before 100 fs (96 fs)
    t, stride = step_save_times(0.010, 0.100, 0.015, dt)
    assert stride == 3
    np.testing.assert_allclose(t, np.arange(2, 17, 3) * dt, rtol=SAVE_TIME_RTOL)
    # a cadence below one step saves every step
    t, stride = step_save_times(0.0, 0.030, 0.001, dt)
    assert stride == 1
    np.testing.assert_allclose(t, np.arange(6) * dt, rtol=SAVE_TIME_RTOL)
    with pytest.raises(ValueError, match="no solver step"):
        step_save_times(0.0071, 0.0119, 0.006, dt)


def test_save_axes_are_on_the_step_grid():
    cfg = _epw_cfg()
    cfg["save"]["fields"]["t"] = {"tmin": "10fs", "tmax": f"{TMAX_FS}fs", "dt": "15fs"}
    cfg = _prepared(cfg)
    dt, nt = cfg["grid"]["dt"], cfg["grid"]["nt"]
    np.testing.assert_allclose(cfg["save"]["default"]["t"]["ax"], np.arange(nt + 1) * dt, rtol=SAVE_TIME_RTOL)
    fields_t = np.asarray(cfg["save"]["fields"]["t"]["ax"])
    np.testing.assert_allclose(fields_t, (2 + 3 * np.arange(fields_t.size)) * dt, rtol=SAVE_TIME_RTOL)
    assert fields_t[-1] <= TMAX_FS * 1e-3 < fields_t[-1] + 3 * dt
    assert cfg["save"]["fields"]["t"]["dt"] == pytest.approx(3 * dt, rel=SAVE_TIME_RTOL)


def test_free_epw_mode_series_is_conserved_at_every_save():
    """A single freely rotating EPW mode through ergoExo: the default series and the field saves are
    the solver's step states, so e_sq and |phi_k|^2 stay constant."""
    import jax.numpy as jnp

    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(_epw_cfg())
    module = exo.adept_module
    cfg = module.cfg
    nx, ny, dt, nt = cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["dt"], cfg["grid"]["nt"]
    phi_k = np.zeros((nx, ny), dtype=np.complex128)
    phi_k[MODE_INDEX, 0] = 1.0e-6 * nx * ny
    module.state = module.state | {"epw": jnp.asarray(phi_k).view(jnp.float64)}

    result = exo(modules)[0]["solver result"]
    t_series = np.asarray(result.ts["default"])
    np.testing.assert_allclose(t_series, np.arange(nt + 1) * dt, rtol=SAVE_TIME_RTOL)
    t_fields = np.asarray(result.ts["fields"])
    np.testing.assert_allclose(t_fields, np.arange(t_fields.size) * dt, rtol=SAVE_TIME_RTOL)

    e_sq = np.asarray(result.ys["default"]["e_sq"])
    np.testing.assert_allclose(e_sq / e_sq[0], 1.0, rtol=CONSERVATION_RTOL)

    mode = np.asarray(result.ys["fields"]["epw"]).view(np.complex128)[:, MODE_INDEX, 0]
    np.testing.assert_allclose(np.abs(mode) ** 2 / np.abs(mode[0]) ** 2, 1.0, rtol=CONSERVATION_RTOL)
    theta = np.angle(mode[2] / mode[1])  # the phase advance over one step
    assert 1.0 - math.cos(theta) >= MIN_ONE_MINUS_COS, f"theta = {theta:.3f} rad per step: the check is insensitive"
