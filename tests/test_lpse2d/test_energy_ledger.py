"""The per-operation EPW energy ledger (terms.epw.energy_ledger) closes exactly.

LPSE asserts that its EPW energy budget closes to 0.1 % per step
(ZakharovSolver.cpp:1700-1712). Here the closure is exact to round-off by
construction, so the test checks that W(t) - W(0) equals the sum of the channels,
that the sign of each channel is what physics demands, and that the ledger is
reported in the default series in epw_energy units.
"""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _finish(cfg):
    from adept._lpse2d.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def _tpd_cfg(noise=True):
    with open("tests/test_lpse2d/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {"boundary_width": "0.6um", "dt": "1fs", "dx": "0.1um", "tmax": "10fs", "ymax": "1.6um", "ymin": "-1.6um"}
    )
    cfg["density"] = {"basis": "linear", "gradient scale length": "30um", "max": 0.27, "min": 0.23}
    cfg["terms"]["epw"]["source"].update({"noise": noise, "noise_seed": 3, "noise_amplitude": 1e-4, "tpd": True})
    cfg["terms"]["epw"]["energy_ledger"] = True
    cfg["terms"]["epw"]["damping"]["collisions"] = True
    return _finish(cfg)


def _srs_cfg():
    with open("tests/test_lpse2d/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update({"boundary_width": "2um", "xmax": "16um", "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um"})
    cfg["terms"]["light"] = {"pump_depletion": True, "coupling": "rotation"}
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"].update({"noise_seed": 4, "noise_amplitude": 1e-4})
    cfg["terms"]["epw"]["energy_ledger"] = True
    return _finish(cfg)


def _state(cfg, seed):
    from adept._lpse2d.core.epw import LEDGER_CHANNELS, LEDGER_KEY

    rng = np.random.default_rng(seed)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    phi = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * 1e-3
    phi *= np.asarray(cfg["grid"]["low_pass_filter_grid"] * cfg["grid"]["zero_mask"])
    E0 = 0.3 * cfg["units"]["derived"]["E0_source"] * (rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2)))
    state = {
        "epw": jnp.asarray(phi).view(jnp.float64),
        "E0": jnp.asarray(E0).view(jnp.float64),
        "E1": jnp.asarray(0.1 * E0).view(jnp.float64),
        LEDGER_KEY: jnp.zeros(len(LEDGER_CHANNELS)),
    }
    return state


def _pump_args(cfg):
    ny = cfg["grid"]["ny"]
    return {
        **cfg["drivers"]["E0"]["derived"],
        "delta_omega": jnp.zeros(1),
        "intensities": jnp.ones((1, ny)),
        "phases": jnp.zeros((1, ny)),
    }


def _run(cfg, state, n_steps):
    from adept._lpse2d.core.vector_field import SplitStep

    step = SplitStep(cfg)
    args = {"drivers": {"E0": _pump_args(cfg)}}
    dt = cfg["grid"]["dt"]
    for i in range(n_steps):
        state = step(jnp.asarray(i * dt), dict(state), args)
    return step, state


def _energy(step, state):
    return float(step.epw.energy(state["epw"].view(jnp.complex128)))


@pytest.mark.parametrize("make_cfg", [_tpd_cfg, _srs_cfg], ids=["tpd+noise+absorbers", "srs+pump_depletion"])
def test_ledger_closes_to_round_off(make_cfg):
    from adept._lpse2d.core.epw import LEDGER_CHANNELS

    cfg = make_cfg()
    state0 = _state(cfg, 1)
    from adept._lpse2d.core.vector_field import SplitStep

    w0 = float(SplitStep(cfg).epw.energy(state0["epw"].view(jnp.complex128)))
    step, state = _run(cfg, state0, 30)
    ledger = dict(zip(LEDGER_CHANNELS, np.asarray(state["epw_ledger"]), strict=False))
    w_end = _energy(step, state)
    total_flow = sum(abs(v) for v in ledger.values()) + abs(w0) + abs(w_end)
    assert abs(w_end - w0 - sum(ledger.values())) < 1e-11 * total_flow, ledger

    assert ledger["dispersion"] == pytest.approx(0.0, abs=1e-12 * total_flow)
    assert ledger["detuning"] == pytest.approx(0.0, abs=1e-12 * total_flow)
    assert ledger["damping"] < 0.0
    assert ledger["boundary"] < 0.0
    assert ledger["dealias"] <= 0.0 and ledger["reprojection"] <= 1e-12 * total_flow
    # the noise channel is 2 Re<phi, kick> + |kick|^2 per step: only its long-run
    # average is positive, its sign over 30 steps is random
    assert ledger["noise"] != 0.0
    assert ledger["driver"] == 0.0
    if cfg["terms"]["epw"]["source"]["tpd"]:
        assert ledger["tpd"] != 0.0 and ledger["srs"] == 0.0
    else:
        assert ledger["srs"] != 0.0 and ledger["tpd"] == 0.0


def test_ledger_reported_in_epw_energy_units():
    from adept._lpse2d.core.epw import LEDGER_CHANNELS
    from adept._lpse2d.helpers import get_default_save_func, get_save_quantities

    cfg = _tpd_cfg()
    cfg = get_save_quantities(cfg)
    save = get_default_save_func(cfg)["func"]
    step, state = _run(cfg, _state(cfg, 2), 10)
    out = save(0.0, state, None)
    for name in LEDGER_CHANNELS:
        assert f"epw_ledger_{name}" in out
    total = sum(float(out[f"epw_ledger_{name}"]) for name in LEDGER_CHANNELS)
    w0 = float(save(0.0, _state(cfg, 2), None)["epw_energy"])
    assert float(out["epw_energy"]) - w0 == pytest.approx(total, rel=1e-9)
    assert float(out["epw_ledger_closure"]) == pytest.approx(w0, rel=1e-9)
    # the fields save function must not try to interpolate the ledger
    fields = cfg["save"]["fields"]["func"](0.0, state, None)
    assert "epw_ledger" not in fields


def test_ledger_off_leaves_state_and_series_unchanged():
    cfg = _tpd_cfg()
    cfg["terms"]["epw"]["energy_ledger"] = False
    from adept._lpse2d.core.vector_field import SplitStep
    from adept._lpse2d.helpers import get_default_save_func, get_save_quantities

    state = _state(cfg, 3)
    del state["epw_ledger"]
    step = SplitStep(cfg)
    out = step(jnp.asarray(0.0), dict(state), {"drivers": {"E0": _pump_args(cfg)}})
    assert "epw_ledger" not in out
    cfg = get_save_quantities(cfg)
    series = get_default_save_func(cfg)["func"](0.0, out, None)
    assert not any(k.startswith("epw_ledger") for k in series)
