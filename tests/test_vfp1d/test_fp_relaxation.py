#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""Spherical Fokker-Planck relaxation tests (FastVFP)."""

from functools import partial

import pytest
from fp_relaxation.problems import (
    bump_on_tail,
    maxwellian,
    monoenergetic_beam,
    shifted_maxwellian,
    supergaussian,
    two_temperature,
)
from fp_relaxation.runner import problem_name, run_relaxation_sweep_and_assert

MODELS = ["FastVFP"]
EXPERIMENT = "vfp1d-fokker-planck-relaxation-tests"
VMAX = 6.0
NV = 128
TEMPERATURE_TOL = 5e-2

PROBLEMS = [
    {"ic_fn": partial(maxwellian, T=1.0), "extra_checks": "rmse", "equilibrium": True},
    {"ic_fn": partial(supergaussian, m=5, T=1.0)},
    {"ic_fn": partial(two_temperature, T_cold=0.5, T_hot=2.0, frac_cold=0.7)},
    {"ic_fn": partial(bump_on_tail, narrow=True)},
    {"ic_fn": partial(bump_on_tail, narrow=False)},
    {"ic_fn": partial(shifted_maxwellian, v_shift=1.8, T=0.162)},
    {"ic_fn": monoenergetic_beam},
]

SLOW_EXTRA_COMBOS = [
    {"sc_iterations": 0, "dt_over_tau": 1.0},
    {"sc_iterations": 1, "dt_over_tau": 1.0},
    {"sc_iterations": 2, "dt_over_tau": 0.1},
    {"sc_iterations": 2, "dt_over_tau": 10.0},
]


@pytest.mark.parametrize(
    "slow",
    [
        pytest.param(False, id="fast"),
        pytest.param(True, id="slow", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: problem_name(p["ic_fn"]))
def test_fp_relaxation(problem, slow):
    run_relaxation_sweep_and_assert(
        "spherical", MODELS, EXPERIMENT, problem, slow, SLOW_EXTRA_COMBOS, NV, VMAX, TEMPERATURE_TOL
    )
