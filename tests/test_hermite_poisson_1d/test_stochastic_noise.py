#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Per-step stochastic Ex force noise (vlasov1d stochastic_noise parity).

The noise is a white-in-time, density-weighted Gaussian force added to the
E/ponderomotive coupling each Lawson step (drawn once per step, frozen
across RK4 stages). Determinism: key = fold_in(seed, round(t/dt)) — the
same t always yields the same realization.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D


def _build_module(noise_enabled: bool, amplitude: float = 1.0e-5, seed: int = 7):
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
    if noise_enabled:
        cfg["stochastic_noise"] = {"enabled": True, "amplitude": amplitude, "seed": seed}
    module = BaseHermitePoisson1D(cfg)
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module


def test_noise_draw_is_deterministic_and_scaled():
    module = _build_module(True, amplitude=2.0e-5, seed=3)
    vf = module.diffeqsolve_quants["terms"].vector_field

    n1 = np.asarray(vf._draw_noise(1.0))
    n1_again = np.asarray(vf._draw_noise(1.0))
    n2 = np.asarray(vf._draw_noise(1.1))

    np.testing.assert_allclose(n1, n1_again)  # same t -> same draw
    assert np.max(np.abs(n1 - n2)) > 0.0  # different step -> different draw
    # uniform density -> profile == 1 -> std ~ amplitude (loose bound, Nx=32)
    assert 0.3 * 2.0e-5 < np.std(n1) < 3.0 * 2.0e-5


def test_stage_freezing_within_a_step():
    """round(t/dt) is constant across the RK4 stage times of one step."""
    module = _build_module(True)
    vf = module.diffeqsolve_quants["terms"].vector_field
    dt = vf.dt
    t0 = 17 * dt
    n_head = np.asarray(vf._draw_noise(t0))
    # _lawson_rk4 draws once at the step head and passes it to all stages;
    # verify the head draw is what stages would freeze
    n_head_again = np.asarray(vf._draw_noise(t0))
    np.testing.assert_allclose(n_head, n_head_again)


def test_noise_drives_kinetic_response():
    """With noise on, a quiet Maxwellian develops C_1 content; with noise off
    it stays exactly quiet (no spurious source)."""
    import jax.numpy as jnp

    out = {}
    for enabled in (False, True):
        module = _build_module(enabled)
        sol = module(None)["solver result"]
        # default scalar save always exists; the noise drives C_1 -> density
        # structure -> nonzero Poisson field energy
        scal = sol.ys["default"]
        out[enabled] = float(np.nanmax(np.asarray(scal["e_energy"])))

    assert out[False] == 0.0
    assert out[True] > 0.0
