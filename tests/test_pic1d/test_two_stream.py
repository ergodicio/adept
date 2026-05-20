"""Two-stream cold-beam instability — integrated physics test.

We initialize two counter-streaming, nearly-cold electron populations of equal
density. The fastest-growing mode of the cold dispersion relation,

    1 = (ωp²/2) / (ω - k v0)² + (ωp²/2) / (ω + k v0)²,

satisfies ``k_max v0 = ωp * sqrt(3/8) ≈ 0.6124 ωp`` and grows at
``γ_max = ωp/2``. We size the box so this mode is the fundamental and seed it
with a tiny sinusoidal density perturbation (via a small displacement of the
quietly-loaded particles), then fit a log-linear growth rate to the m=1
Fourier amplitude during the linear phase.
"""

from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
import pytest

from adept.pic1d import BasePIC1D

V_BEAM = 1.0
WP = 1.0  # plasma frequency in code units (ωp = 1)
# Cold counter-streaming beams of equal density n0/2 each:
#   1 = (ωp²/2)/(ω - kv0)² + (ωp²/2)/(ω + kv0)²
# → instability for k v0 < ωp, peak γ at k v0 = ωp √(3/8), γ_max = ωp/(2√2).
K_MAX = WP * np.sqrt(3.0 / 8.0)  # ≈ 0.6124
GAMMA_THEORY = WP / (2.0 * np.sqrt(2.0))  # ≈ 0.3536


def _two_stream_cfg(ppc: int, nx: int, tmax: float, dt: float, shape: str) -> dict:
    L = 2 * np.pi / K_MAX
    return {
        "units": {"normalizing_temperature": "1eV", "normalizing_density": "1e21/cc"},
        "density": {
            "quasineutrality": True,
            "species-beam-pos": {
                "noise_seed": 1, "noise_type": "gaussian", "noise_val": 0.0,
                "v0": V_BEAM, "T0": 1e-4, "m": 2.0,
                "basis": "uniform", "baseline": 0.5,
            },
            "species-beam-neg": {
                "noise_seed": 2, "noise_type": "gaussian", "noise_val": 0.0,
                "v0": -V_BEAM, "T0": 1e-4, "m": 2.0,
                "basis": "uniform", "baseline": 0.5,
            },
        },
        "grid": {
            "dt": dt, "nx": nx, "tmin": 0.0, "tmax": tmax,
            "xmin": 0.0, "xmax": L, "ppc": ppc,
            "particle_shape": shape,
        },
        "save": {"fields": {"t": {"tmin": 0.0, "tmax": tmax, "nt": int(tmax / dt) + 1}}},
        "solver": "pic-1d",
        "mlflow": {"experiment": "pic1d-tests", "run": "two-stream"},
        "drivers": {"ex": {}, "ey": {}},
        "diagnostics": {},
        "terms": {
            "field": "poisson",
            "time": "leapfrog",
            "species": [
                {"name": "beam_pos", "charge": -1.0, "mass": 1.0,
                 "density_components": ["species-beam-pos"],
                 "loading": "quiet", "vmax_load": 4.0},
                {"name": "beam_neg", "charge": -1.0, "mass": 1.0,
                 "density_components": ["species-beam-neg"],
                 "loading": "quiet", "vmax_load": 4.0},
            ],
        },
    }


def _seed_perturbation(state: dict, amplitude: float, L: float) -> dict:
    """Apply a sinusoidal x-displacement to seed the k_max mode in both beams.

    A displacement ``δx = (A/k) sin(k x)`` produces a density perturbation
    ``δn/n0 = -A cos(k x)`` at leading order. We perturb both beams identically
    so the total electron density wave (and hence E) is seeded coherently at
    ``k = k_max``.
    """
    out = dict(state)
    for name in ("beam_pos", "beam_neg"):
        key = f"x_{name}"
        x = state[key]
        x_new = x + (amplitude / K_MAX) * jnp.sin(K_MAX * x)
        out[key] = jnp.mod(x_new, L)
    return out


@pytest.mark.parametrize("shape", ["tsc", "cubic"])
def test_two_stream_growth_rate(shape):
    tmax = 18.0
    dt = 0.05
    cfg = _two_stream_cfg(ppc=128, nx=32, tmax=tmax, dt=dt, shape=shape)
    L = cfg["grid"]["xmax"]

    m = BasePIC1D(copy.deepcopy(cfg))
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.state = _seed_perturbation(m.state, amplitude=0.01, L=L)
    m.init_diffeqsolve()
    sol = m({})["solver result"]

    ts = np.asarray(sol.ts["fields"])
    E_xt = np.asarray(sol.ys["fields"]["e"])
    A1 = np.abs(np.fft.fft(E_xt, axis=1)[:, 1])

    # The seeded density perturbation excites both stable and unstable modes.
    # The stable Langmuir-like component oscillates and decays in the first
    # few plasma periods; the unstable component then dominates and grows
    # exponentially up to particle-trapping saturation around |E1| ~ 10. We
    # fit in the regime where the unstable branch is clearly dominant.
    in_window = (A1 > 0.5) & (A1 < 6.0)
    assert in_window.sum() >= 30, (
        f"shape={shape}: fit window has only {int(in_window.sum())} samples "
        f"(A1 range over sim: [{A1.min():.2e}, {A1.max():.2e}])"
    )

    gamma = float(np.polyfit(ts[in_window], np.log(A1[in_window]), 1)[0])
    rel_err = abs(gamma - GAMMA_THEORY) / GAMMA_THEORY
    assert rel_err < 0.10, (
        f"shape={shape}: measured γ={gamma:.4f}, theory γ={GAMMA_THEORY:.4f}, "
        f"rel_err={rel_err:.2%}"
    )
