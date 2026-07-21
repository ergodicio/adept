#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Radiation-amplitude calibration tests for TransverseWaveDriver.

Regression context: the extended source originally injected
S = −w²·a0·env·sin(kx−wt) with NO calibration, so the launched wave had
amplitude a0·(w·G/2c) with G = ∫env dx — a factor 9.0 (81× intensity) for the
production SRS geometry (1 µm antenna, w=1). Every HP campaign before
2026-07-04 was pumped at ~81× nominal intensity; the pump measured at the
probes was 9.00× nominal at all four campaign intensities (the ep = Ey + Bz
probe convention contributes a further ×2, giving the observed 18.00).
The vlasov1d reference used its point source, which divides the Green's
factor out explicitly (F0 = 2·w·c·a0) and was unaffected.

These tests launch each source into vacuum with the production WaveSolver and
assert the radiated |Ey| equals w·a0 — the check that was always missing.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.vector_field import TransverseWaveDriver
from adept._vlasov1d.solvers.pushers.field import WaveSolver


def _radiated_amplitude(pulse: dict, Lx=200.0, Nx=800, dt=0.1, t_end=350.0, x_probe=140.0):
    """Launch one pulse into vacuum; return radiated |Ey| at x_probe.

    Ey = -(a - a_old)/dt; amplitude from the last third of the run (steady state:
    source at x=30, probe transit 110/c plus margin).
    """
    dx = Lx / Nx
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    x_a = jnp.concatenate([jnp.array([x[0] - dx]), x, jnp.array([x[-1] + dx])])

    driver = TransverseWaveDriver(x_a, {"0": pulse}, c=1.0)
    wave = WaveSolver(c=1.0, dx=dx, dt=dt)
    n_e = jnp.zeros(Nx)  # vacuum

    @jax.jit
    def step(carry, t):
        a, prev_a = carry
        res = wave(a=a, aold=prev_a, djy_array=driver(t, {}), electron_density=n_e)
        return (res["a"], res["prev_a"]), (res["a"] - a)[1:-1]

    n_steps = int(t_end / dt)
    a0_arr = jnp.zeros(Nx + 2)
    (_, _), da_hist = jax.lax.scan(step, (a0_arr, a0_arr), dt * jnp.arange(n_steps))
    ey = -np.asarray(da_hist) / dt  # (nt, Nx)

    ip = int(x_probe / dx)
    sig = ey[int(0.67 * n_steps) :, ip]
    return float(np.sqrt(2.0 * np.mean(sig**2)))


def test_point_source_radiates_a0():
    """Point source (vlasov1d convention): radiated |Ey| = w·a0 within 3%."""
    w0, a0 = 1.0, 1.0e-3
    pulse = {
        "source": "point",
        "k0": w0,
        "w0": w0,
        "a0": a0,
        "t_center": 0.0,
        "t_width": 1.0e10,
        "t_rise": 5.0,
        "x_center": 30.0,
    }
    amp = _radiated_amplitude(pulse)
    np.testing.assert_allclose(amp, w0 * a0, rtol=0.03)


def test_extended_source_radiates_a0():
    """Calibrated extended source: radiated |Ey| = w·a0 within 10%.

    Antenna width 18 c/wp0 (the production 1 µm geometry, ~3 wavelengths at
    w=1). Pre-fix this radiated 9.0× nominal."""
    w0, a0 = 1.0, 1.0e-3
    pulse = {
        "source": "extended",
        "k0": w0,
        "w0": w0,
        "a0": a0,
        "t_center": 0.0,
        "t_width": 1.0e10,
        "t_rise": 5.0,
        "x_center": 30.0,
        "x_width": 17.9,
        "x_rise": 1.8,
    }
    amp = _radiated_amplitude(pulse)
    np.testing.assert_allclose(amp, w0 * a0, rtol=0.10)
