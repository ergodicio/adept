"""Self-consistent validation of the longitudinal EM-PIC-1D path.

Two checks on the deposit → push → current → Ampère loop:

1. **Gauss's law** ``D_x E = ρ`` is preserved to machine precision by the
   charge-conserving current (the whole point of the Esirkepov-equivalent
   construction + Ampère update).
2. **Cold plasma oscillation** of a small density perturbation rings at the
   plasma frequency. Units are chosen so the equilibrium density is 1, hence
   ``ω_pe = 1``; ``c`` is large so the small-amplitude motion is non-relativistic.

These exercise the full longitudinal solver short of the ``ergoExo`` run path.
``jax_enable_x64`` is on for the suite (``tests/conftest.py``).
"""

import jax
import numpy as np
from jax import numpy as jnp

from adept._empic1d.solvers.pushers.field import (
    charge_density_nodes,
    divergence_ex,
    solve_ex_from_gauss,
)
from adept._empic1d.solvers.vector_field import longitudinal_step

C = 50.0  # speed of light; large ⇒ small-amplitude oscillation is non-relativistic
SHAPE = "tsc"


def _cold_plasma(L, nx, ppc, delta):
    """Quiet-loaded cold electrons with a sinusoidal density perturbation.

    Weights are set so the equilibrium electron density is 1 (⇒ ω_pe = 1), with
    a uniform neutralizing ion background. Returns (state, params, background).
    """
    n_particles = nx * ppc
    dx = L / nx
    x0 = (jnp.arange(n_particles) + 0.5) * (L / n_particles)
    k1 = 2.0 * jnp.pi / L
    x = jnp.mod(x0 + delta * jnp.sin(k1 * x0), L)
    w = jnp.full((n_particles,), L / n_particles)
    u = jnp.zeros((n_particles, 3))
    charge, qm = -1.0, -1.0

    rho_e = charge_density_nodes(x, w, charge, nx, dx, 0.0, SHAPE)
    background = float(-jnp.mean(rho_e))  # uniform ions ⇒ zero-mean total charge
    e_face = solve_ex_from_gauss(rho_e + background, dx)

    state = {"x": x, "u": u, "w": w, "E": e_face}
    params = dict(charge=charge, qm=qm, dt=0.05, c=C, nx=nx, dx=dx, xmin=0.0, length=L, shape=SHAPE)
    return state, params, background


def _run(state, params, background, n_steps):
    """Scan the step, recording the mode-1 E coefficient and Gauss residual."""

    def scan_fn(s, _):
        s = longitudinal_step(s, **params)
        rho = charge_density_nodes(s["x"], s["w"], params["charge"], params["nx"], params["dx"], 0.0, SHAPE)
        resid = jnp.max(jnp.abs(divergence_ex(s["E"], params["dx"]) - (rho + background)))
        mode1 = jnp.fft.rfft(s["E"])[1]
        return s, (mode1, resid)

    _, (mode1_hist, resid_hist) = jax.lax.scan(scan_fn, state, None, length=n_steps)
    return np.asarray(mode1_hist), np.asarray(resid_hist)


def _dominant_frequency(signal, dt):
    """Peak angular frequency of a real-valued tone, parabolically interpolated."""
    sig = signal - signal.mean()
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=dt) * 2.0 * np.pi
    k = int(np.argmax(spec[1:])) + 1  # skip DC
    # Parabolic interpolation around the peak bin for sub-bin resolution.
    if 0 < k < len(spec) - 1:
        a, b, cc = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2.0 * b + cc
        offset = 0.5 * (a - cc) / denom if denom != 0 else 0.0
    else:
        offset = 0.0
    dw = freqs[1] - freqs[0]
    return freqs[k] + offset * dw


def test_gauss_law_preserved():
    state, params, background = _cold_plasma(L=2.0 * np.pi, nx=64, ppc=64, delta=1e-3)
    _, resid_hist = _run(state, params, background, n_steps=400)
    assert resid_hist.max() < 1e-9


def test_cold_plasma_oscillation_frequency():
    L = 2.0 * np.pi
    state, params, background = _cold_plasma(L=L, nx=64, ppc=64, delta=1e-3)
    n_steps = 2600  # ~20 plasma periods at dt=0.05 ⇒ fine frequency resolution
    mode1_hist, _ = _run(state, params, background, n_steps)

    omega = _dominant_frequency(np.real(mode1_hist), params["dt"])
    # ω_pe = 1 by construction (equilibrium density = 1, q = m = 1).
    assert abs(omega - 1.0) < 0.02
