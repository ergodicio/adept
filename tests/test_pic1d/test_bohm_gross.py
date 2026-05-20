"""EPW dispersion test: oscillation frequency matches kinetic dispersion.

We initialize a Maxwellian quietly, inject a small sinusoidal density
perturbation, and verify the resulting oscillation frequency at the seeded
mode matches the kinetic (Vlasov-Poisson) dispersion of a Langmuir wave.

We use the seeded mode at ``k = 0.3 / λ_D`` — Landau damping there is
~ 10⁻³ ωp, so the wave is essentially a clean oscillator over the simulation
time. We extract ω from the dominant peak of the time-FFT of E_k(t).

The reference frequency is obtained by solving ``Re ε(k, ω) = 0`` for the
real part of the dispersion via the plasma dispersion function. This is
distinct from the leading-order Bohm-Gross ``ω² ≈ ωp² + 3 k² v_th²``, which
is only the small-k Taylor expansion and underestimates the kinetic ω at
``k ~ 0.3`` by a few percent.
"""

from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import wofz

K_MODE = 0.3
L_BOX = 2 * np.pi / K_MODE  # one fundamental wavelength


def _plasma_dispersion_real_part(omega: float, k: float) -> float:
    """``Re ε(k, ω)`` for a Maxwellian with v_th=1, ωp=1 (code units)."""
    xi = omega / (np.sqrt(2) * k)
    Z = 1j * np.sqrt(np.pi) * wofz(xi)
    return float((1.0 + (1.0 / k**2) * (1.0 + xi * Z)).real)


# Real-frequency root of Re ε(k=0.3, ω) = 0: the kinetic Langmuir frequency.
OMEGA_KINETIC = brentq(_plasma_dispersion_real_part, 1.05, 1.5, args=(K_MODE,))


def _cfg(ppc: int, nx: int, tmax: float, dt: float) -> dict:
    return {
        "units": {"normalizing_temperature": "1eV", "normalizing_density": "1e21/cc"},
        "density": {
            "quasineutrality": True,
            "species-background": {
                "noise_seed": 42, "noise_type": "gaussian", "noise_val": 0.0,
                "v0": 0.0, "T0": 1.0, "m": 2.0,
                "basis": "uniform", "baseline": 1.0,
            },
        },
        "grid": {
            "dt": dt, "nx": nx, "tmin": 0.0, "tmax": tmax,
            "xmin": 0.0, "xmax": L_BOX, "ppc": ppc,
            "particle_shape": "tsc",
        },
        "save": {"fields": {"t": {"tmin": 0.0, "tmax": tmax, "nt": int(tmax / dt) + 1}}},
        "solver": "pic-1d",
        "mlflow": {"experiment": "pic1d-tests", "run": "epw-dispersion"},
        "drivers": {"ex": {}, "ey": {}},
        "diagnostics": {},
        "terms": {
            "field": "poisson",
            "time": "leapfrog",
            "species": [
                {"name": "electron", "charge": -1.0, "mass": 1.0,
                 "density_components": ["species-background"],
                 "loading": "quiet", "vmax_load": 8.0},
            ],
        },
    }


@pytest.mark.parametrize("integrator", ["leapfrog", "yoshida4"])
def test_epw_dispersion_frequency(integrator):
    from adept.pic1d import BasePIC1D  # local import to keep pytest collection cheap

    tmax = 60.0
    dt = 0.05
    cfg = _cfg(ppc=512, nx=32, tmax=tmax, dt=dt)
    cfg["terms"]["time"] = integrator

    m = BasePIC1D(copy.deepcopy(cfg))
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()

    # Seed a small density perturbation at the fundamental mode.
    A_seed = 0.005
    x = m.state["x_electron"]
    x_new = x + (A_seed / K_MODE) * jnp.sin(K_MODE * x)
    m.state["x_electron"] = jnp.mod(x_new, L_BOX)

    m.init_diffeqsolve()
    sol = m({})["solver result"]

    ts = np.asarray(sol.ts["fields"])
    E_xt = np.asarray(sol.ys["fields"]["e"])
    Ek1 = np.fft.fft(E_xt, axis=1)[:, 1]  # complex mode-1 amplitude over time

    # Drop the first ~5 plasma periods (start-up transient) and zero-pad
    # heavily for sub-bin frequency resolution.
    keep = int(5.0 / dt)
    sig = Ek1[keep:] - Ek1[keep:].mean()
    n_pad = 16 * len(sig)
    spectrum = np.fft.fft(sig, n=n_pad)
    freqs = np.fft.fftfreq(n_pad, d=dt) * 2 * np.pi  # angular frequency
    pos = freqs > 0
    omega_meas = float(freqs[pos][np.argmax(np.abs(spectrum[pos]))])

    rel_err = abs(omega_meas - OMEGA_KINETIC) / OMEGA_KINETIC
    assert rel_err < 0.03, (
        f"integrator={integrator}: ω_meas={omega_meas:.4f}, "
        f"ω_kinetic={OMEGA_KINETIC:.4f}, rel_err={rel_err:.2%}"
    )
