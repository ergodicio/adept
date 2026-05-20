"""Linear Landau damping of a driven EPW with the noise-subtraction technique.

We drive a small-amplitude Langmuir wave at ``k = 0.5`` (where the linear
Landau rate is large enough to measure in a tractable run) using a brief tanh
time-envelope, then run a second simulation with the **same noise seed** but
no driver. Subtracting ``E_A − E_B`` cancels the shared particle shot-noise
and exposes the driven response, whose mode-1 amplitude decays at the Landau
rate after the driver shuts off.

This test is intentionally expensive (1M PPC × nx=32 = 32M particles per
run) — the user explicitly asked for that resolution because the subtraction
floor is set by trajectory divergence between the two runs and only gets
clean at large N. Even at this scale we only expect order-of-magnitude
agreement with the analytic rate: a finite-grid PIC code with TSC shape and
``dt = 0.1`` has its own numerical damping/heating contributions on top of
true Landau damping. We assert that:

  1. The driven mode-1 amplitude is well above the single-run noise floor.
  2. The subtracted signal decays (γ < 0) after the driver shuts off.
  3. The measured decay rate is within a factor of 2 of the analytic
     ``γ_L = −√(π/8) · (1/k³) · ω_BG · exp(−ω_BG²/(2k²) − 3/2) ≈ −0.045``.

Marked ``slow`` — runs ~6 minutes wall clock on a single-CPU JAX backend.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from adept.pic1d import BasePIC1D

K_MODE = 0.5
L_BOX = 2 * np.pi / K_MODE
OMEGA_BG = float(np.sqrt(1.0 + 3.0 * K_MODE**2))  # ≈ 1.3229
GAMMA_LANDAU = float(
    -np.sqrt(np.pi / 8.0)
    * (1.0 / K_MODE**3)
    * OMEGA_BG
    * np.exp(-OMEGA_BG**2 / (2 * K_MODE**2) - 1.5)
)  # ≈ -0.0447


def _base_cfg(ppc: int, nx: int, tmax: float, dt: float) -> dict:
    return {
        "units": {"normalizing_temperature": "1eV", "normalizing_density": "1e21/cc"},
        "density": {
            "quasineutrality": True,
            "species-background": {
                "noise_seed": 12345, "noise_type": "gaussian", "noise_val": 0.0,
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
        "mlflow": {"experiment": "pic1d-tests", "run": "landau"},
        "drivers": {"ex": {}, "ey": {}},
        "diagnostics": {},
        "terms": {
            "field": "poisson",
            "time": "leapfrog",
            "species": [
                {"name": "electron", "charge": -1.0, "mass": 1.0,
                 "density_components": ["species-background"],
                 "loading": "random", "vmax_load": 6.0},
            ],
        },
    }


def _with_driver(cfg: dict) -> dict:
    """Drive at the Bohm-Gross frequency, ramped on at t≈10 and off by t≈20."""
    cfg = copy.deepcopy(cfg)
    cfg["drivers"]["ex"] = {
        "0": {
            "params": {"a0": 1.0e-3, "k0": K_MODE, "w0": OMEGA_BG, "dw0": 0.0},
            "envelope": {
                "time": {"center": 12.5, "rise": 1.5, "width": 10.0,
                          "baseline": 0.0, "bump_height": 1.0, "bump_or_trough": "bump",
                          "slope": 0.0},
                "space": {"center": L_BOX / 2, "rise": 1.0, "width": 1.0e6,
                          "baseline": 0.0, "bump_height": 1.0, "bump_or_trough": "bump",
                          "slope": 0.0},
            },
        }
    }
    return cfg


def _run(cfg: dict):
    m = BasePIC1D(copy.deepcopy(cfg))
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    return m({})["solver result"]


@pytest.mark.slow
def test_landau_damping_with_subtraction():
    base = _base_cfg(ppc=1_000_000, nx=32, tmax=50.0, dt=0.1)

    t0 = time.time()
    sol_A = _run(_with_driver(base))
    print(f"\nRun A (with driver): {time.time() - t0:.1f}s")
    t0 = time.time()
    sol_B = _run(base)
    print(f"Run B (control):     {time.time() - t0:.1f}s")

    ts = np.asarray(sol_A.ts["fields"])
    E_A = np.asarray(sol_A.ys["fields"]["e"])
    E_B = np.asarray(sol_B.ys["fields"]["e"])
    E_driven = E_A - E_B

    A1_driven = np.abs(np.fft.fft(E_driven, axis=1)[:, 1])
    A1_noise = np.abs(np.fft.fft(E_B, axis=1)[:, 1])

    # (1) Driver pumps the mode well above the single-run noise floor.
    i_peak = int(np.argmax(A1_driven[: int(0.6 * len(ts))]))
    assert A1_driven[i_peak] > 5 * A1_noise[i_peak], (
        f"Driven signal not visible above noise at t={ts[i_peak]:.1f}: "
        f"|E_d|={A1_driven[i_peak]:.3e}, |E_noise|={A1_noise[i_peak]:.3e}"
    )

    # (2-3) Fit decay in [t_off, t_off + 10]. Driver tanh tail is ~ 0.1% by
    # t=22 and effectively zero by t=24. We stop the fit at t=32 — beyond
    # that the subtraction-floor (trajectory divergence between A and B
    # builds up ~ a0² · t²) dominates the residual and contaminates the rate.
    fit_mask = (ts >= 22.0) & (ts <= 32.0)
    assert fit_mask.sum() >= 8, "Not enough samples in fit window"
    gamma_measured = float(
        np.polyfit(ts[fit_mask], np.log(A1_driven[fit_mask]), 1)[0]
    )

    print(f"  measured γ = {gamma_measured:+.4f}")
    print(f"  theory   γ = {GAMMA_LANDAU:+.4f}")

    # Decay direction.
    assert gamma_measured < 0, f"Expected damping, got γ = {gamma_measured:+.4f}"

    # Order-of-magnitude agreement (factor of 2).
    rel_err = abs(gamma_measured - GAMMA_LANDAU) / abs(GAMMA_LANDAU)
    assert rel_err < 1.0, (
        f"Measured γ={gamma_measured:.4f} differs from analytic Landau "
        f"γ_L={GAMMA_LANDAU:.4f} by {rel_err:.1%} (>100% allowed)"
    )
