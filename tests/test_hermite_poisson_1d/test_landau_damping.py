#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""EPW dispersion test for the Hermite-Poisson 1D solver (full solve path).

Runs the actual production path — BaseHermitePoisson1D lifecycle with the
Lawson-RK4 stepper and the explicit E·∂_v f coupling — on a uniform plasma
with a small single-mode density perturbation, then checks the ringing
frequency and Landau damping rate of E_x(k1, t) against the kinetic
electrostatic dispersion relation.

This is the physics gate the solver lacked when the inverted force coupling
shipped: with that bug the perturbation produced no Langmuir oscillation at
all (E never did work on the plasma), so any version of this test would have
failed immediately.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pytest

from adept import electrostatic
from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D


def _measure_ringing(e_xt: np.ndarray, t: np.ndarray, mode: int) -> tuple[float, float]:
    """Fit oscillation frequency and damping rate of the k=mode component.

    The initial standing perturbation excites both ±ω branches, so
    |A(t)| oscillates through near-zeros twice per period. We fit:
      - omega from the mean spacing of |A| maxima (= pi / omega apart),
      - gamma from a linear fit to log|A| at those maxima.
    """
    A = np.fft.fft(e_xt, axis=1)[:, mode] / e_xt.shape[1]
    absA = np.abs(A)

    # local maxima, skipping the initial transient (first 10% of the run)
    i0 = len(t) // 10
    peaks = [
        i
        for i in range(i0 + 1, len(t) - 1)
        if absA[i] >= absA[i - 1] and absA[i] > absA[i + 1]
    ]
    assert len(peaks) >= 4, f"too few |A| maxima to fit ({len(peaks)})"

    t_pk = t[peaks]
    omega = np.pi / np.mean(np.diff(t_pk))
    gamma = np.polyfit(t_pk, np.log(absA[peaks]), 1)[0]
    return float(omega), float(gamma)


@pytest.mark.parametrize("klambda_D", [0.30, 0.35])
def test_epw_dispersion(klambda_D: float):
    mode = 1
    Lx = 4.0 * np.pi
    k = 2.0 * np.pi * mode / Lx
    alpha_e = klambda_D * np.sqrt(2.0) / k  # same convention as test_spectrax1d

    root = electrostatic.get_roots_to_electrostatic_dispersion(
        wp_e=1.0, vth_e=1.0, k0=klambda_D, maxwellian_convention_factor=2.0
    )
    expected_freq = float(np.real(root))
    expected_damp = float(np.imag(root))

    tmax = 80.0
    nt_save = 1601

    cfg = {
        "physics": {
            "Lx": Lx,
            "alpha_e": alpha_e,
            "alpha_i": alpha_e,
            "n0_e": 1.0,
            "n0_i": 1.0,
            "nu": 0.0,
            "static_ions": True,
            "c_light": 1.0,
        },
        "grid": {
            "Nn": 256,
            "Ni": 2,
            "Nx": 64,
            "tmax": tmax,
            "dt": 0.05,
        },
        "density": {"perturbation": {"mode": mode, "amplitude": 1.0e-4}},
        "drivers": {},
        "save": {"fields": {"t": {"tmin": 0.0, "tmax": tmax, "nt": nt_save}}},
    }

    module = BaseHermitePoisson1D(cfg)
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    sol = module(None)["solver result"]

    e_xt = np.asarray(sol.ys["fields"]["e"])
    t = np.asarray(sol.ts["fields"])

    measured_freq, measured_damp = _measure_ringing(e_xt, t, mode)

    print(
        f"\nklambda_D={klambda_D:.2f}  freq: {measured_freq:.4f} (expected {expected_freq:.4f})"
        f"  damp: {measured_damp:.5f} (expected {expected_damp:.5f})"
    )

    # Observed agreement at these parameters is ~0.2% or better; the bounds
    # below leave headroom for platform jitter while still catching any
    # physics-level regression.
    np.testing.assert_allclose(measured_freq, expected_freq, rtol=0.02)
    np.testing.assert_allclose(measured_damp, expected_damp, rtol=0.05)
