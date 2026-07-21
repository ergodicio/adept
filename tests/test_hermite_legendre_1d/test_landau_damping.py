#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Driven Landau-damping gate for the mixed Hermite-Legendre solver.

Mirrors the BaseVlasov1D driven-resonance test (tests/test_vlasov1d/test_landau_damping.py):
a uniform Maxwellian is driven by a small external longitudinal field `ex` at the
resonant wavenumber `k0` and frequency `w0 = Re(omega)` of the kinetic electrostatic
dispersion relation, on a box of length `2*pi/k0` (so the driven mode is k=1). After
the driver ramps off, the electron plasma wave free-rings at its natural frequency
and decays at the Landau rate; we measure both from `E_x(k=1, t)` and compare to the
dispersion-relation root.

The bulk is carried by the AW-Hermite expansion, which captures Landau damping; the
Legendre part stays negligible in this linear regime.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pytest

from adept import electrostatic
from adept._hermite_legendre_1d.modules import BaseHermiteLegendre1D


def _run_driven(klambda_D: float, w0: float, Nh: int = 256, tmax: float = 80.0, dt: float = 0.05):
    Lx = 2.0 * np.pi / klambda_D
    cfg = {
        "solver": "hermite-legendre-1d",
        "physics": {
            "Lx": Lx,
            "alpha": np.sqrt(2.0),
            "u": 0.0,
            "v_a": -6.0,
            "v_b": 6.0,
            "gamma": 0.5,
            "nu_H": 0.0,
            "nu_L": 0.0,
            "enforce_conservation": True,
            "field": True,
        },
        "grid": {"Nx": 64, "Nh": Nh, "Nl": 16, "tmax": tmax, "dt": dt},
        "initialization": {"type": "linear-advection", "eps": 0.0, "mode": 1},  # uniform Maxwellian
        "drivers": {
            "ex": {
                "0": {
                    "k0": 2.0 * np.pi / Lx,
                    "w0": w0,
                    "dw0": 0.0,
                    "a0": 1.0e-3,
                    "t_center": 20.0,
                    "t_width": 20.0,
                    "t_rise": 5.0,
                    "x_center": 0.5 * Lx,
                    "x_width": 1.0e6,
                    "x_rise": 1.0,
                }
            }
        },
        "save": {"fields": {"t": {"nt": int(tmax * 4)}}},
        "units": {},
    }
    m = BaseHermiteLegendre1D(cfg)
    m.write_units()
    m.get_derived_quantities()
    m.get_solver_quantities()
    m.init_state_and_args()
    m.init_diffeqsolve()
    sol = m(trainable_modules={})["solver result"]
    e = np.asarray(sol.ys["fields"]["e"])
    t = np.asarray(sol.ts["fields"])
    ek1 = np.fft.fft(e, axis=1)[:, 1] / e.shape[1]
    return t, ek1


@pytest.mark.parametrize("klambda_D", [0.30, 0.35])
def test_driven_landau_damping(klambda_D):
    root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, klambda_D, maxwellian_convention_factor=2.0)
    expected_freq = float(np.real(root))
    expected_damp = float(np.imag(root))

    t, ek1 = _run_driven(klambda_D, w0=expected_freq)
    assert np.all(np.isfinite(ek1)), "driven run went non-finite"

    # free-decay window: driver is off by ~t=40, recurrence is well beyond t=70 at Nh=256
    win = (t > 45.0) & (t < 70.0)
    mag = np.abs(ek1[win])
    measured_damp = float(np.polyfit(t[win], np.log(mag), 1)[0])  # d/dt ln|E_k1|
    measured_freq = float(-np.polyfit(t[win], np.unwrap(np.angle(ek1[win])), 1)[0])  # -d(arg)/dt

    print(
        f"\nklambda_D={klambda_D:.2f}  freq {measured_freq:.4f} (exp {expected_freq:.4f})  "
        f"damp {measured_damp:.5f} (exp {expected_damp:.5f})"
    )
    # frequency is captured to <1%; finite-Nh Hermite slightly under-damps (~5-9%).
    np.testing.assert_allclose(measured_freq, expected_freq, rtol=0.02)
    np.testing.assert_allclose(measured_damp, expected_damp, rtol=0.15)
