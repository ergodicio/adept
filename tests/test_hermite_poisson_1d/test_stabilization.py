#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Tests for the LPSE-style stabilization clamps (terms.stabilize).

Purpose: SRS-threshold campaigns need runs that complete both below AND above
threshold. Above threshold the truncated AW-Hermite hierarchy detonates once
local fields reach the strong-forcing zone (hierarchy-forcing parameter
x = dt·|accel|·√(4Nn)/α ≳ O(1)) — an instability of the truncated system
itself, insensitive to filter strength/shape, LB nu, and Nn (absorber-bracket
campaign, kinetic-srs 2026-07-04). The clamp caps the acceleration with a
smooth tanh at x ≈ 0.3 (auto default), which is transparent through the
linear/threshold phase (fields 10–100× below the cap; relative distortion
(a/cap)²/3) and artificially saturates the parametric loop above it —
stable-but-inaccurate at saturation, like an enveloped LPSE run that just
keeps growing. A clip on the wave-equation density is the second safety net.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D


def _make_module(e_amplitude: float, stabilize: bool, w0: float = 3.0):
    """Uniform plasma + uniform-envelope Ex driver with |E| = e_amplitude
    (same rig as test_integrators; drive above ωpe so shielding is weak)."""
    Lx = 20.0
    cfg = {
        "physics": {
            "Lx": Lx,
            "alpha_e": 0.1,
            "alpha_i": 0.1,
            "n0_e": 1.0,
            "n0_i": 1.0,
            "nu": 0.02,
            "collision_model": "lenard-bernstein",
            "static_ions": True,
            "c_light": 1.0,
        },
        "grid": {"Nn": 512, "Ni": 2, "Nx": 32, "tmax": 40.0, "dt": 0.1},
        "density": {},
        "terms": {"integrator": "strang-exp", "stabilize": stabilize},
        "drivers": {
            "ex": {
                "0": {
                    "k0": 2.0 * np.pi / Lx,
                    "w0": w0,
                    "a0": e_amplitude / w0,
                    "t_rise": 1.0,
                    "t_center": 0.0,
                    "t_width": 1.0e10,
                }
            }
        },
        "save": {"fields": {"t": {"tmin": 0.0, "tmax": 40.0, "nt": 3}}},
    }
    module = BaseHermitePoisson1D(cfg)
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    return module


def _run(module, n_steps: int):
    vf = module.diffeqsolve_quants["terms"].vector_field
    step = jax.jit(lambda t, y, vf=vf: vf(t, y, {}))
    y = module.state
    peak = 0.0
    for i in range(n_steps):
        y = step(0.1 * i, y)
        m = float(jnp.max(jnp.abs(y["Ck_electrons"])))
        if not np.isfinite(m):
            return y, np.inf
        peak = max(peak, m)
    return y, peak


def test_stabilize_transparent_at_small_amplitude():
    """Weak drive (|accel|/cap ≈ 0.05): clamped and unclamped runs agree to
    well under a percent of the driven response — the clamp must not touch
    below-threshold / linear-growth physics."""
    states = {}
    for stab in (False, True):
        module = _make_module(e_amplitude=3e-4, stabilize=stab)
        y, peak = _run(module, 100)
        assert np.isfinite(peak)
        states[stab] = np.asarray(y["Ck_electrons"].view(jnp.complex128))
    a, b = states[False], states[True]
    scale = np.max(np.abs(a[1:]))  # response lives in n >= 1
    assert scale > 0
    diff = np.max(np.abs(a - b))
    assert diff < 1e-2 * scale, f"clamp not transparent: diff={diff:.3e} vs response={scale:.3e}"


def test_stabilize_survives_strong_forcing():
    """The regression: |E|/α = 1.5 at Nn=512, dt=0.1 detonates the truncated
    hierarchy for EVERY uncapped integrator at any dt (development
    dt-convergence study: γ ≈ 1 ωp persisting as dt→0, filter on). With
    terms.stabilize the same drive must run 400 steps finite and bounded."""
    module = _make_module(e_amplitude=0.15, stabilize=True)
    # auto cap at Nn=512, dt=0.1, alpha=0.1: 0.3*0.1/(2*sqrt(512)*0.1) ≈ 6.6e-3
    vf = module.diffeqsolve_quants["terms"].vector_field
    assert vf.force_cap is not None and 5e-3 < vf.force_cap < 8e-3
    _, peak = _run(module, 400)
    assert np.isfinite(peak), "stabilized run went non-finite"
    assert peak < 1e6, f"stabilized run unbounded: peak/C0 ~ {peak / 1e3:.2e}"


def test_epw_dispersion_unchanged_with_stabilize():
    """Landau gate with the clamp on: perturbation fields (~1e-4) sit far
    below the cap, so frequency and damping must match kinetic theory at the
    same tolerances as the unclamped suite."""
    from adept import electrostatic

    klambda_D = 0.30
    mode = 1
    Lx = 4.0 * np.pi
    k = 2.0 * np.pi * mode / Lx
    alpha_e = klambda_D * np.sqrt(2.0) / k

    root = electrostatic.get_roots_to_electrostatic_dispersion(
        wp_e=1.0, vth_e=1.0, k0=klambda_D, maxwellian_convention_factor=2.0
    )

    tmax = 80.0
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
        "grid": {"Nn": 256, "Ni": 2, "Nx": 64, "tmax": tmax, "dt": 0.05},
        "density": {"perturbation": {"mode": mode, "amplitude": 1.0e-4}},
        "drivers": {},
        "terms": {"integrator": "strang-exp", "stabilize": True},
        "save": {"fields": {"t": {"tmin": 0.0, "tmax": tmax, "nt": 1601}}},
    }
    module = BaseHermitePoisson1D(cfg)
    module.get_derived_quantities()
    module.get_solver_quantities()
    module.init_state_and_args()
    module.init_diffeqsolve()
    sol = module(None)["solver result"]

    e_xt = np.asarray(sol.ys["fields"]["e"])
    t = np.asarray(sol.ts["fields"])
    A = np.abs(np.fft.fft(e_xt, axis=1)[:, mode]) / e_xt.shape[1]
    i0 = len(t) // 10
    peaks = [i for i in range(i0 + 1, len(t) - 1) if A[i] >= A[i - 1] and A[i] > A[i + 1]]
    t_pk = t[peaks]
    omega = np.pi / np.mean(np.diff(t_pk))
    gamma = np.polyfit(t_pk, np.log(A[peaks]), 1)[0]

    np.testing.assert_allclose(omega, float(np.real(root)), rtol=0.02)
    np.testing.assert_allclose(gamma, float(np.imag(root)), rtol=0.05)
