#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Tests for LPSE mode (terms.linear_response).

The velocity-space force term is linearized about the initialized
equilibrium: dC_n/dt|F = kappa_n*accel*C_eq_{n-1} — the textbook
-(qE/m)*df0/dv source. The force-driven ladder cascade, and with it the
truncated-AW blow-up under strong forcing, is removed structurally: linear
kinetics are exact, and above-threshold instabilities grow exponentially
without saturation (enveloped-LPSE semantics). Scan-5 context: capping the
force on the FULL coupling (terms.stabilize) only converts the detonation
into a bounded-coefficient exponential at gamma ~ kappa_max*cap ~ 0.5 wp
that overflows float64 within ~1000/wp; linearization is the correct
stable-at-any-intensity mode.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D


def _make_module(e_amplitude: float, linear_response: bool, w0: float = 3.0):
    """Uniform plasma + uniform-envelope Ex driver with |E| = e_amplitude
    (same rig as test_stabilization)."""
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
        "terms": {"integrator": "strang-exp", "linear_response": linear_response},
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


def test_linear_response_matches_full_at_small_amplitude():
    """Weak drive: the linearized and full couplings agree to O(perturbation²)
    — well under a percent of the driven response."""
    states = {}
    for lin in (False, True):
        module = _make_module(e_amplitude=3e-4, linear_response=lin)
        y, peak = _run(module, 100)
        assert np.isfinite(peak)
        states[lin] = np.asarray(y["Ck_electrons"].view(np.complex128))
    a, b = states[False], states[True]
    scale = np.max(np.abs(a[1:]))
    assert scale > 0
    diff = np.max(np.abs(a - b))
    assert diff < 1e-2 * scale, f"linearization not transparent: diff={diff:.3e} vs response={scale:.3e}"


def test_linear_response_survives_strong_forcing():
    """|E|/α = 1.5 at Nn=512: detonates the full coupling at any dt and
    overflows the tanh-capped variant; the linearized coupling is a driven
    LINEAR system — it must run 400 steps finite."""
    module = _make_module(e_amplitude=0.15, linear_response=True)
    _, peak = _run(module, 400)
    assert np.isfinite(peak), "linear-response run went non-finite"


def test_linear_response_epw_dispersion():
    """Landau gate through the linearized path: this IS textbook linearized
    Vlasov, so frequency and damping must match kinetic theory at the
    standard tolerances."""
    from adept import electrostatic
    from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D

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
        "terms": {"integrator": "strang-exp", "linear_response": True},
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
