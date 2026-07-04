#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Tests for the Strang / exact-force-exponential integrator ("strang-exp").

Regression context: the legacy Lawson-RK4 path treats the E/ponderomotive
force coupling explicitly, with stability bound |F|·√(2Nn)·√2/α·dt ≲ 2.8 —
E_crit ≈ 0.034 at Nn=1024, dt=0.1, α=0.108. In the first threshold campaign
on the sign-fixed solver, quarter-critical SRS fields reached that bound and
every run detonated super-exponentially to ~1e35 before NaN.

The strang-exp force substep applies the EXACT exponential of the nilpotent
lower-bidiagonal coupling (the frozen-force flow is a velocity shift — the
Hermite analog of vlasov1d's unitary e^{i·k_v·E·dt} push), as a finite series
of cheap bidiagonal products. Deliberately NOT Crank-Nicolson: CN is
spectrally stable but its non-normal transient response down the nilpotent
ladder behaves like (2x)^k where the true flow has x^k/k! — measured during
development, CN detonated where the exact series is machine-exact.

There is deliberately NO lawson-diverges/strang-survives contrast test here:
at any unit-test-accessible parameters strong enough to break the explicit
RK4 stage (per-level x = dt·|E|·√(2Nn)·√2/α ≳ 2.8), the truncated AW-Hermite
SYSTEM itself is unstable (dt-convergence study during development: growth
γ ≈ 1 ωp at E/α = 0.5, Nn=512, persisting as dt→0 with the o8-s36 filter on —
the Schumer-Holloway non-normal unboundedness under strong forcing, not an
integrator artifact). The integrator contract is therefore: exactness of the
sub-flow, conservation, and agreement with Lawson-RK4 in the smooth regime;
production-scale behavior is validated by SRS campaign A/Bs.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.modules import BaseHermitePoisson1D
from adept._hermite_poisson_1d.vector_field import _exp_force_substep


def _dense_nilpotent_exp(Nn: int, kappa_F_dt: np.ndarray) -> np.ndarray:
    """Dense exp(dt·A) for the nilpotent lower-bidiagonal coupling A (one x point).

    A[n, n-1] = kappa_n·F; nilpotent (A^Nn = 0) so the series is finite/exact.
    """
    A = np.zeros((Nn, Nn))
    for n in range(1, Nn):
        A[n, n - 1] = kappa_F_dt[n]
    E = np.eye(Nn)
    term = np.eye(Nn)
    for k in range(1, Nn):
        term = term @ A / k
        E = E + term
    return E


def test_exp_substep_matches_dense_nilpotent_exponential():
    """With n_terms >= Nn-1 the series is the EXACT matrix exponential —
    machine-precision agreement with a dense reference, even at large dt·F
    (here per-level x up to ~1.2, well past where any rational approximation
    would distort the transient)."""
    Nn, Nx = 8, 4
    alpha, F0, dt = 1.2, 0.5, 0.8
    rng = np.random.default_rng(3)
    C0 = rng.normal(size=(Nn, Nx)) + 1j * rng.normal(size=(Nn, Nx))
    Ck = jnp.asarray(np.fft.fft(C0, axis=-1) / Nx)  # k-space, norm="forward" convention
    sqrt_n = jnp.sqrt(jnp.arange(Nn, dtype=jnp.float64))

    out = _exp_force_substep(Ck, F0 * jnp.ones(Nx), sqrt_n, alpha, 1.0, dt, n_terms=24)
    C_num = np.fft.ifft(np.asarray(out), axis=-1) * Nx

    E = _dense_nilpotent_exp(Nn, dt * np.sqrt(2.0) / alpha * np.asarray(sqrt_n) * F0)
    C_exact = E @ C0

    np.testing.assert_allclose(C_num, C_exact, rtol=1e-13, atol=1e-13)


def test_exp_substep_conserves_density_and_momentum_response_sign():
    """Uniform accel on a Maxwellian: C_0 untouched exactly; the exact one-step
    momentum response is dC_1 = dt·(√2/α)·accel·C_0 (the same convention the
    ponderomotive-sign fix locked in for _hermite_e_coupling)."""
    Nn, Nx = 8, 16
    alpha, n0, F0, dt = 1.3, 0.7, 0.2, 0.05
    Ck = jnp.zeros((Nn, Nx), dtype=jnp.complex128).at[0, 0].set(n0 / alpha**3)
    sqrt_n = jnp.sqrt(jnp.arange(Nn, dtype=jnp.float64))

    out = _exp_force_substep(Ck, F0 * jnp.ones(Nx), sqrt_n, alpha, 1.0, dt)
    C = np.fft.ifft(np.asarray(out), axis=-1) * Nx

    np.testing.assert_allclose(C[0].real, n0 / alpha**3, rtol=1e-14)
    expected_c1 = dt * np.sqrt(2.0) / alpha * F0 * (n0 / alpha**3)
    np.testing.assert_allclose(C[1].real, expected_c1, rtol=1e-12)


def _make_module(integrator: str, e_amplitude: float, w0: float = 3.0):
    """Uniform plasma + uniform-envelope Ex driver with |E| = e_amplitude.

    w0=3 (above ωpe=1) so Debye shielding doesn't cancel the applied field and
    the oscillation velocity stays modest (v_osc/vth = E/(w0·α))."""
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
        "terms": {"integrator": integrator},
        "drivers": {
            "ex": {
                "0": {
                    "k0": 2.0 * np.pi / Lx,
                    "w0": w0,
                    "a0": e_amplitude / w0,  # driver amplitude prefactor is w0*a0
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


def test_both_integrators_agree_in_smooth_regime():
    """At weak drive (explicit-stage parameter ≪ 1) the two integrators solve
    the same physics: after 100 steps the electron states should agree to
    second-order accuracy levels."""
    import jax

    states = {}
    for integrator in ("lawson-rk4", "strang-exp"):
        module = _make_module(integrator, e_amplitude=3e-4)
        vf = module.diffeqsolve_quants["terms"].vector_field
        step = jax.jit(lambda t, y, vf=vf: vf(t, y, {}))
        y = module.state
        for i in range(100):
            y = step(0.1 * i, y)
        states[integrator] = np.asarray(y["Ck_electrons"].view(jnp.complex128))

    a, b = states["lawson-rk4"], states["strang-exp"]
    scale = np.max(np.abs(a[1:]))  # driven response lives in n >= 1 (n=0 is the Maxwellian)
    diff = np.max(np.abs(a - b))
    assert scale > 0
    assert diff < 0.05 * scale, f"integrators disagree: diff={diff:.3e} vs response scale={scale:.3e}"
