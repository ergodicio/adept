#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Regression tests for the ponderomotive force sign in the HP force assembly.

Regression context: _nonlinear_rhs originally computed Fp = -0.5*d(a²)/dx and
passed Ex + Fp into _hermite_e_coupling, which multiplies the WHOLE sum by
q/m (= -1 for electrons). But the ponderomotive acceleration is ∝ q²/m² —
charge-sign independent — so folding it under q/m flipped its sign for
electrons: acceleration = -Ex + ½∂x(a²) instead of -Ex - ½∂x(a²).

The E-field leg passed every dispersion/Landau test (those never exercise a),
while the sign flip inverted one leg of the SRS three-wave coupling
(γ² ∝ product of both legs): Stokes backscatter had γ² < 0 (SRS structurally
forbidden in every run) and the normally-stable anti-Stokes pair had γ² > 0.
In the homogeneous 0.2 n_c / 3 keV bisection campaign the observed detonating
electrostatic mode sat at k = 0.49 c/ωp0 — anti-Stokes matching
ω_em = ω0 + ω_BG(k), k_em = k0 + k closes at k = 0.4919, within one spectral
bin, while Stokes matching was 14% off.

Ground truth for the force assembly is the vlasov1d reference
(solvers/pushers/vlasov.py): force = q*e + (q²/m)*pond, accel = force/m.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.vector_field import (
    HermitePoisson1DVectorField,
    LongitudinalElectricFieldDriver,
    PoissonSolver1D,
)


def _make_vf(Nn, Nx, Lx, alpha_e, alpha_i, mi_me, static_ions, static_ion_density):
    """Minimal vector field: only the pieces _nonlinear_rhs touches are real."""
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx) * Nx / Lx
    one_over_kx = jnp.where(kx != 0.0, 1.0 / jnp.where(kx != 0.0, kx, 1.0), 0.0)

    poisson = PoissonSolver1D(
        one_over_kx=one_over_kx,
        alpha_e=alpha_e,
        alpha_i=alpha_i,
        static_ion_density=static_ion_density,
    )
    n = jnp.arange(Nn, dtype=jnp.float64)
    return HermitePoisson1DVectorField(
        combined_exp=None,
        poisson=poisson,
        wave_solver=None,
        ey_driver=None,
        ex_driver=LongitudinalElectricFieldDriver(x, {}),
        sqrt_n_minus_e=jnp.sqrt(n),
        sqrt_n_minus_i=jnp.sqrt(n),
        alpha_e=alpha_e,
        alpha_i=alpha_i,
        q_e=-1.0,
        q_i=1.0,
        mi_me=mi_me,
        Omega_ce_tau=1.0,
        dx=Lx / Nx,
        dt=0.1,
        static_ions=static_ions,
    )


def _maxwellian_state(Nn, Nx, n0, alpha_e, alpha_i):
    """Uniform Maxwellians for both species (only C_0 ≠ 0), density n0."""
    Ck_e = jnp.zeros((Nn, Nx), dtype=jnp.complex128).at[0, 0].set(n0 / alpha_e**3)
    Ck_i = jnp.zeros((Nn, Nx), dtype=jnp.complex128).at[0, 0].set(n0 / alpha_i**3)
    return {"Ck_electrons": Ck_e.view(jnp.float64), "Ck_ions": Ck_i.view(jnp.float64)}


def test_electron_ponderomotive_sign_and_value():
    """Static a(x) bump on a uniform Maxwellian (Ex = 0): the ONLY force is
    ponderomotive, and the electron momentum response must be

        dC_1/dt = √2/α_e · (-½ ∂x a²) · C_0

    i.e. electrons are pushed DOWN the intensity gradient, independent of
    their charge sign. The pre-fix code returned exactly the opposite sign
    (ponderomotive folded under q/m = -1)."""
    Nn, Nx, Lx = 8, 64, 2.0 * jnp.pi
    alpha_e, alpha_i, n0 = 1.3, 0.05, 0.7
    dx = Lx / Nx

    vf = _make_vf(
        Nn,
        Nx,
        Lx,
        alpha_e,
        alpha_i,
        mi_me=1836.0,
        static_ions=True,
        static_ion_density=n0 * jnp.ones(Nx),
    )
    state = _maxwellian_state(Nn, Nx, n0, alpha_e, alpha_i)

    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    a_frozen = jnp.asarray(0.3 * np.sin(x))

    out = vf._nonlinear_rhs(0.0, state, a_frozen, {})
    dC_e = np.fft.ifft(np.asarray(out["Ck_electrons"]).view(np.complex128), axis=-1) * Nx

    Fp = np.asarray(-0.5 * jnp.gradient(a_frozen**2, dx))
    expected_c1 = np.sqrt(2.0) / alpha_e * Fp * (n0 / alpha_e**3)

    np.testing.assert_allclose(dC_e[1].real, expected_c1, rtol=1e-12, atol=1e-15)

    # Only the momentum mode responds (force ladder feeds n=1 from n=0)
    mask = np.ones(Nn, dtype=bool)
    mask[1] = False
    assert np.max(np.abs(dC_e[mask])) < 1e-14


def test_ponderomotive_is_charge_sign_independent():
    """Mobile ions with q = +1: the ponderomotive response must have the SAME
    sign as the electrons' (both species pushed down the a² gradient), with
    magnitude scaled by (q/m)² = 1/mi_me². The pre-fix code dropped the ion
    ponderomotive entirely AND sign-flipped the electron one, so both species
    fail this test there."""
    Nn, Nx, Lx = 8, 64, 2.0 * jnp.pi
    alpha_e, alpha_i, n0, mi_me = 1.1, 0.04, 0.5, 100.0
    dx = Lx / Nx

    vf = _make_vf(
        Nn,
        Nx,
        Lx,
        alpha_e,
        alpha_i,
        mi_me=mi_me,
        static_ions=False,
        static_ion_density=None,
    )
    state = _maxwellian_state(Nn, Nx, n0, alpha_e, alpha_i)

    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    a_frozen = jnp.asarray(0.2 * np.cos(2.0 * x))

    out = vf._nonlinear_rhs(0.0, state, a_frozen, {})
    dC_e = np.fft.ifft(np.asarray(out["Ck_electrons"]).view(np.complex128), axis=-1) * Nx
    dC_i = np.fft.ifft(np.asarray(out["Ck_ions"]).view(np.complex128), axis=-1) * Nx

    Fp = np.asarray(-0.5 * jnp.gradient(a_frozen**2, dx))
    expected_e = np.sqrt(2.0) / alpha_e * Fp * (n0 / alpha_e**3)
    expected_i = (1.0 / mi_me**2) * np.sqrt(2.0) / alpha_i * Fp * (n0 / alpha_i**3)

    np.testing.assert_allclose(dC_e[1].real, expected_e, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dC_i[1].real, expected_i, rtol=1e-12, atol=1e-14)


def test_e_field_leg_unchanged_by_fix():
    """The E-field leg (uniform charge separation → Poisson Ex) must still
    carry q/m: electron and ion accelerations from the same Ex have OPPOSITE
    signs, ratio -mi_me (per unit C_0 response). Guards against 'fixing' the
    ponderomotive by dropping the charge factor everywhere."""
    Nn, Nx, Lx = 8, 64, 2.0 * jnp.pi
    alpha_e, alpha_i, n0, mi_me = 1.0, 0.05, 0.6, 25.0

    vf = _make_vf(
        Nn,
        Nx,
        Lx,
        alpha_e,
        alpha_i,
        mi_me=mi_me,
        static_ions=False,
        static_ion_density=None,
    )
    # Perturb electron density at one k so Poisson gives Ex ≠ 0
    state = _maxwellian_state(Nn, Nx, n0, alpha_e, alpha_i)
    Ck_e = state["Ck_electrons"].view(jnp.complex128).at[0, 1].set(0.01 / alpha_e**3)
    state["Ck_electrons"] = Ck_e.view(jnp.float64)

    out = vf._nonlinear_rhs(0.0, state, jnp.zeros(Nx), {})
    dC_e = np.fft.ifft(np.asarray(out["Ck_electrons"]).view(np.complex128), axis=-1) * Nx
    dC_i = np.fft.ifft(np.asarray(out["Ck_ions"]).view(np.complex128), axis=-1) * Nx

    # Expected: dC_1 = √2/α_s · (q_s/m_s)·Ex · C_0_s(x), with q/m = -1 (e), +1/mi_me (i)
    Ex = np.asarray(vf.poisson(state["Ck_electrons"].view(jnp.complex128), state["Ck_ions"].view(jnp.complex128)))
    assert np.max(np.abs(Ex)) > 1e-12  # non-trivial field
    C0_e = np.fft.ifft(np.asarray(state["Ck_electrons"].view(jnp.complex128))[0]) * Nx
    C0_i = np.fft.ifft(np.asarray(state["Ck_ions"].view(jnp.complex128))[0]) * Nx
    expected_e = np.sqrt(2.0) / alpha_e * (-1.0) * Ex * C0_e
    expected_i = np.sqrt(2.0) / alpha_i * (1.0 / mi_me) * Ex * C0_i

    np.testing.assert_allclose(dC_e[1], expected_e, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dC_i[1], expected_i, rtol=1e-12, atol=1e-13)
