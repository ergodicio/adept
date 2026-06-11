#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Unit tests for the Hermite-Poisson 1D E-field/ponderomotive coupling.

Regression context: _hermite_e_coupling originally read C[n+1] instead of
C[n−1] — transcribed from a (then-wrong) docstring in spectrax1d's
shift_multi rather than from its behavior. With that inversion, a uniform
electric field applied to a Maxwellian drove zero current (the momentum
equation was structurally absent), and every gradient-SRS production run
blew up from a spurious, field-free low-n instability.

The correct AW-Hermite force term (Parker & Dellar 2015 eq. 3.11):

    dC_n/dt|E = +(q/m)·√(2n)/α · F · C_{n−1}
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adept._hermite_poisson_1d.vector_field import _hermite_e_coupling


def _coupling(Ck, F, Nn, q_over_m=-1.0, alpha=1.0, Omega=1.0):
    sqrt_n_minus = jnp.sqrt(jnp.arange(Nn, dtype=jnp.float64))
    return np.asarray(_hermite_e_coupling(Ck, F, sqrt_n_minus, q_over_m, alpha, Omega))


def test_uniform_E_on_maxwellian_drives_current():
    """Uniform F on a Maxwellian (only C_0 ≠ 0) excites ONLY n=1, with
    dC_1/dt = (q/m)·√2/α·F·C_0. This is the convention-independent momentum
    equation; the inverted coupling returned identically zero here."""
    Nn, Nx = 8, 16
    c0, E0, alpha, q_over_m = 0.6, 0.3, 1.4, -1.0

    Ck = jnp.zeros((Nn, Nx), dtype=jnp.complex128).at[0, 0].set(c0)
    F = E0 * jnp.ones(Nx)

    out = _coupling(Ck, F, Nn, q_over_m=q_over_m, alpha=alpha)

    expected_c1 = q_over_m * np.sqrt(2.0) / alpha * E0 * c0
    np.testing.assert_allclose(out[1, 0].real, expected_c1, rtol=1e-12)

    mask = np.ones_like(out, dtype=bool)
    mask[1, 0] = False
    assert np.max(np.abs(out[mask])) < 1e-14


def test_force_ladder_is_one_sided_upward():
    """Seeding only C_2 excites only C_3 with coefficient √(2·3)/α."""
    Nn, Nx = 8, 16
    c2, E0, alpha, q_over_m = 0.5, 0.2, 0.9, -1.0

    Ck = jnp.zeros((Nn, Nx), dtype=jnp.complex128).at[2, 0].set(c2)
    F = E0 * jnp.ones(Nx)

    out = _coupling(Ck, F, Nn, q_over_m=q_over_m, alpha=alpha)

    expected_c3 = q_over_m * np.sqrt(6.0) / alpha * E0 * c2
    np.testing.assert_allclose(out[3, 0].real, expected_c3, rtol=1e-12)

    mask = np.ones_like(out, dtype=bool)
    mask[3, 0] = False
    assert np.max(np.abs(out[mask])) < 1e-14


def test_density_is_never_forced():
    """The force term must not alter C_0 (density) regardless of state."""
    Nn, Nx = 8, 16
    rng = np.random.default_rng(0)
    Ck = jnp.asarray(rng.normal(size=(Nn, Nx)) + 1j * rng.normal(size=(Nn, Nx)))
    F = jnp.asarray(rng.normal(size=Nx))

    out = _coupling(Ck, F, Nn)
    assert np.max(np.abs(out[0])) < 1e-14


def test_matches_spectrax1d_lorentz_term():
    """The HP coupling must equal the validated spectrax1d Lorentz E-term
    (the path exercised by the passing spectrax EPW dispersion tests),
    specialized to 1D (Nm=Np=1, E_x only)."""
    from adept._spectrax1d.hermite_fourier_ode import HermiteFourierODE

    Nn, Nx = 6, 8
    alpha, q_over_m = 1.2, -1.0
    rng = np.random.default_rng(1)
    Ck = jnp.asarray(rng.normal(size=(Nn, Nx)) + 1j * rng.normal(size=(Nn, Nx)))
    F = jnp.asarray(rng.normal(size=Nx))

    out_hp = _coupling(Ck, F, Nn, q_over_m=q_over_m, alpha=alpha)

    # spectrax reference
    Lx = 2.0 * np.pi
    kx_1d = jnp.fft.fftfreq(Nx) * Nx * 2 * jnp.pi / Lx
    n = jnp.arange(Nn, dtype=jnp.float64)
    ode = HermiteFourierODE(
        Nn=Nn, Nm=1, Np=1, Nx=Nx,
        kx_grid=kx_1d[None, :, None],
        ky_grid=jnp.zeros((1, Nx, 1)),
        kz_grid=jnp.zeros((1, Nx, 1)),
        k2_grid=(kx_1d**2)[None, :, None],
        Lx=Lx, Ly=1.0, Lz=1.0,
        col=jnp.zeros((1, 1, Nn)),
        sqrt_n_plus=jnp.sqrt(n + 1.0), sqrt_n_minus=jnp.sqrt(n),
        sqrt_m_plus=jnp.array([1.0]), sqrt_m_minus=jnp.array([0.0]),
        sqrt_p_plus=jnp.array([1.0]), sqrt_p_minus=jnp.array([0.0]),
        mask23=jnp.ones((1, Nx, 1)),
    )
    C_real = jnp.fft.ifft(Ck, axis=-1, norm="forward")
    C6 = C_real[None, None, :, None, :, None]
    F6 = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128).at[0, 0, :, 0].set(F)
    out_ref = ode._compute_lorentz_rhs(
        C=C6, F=F6, alpha=jnp.array([alpha, 1.0, 1.0]), u=jnp.zeros(3),
        q=q_over_m, Omega_ce_tau=1.0, m=1.0,
    )
    out_ref = np.asarray(out_ref[0, 0, :, 0, :, 0])

    np.testing.assert_allclose(out_hp, out_ref, atol=1e-12)
