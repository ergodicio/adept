#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""Lock shift_multi's direction semantics and the E-field Lorentz coupling.

These tests exist because shift_multi's docstring once stated the OPPOSITE
direction mapping. Code transcribed from that wording (rather than from the
function's behavior) shipped an inverted E·∂_v f term in a downstream solver
(_hermite_poisson_1d), which produced a spurious low-n instability while a
uniform E field drove zero current. The probes below pin the actual behavior
so documentation and code can never silently diverge again.

Physics reference: Parker & Dellar (2015). For the asymmetrically-weighted
Hermite basis, ∂φ_m/∂v = −√(2(m+1)) φ_{m+1} (differentiation raises the
index), so the projection of (q/m)E·∂_v f onto mode n draws from mode n−1:

    dC_n/dt|E = +(q/m)·√(2n)/α · E · C_{n−1}

The convention-independent acid test: a uniform E applied to a pure
Maxwellian (only C_0 ≠ 0) must drive current, dC_1/dt = (q/m)·√2/α·E·C_0,
and must excite no other mode.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from adept._spectrax1d.hermite_fourier_ode import HermiteFourierODE


def _make_ode(Nn: int = 6, Nm: int = 1, Np: int = 1, Nx: int = 8) -> HermiteFourierODE:
    """Minimal 1D-in-x HermiteFourierODE for unit probes."""
    Lx = 2.0 * np.pi
    kx_1d = jnp.fft.fftfreq(Nx) * Nx * 2 * jnp.pi / Lx
    kx_grid = kx_1d[None, :, None]  # (Ny, Nx, Nz) = (1, Nx, 1)
    zeros_grid = jnp.zeros_like(kx_grid)
    n = jnp.arange(Nn, dtype=jnp.float64)
    m = jnp.arange(Nm, dtype=jnp.float64)
    p = jnp.arange(Np, dtype=jnp.float64)
    return HermiteFourierODE(
        Nn=Nn,
        Nm=Nm,
        Np=Np,
        Nx=Nx,
        kx_grid=kx_grid,
        ky_grid=zeros_grid,
        kz_grid=zeros_grid,
        k2_grid=kx_grid**2,
        Lx=Lx,
        Ly=1.0,
        Lz=1.0,
        col=jnp.zeros((Np, Nm, Nn)),
        sqrt_n_plus=jnp.sqrt(n + 1.0),
        sqrt_n_minus=jnp.sqrt(n),
        sqrt_m_plus=jnp.sqrt(m + 1.0),
        sqrt_m_minus=jnp.sqrt(m),
        sqrt_p_plus=jnp.sqrt(p + 1.0),
        sqrt_p_minus=jnp.sqrt(p),
        mask23=jnp.ones((1, Nx, 1)),
    )


def test_shift_multi_direction_semantics():
    """dn=+1 → result[n] = C[n+1]; dn=−1 → result[n] = C[n−1]; dn=0 → identity."""
    ode = _make_ode(Nn=4, Nx=2)
    vals = jnp.array([10.0, 20.0, 30.0, 40.0])
    C = jnp.zeros((1, 1, 4, 1, 2, 1)).at[0, 0, :, 0, 0, 0].set(vals)

    up = np.asarray(ode.shift_multi(C, dn=+1)[0, 0, :, 0, 0, 0])
    down = np.asarray(ode.shift_multi(C, dn=-1)[0, 0, :, 0, 0, 0])
    same = np.asarray(ode.shift_multi(C, dn=0)[0, 0, :, 0, 0, 0])

    np.testing.assert_allclose(up, [20.0, 30.0, 40.0, 0.0])  # source at n+1
    np.testing.assert_allclose(down, [0.0, 10.0, 20.0, 30.0])  # source at n-1
    np.testing.assert_allclose(same, [10.0, 20.0, 30.0, 40.0])


def test_uniform_E_on_maxwellian_drives_current():
    """Uniform E_x on a Maxwellian responds ONLY at n=1 with (q/m)·√2/α·E·C_0."""
    Nn, Nx = 6, 8
    ode = _make_ode(Nn=Nn, Nx=Nx)

    c0 = 0.7
    E0 = 0.3
    alpha = jnp.array([1.3, 1.0, 1.0])
    u = jnp.zeros(3)
    q, m_s, Omega = -1.0, 1.0, 1.0

    # Real-space state: Maxwellian (C_0 = c0 everywhere), all higher modes zero
    C = jnp.zeros((1, 1, Nn, 1, Nx, 1), dtype=jnp.complex128).at[0, 0, 0].set(c0)
    # Uniform E_x; all other field components zero
    F = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128).at[0].set(E0)

    dCk = ode._compute_lorentz_rhs(C=C, F=F, alpha=alpha, u=u, q=q, Omega_ce_tau=Omega, m=m_s)
    dCk = np.asarray(dCk)

    # fftn(norm="forward") of a uniform field puts the value in the k=0 bin
    expected_c1 = (q / m_s) * Omega * np.sqrt(2.0) / float(alpha[0]) * E0 * c0
    np.testing.assert_allclose(dCk[0, 0, 1, 0, 0, 0].real, expected_c1, rtol=1e-12)

    # No response anywhere else (in particular: n=0 must be untouched —
    # the force term must not alter density)
    mask = np.ones_like(dCk, dtype=bool)
    mask[0, 0, 1, 0, 0, 0] = False
    assert np.max(np.abs(dCk[mask])) < 1e-14


def test_force_ladder_is_one_sided_upward():
    """Seeding only C_2 must excite only C_3 (coefficient √(2·3)/α): the AW
    force term couples n−1 → n and nothing else."""
    Nn, Nx = 6, 8
    ode = _make_ode(Nn=Nn, Nx=Nx)

    c2 = 0.4
    E0 = 0.25
    alpha = jnp.array([0.9, 1.0, 1.0])
    q, m_s, Omega = -1.0, 1.0, 1.0

    C = jnp.zeros((1, 1, Nn, 1, Nx, 1), dtype=jnp.complex128).at[0, 0, 2].set(c2)
    F = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128).at[0].set(E0)

    dCk = np.asarray(
        ode._compute_lorentz_rhs(C=C, F=F, alpha=alpha, u=jnp.zeros(3), q=q, Omega_ce_tau=Omega, m=m_s)
    )

    expected_c3 = (q / m_s) * Omega * np.sqrt(2.0 * 3.0) / float(alpha[0]) * E0 * c2
    np.testing.assert_allclose(dCk[0, 0, 3, 0, 0, 0].real, expected_c3, rtol=1e-12)

    mask = np.ones_like(dCk, dtype=bool)
    mask[0, 0, 3, 0, 0, 0] = False
    assert np.max(np.abs(dCk[mask])) < 1e-14
