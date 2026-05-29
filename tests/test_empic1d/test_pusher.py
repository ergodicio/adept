"""Analytic single-particle validation of the Higuera–Cary pusher.

These exercise the relativistic momentum update against closed-form orbits:

1. **Static E field** — constant longitudinal force gives ``u_x = (q/m) E_x t``
   exactly (leapfrog is exact for a constant force) and the Lorentz factor
   follows ``γ = sqrt(1 + u_x²/c²)``.
2. **Pure B field** — the magnetic rotation conserves energy to machine
   precision (``γ`` constant), validating that the rotation is a true rotation.
3. **Gyrofrequency** — the small-step rotation rate matches the *relativistic*
   gyrofrequency ``ω_c = (q/m) B / γ``, validating the HC Lorentz factor.
4. **E×B drift** — a particle released from rest in crossed fields drifts at the
   ``E×B/B²`` velocity.
5. **Free streaming** — with no fields, momentum is unchanged and the position
   advances at ``v_x = u_x/γ``.

``c`` is set to a finite value so the relativistic regime is reachable with
order-unity momenta. ``jax_enable_x64`` is on for the test suite (see
``tests/conftest.py``), so machine-precision assertions are at the 1e-12 level.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from adept._empic1d.solvers.pushers.push import (
    advance_position_x,
    higuera_cary_momentum,
    lorentz_gamma,
)

C = 10.0  # speed of light in solver units


def _integrate_momentum(u0, E, B, qm, dt, n_steps, c=C):
    """Scan the HC momentum push under static fields; return (u_final, traj)."""

    def step(u, _):
        u_next = higuera_cary_momentum(u, E, B, qm, dt, c)
        return u_next, u_next

    return jax.lax.scan(step, u0, None, length=n_steps)


def test_static_efield_relativistic_energy_gain():
    qm, dt, n = -1.0, 0.01, 500
    E = jnp.array([0.3, 0.0, 0.0])
    B = jnp.zeros(3)
    _, traj = _integrate_momentum(jnp.zeros(3), E, B, qm, dt, n)
    traj = np.asarray(traj)

    # Constant force ⇒ u_x increments by (q/m) E_x dt every step, exactly.
    expected_ux = qm * float(E[0]) * dt * np.arange(1, n + 1)
    assert np.allclose(traj[:, 0], expected_ux, atol=1e-10)
    assert np.allclose(traj[:, 1:], 0.0, atol=1e-12)

    # Lorentz factor and subluminal speed.
    gamma = np.asarray(lorentz_gamma(jnp.asarray(traj), C))
    assert np.allclose(gamma, np.sqrt(1.0 + expected_ux**2 / C**2), atol=1e-10)
    assert np.all(np.abs(traj[:, 0] / gamma) < C)


def test_pure_bfield_conserves_energy():
    qm, dt, n = -1.0, 0.05, 4000
    E = jnp.zeros(3)
    B = jnp.array([0.0, 0.0, 1.0])
    u_perp = C * np.sqrt(3.0)  # γ = 2
    _, traj = _integrate_momentum(jnp.array([u_perp, 0.0, 0.0]), E, B, qm, dt, n)
    traj = np.asarray(traj)

    gamma = np.asarray(lorentz_gamma(jnp.asarray(traj), C))
    assert np.allclose(gamma, 2.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(traj, axis=1), u_perp, rtol=1e-9)


def test_relativistic_gyrofrequency():
    qm, dt = -1.0, 0.01
    B0 = 1.0
    E = jnp.zeros(3)
    B = jnp.array([0.0, 0.0, B0])
    gamma0 = 2.0
    u_perp = C * np.sqrt(gamma0**2 - 1.0)

    u1 = higuera_cary_momentum(jnp.array([u_perp, 0.0, 0.0]), E, B, qm, dt, C)
    theta = float(jnp.arctan2(u1[1], u1[0]))  # signed rotation in one step
    omega_measured = theta / dt
    # du/dt = (q/m) v×B = ω×u with ω = -(q/m)(B/γ) ẑ, so the signed rotation
    # rate about +ẑ is the relativistic gyrofrequency -(q/m) B / γ.
    omega_c = -qm * B0 / gamma0

    assert abs(omega_measured - omega_c) / abs(omega_c) < 1e-4


def test_exb_drift_from_rest():
    qm, dt, n = -1.0, 0.005, 40000
    E0, B0 = 2.0, 1.0  # E/(cB) = 0.2 ⇒ subluminal drift
    E = jnp.array([E0, 0.0, 0.0])
    B = jnp.array([0.0, 0.0, B0])
    _, traj = _integrate_momentum(jnp.zeros(3), E, B, qm, dt, n)
    traj = np.asarray(traj)

    gamma = np.asarray(lorentz_gamma(jnp.asarray(traj), C))
    v = traj / gamma[:, None]
    v_drift_y = -E0 / B0  # (E×B/B²)_y

    # Time-averaged velocity equals the E×B drift; transverse-to-drift mean ≈ 0.
    assert abs(v[:, 1].mean() - v_drift_y) / abs(v_drift_y) < 0.02
    assert abs(v[:, 0].mean()) < 0.05 * abs(v_drift_y)


def test_free_streaming():
    qm, dt, n = -1.0, 0.1, 100
    u0 = jnp.array([3.0, 1.0, 0.0])
    _, traj = _integrate_momentum(u0, jnp.zeros(3), jnp.zeros(3), qm, dt, n)
    assert np.allclose(np.asarray(traj), np.asarray(u0)[None, :], atol=1e-12)

    # Position drifts at v_x = u_x/γ (large box ⇒ no wrap).
    gamma = float(lorentz_gamma(u0, C))
    x = jnp.zeros(1)
    for _ in range(n):
        x = advance_position_x(x, u0[None, :], dt, C, 0.0, 1.0e6)
    assert np.allclose(np.asarray(x), n * dt * float(u0[0]) / gamma, atol=1e-9)
