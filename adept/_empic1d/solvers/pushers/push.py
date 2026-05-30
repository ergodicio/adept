"""Relativistic Higuera–Cary particle pusher.

Reference: A. V. Higuera & J. R. Cary, "Structure-preserving second-order
integration of relativistic charged particle trajectories in electromagnetic
fields", Phys. Plasmas 24, 052104 (2017).

The Higuera–Cary (HC) scheme advances momentum from the half-integer levels
``u_{n-1/2} -> u_{n+1/2}`` using the fields at integer level ``n``. It is a
second-order, phase-space-volume-preserving leapfrog. Like the relativistic
Boris push it splits each step into

    half electric kick  →  magnetic rotation  →  half electric kick,

and the magnetic rotation is an *exact* rotation of the momentum (so energy is
conserved when ``E = 0``). HC differs from Boris only in the Lorentz factor
used to set the rotation angle: instead of ``γ⁻ = γ(u⁻)`` it uses an averaged
factor ``γ*`` obtained from a closed-form solution that makes the map
volume-preserving and gives the correct E×B drift.

Conventions
-----------
- We evolve the *proper velocity* ``u = γ v`` (same units as velocity), so the
  Lorentz factor is ``γ = sqrt(1 + |u|² / c²)`` and ``v = u / γ``.
- Vectors carry their components on the last axis, size 3 ``(x, y, z)``, with an
  arbitrary leading particle shape. For the 1D2V solver only ``(u_x, u_y)``,
  ``(E_x, E_y)`` and ``B_z`` are non-zero, but the kernel is written for general
  3-vectors so it also serves a future 1D3V (circular-polarization) extension.
- ``qm = q / m`` is the (scalar, per-species) charge-to-mass ratio.
- The magnetic rotation vector ``τ = (qm dt / 2) B`` is dimensionless; ``c``
  only enters through ``γ``.
"""

from jax import numpy as jnp


def lorentz_gamma(u: jnp.ndarray, c: float) -> jnp.ndarray:
    """Lorentz factor ``γ = sqrt(1 + |u|²/c²)`` for proper velocity ``u``.

    Args:
        u: proper-velocity array ``(..., 3)`` (``u = γ v``).
        c: speed of light in the solver's units.

    Returns:
        ``γ`` with the leading (particle) shape of ``u``.
    """
    return jnp.sqrt(1.0 + jnp.sum(u * u, axis=-1) / c**2)


def higuera_cary_momentum(
    u: jnp.ndarray,
    E: jnp.ndarray,
    B: jnp.ndarray,
    qm: float,
    dt: float,
    c: float,
) -> jnp.ndarray:
    """Advance proper velocity one step with the Higuera–Cary push.

    Implements ``u_{n-1/2} -> u_{n+1/2}`` given the fields ``E``, ``B`` sampled
    at the particle position at integer time ``n``.

    Args:
        u: proper velocity ``(..., 3)`` at level ``n-1/2``.
        E: electric field ``(..., 3)`` at the particle, level ``n``.
        B: magnetic field ``(..., 3)`` at the particle, level ``n``.
        qm: charge-to-mass ratio ``q/m`` (scalar).
        dt: time step.
        c: speed of light.

    Returns:
        Proper velocity ``(..., 3)`` at level ``n+1/2``.
    """
    eps = (0.5 * dt * qm) * E  # half electric impulse (velocity units)
    tau = (0.5 * dt * qm) * B  # magnetic rotation vector (dimensionless)

    # 1. First half electric kick.
    u_minus = u + eps
    gamma_minus = lorentz_gamma(u_minus, c)

    # 2. Higuera–Cary averaged Lorentz factor for the rotation.
    #    σ = γ⁻² − |τ|² ;  γ*² = (σ + sqrt(σ² + 4(|τ|² + (τ·u⁻)²/c²))) / 2
    tau_sq = jnp.sum(tau * tau, axis=-1)
    tau_dot_u = jnp.sum(tau * u_minus, axis=-1)
    sigma = gamma_minus**2 - tau_sq
    gamma_star = jnp.sqrt(0.5 * (sigma + jnp.sqrt(sigma**2 + 4.0 * (tau_sq + (tau_dot_u / c) ** 2))))

    # 3. Exact magnetic rotation by t = τ / γ* (norm-preserving).
    t = tau / gamma_star[..., None]
    t_sq = jnp.sum(t * t, axis=-1)
    s = 1.0 / (1.0 + t_sq)
    t_dot_u = jnp.sum(t * u_minus, axis=-1)
    u_m = s[..., None] * (u_minus + t_dot_u[..., None] * t + jnp.cross(u_minus, t))

    # 4. Second half electric kick (with the HC magnetic-correction term).
    return u_m + eps + jnp.cross(u_m, t)


def advance_position_x(
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
    c: float,
    xmin: float,
    length: float,
) -> jnp.ndarray:
    """Drift the (1D) position with the longitudinal velocity ``v_x = u_x/γ``.

    Only the ``x`` component advances the spatial coordinate in 1D; the
    transverse momentum ``u_y`` is carried but does not move the particle.
    Positions are wrapped periodically onto ``[xmin, xmin + length)``.

    Args:
        x: particle positions ``(...,)``.
        u: proper velocity ``(..., 3)`` (at level ``n+1/2``).
        dt: time step.
        c: speed of light.
        xmin: lower domain edge.
        length: domain length ``xmax - xmin``.

    Returns:
        Updated, periodically wrapped positions ``(...,)``.
    """
    v_x = u[..., 0] / lorentz_gamma(u, c)
    x_new = x + dt * v_x
    return jnp.mod(x_new - xmin, length) + xmin
