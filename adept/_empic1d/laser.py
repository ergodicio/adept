"""Laser injection for the EM-PIC-1D solver.

A **soft source**: a localized transverse current ``j_y(x, t)`` added to the
Ampère update each step (see ``em_step``'s ``j_y_source`` argument). A current
antenna radiates EM waves symmetrically in ±x; in 1D this is the simplest way to
launch a laser pulse of a chosen frequency and envelope into the box.

Boundaries are periodic (no absorbing layer yet — use a large box and stop
before the pulse wraps, as for the PWFA wake studies). A Mur ABC is a planned
downstream refinement.
"""

from jax import numpy as jnp


def soft_source_jy(
    x_nodes: jnp.ndarray,
    t: float,
    *,
    x0: float,
    sigma_x: float,
    omega0: float,
    amplitude: float,
    t0: float,
    tau: float,
) -> jnp.ndarray:
    """Transverse-current antenna at ``x0`` emitting a Gaussian-envelope tone.

    ``j_y(x, t) = amplitude · exp(-(x-x0)²/2σ_x²) · sin(ω0 (t-t0)) ·
                  exp(-((t-t0)/τ)²)``

    Args:
        x_nodes: node positions where ``E_y``/``j_y`` live.
        t: current time.
        x0, sigma_x: antenna center and spatial width.
        omega0: laser (carrier) angular frequency.
        amplitude: source-current amplitude (sets the launched field amplitude).
        t0, tau: temporal envelope center and width.

    Returns:
        ``j_y`` source array on the node grid at time ``t``.
    """
    spatial = jnp.exp(-((x_nodes - x0) ** 2) / (2.0 * sigma_x**2))
    carrier = jnp.sin(omega0 * (t - t0))
    envelope = jnp.exp(-(((t - t0) / tau) ** 2))
    return amplitude * spatial * carrier * envelope
