"""Co-moving wake diagnostics for the EM-PIC-1D solver.

Utilities to view the longitudinal field in the frame co-moving with a driver
(beam or laser) at velocity ``v_b``, and to extract the **transformer ratio** —
the figure of merit for PWFA drive-bunch shaping.

Written with ``jax.numpy`` so the transformer ratio is differentiable end-to-end
(the Inc 4 optimization objective).
"""

import jax
from jax import numpy as jnp


def face_positions(nx: int, dx: float, xmin: float) -> jnp.ndarray:
    """Physical positions of the face-centered field samples."""
    return xmin + (jnp.arange(nx) + 0.5) * dx


def to_comoving(
    x_face: jnp.ndarray,
    e_z: jnp.ndarray,
    v_b: float,
    t: float,
    length: float,
    center: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map the field to the co-moving coordinate ``ξ = x - v_b·t``.

    ``ξ`` is wrapped into a length-``length`` window centered on ``center`` (the
    driver's co-moving position) and the samples are returned sorted by ``ξ``.
    """
    xi = x_face - v_b * t
    xi = ((xi - center + 0.5 * length) % length) - 0.5 * length + center
    order = jnp.argsort(xi)
    return xi[order], e_z[order]


def transformer_ratio(
    xi: jnp.ndarray,
    e_z: jnp.ndarray,
    beam_center: float,
    beam_halfwidth: float,
) -> jnp.ndarray:
    """Transformer ratio ``R = max|E_z| behind the driver / max|E_z| within it``.

    ``xi`` is the co-moving coordinate (driver at ``beam_center``); ``behind`` is
    ``ξ < beam_center - beam_halfwidth`` (smaller ``x`` trails a driver moving in
    +x). For a symmetric driver ``R ≤ 2``; shaped drivers can exceed it.
    """
    inside = jnp.abs(xi - beam_center) < beam_halfwidth
    behind = xi < (beam_center - beam_halfwidth)
    decel = jnp.max(jnp.where(inside, jnp.abs(e_z), 0.0))
    accel = jnp.max(jnp.where(behind, jnp.abs(e_z), 0.0))
    return accel / decel


def _soft_max(vals: jnp.ndarray, mask: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Smooth approximation to ``max(vals[mask])`` via log-sum-exp.

    Differentiable everywhere with dense gradients (unlike ``jnp.max``); recovers
    the hard max as ``beta → ∞``.
    """
    masked = jnp.where(mask, vals, -jnp.inf)
    return jax.scipy.special.logsumexp(beta * masked) / beta


def soft_transformer_ratio(
    xi: jnp.ndarray,
    e_z: jnp.ndarray,
    beam_center: float,
    beam_halfwidth: float,
    beta: float = 30.0,
) -> jnp.ndarray:
    """Smooth transformer ratio for use as a gradient-based optimization objective.

    Same definition as :func:`transformer_ratio` but with log-sum-exp soft maxima
    so the gradient is dense. ``beta`` sets the sharpness (larger ⇒ closer to the
    hard ratio).
    """
    abs_e = jnp.abs(e_z)
    inside = jnp.abs(xi - beam_center) < beam_halfwidth
    behind = xi < (beam_center - beam_halfwidth)
    return _soft_max(abs_e, behind, beta) / _soft_max(abs_e, inside, beta)
