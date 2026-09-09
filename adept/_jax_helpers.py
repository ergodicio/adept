"""Small numerical helpers that do not import ADEPT runtime services."""

import jax.numpy as jnp


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    """Return the historical two-sided hyperbolic-tangent envelope."""

    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))


__all__ = ["get_envelope"]
