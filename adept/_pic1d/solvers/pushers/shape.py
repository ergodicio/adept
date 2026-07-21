"""Particle shape functions (B-splines) for charge deposition and field gather.

We use the standard cell-centered grid convention used by the Vlasov-1D solver:
``x_g = xmin + (g + 0.5) * dx`` for ``g = 0..nx-1`` with periodic boundaries.

For a particle at position ``x_p``, define ``s_real = (x_p - xmin)/dx - 0.5`` —
the (fractional) "grid index" of the particle. The B-spline weights below
are evaluated relative to the chosen reference cell ``ig``.

All routines are JIT-compatible: the ``shape`` argument is a static Python
string that selects a Python-time branch.
"""

from jax import numpy as jnp


def _shape_indices_weights(x: jnp.ndarray, dx: float, xmin: float, nx: int, shape: str):
    """Return (indices, weights) arrays of shape (S, N_particles) where S is
    the stencil size (2 for linear, 3 for TSC, 4 for cubic). Indices are
    already wrapped modulo ``nx`` for periodic boundaries.
    """
    s_real = (x - xmin) / dx - 0.5

    if shape == "linear":
        ig = jnp.floor(s_real).astype(jnp.int32)
        s = s_real - ig  # in [0, 1)
        idx = jnp.stack([ig, ig + 1], axis=0) % nx
        w = jnp.stack([1.0 - s, s], axis=0)
        return idx, w

    if shape == "tsc":
        ig = jnp.round(s_real).astype(jnp.int32)
        s = s_real - ig  # in [-0.5, 0.5]
        w_left = 0.5 * (0.5 - s) ** 2
        w_cent = 0.75 - s**2
        w_right = 0.5 * (0.5 + s) ** 2
        idx = jnp.stack([ig - 1, ig, ig + 1], axis=0) % nx
        w = jnp.stack([w_left, w_cent, w_right], axis=0)
        return idx, w

    if shape == "cubic":
        ig = jnp.floor(s_real).astype(jnp.int32)
        s = s_real - ig  # in [0, 1)
        # Cubic B-spline weights for cells at offsets {-1, 0, 1, 2} from ig
        w_m1 = (1.0 - s) ** 3 / 6.0
        w_0 = 2.0 / 3.0 - s**2 + s**3 / 2.0
        w_p1 = 2.0 / 3.0 - (1.0 - s) ** 2 + (1.0 - s) ** 3 / 2.0
        w_p2 = s**3 / 6.0
        idx = jnp.stack([ig - 1, ig, ig + 1, ig + 2], axis=0) % nx
        w = jnp.stack([w_m1, w_0, w_p1, w_p2], axis=0)
        return idx, w

    raise ValueError(f"Unknown particle shape: {shape!r}")


def deposit(
    x: jnp.ndarray,
    w_particles: jnp.ndarray,
    nx: int,
    dx: float,
    xmin: float,
    shape: str = "tsc",
) -> jnp.ndarray:
    """Scatter particle weights onto a periodic grid using B-spline shape.

    Args:
        x: particle positions (N_p,).
        w_particles: particle "macro-weights" (N_p,), i.e. the number-density
            contribution per particle. The grid output has units of
            ``[w_particles] / dx`` (number density).
        nx, dx, xmin: spatial grid parameters.
        shape: one of ``"linear" | "tsc" | "cubic"``.

    Returns:
        Density-like array (nx,) = ``(1/dx) Σ_p w_p W(x_g - x_p)``.
    """
    idx, w = _shape_indices_weights(x, dx, xmin, nx, shape)
    out = jnp.zeros(nx)
    # idx, w have shape (S, Np). Sum each stencil entry with one scatter.
    for k in range(idx.shape[0]):
        out = out.at[idx[k]].add(w[k] * w_particles)
    return out / dx


def gather(
    field: jnp.ndarray,
    x: jnp.ndarray,
    dx: float,
    xmin: float,
    shape: str = "tsc",
) -> jnp.ndarray:
    """Interpolate ``field`` (nx,) onto particle positions using same B-spline."""
    nx = field.shape[0]
    idx, w = _shape_indices_weights(x, dx, xmin, nx, shape)
    out = jnp.zeros_like(x)
    for k in range(idx.shape[0]):
        out = out + w[k] * field[idx[k]]
    return out
