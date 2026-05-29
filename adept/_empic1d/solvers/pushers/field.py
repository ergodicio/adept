"""Longitudinal field kernels for EM-PIC-1D.

Staggered charge deposition, a charge-conserving (Esirkepov-equivalent in 1D)
current, the Ampère ``E_x`` update, and an initial Gauss solve.

Grid staggering (Yee-compatible, so the transverse Yee solver slots in later):

- **nodes** ``i``      at ``xmin + i·dx``        carry charge density ``ρ``
- **faces** ``i+½`` at ``xmin + (i+½)·dx`` carry ``E_x`` and ``j_x``

so node ``i`` sits between faces ``i-1`` and ``i`` and the discrete Gauss law is
``(E_face[i] - E_face[i-1]) / dx = ρ_node[i]``.

We reuse the B-spline shape functions from :mod:`adept._pic1d.solvers.pushers.shape`.
Because that shared ``deposit``/``gather`` places its grid points at
``(g+½)·dx + origin``, depositing charge to **nodes** uses a half-cell-shifted
origin ``xmin - dx/2`` while gathering face-centered ``E_x`` uses ``xmin``.

Charge conservation
-------------------
With the current built from the per-step node-charge change,

    j_face = -(dx/dt)·cumsum(ρ_new - ρ_old)  (+ uniform drift current),

the Ampère update ``E ← E - dt·j`` satisfies ``D_x E^{n+1} = ρ^{n+1}`` exactly
whenever ``D_x E^n = ρ^n`` — the discrete divergence difference telescopes to
``ρ_new - ρ_old`` independent of the uniform (k=0) part. The uniform part is the
physical net current ``⟨j⟩``, which carries beam drift and must be supplied.
"""

from jax import numpy as jnp

from adept._pic1d.solvers.pushers.shape import deposit, gather


def charge_density_nodes(
    x: jnp.ndarray,
    w: jnp.ndarray,
    charge: float,
    nx: int,
    dx: float,
    xmin: float,
    shape: str,
) -> jnp.ndarray:
    """Deposit ``charge · Σ_p w_p S(x_node - x_p)`` onto the node grid."""
    return charge * deposit(x, w, nx, dx, xmin - 0.5 * dx, shape)


def gather_ex_faces(
    e_face: jnp.ndarray,
    x: jnp.ndarray,
    dx: float,
    xmin: float,
    shape: str,
) -> jnp.ndarray:
    """Interpolate the face-centered longitudinal field onto particles."""
    return gather(e_face, x, dx, xmin, shape)


def solve_ex_from_gauss(rho_total: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Initial ``E_x`` on faces from the discrete Gauss law, zero spatial mean.

    Solves ``(E[i] - E[i-1])/dx = ρ_total[i]`` periodically. Requires
    ``Σ ρ_total = 0`` (a neutralizing background), which makes the periodic
    solve consistent.
    """
    e_face = dx * jnp.cumsum(rho_total)
    return e_face - jnp.mean(e_face)


def charge_conserving_current(
    rho_old: jnp.ndarray,
    rho_new: jnp.ndarray,
    mean_current: float,
    dx: float,
    dt: float,
) -> jnp.ndarray:
    """Face current from the node-charge change, with the uniform part set to
    the physical net current ``mean_current = ⟨j_x⟩``."""
    drho = rho_new - rho_old
    j_face = -(dx / dt) * jnp.cumsum(drho)
    return j_face - jnp.mean(j_face) + mean_current


def divergence_ex(e_face: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Discrete Gauss divergence at nodes: ``(E_face[i] - E_face[i-1]) / dx``."""
    return (e_face - jnp.roll(e_face, 1)) / dx
