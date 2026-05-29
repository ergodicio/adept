"""One explicit step of the longitudinal relativistic EM-PIC-1D scheme.

This is the "electrostatic-via-Ampère" path: only ``E_x`` is evolved (transverse
``(E_y, B_z)`` via Yee FDTD come in a later increment, so ``B = 0`` and the
transverse momentum ``u_y`` is inert here). The leapfrog ordering is the
canonical explicit EM-PIC cycle

    gather E^n → Higuera–Cary momentum push (n-½ → n+½)
                → position drift (n → n+1)
                → charge-conserving current j^{n+½}
                → Ampère E^{n+1} = E^n - dt·j^{n+½},

which preserves Gauss's law to machine precision (see
:mod:`adept._empic1d.solvers.pushers.field`).

State is a flat pytree dict so it composes with ``jax.lax.scan`` (and, later, a
diffrax ``ODETerm``):

    {"x": (N,), "u": (N, 3), "w": (N,), "E": (nx,)}   # E on faces
"""

from jax import numpy as jnp

from adept._empic1d.solvers.pushers.field import (
    charge_conserving_current,
    charge_density_nodes,
    gather_ex_faces,
)
from adept._empic1d.solvers.pushers.push import (
    advance_position_x,
    higuera_cary_momentum,
    lorentz_gamma,
)


def longitudinal_step(
    state: dict,
    *,
    charge: float,
    qm: float,
    dt: float,
    c: float,
    nx: int,
    dx: float,
    xmin: float,
    length: float,
    shape: str,
) -> dict:
    """Advance ``{x, u, w, E}`` one step. ``E`` is the face-centered ``E_x``."""
    x, u, w, e_face = state["x"], state["u"], state["w"], state["E"]

    # Gather E^n and push momentum n-½ → n+½ (B = 0 in the longitudinal path).
    ex_p = gather_ex_faces(e_face, x, dx, xmin, shape)
    e_vec = jnp.stack([ex_p, jnp.zeros_like(ex_p), jnp.zeros_like(ex_p)], axis=-1)
    u_new = higuera_cary_momentum(u, e_vec, jnp.zeros_like(e_vec), qm, dt, c)

    # Charge before and after the position drift, for the charge-conserving current.
    rho_old = charge_density_nodes(x, w, charge, nx, dx, xmin, shape)
    x_new = advance_position_x(x, u_new, dt, c, xmin, length)
    rho_new = charge_density_nodes(x_new, w, charge, nx, dx, xmin, shape)

    v_x = u_new[..., 0] / lorentz_gamma(u_new, c)
    mean_current = charge * jnp.sum(w * v_x) / (nx * dx)
    j_face = charge_conserving_current(rho_old, rho_new, mean_current, dx, dt)

    e_new = e_face - dt * j_face
    return {"x": x_new, "u": u_new, "w": w, "E": e_new}
