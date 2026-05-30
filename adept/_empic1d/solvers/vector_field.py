"""One explicit step of the longitudinal relativistic EM-PIC-1D scheme.

This is the "electrostatic-via-Ampère" path: only ``E_x`` is evolved (transverse
``(E_y, B_z)`` via Yee FDTD come in a later increment, so ``B = 0`` and the
transverse momentum ``u_y`` is inert here). The leapfrog ordering is the
canonical explicit EM-PIC cycle

    gather E^n → Higuera–Cary momentum push (n-½ → n+½)
                → position drift (n → n+1)
                → charge-conserving current j^{n+½} (summed over species)
                → Ampère E^{n+1} = E^n - dt·j^{n+½},

which preserves Gauss's law to machine precision (see
:mod:`adept._empic1d.solvers.pushers.field`).

State is a flat pytree dict so it composes with ``jax.lax.scan`` (and, later, a
diffrax ``ODETerm``). Multiple species (e.g. a background plasma plus a PWFA
drive beam) share the field; each carries its own particle arrays::

    {"species": {name: {"x": (N,), "u": (N, 3), "w": (N,)}}, "E": (nx,)}  # E on faces

``species_params[name]`` provides each species' ``charge`` and ``qm = q/m``.
"""

from jax import numpy as jnp

from adept._empic1d.solvers.pushers.field import (
    advance_bz_faces,
    advance_ey_nodes,
    charge_conserving_current,
    charge_density_nodes,
    deposit_jy_nodes,
    gather_ex_faces,
    gather_nodes,
)
from adept._empic1d.solvers.pushers.push import (
    advance_position_x,
    higuera_cary_momentum,
    lorentz_gamma,
)


def longitudinal_step(
    state: dict,
    *,
    species_params: dict,
    dt: float,
    c: float,
    nx: int,
    dx: float,
    xmin: float,
    length: float,
    shape: str,
) -> dict:
    """Advance ``{species, E}`` one step. ``E`` is the face-centered ``E_x``."""
    e_face = state["E"]

    new_species = {}
    rho_old_total = jnp.zeros(nx)
    rho_new_total = jnp.zeros(nx)
    mean_current = 0.0

    for name, p in state["species"].items():
        charge = species_params[name]["charge"]
        qm = species_params[name]["qm"]

        # Gather E^n and push momentum n-½ → n+½ (B = 0 in the longitudinal path).
        ex_p = gather_ex_faces(e_face, p["x"], dx, xmin, shape)
        e_vec = jnp.stack([ex_p, jnp.zeros_like(ex_p), jnp.zeros_like(ex_p)], axis=-1)
        u_new = higuera_cary_momentum(p["u"], e_vec, jnp.zeros_like(e_vec), qm, dt, c)

        rho_old_total += charge_density_nodes(p["x"], p["w"], charge, nx, dx, xmin, shape)
        x_new = advance_position_x(p["x"], u_new, dt, c, xmin, length)
        rho_new_total += charge_density_nodes(x_new, p["w"], charge, nx, dx, xmin, shape)

        v_x = u_new[..., 0] / lorentz_gamma(u_new, c)
        mean_current += charge * jnp.sum(p["w"] * v_x) / (nx * dx)

        new_species[name] = {"x": x_new, "u": u_new, "w": p["w"]}

    j_face = charge_conserving_current(rho_old_total, rho_new_total, mean_current, dx, dt)
    e_new = e_face - dt * j_face
    return {"species": new_species, "E": e_new}


def em_step(
    state: dict,
    *,
    species_params: dict,
    dt: float,
    c: float,
    nx: int,
    dx: float,
    xmin: float,
    length: float,
    shape: str,
    j_y_source: jnp.ndarray | None = None,
) -> dict:
    """Full relativistic EM step: longitudinal (Ampère/Esirkepov ``E_x``) + transverse
    Yee ``(E_y, B_z)``, coupled through the Higuera–Cary push.

    State adds the transverse fields to the longitudinal one::

        {"species": {name: {x, u, w}}, "E": E_x(faces), "Ey": E_y(nodes), "Bz": B_z(faces)}

    Fields ``E_x, E_y`` are at integer time ``n``; ``B_z`` at ``n-½``. ``B_z`` is
    half-advanced to integer time for the particle gather, the particles are
    pushed under the full ``(E_x, E_y, B_z)``, currents are deposited, and the
    fields are advanced to ``n+1`` (``E``) / ``n+½`` (``B_z``).
    """
    e_x, e_y, b_z = state["E"], state["Ey"], state["Bz"]

    # B_z to integer time for the gather (Faraday half-step with E_y^n).
    b_z_n = advance_bz_faces(b_z, e_y, dx, 0.5 * dt)

    new_species = {}
    rho_old_total = jnp.zeros(nx)
    rho_new_total = jnp.zeros(nx)
    mean_jx = 0.0
    j_y = jnp.zeros(nx)

    for name, p in state["species"].items():
        charge = species_params[name]["charge"]
        qm = species_params[name]["qm"]

        ex_p = gather_ex_faces(e_x, p["x"], dx, xmin, shape)
        ey_p = gather_nodes(e_y, p["x"], dx, xmin, shape)
        bz_p = gather_ex_faces(b_z_n, p["x"], dx, xmin, shape)  # B_z is face-centered like E_x
        e_vec = jnp.stack([ex_p, ey_p, jnp.zeros_like(ex_p)], axis=-1)
        b_vec = jnp.stack([jnp.zeros_like(bz_p), jnp.zeros_like(bz_p), bz_p], axis=-1)
        u_new = higuera_cary_momentum(p["u"], e_vec, b_vec, qm, dt, c)

        rho_old_total += charge_density_nodes(p["x"], p["w"], charge, nx, dx, xmin, shape)
        x_new = advance_position_x(p["x"], u_new, dt, c, xmin, length)
        rho_new_total += charge_density_nodes(x_new, p["w"], charge, nx, dx, xmin, shape)

        gamma = lorentz_gamma(u_new, c)
        mean_jx += charge * jnp.sum(p["w"] * u_new[..., 0] / gamma) / (nx * dx)
        j_y += deposit_jy_nodes(x_new, p["w"], u_new[..., 1] / gamma, charge, nx, dx, xmin, shape)

        new_species[name] = {"x": x_new, "u": u_new, "w": p["w"]}

    # Optional external transverse-current antenna (laser soft source).
    if j_y_source is not None:
        j_y = j_y + j_y_source

    # Longitudinal Ampère (charge-conserving) and transverse Yee advance.
    j_x = charge_conserving_current(rho_old_total, rho_new_total, mean_jx, dx, dt)
    e_x_new = e_x - dt * j_x
    b_z_np12 = advance_bz_faces(b_z_n, e_y, dx, 0.5 * dt)  # complete Faraday → n+½
    e_y_new = advance_ey_nodes(e_y, b_z_np12, j_y, c, dx, dt)

    return {"species": new_species, "E": e_x_new, "Ey": e_y_new, "Bz": b_z_np12}
