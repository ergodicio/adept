"""Fixed-panel FARSIGHT numerics, independently implemented in JAX.

The method follows Sandberg, Krasny & Thomas, JCP 523 (2025), 113664:
regularized periodic field quadrature, coupled RK4 characteristics, and a
physical-coordinate biquadratic fit on each deformed nine-node panel. No
positivity clipping or invariant renormalization is performed.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def make_mesh(nx, nv, xmin, xmax, vmin, vmax, quadrature="trapezoid"):
    """Make endpoint-inclusive nodes and assembled panel quadrature weights.

    ``nx`` and ``nv`` count intervals; each 3x3 panel spans two intervals in
    either direction. Both x endpoints are retained for panel connectivity,
    with half endpoint weights so periodic sources are not double-counted.
    """
    if isinstance(nx, bool) or isinstance(nv, bool) or int(nx) != nx or int(nv) != nv:
        raise ValueError("nx and nv must be integers")
    if nx < 4 or nv < 2 or nx % 2 or nv % 2:
        raise ValueError("nx must be even and >= 4; nv must be even and >= 2")
    if not np.isfinite([xmin, xmax, vmin, vmax]).all() or xmax <= xmin or vmax <= vmin:
        raise ValueError("mesh bounds must be finite and increasing")
    if quadrature not in ("trapezoid", "simpson"):
        raise ValueError("quadrature must be 'trapezoid' or 'simpson'")

    def axis_weights(n, spacing):
        if quadrature == "trapezoid":
            weights = jnp.ones(n + 1).at[jnp.array([0, n])].set(0.5)
            return weights * spacing
        weights = jnp.where(jnp.arange(n + 1) % 2 == 1, 4.0, 2.0)
        return weights.at[jnp.array([0, n])].set(1.0) * (spacing / 3)

    x, v = jnp.meshgrid(jnp.linspace(xmin, xmax, nx + 1), jnp.linspace(vmin, vmax, nv + 1), indexing="ij")
    weights = axis_weights(nx, (xmax - xmin) / nx)[:, None] * axis_weights(nv, (vmax - vmin) / nv)[None, :]
    return x, v, weights


def periodic_kernel(displacement, length, epsilon):
    """Odd, mean-zero softened periodic Poisson kernel; epsilon is a length."""
    r = displacement / length
    r = r - jnp.floor(r + 0.5)
    a = epsilon / length
    return 0.5 * r * jnp.sqrt(1 + 4 * a**2) / jnp.sqrt(r**2 + a**2) - r


def electric_field(targets, sources, charges, length, epsilon, chunk_size=64):
    """Direct field sum with bounded target batches (O(N*M) work).

    ``charges`` already includes species charge, nodal f, and volume weights.
    A homogeneous neutralizing background contributes zero to this kernel.
    The zero charge mode is projected out even if remeshing changes the mass.
    """
    sources, charges = sources.reshape(-1), charges.reshape(-1)

    def at_target(target):
        return jnp.sum(charges * periodic_kernel(target - sources, length, epsilon))

    values = jax.lax.map(at_target, targets.reshape(-1), batch_size=min(chunk_size, targets.size))
    return values.reshape(targets.shape)


def rk4_push(x, v, charge_weights, dt, length, epsilon, charge_to_mass, chunk_size=64):
    """Advance all characteristics together, evaluating E at every RK stage.

    Keep x unwrapped so neighboring nodes describe a continuous panel across
    the periodic seam. Nodal f and material quadrature weights remain fixed.
    """

    def acceleration(position):
        return charge_to_mass * electric_field(position, position, charge_weights, length, epsilon, chunk_size)

    k1x, k1v = v, acceleration(x)
    k2x, k2v = v + 0.5 * dt * k1v, acceleration(x + 0.5 * dt * k1x)
    k3x, k3v = v + 0.5 * dt * k2v, acceleration(x + 0.5 * dt * k2x)
    k4x, k4v = v + dt * k3v, acceleration(x + dt * k3x)
    return x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6, v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6


def _panels(values):
    """Gather each shared 3x3 panel in x-major, v-minor order."""
    ix = jnp.arange(0, values.shape[0] - 1, 2)
    iv = jnp.arange(0, values.shape[1] - 1, 2)
    ix, iv = jnp.meshgrid(ix, iv, indexing="ij")
    ox, ov = jnp.meshgrid(jnp.arange(3), jnp.arange(3), indexing="ij")
    return values[ix.reshape(-1, 1) + ox.reshape(-1), iv.reshape(-1, 1) + ov.reshape(-1)]


def _basis(x, v):
    """The nine physical monomials x**i * v**j, i,j=0,1,2."""
    xp = jnp.stack((jnp.ones_like(x), x, x * x), axis=-1)
    vp = jnp.stack((jnp.ones_like(v), v, v * v), axis=-1)
    return (xp[..., :, None] * vp[..., None, :]).reshape((*x.shape, 9))


def _cross(ax, av, bx, bv):
    return ax * bv - av * bx


def remesh(x, v, f, x0, v0, *, chunk_size=64):
    """Fit physical biquadratics on deformed panels and sample reference nodes.

    Containment uses straight quadrilateral edges, as in the reference method.
    The first containing panel owns shared edges deterministically. Targets
    outside the advected velocity boundary receive zero (zero inflow).

    A convex panel must fit within one nearest-image neighborhood about its
    center. Folded, overly extended, singular or nonfinite panels mark the
    result invalid; they are never silently replaced with a lower-order fit.
    """
    length = x0[-1, 0] - x0[0, 0]
    px, pv, pf = _panels(x), _panels(v), _panels(f)
    cx, cv = px[:, 4], pv[:, 4]
    offsets_x, offsets_v = px - cx[:, None], pv - cv[:, None]
    sx = jnp.maximum(jnp.max(jnp.abs(offsets_x), axis=1), jnp.finfo(x.dtype).tiny)
    sv = jnp.maximum(jnp.max(jnp.abs(offsets_v), axis=1), jnp.finfo(v.dtype).tiny)
    ux, uv = offsets_x / sx[:, None], offsets_v / sv[:, None]
    corners = jnp.array([0, 6, 8, 2])  # counterclockwise
    qx, qv = ux[:, corners], uv[:, corners]
    ex, ev = jnp.roll(qx, -1, axis=1) - qx, jnp.roll(qv, -1, axis=1) - qv
    turns = _cross(ex, ev, jnp.roll(ex, -1, axis=1), jnp.roll(ev, -1, axis=1))
    area = 0.5 * jnp.sum(_cross(qx, qv, jnp.roll(qx, -1, axis=1), jnp.roll(qv, -1, axis=1)), axis=1) * sx * sv
    reference_area = (x0[2, 0] - x0[0, 0]) * (v0[0, 2] - v0[0, 0])
    geometry_ok = jnp.all(turns > 1e-12, axis=1) & (sx < 0.5 * length)
    matrix = _basis(ux, uv)
    # A scaled determinant guard keeps invalid panels out of the linear solve;
    # they still set the persistent failure flag and NaN output below.
    determinant = jnp.linalg.det(matrix)
    panel_ok = geometry_ok & jnp.isfinite(determinant) & (jnp.abs(determinant) > 1e-12)
    safe_matrix = jnp.where(panel_ok[:, None, None], matrix, jnp.eye(9, dtype=f.dtype))
    safe_values = jnp.where(panel_ok[:, None], pf, 0.0)
    coefficients = jnp.linalg.solve(safe_matrix, safe_values[..., None])[..., 0]
    panel_ok &= jnp.all(jnp.isfinite(coefficients), axis=1)

    def sample(target):
        tx, tv = target
        delta_x = tx - cx
        delta_x -= length * jnp.floor(delta_x / length + 0.5)
        txs, tvs = delta_x / sx, (tv - cv) / sv
        side = _cross(ex, ev, txs[:, None] - qx, tvs[:, None] - qv)
        inside = jnp.all(side >= -1e-11, axis=1) & panel_ok
        owner = jnp.argmax(inside)
        covered = jnp.any(inside)
        value = jnp.sum(_basis(txs[owner], tvs[owner]) * coefficients[owner])
        return jnp.where(covered, value, 0.0), ~covered

    targets = jnp.stack((x0.reshape(-1), v0.reshape(-1)), axis=-1)
    values, uncovered = jax.lax.map(sample, targets, batch_size=min(chunk_size, targets.shape[0]))
    # Periodic endpoints denote the same point. Use one evaluation for both,
    # avoiding seam roundoff selecting opposite sides of a shared panel edge.
    values = values.reshape(f.shape).at[-1, :].set(values.reshape(f.shape)[0, :])
    valid = jnp.all(panel_ok) & jnp.all(jnp.isfinite(values))
    values = jnp.where(valid, values, jnp.nan)
    info = {
        "uncovered_nodes": jnp.sum(uncovered, dtype=jnp.int32),
        "invalid_panels": jnp.sum(~panel_ok, dtype=jnp.int32),
        "max_panel_area_error": jnp.max(jnp.abs(area / reference_area - 1)),
        "valid": valid,
    }
    return values, info


def initial_state(x, v, f, weights):
    """Explicit particle-panel state with cumulative remeshing error budgets."""
    del weights
    zero = jnp.zeros((), dtype=f.dtype)
    return {
        "x": x,
        "v": v,
        "f": f,
        "remesh_count": jnp.asarray(0, dtype=jnp.int32),
        "remap_mass_change": zero,
        "remap_c2_change": zero,
        "remap_mass_abs_change": zero,
        "remap_c2_abs_change": zero,
        "remap_uncovered_nodes": jnp.asarray(0, dtype=jnp.int32),
        "invalid_panels": jnp.asarray(0, dtype=jnp.int32),
        "max_panel_area_error": zero,
        "valid": jnp.asarray(True),
    }


def diagnose(state, weights):
    """Material quadrature moments, negativity and accumulated remap defects.

    Between remeshes, mass and C2 are fixed-weight material sums, not a
    measurement of deformed panel volume or of fine-scale resolution.
    Momentum and kinetic energy below assume unit particle mass.
    """
    f, v = state["f"], state["v"]
    weights = state.get("weights", weights)
    min_f = jnp.min(jnp.where(state["active"][:, None], f, jnp.inf)) if "active" in state else jnp.min(f)
    return {
        "mass": jnp.sum(weights * f),
        "c2": jnp.sum(weights * f**2),
        "momentum": jnp.sum(weights * f * v),
        "kinetic_energy": 0.5 * jnp.sum(weights * f * v**2),
        "min_f": min_f,
        "negative_mass": jnp.sum(weights * jnp.maximum(-f, 0)),
        **{
            key: value
            for key, value in state.items()
            if key not in ("x", "v", "f", "weights", "active", "panel_id", "level")
        },
    }


class FarsightSystem(eqx.Module):
    """A complete gridless characteristic push and optional panel remesh."""

    x0: Any
    v0: Any
    weights: Any
    length: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    charge: float = eqx.field(static=True, default=-1.0)
    mass: float = eqx.field(static=True, default=1.0)
    remesh_every: int = eqx.field(static=True, default=1)
    chunk_size: int = eqx.field(static=True, default=64)

    def step(self, step, state, params, inputs, key):
        del params, inputs, key
        x, v = rk4_push(
            state["x"],
            state["v"],
            self.charge * self.weights * state["f"],
            self.dt,
            self.length,
            self.epsilon,
            self.charge / self.mass,
            self.chunk_size,
        )
        pushed = {**state, "x": x, "v": v}
        pushed["valid"] = state["valid"] & jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(v))
        if self.remesh_every == 0:
            return pushed

        def reset(current):
            f, info = remesh(current["x"], current["v"], current["f"], self.x0, self.v0, chunk_size=self.chunk_size)
            mass_delta = jnp.sum(self.weights * (f - current["f"]))
            c2_delta = jnp.sum(self.weights * (f**2 - current["f"] ** 2))
            return {
                **current,
                "x": self.x0,
                "v": self.v0,
                "f": f,
                "remesh_count": current["remesh_count"] + 1,
                "remap_mass_change": current["remap_mass_change"] + mass_delta,
                "remap_c2_change": current["remap_c2_change"] + c2_delta,
                "remap_mass_abs_change": current["remap_mass_abs_change"] + jnp.abs(mass_delta),
                "remap_c2_abs_change": current["remap_c2_abs_change"] + jnp.abs(c2_delta),
                "remap_uncovered_nodes": current["remap_uncovered_nodes"] + info["uncovered_nodes"],
                "invalid_panels": current["invalid_panels"] + info["invalid_panels"],
                "max_panel_area_error": jnp.maximum(current["max_panel_area_error"], info["max_panel_area_error"]),
                "valid": current["valid"] & info["valid"],
            }

        return jax.lax.cond((step + 1) % self.remesh_every == 0, reset, lambda current: current, pushed)
