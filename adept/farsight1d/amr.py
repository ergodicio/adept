"""Bounded, static-shape quadtree panels for the FARSIGHT characteristic method.

The nine-point range indicator and rebuild-from-roots policy follow the
reference method. Nearest-polygon extension across nonconforming interfaces
is an explicit ADEPT approximation, not the author's neighbor-routing code.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .numerics import _basis, _cross, initial_state, make_mesh, rk4_push


class PanelHierarchy(eqx.Module):
    x: Any
    v: Any
    weights: Any
    level: Any
    parent: Any
    children: Any
    neighbors: Any
    root_count: int = eqx.field(static=True)
    max_level: int = eqx.field(static=True)
    xmin: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    vmin: float = eqx.field(static=True)
    vmax: float = eqx.field(static=True)


def make_hierarchy(nx, nv, xmin, xmax, vmin, vmax, max_level, quadrature="trapezoid"):
    """Precompute a bounded complete candidate hierarchy (not evolving leaves).

    Nodes are drawn from one integer lattice so duplicated nodes coincide
    exactly, including at hanging interfaces. Leaves have local quadrature;
    duplicates carry their incident panels' separate weight contributions.
    """
    make_mesh(nx, nv, xmin, xmax, vmin, vmax, quadrature)
    if isinstance(max_level, bool) or int(max_level) != max_level or not 0 <= max_level <= 4:
        raise ValueError("max_level must be an integer between 0 and 4")
    roots = (nx // 2) * (nv // 2)
    count = roots * sum(4**level for level in range(max_level + 1))
    if count > 32768:
        raise ValueError("AMR candidate hierarchy exceeds 32768 panels; reduce base grid or max_level")
    ox, ov = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    ox, ov = ox.ravel(), ov.ravel()
    unit = np.array([1, 2, 1]) if quadrature == "trapezoid" else np.array([1, 4, 1])
    unit = np.outer(unit, unit).ravel() / unit.sum() ** 2
    xs, vs, weights, levels, parents, children, neighbors = [], [], [], [], [], [], []
    offsets = np.cumsum([0] + [roots * 4**level for level in range(max_level)])
    for level in range(max_level + 1):
        npx, npv = nx // 2 * 2**level, nv // 2 * 2**level
        spacing_x, spacing_v = (xmax - xmin) / (nx * 2**max_level), (vmax - vmin) / (nv * 2**max_level)
        lattice_stride = 2 ** (max_level - level)
        for ix in range(npx):
            for iv in range(npv):
                xs.append(xmin + ((2 * ix + ox) * lattice_stride) * spacing_x)
                vs.append(vmin + ((2 * iv + ov) * lattice_stride) * spacing_v)
                weights.append(unit * ((xmax - xmin) / npx) * ((vmax - vmin) / npv))
                levels.append(level)
                parents.append(-1 if level == 0 else offsets[level - 1] + (ix // 2) * (npv // 2) + iv // 2)
                children.append(
                    [-1] * 4
                    if level == max_level
                    else [
                        offsets[level + 1] + (2 * ix + dx) * (2 * npv) + 2 * iv + dv
                        for dx, dv in ((0, 0), (0, 1), (1, 0), (1, 1))
                    ]
                )
                neighbors.append(
                    [
                        offsets[level] + ((ix - 1) % npx) * npv + iv,
                        offsets[level] + ((ix + 1) % npx) * npv + iv,
                        offsets[level] + ix * npv + iv - 1 if iv > 0 else -1,
                        offsets[level] + ix * npv + iv + 1 if iv + 1 < npv else -1,
                    ]
                )
    return PanelHierarchy(
        jnp.asarray(np.array(xs)),
        jnp.asarray(np.array(vs)),
        jnp.asarray(np.array(weights)),
        jnp.asarray(levels, dtype=jnp.int32),
        jnp.asarray(parents, dtype=jnp.int32),
        jnp.asarray(children, dtype=jnp.int32),
        jnp.asarray(neighbors, dtype=jnp.int32),
        roots,
        max_level,
        float(xmin),
        float(xmax - xmin),
        float(vmin),
        float(vmax),
    )


def _leaf_mask(hierarchy, f, min_level, atol, rtol):
    indicator = jnp.max(f, axis=1) - jnp.min(f, axis=1)
    wants_refinement = indicator > atol + rtol * jnp.max(jnp.abs(f), axis=1)
    leaf = hierarchy.level == 0

    def split(mask, selected):
        child_ids = jnp.maximum(hierarchy.children, 0).reshape(-1)
        additions = jnp.zeros_like(mask).at[child_ids].max(jnp.repeat(selected, 4))
        return (mask & ~selected) | additions

    # 2:1 face balance, including the periodic seam. Propagate finest selected
    # descendant level upward, then split coarse leaves beside deeper subtrees.
    # Re-test newly created leaves, including balancing-induced children. This
    # monotone finite closure carries only discrete topology, not floats, so
    # reverse derivatives of the selected continuous values remain available.
    def refine(carry):
        leaf, _ = carry
        deepest = jnp.where(leaf, hierarchy.level, -1)
        for level in range(hierarchy.max_level, 0, -1):
            descendant = jnp.where(hierarchy.level == level, deepest, -1)
            deepest = deepest.at[jnp.maximum(hierarchy.parent, 0)].max(descendant)
        neighbor_depth = jnp.where(hierarchy.neighbors >= 0, deepest[jnp.maximum(hierarchy.neighbors, 0)], -1)
        selected = (
            leaf
            & (hierarchy.level < hierarchy.max_level)
            & (
                (hierarchy.level < min_level)
                | wants_refinement
                | jnp.any(neighbor_depth > hierarchy.level[:, None] + 1, axis=1)
            )
        )
        return split(leaf, selected), jnp.any(selected)

    leaf, _ = jax.lax.while_loop(lambda carry: carry[1], refine, (leaf, jnp.asarray(True)))
    limited = jnp.sum(leaf & wants_refinement & (hierarchy.level == hierarchy.max_level), dtype=jnp.int32)
    return leaf, limited


def initialize_amr(hierarchy, candidate_f, *, max_panels, min_level=0, atol=0.05, rtol=0.0):
    """Choose/pack a leaf partition, with explicit failure on capacity overflow.

    No normalization is performed here. The public builder normalizes initial
    mass once; every later regrid preserves the raw interpolated values.
    """
    if max_panels < 1 or not 0 <= min_level <= hierarchy.max_level:
        raise ValueError("positive max_panels and 0 <= min_level <= max_level required")
    leaf, limited = _leaf_mask(hierarchy, candidate_f, min_level, atol, rtol)
    requested = jnp.sum(leaf, dtype=jnp.int32)
    ids = jnp.nonzero(leaf, size=max_panels, fill_value=0)[0]
    active = jnp.arange(max_panels) < requested
    f = jnp.where(active[:, None], candidate_f[ids], 0.0)
    weights = jnp.where(active[:, None], hierarchy.weights[ids], 0.0)
    state = initial_state(hierarchy.x[ids], hierarchy.v[ids], f, weights)
    zero = jnp.zeros((), dtype=f.dtype)
    return {
        **state,
        "weights": weights,
        "active": active,
        "panel_id": jnp.where(active, ids, -1),
        "level": jnp.where(active, hierarchy.level[ids], -1),
        "active_panels": jnp.minimum(requested, max_panels),
        "requested_panels": requested,
        "refinement_limited_panels": limited,
        "capacity_exceeded": requested > max_panels,
        "valid": (requested <= max_panels) & jnp.all(jnp.isfinite(candidate_f)),
        "regrid_mass_change": zero,
        "regrid_c2_change": zero,
        "remap_gap_nodes": jnp.asarray(0, dtype=jnp.int32),
        "max_gap_fraction": zero,
    }


def remesh_amr(state, hierarchy, *, chunk_size=64, max_gap_fraction=0.01):
    """Sample the hierarchy from deformed active-panel polynomials.

    Finest-containing leaf owns overlaps. At an interior polygon crack use
    the nearest polygon in fixed (x/L, v/velocity_span) distance, recording
    and bounding polynomial extension. Only crossing an actual outer velocity
    edge gives zero inflow. Those edges must remain graphs in x; folds fail.
    """
    px, pv, pf, active = state["x"], state["v"], state["f"], state["active"]
    length, velocity_span = hierarchy.length, hierarchy.vmax - hierarchy.vmin
    cx, cv = px[:, 4], pv[:, 4]
    sx = jnp.maximum(jnp.max(jnp.abs(px - cx[:, None]), axis=1), jnp.finfo(px.dtype).tiny)
    sv = jnp.maximum(jnp.max(jnp.abs(pv - cv[:, None]), axis=1), jnp.finfo(pv.dtype).tiny)
    ux, uv = (px - cx[:, None]) / sx[:, None], (pv - cv[:, None]) / sv[:, None]
    corners = jnp.array([0, 6, 8, 2])
    qx, qv = ux[:, corners], uv[:, corners]
    ex, ev = jnp.roll(qx, -1, axis=1) - qx, jnp.roll(qv, -1, axis=1) - qv
    turns = _cross(ex, ev, jnp.roll(ex, -1, axis=1), jnp.roll(ev, -1, axis=1))
    area = 0.5 * jnp.sum(_cross(qx, qv, jnp.roll(qx, -1, axis=1), jnp.roll(qv, -1, axis=1)), axis=1) * sx * sv
    reference_ids = jnp.maximum(state["panel_id"], 0)
    reference_area = jnp.sum(hierarchy.weights[reference_ids], axis=1)
    lower = active & (hierarchy.neighbors[reference_ids, 2] < 0)
    upper = active & (hierarchy.neighbors[reference_ids, 3] < 0)
    lower_dx, upper_dx = px[:, 6] - px[:, 0], px[:, 8] - px[:, 2]
    graph_ok = (~lower | (lower_dx > 0)) & (~upper | (upper_dx > 0))
    matrix = _basis(ux, uv)
    determinant = jnp.linalg.det(matrix)
    panel_ok = jnp.all(turns > 1e-12, axis=1) & (sx < length / 2) & graph_ok
    panel_ok &= jnp.isfinite(determinant) & (jnp.abs(determinant) > 1e-12)
    safe_matrix = jnp.where((active & panel_ok)[:, None, None], matrix, jnp.eye(9, dtype=pf.dtype))
    coefficients = jnp.linalg.solve(safe_matrix, jnp.where((active & panel_ok)[:, None], pf, 0.0)[..., None])[..., 0]
    panel_ok &= jnp.all(jnp.isfinite(coefficients), axis=1)
    usable = active & panel_ok
    # Physical corners/edges in ONE global metric, not per-panel scaling.
    gx, gv = px[:, corners] / length, pv[:, corners] / velocity_span
    dx, dv = jnp.roll(gx, -1, axis=1) - gx, jnp.roll(gv, -1, axis=1) - gv
    edge_norm = jnp.maximum(dx * dx + dv * dv, jnp.finfo(px.dtype).tiny)

    def sample(target):
        tx, tv = target
        delta_x = tx - cx
        delta_x -= length * jnp.floor(delta_x / length + 0.5)
        image_x = cx + delta_x
        txs, tvs = delta_x / sx, (tv - cv) / sv
        side = _cross(ex, ev, txs[:, None] - qx, tvs[:, None] - qv)
        inside = jnp.all(side >= -1e-11, axis=1) & usable
        covered = jnp.any(inside)
        containing = jnp.argmax(jnp.where(inside, state["level"] + 1, 0))
        # Piecewise linear OUTER boundary, with explicit topology validity.
        lo_alpha = (image_x - px[:, 0]) / jnp.where(lower_dx > 0, lower_dx, 1.0)
        hi_alpha = (image_x - px[:, 2]) / jnp.where(upper_dx > 0, upper_dx, 1.0)
        on_lower = lower & (lo_alpha >= -1e-11) & (lo_alpha <= 1 + 1e-11)
        on_upper = upper & (hi_alpha >= -1e-11) & (hi_alpha <= 1 + 1e-11)
        low_v = pv[:, 0] + lo_alpha * (pv[:, 6] - pv[:, 0])
        high_v = pv[:, 2] + hi_alpha * (pv[:, 8] - pv[:, 2])
        outside = jnp.any(on_lower & (tv < low_v - 1e-11 * velocity_span)) | jnp.any(
            on_upper & (tv > high_v + 1e-11 * velocity_span)
        )
        boundary_found = jnp.any(on_lower) & jnp.any(on_upper)
        rx, rv = image_x[:, None] / length - gx, tv / velocity_span - gv
        alpha = jnp.clip((rx * dx + rv * dv) / edge_norm, 0.0, 1.0)
        distances = jnp.min((rx - alpha * dx) ** 2 + (rv - alpha * dv) ** 2, axis=1)
        distances = jnp.where(usable, distances, jnp.inf)
        nearest = jnp.argmin(distances)
        gap = ~covered & ~outside
        extension = jnp.where(gap, jnp.sqrt(distances[nearest]), 0.0)
        owner = jnp.where(covered, containing, nearest)
        value = jnp.sum(_basis(txs[owner], tvs[owner]) * coefficients[owner])
        return jnp.where(outside, 0.0, value), ~covered, gap, extension, boundary_found

    # Canonicalize targets before ownership tests so coincident periodic nodes
    # select exactly the same polynomial, including when panels overlap.
    targets_x = hierarchy.x.reshape(-1)
    targets_x = hierarchy.xmin + jnp.mod(targets_x - hierarchy.xmin, length)
    targets = jnp.stack((targets_x, hierarchy.v.reshape(-1)), axis=-1)
    values, uncovered, gaps, extension, boundary_found = jax.lax.map(
        sample, targets, batch_size=min(chunk_size, targets.shape[0])
    )
    valid = (
        jnp.all(~active | panel_ok)
        & jnp.all(boundary_found)
        & jnp.all(jnp.isfinite(values))
        & (jnp.max(extension) <= max_gap_fraction)
    )
    info = {
        "uncovered_nodes": jnp.sum(uncovered, dtype=jnp.int32),
        "gap_nodes": jnp.sum(gaps, dtype=jnp.int32),
        "max_gap_fraction": jnp.max(extension),
        "invalid_panels": jnp.sum(active & ~panel_ok, dtype=jnp.int32),
        "max_panel_area_error": jnp.max(jnp.where(active, jnp.abs(area / reference_area - 1), 0.0)),
        "valid": valid,
    }
    return values.reshape(hierarchy.x.shape), info


class AdaptiveFarsightSystem(eqx.Module):
    hierarchy: PanelHierarchy
    max_panels: int = eqx.field(static=True)
    min_level: int = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    charge: float = eqx.field(static=True, default=-1.0)
    mass: float = eqx.field(static=True, default=1.0)
    remesh_every: int = eqx.field(static=True, default=1)
    chunk_size: int = eqx.field(static=True, default=64)
    max_gap_fraction: float = eqx.field(static=True, default=0.01)
    field_solver: Any = None

    def step(self, step, state, params, inputs, key):
        del params, inputs, key
        x, v = rk4_push(
            state["x"],
            state["v"],
            self.charge * state["weights"] * state["f"],
            self.dt,
            self.length,
            self.epsilon,
            self.charge / self.mass,
            self.chunk_size,
            field_solver=self.field_solver,
        )
        x, v = jnp.where(state["active"][:, None], x, state["x"]), jnp.where(state["active"][:, None], v, state["v"])
        pushed = {
            **state,
            "x": x,
            "v": v,
            "valid": state["valid"] & jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(v)),
        }
        if self.remesh_every == 0:
            return pushed

        def reset(current):
            candidates, info = remesh_amr(
                current, self.hierarchy, chunk_size=self.chunk_size, max_gap_fraction=self.max_gap_fraction
            )
            selected = initialize_amr(
                self.hierarchy,
                candidates,
                max_panels=self.max_panels,
                min_level=self.min_level,
                atol=self.atol,
                rtol=self.rtol,
            )
            weights, f = current["weights"], current["f"]
            new_weights, new_f = selected["weights"], selected["f"]
            mass_delta = jnp.sum(new_weights * new_f) - jnp.sum(weights * f)
            c2_delta = jnp.sum(new_weights * new_f**2) - jnp.sum(weights * f**2)
            same_layout_f = candidates[jnp.maximum(current["panel_id"], 0)]
            regrid_mass = jnp.sum(new_weights * new_f) - jnp.sum(weights * same_layout_f)
            regrid_c2 = jnp.sum(new_weights * new_f**2) - jnp.sum(weights * same_layout_f**2)
            return {
                **selected,
                "valid": current["valid"] & info["valid"] & selected["valid"],
                "capacity_exceeded": current["capacity_exceeded"] | selected["capacity_exceeded"],
                "remesh_count": current["remesh_count"] + 1,
                "remap_mass_change": current["remap_mass_change"] + mass_delta,
                "remap_c2_change": current["remap_c2_change"] + c2_delta,
                "remap_mass_abs_change": current["remap_mass_abs_change"] + jnp.abs(mass_delta),
                "remap_c2_abs_change": current["remap_c2_abs_change"] + jnp.abs(c2_delta),
                "regrid_mass_change": current["regrid_mass_change"] + regrid_mass,
                "regrid_c2_change": current["regrid_c2_change"] + regrid_c2,
                "remap_uncovered_nodes": current["remap_uncovered_nodes"] + info["uncovered_nodes"],
                "remap_gap_nodes": current["remap_gap_nodes"] + info["gap_nodes"],
                "max_gap_fraction": jnp.maximum(current["max_gap_fraction"], info["max_gap_fraction"]),
                "invalid_panels": current["invalid_panels"] + info["invalid_panels"],
                "max_panel_area_error": jnp.maximum(current["max_panel_area_error"], info["max_panel_area_error"]),
            }

        return jax.lax.cond((step + 1) % self.remesh_every == 0, reset, lambda current: current, pushed)
