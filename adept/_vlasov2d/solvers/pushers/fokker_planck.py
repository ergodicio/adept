"""Collisional operators for the Vlasov-2D solver.

Dougherty Fokker-Planck applied separably in vx and vy (Lie split), plus a
Krook relaxation toward a bi-Maxwellian at the local density.

Each 1D operator on its own axis is:
    df/dt = nu(x,y) * d/dv [ (v - vbar) f + T df/dv ]

with vbar = <v> and T = <(v-vbar)^2> computed from f along the same axis.
This is the standard Dougherty form that relaxes toward a Maxwellian centred
at the local mean velocity with the local temperature, conserving density,
mean velocity, and energy along that axis exactly.

We discretize with central differencing (matching the default 1D scheme) and
solve backward-Euler tridiagonally.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import vmap


def _dougherty_step_1d(f_v: jax.Array, v: jax.Array, dv: float, nu: float, dt: float) -> jax.Array:
    """Backward-Euler step of the 1D Dougherty operator on one column of f.

    f_v: (nv,) distribution along the active velocity axis
    """
    n = jnp.sum(f_v) * dv
    safe_n = jnp.where(n > 0, n, 1.0)
    vbar = jnp.sum(f_v * v) * dv / safe_n
    T = jnp.sum(f_v * (v - vbar) ** 2) * dv / safe_n
    T = jnp.maximum(T, 1e-12)

    # edge-centered drift C = (v - vbar) evaluated at edges
    v_edge = 0.5 * (v[1:] + v[:-1])
    C_edge = v_edge - vbar
    D = T  # scalar diffusion coefficient

    nv = f_v.shape[0]

    # Bare flux divergence (central differencing, zero-flux BC):
    # F_{i+1/2} = (C/2 - D/dv) f_i + (C/2 + D/dv) f_{i+1}
    bare_diag = jnp.zeros(nv)
    bare_diag = bare_diag.at[:-1].add((C_edge / 2.0 - D / dv) / dv)
    bare_diag = bare_diag.at[1:].add(-(C_edge / 2.0 + D / dv) / dv)
    bare_upper = (C_edge / 2.0 + D / dv) / dv
    bare_lower = (-C_edge / 2.0 + D / dv) / dv

    diag = 1.0 - dt * nu * bare_diag
    upper = -dt * nu * bare_upper
    lower = -dt * nu * bare_lower

    dl_padded = jnp.pad(lower, (1, 0))
    du_padded = jnp.pad(upper, (0, 1))

    # Solve (I - dt*nu*L) f_new = f_old
    return jax.lax.linalg.tridiagonal_solve(dl_padded, diag, du_padded, f_v[:, None])[:, 0]


class DoughertyFP:
    """Separable Dougherty FP applied to vx then vy at every (x, y)."""

    def __init__(self, species_grids, species_params):
        self.species_grids = species_grids
        self.species_params = species_params

    def __call__(self, nu_xy: jax.Array, f_dict: dict, dt: float) -> dict:
        out = {}
        for name, f in f_dict.items():
            vx = self.species_grids[name]["vx"]
            vy = self.species_grids[name]["vy"]
            dvx = self.species_grids[name]["dvx"]
            dvy = self.species_grids[name]["dvy"]

            # vmap _dougherty_step_1d over (x, y) and the orthogonal velocity axis.
            # vx step: f has shape (nx, ny, nvx, nvy); collapse to nvx vector per (x, y, vy).
            def _step_vx(f_one, nu_one):
                return _dougherty_step_1d(f_one, vx, dvx, nu_one, dt)

            # vmap order: x, y, vy → axes (0, 1, 3); nu over (x, y)
            f = vmap(  # over x
                vmap(  # over y
                    vmap(_step_vx, in_axes=(1, None), out_axes=1),  # over vy, share nu
                    in_axes=(0, 0),
                ),
                in_axes=(0, 0),
            )(f, nu_xy)

            def _step_vy(f_one, nu_one):
                return _dougherty_step_1d(f_one, vy, dvy, nu_one, dt)

            f = vmap(
                vmap(
                    vmap(_step_vy, in_axes=(0, None), out_axes=0),  # over vx, share nu
                    in_axes=(0, 0),
                ),
                in_axes=(0, 0),
            )(f, nu_xy)

            out[name] = f
        return out


class Krook2D:
    """Krook relaxation toward a local bi-Maxwellian at the species-stored T0=1.

    f_relax = n(x,y) * M(vx) * M(vy)  with M = unit-variance Maxwellian.
    Computes only over (x, y), not over species — for simplicity uses each
    species' own velocity grid and grid spacing.
    """

    def __init__(self, species_grids):
        self.species_grids = species_grids

    def __call__(self, nu_xy: jax.Array, f_dict: dict, dt: float) -> dict:
        out = {}
        nu_dt = nu_xy * dt
        decay = jnp.exp(-nu_dt)
        for name, f in f_dict.items():
            vx = self.species_grids[name]["vx"]
            vy = self.species_grids[name]["vy"]
            dvx = self.species_grids[name]["dvx"]
            dvy = self.species_grids[name]["dvy"]

            mx = jnp.exp(-0.5 * vx**2)
            mx = mx / (jnp.sum(mx) * dvx)
            my = jnp.exp(-0.5 * vy**2)
            my = my / (jnp.sum(my) * dvy)
            m_bi = mx[:, None] * my[None, :]  # (nvx, nvy)

            n_xy = jnp.sum(f, axis=(2, 3)) * (dvx * dvy)  # (nx, ny)
            target = n_xy[:, :, None, None] * m_bi[None, None, :, :]

            out[name] = f * decay[:, :, None, None] + target * (1.0 - decay[:, :, None, None])
        return out


class Collisions:
    """Container that applies FP then Krook if enabled."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        fp_cfg = cfg["terms"]["fokker_planck"]
        kr_cfg = cfg["terms"]["krook"]
        self.fp_on = bool(fp_cfg["is_on"])
        self.krook_on = bool(kr_cfg["is_on"])
        if self.fp_on:
            fp_type = fp_cfg["type"].casefold()
            if fp_type != "dougherty":
                raise NotImplementedError(f"vlasov-2d FP only supports 'dougherty' (got {fp_type})")
            self.fp = DoughertyFP(cfg["grid"]["species_grids"], cfg["grid"]["species_params"])
        if self.krook_on:
            self.krook = Krook2D(cfg["grid"]["species_grids"])

    def __call__(self, nu_fp: jax.Array | None, nu_K: jax.Array | None, f_dict: dict, dt: float) -> dict:
        if self.fp_on and nu_fp is not None:
            f_dict = self.fp(nu_fp, f_dict, dt)
        if self.krook_on and nu_K is not None:
            f_dict = self.krook(nu_K, f_dict, dt)
        return f_dict
