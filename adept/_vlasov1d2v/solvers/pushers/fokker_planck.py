"""v_par-only Fokker-Planck collisions for the Vlasov-1D2V solver.

This is the "1D-Dougherty limit" rung of the operator-fidelity ladder: the
1D Dougherty (or dougherty_nodrag) operator applied along v_par to every
(x, v_perp) slice, with the drift/diffusion coefficients computed from the
MARGINAL F(v_par) = int f w_perp dv_perp.

Because the discrete energy-flux condition that find_self_consistent_beta
zeroes is linear in f, and the marginal is a w_perp-weighted sum of slices,
computing (vbar, beta) from the marginal and applying the same (C_edge, D)
to every slice conserves total n, P_par, and E_par to the same standard as
the 1D operator. For a v_perp-separable distribution f = F(v_par) M(v_perp),
the marginal dynamics are EXACTLY the 1D operator's (each slice remains
proportional to F), which is the basis of the 1D-limit equivalence test.

The full-geometry cylindrical operator (Lorentz + speed channels) is added
at rung A (D2 of the PRL plan); it will slot in as additional fp types here.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import Array, vmap
from jax import numpy as jnp

from adept._vlasov1d.solvers.pushers.fokker_planck import Dougherty, LenardBernstein
from adept.driftdiffusion import CentralDifferencing, find_self_consistent_beta


class Collisions:
    """v_par drift-diffusion collisions with marginal-moment coefficients."""

    def __init__(self, cfg: Mapping[str, Any]):
        """Build the collision model, scheme, and marginal-moment machinery."""
        self.cfg = cfg

        grid = cfg["grid"]["species_grids"]["electron"]
        self.v = grid["v"]
        self.dv = grid["dv"]
        self.wperp = grid["wperp"]

        fp_type = cfg["terms"]["fokker_planck"]["type"].casefold()
        if fp_type in ("dougherty", "dougherty_nodrag"):
            self.fp_model = Dougherty(v=self.v, dv=self.dv)
        elif fp_type == "lenard_bernstein":
            self.fp_model = LenardBernstein(v=self.v, dv=self.dv)
        else:
            raise NotImplementedError(
                f"Unknown Fokker-Planck type for vlasov-1d2v: {fp_type} (rung-A cylindrical channels arrive with D2)"
            )
        self.fp_scheme = CentralDifferencing(dv=self.dv)
        self._nodrag = fp_type == "dougherty_nodrag"

        if cfg["terms"]["krook"]["is_on"]:
            raise NotImplementedError("Krook is not implemented for vlasov-1d2v")

        fp_cfg = cfg["terms"]["fokker_planck"]
        sc_cfg = fp_cfg.get("self_consistent_beta", {})
        sc_enabled = sc_cfg.get("enabled", False)
        self._sc_max_steps = sc_cfg.get("max_steps", 3) if sc_enabled else 0
        self._sc_rtol = sc_cfg.get("rtol", 1e-8)
        self._sc_atol = sc_cfg.get("atol", 1e-12)

    def marginal(self, f: Array) -> Array:
        """Return F(x, v_par) = int f w_perp dv_perp."""
        return jnp.einsum("xvp,p->xv", f, self.wperp)

    def _solve_slices_one_x(self, C_edge: Array, D_scalar: Array, nu: Array, f_slices: Array, dt: float) -> Array:
        """Apply the implicit v_par operator to every v_perp slice at one x.

        f_slices has shape (nvperp, nv); the tridiagonal operator is built once
        from the shared marginal coefficients and solved per slice in delta
        form to reduce floating-point error (same as the 1D solver).
        """
        op = self.fp_scheme.get_operator(C_edge=C_edge, D=D_scalar, nu=nu, dt=dt)
        dl_padded = jnp.pad(op.lower_diagonal, (1, 0))
        du_padded = jnp.pad(op.upper_diagonal, (0, 1))

        def solve_one(f_v):
            rhs = f_v - op.mv(f_v)
            delta = jax.lax.linalg.tridiagonal_solve(dl_padded, op.diagonal, du_padded, rhs[..., None])[..., 0]
            return f_v + delta

        return vmap(solve_one)(f_slices)

    def __call__(self, nu_fp: Array, nu_K: Array, f, dt: float):
        """Apply collisions to the (dict of) species distributions."""
        if isinstance(f, dict):
            return {name: self._apply(nu_fp, f_s, dt) if name == "electron" else f_s for name, f_s in f.items()}
        return self._apply(nu_fp, f, dt)

    def _apply(self, nu_fp: Array, f: Array, dt: float) -> Array:
        """Apply the v_par operator to one species distribution (nx, nv, nvperp)."""
        if not self.cfg["terms"]["fokker_planck"]["is_on"]:
            return f

        nu_fp_in = nu_fp if nu_fp is not None else jnp.zeros(f.shape[0])

        F = self.marginal(f)
        vbar = self.fp_model.compute_vbar(F)
        beta = find_self_consistent_beta(
            F,
            self.v,
            self.dv,
            spherical=False,
            vbar=vbar,
            rtol=self._sc_rtol,
            atol=self._sc_atol,
            max_steps=self._sc_max_steps,
        )
        C_edge, D = self.fp_model.compute_C_and_D(F, beta)
        if self._nodrag:
            C_edge = jnp.zeros_like(C_edge)

        ft = jnp.transpose(f, (0, 2, 1))  # (nx, nvperp, nv)
        ft_new = vmap(self._solve_slices_one_x, in_axes=(0, 0, 0, 0, None))(C_edge, D, nu_fp_in, ft, dt)

        if self._nodrag:
            # Subtract the diffusion of the per-slice Maxwellian with the same
            # zero-flux stencil as the implicit operator (dougherty_nodrag
            # pattern): each slice's Maxwellian share is normalized to the
            # slice density, with (vbar, beta) shared from the marginal, so
            # f_M stays a discrete fixed point and n, P, E are conserved.
            n_slices = jnp.sum(ft, axis=-1) * self.dv  # (nx, nvperp)
            f_mx = jnp.exp(-beta[:, None, None] * (self.v[None, None, :] - vbar[:, None, None]) ** 2)
            f_mx = f_mx * (n_slices / (jnp.sum(f_mx, axis=-1) * self.dv))[..., None]
            DfM = D[:, None, None] * f_mx
            lap = jnp.zeros_like(DfM)
            lap = lap.at[..., 1:-1].set((DfM[..., 2:] - 2.0 * DfM[..., 1:-1] + DfM[..., :-2]) / self.dv**2)
            lap = lap.at[..., 0].set((DfM[..., 1] - DfM[..., 0]) / self.dv**2)
            lap = lap.at[..., -1].set((DfM[..., -2] - DfM[..., -1]) / self.dv**2)
            ft_new = ft_new - dt * nu_fp_in[:, None, None] * lap

        return jnp.transpose(ft_new, (0, 2, 1))
