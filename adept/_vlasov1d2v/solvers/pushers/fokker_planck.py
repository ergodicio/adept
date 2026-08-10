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
        self._cylindrical = fp_type == "cylindrical_landau"
        if fp_type in ("dougherty", "dougherty_nodrag"):
            self.fp_model = Dougherty(v=self.v, dv=self.dv)
        elif fp_type == "lenard_bernstein":
            self.fp_model = LenardBernstein(v=self.v, dv=self.dv)
        elif self._cylindrical:
            fp_cfg = cfg["terms"]["fokker_planck"]
            channels = fp_cfg.get("channels", {})
            self.cyl = CylindricalLandau(
                v=grid["v"],
                dv=grid["dv"],
                vperp=grid["vperp"],
                dvperp=grid["dvperp"],
                wperp=grid["wperp"],
                alpha_speed=float(channels.get("speed", 1.0)),
                alpha_lorentz=float(channels.get("lorentz", 1.0)),
                restore=bool(fp_cfg.get("moment_restoration", True)),
                explicit_substeps=int(fp_cfg.get("explicit_substeps", 1)),
            )
        else:
            raise NotImplementedError(f"Unknown Fokker-Planck type for vlasov-1d2v: {fp_type}")
        if not self._cylindrical:
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
        """Apply the configured collision operator to one species (nx, nv, nvperp)."""
        if not self.cfg["terms"]["fokker_planck"]["is_on"]:
            return f

        nu_fp_in = nu_fp if nu_fp is not None else jnp.zeros(f.shape[0])

        if self._cylindrical:
            return self.cyl.apply_with_restoration(nu_fp_in, f, dt)

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


class CylindricalLandau:
    """Linearized full-velocity-geometry e-e collision operator (rung A).

    C[f] = nu * div[ D(v) . M grad(f / M) ],   M = exp(-|v - u|^2 / 2T),

    with the anisotropic test-particle tensor about the bulk frame,

        D = alpha_speed * Dhat_par(s) shat shat
          + alpha_lorentz * Dhat_perp(s) (I - shat shat),
        s = |v - u| / sqrt(T),

    where Dhat_par = psi(x)/s^3 and Dhat_perp = [(1 - 1/s^2) psi + psi']/(2s)
    (x = s^2/2) are the erf-exact Rosenbluth-potential coefficients of a
    Maxwellian field species, re-derived and verified in the campaign's
    theory/derivation_3v_sympy.py. M grad(f/M) is the Einstein-relation form
    (identical to grad f + (v-u) f / T), so EACH channel annihilates the bulk
    Maxwellian; discretizing the flux as D * M_edge * (g_{i+1} - g_i)/dv with
    g = f/M and M_edge the geometric mean makes the SAMPLED Maxwellian an
    exact discrete fixed point of every channel mix at any dt.

    Discretization: finite-volume on the cylindrical (v_par, v_perp) grid
    (weight 2 pi v_perp dv_perp dv_par) with zero-flux outer boundaries, so
    density is conserved to solver precision. BOTH diagonal diffusion
    sweeps (D_parpar along v_par, D_perpperp along v_perp) are implicit
    batched tridiagonals; only the cross terms are explicit (optionally
    substepped via ``explicit_substeps``) — the isolated cross term is
    sign-indefinite at high k and is stabilized by the implicit damping of
    the same modes (|D_pq| <= sqrt(D_pp D_qq) for the PSD tensor).
    Momentum/energy exchanged with the (implicit) field particles is
    restored each step by projecting onto shifted/heated bulk-Maxwellian
    modes (2x2 solve; the discrete analog of the field-particle
    back-reaction), keeping n, P_par, E exact.

    Quasi-static re-linearization: (n, u, T) are recomputed from f at every
    call, so the operator always relaxes toward the current bulk. Moment
    restoration is REQUIRED for long-run stability, not just bookkeeping:
    without it the live-moment loop chases the O(dv_perp^2) midpoint-rule
    bias of the measured T and drifts secularly (~5e-5/collision time at
    dv_perp = 0.1); with it P_par and E are pinned and the loop closes.
    """

    def __init__(
        self,
        v: Array,
        dv: float,
        vperp: Array,
        dvperp: float,
        wperp: Array,
        alpha_speed: float = 1.0,
        alpha_lorentz: float = 1.0,
        restore: bool = True,
        explicit_substeps: int = 1,
    ):
        """Precompute grids, edges, and integration weights."""
        self.v = v
        self.dv = dv
        self.vperp = vperp
        self.dvperp = dvperp
        self.wperp = wperp
        self.alpha_speed = alpha_speed
        self.alpha_lorentz = alpha_lorentz
        self.restore = restore
        self.n_sub = max(int(explicit_substeps), 1)

        self.v_edge = 0.5 * (v[1:] + v[:-1])
        self.vperp_edge = 0.5 * (vperp[1:] + vperp[:-1])
        # full phase-space weight w[i, j] = dv * 2 pi vperp_j dvperp
        self.w = dv * wperp[None, :] * jnp.ones_like(v)[:, None]

    # ---- erf-exact coefficients (Gamma n = 1 units; see derivation_3v_sympy) --
    @staticmethod
    def _psi(x: Array) -> Array:
        return jax.scipy.special.erf(jnp.sqrt(x)) - 2.0 * jnp.sqrt(x / jnp.pi) * jnp.exp(-x)

    @classmethod
    def _dhat_par(cls, s: Array) -> Array:
        """Speed-channel (parallel) diffusion coefficient psi(s^2/2)/s^3.

        D_par = nu_par v^2/2 = [psi(x)/x] nu_0 v^2/2 = psi/s^3 (Gamma n = 1);
        its s -> 0 limit sqrt(2/pi)/3 equals D_perp(0), the isotropic value,
        and its tail is nu_0(s) = 1/s^3 (both verified in Stage 1 of
        theory/derivation_3v_sympy.py).
        """
        s = jnp.maximum(s, 1.0e-6)
        return cls._psi(s**2 / 2.0) / s**3

    @classmethod
    def _dhat_perp(cls, s: Array) -> Array:
        """Lorentz-channel (perpendicular) diffusion coefficient."""
        s = jnp.maximum(s, 1.0e-6)
        x = s**2 / 2.0
        psi = cls._psi(x)
        psip = jnp.sqrt(2.0 / jnp.pi) * s * jnp.exp(-x)
        return ((1.0 - 1.0 / s**2) * psi + psip) / (2.0 * s)

    def _tensor(self, wpar: Array, vperp: Array, T: Array) -> tuple[Array, Array, Array]:
        """Return (D_parpar, D_parperp, D_perpperp) at points (wpar, vperp).

        wpar is the parallel velocity relative to the bulk. All arrays
        broadcast; T broadcasts from (nx, 1, 1). Units of Gamma n sqrt(T).
        """
        s2 = wpar**2 + vperp**2
        s = jnp.sqrt(s2 + 1.0e-30)
        shat = s / jnp.sqrt(T)
        dpar = self.alpha_speed * self._dhat_par(shat)
        dperp = self.alpha_lorentz * self._dhat_perp(shat)
        dpp = (dpar * wpar**2 + dperp * vperp**2) / s2
        dpq = (dpar - dperp) * wpar * vperp / s2
        dqq = (dpar * vperp**2 + dperp * wpar**2) / s2
        return dpp, dpq, dqq

    def _moments(self, f: Array) -> tuple[Array, Array, Array]:
        """Bulk (n, u, T) per x from the full distribution."""
        n = jnp.einsum("xvp,vp->x", f, self.w)
        u = jnp.einsum("xvp,vp->x", f, self.v[:, None] * self.w) / n
        wpar = self.v[None, :, None] - u[:, None, None]
        T = jnp.einsum("xvp,xvp->x", f, (wpar**2 + self.vperp[None, None, :] ** 2) * self.w[None, ...]) / (3.0 * n)
        return n, u, T

    def _dperp_g(self, g: Array) -> Array:
        """Centered d/dv_perp of g with replicated boundary cells, shape of g."""
        gpad = jnp.concatenate([g[..., :1], g, g[..., -1:]], axis=-1)
        return (gpad[..., 2:] - gpad[..., :-2]) / (2.0 * self.dvperp)

    def __call__(self, nu: Array, f: Array, dt: float) -> Array:
        """Advance f (nx, nv, nvperp) one collision step of size dt."""
        n, u, T = self._moments(f)
        nb = n[:, None, None]
        ub = u[:, None, None]
        Tb = T[:, None, None]
        scale = nb * jnp.sqrt(Tb)  # coefficient magnitude ~ Gamma n sqrt(T)
        nudt = dt * nu[:, None, None]

        vpar = self.v[None, :, None]
        vprp = self.vperp[None, None, :]
        M = jnp.exp(-((vpar - ub) ** 2 + vprp**2) / (2.0 * Tb))

        # ---------------- implicit v_par (D_parpar) sweep -------------------
        # flux at interior v_par edges: c_e (g_{i+1} - g_i), c_e = D_pp M_e / dv^2
        wpar_e = self.v_edge[None, :, None] - ub
        dpp_e, _, _ = self._tensor(wpar_e, vprp, Tb)
        M_e = jnp.sqrt(M[:, 1:, :] * M[:, :-1, :])
        c_e = scale * dpp_e * M_e / self.dv**2  # (nx, nv-1, nvperp)

        # tridiagonal in i for each (x, j): batch layout (nx, nvperp, nv)
        cT = jnp.transpose(c_e, (0, 2, 1))
        MT = jnp.transpose(M, (0, 2, 1))
        fT = jnp.transpose(f, (0, 2, 1))
        nudtT = dt * nu[:, None, None]

        c_lo = jnp.pad(cT, ((0, 0), (0, 0), (1, 0)))  # edge below cell i
        c_hi = jnp.pad(cT, ((0, 0), (0, 0), (0, 1)))  # edge above cell i
        diag = 1.0 + nudtT * (c_lo + c_hi) / MT
        lower = -nudtT * cT / MT[:, :, :-1]  # multiplies f_{i-1}
        upper = -nudtT * cT / MT[:, :, 1:]  # multiplies f_{i+1}
        dl = jnp.pad(lower, ((0, 0), (0, 0), (1, 0)))
        du = jnp.pad(upper, ((0, 0), (0, 0), (0, 1)))
        fT_new = jax.lax.linalg.tridiagonal_solve(dl, diag, du, fT[..., None])[..., 0]
        f = jnp.transpose(fT_new, (0, 2, 1))

        # ---------------- implicit v_perp (D_perpperp) sweep -----------------
        # Both diagonal diffusion sweeps are implicit; only the cross terms
        # stay explicit. The isolated cross term has a sign-indefinite symbol
        # at high k (substepping alone cannot stabilize it), but the implicit
        # sweeps damp exactly those modes and |D_pq| <= sqrt(D_pp D_qq) (PSD
        # tensor) makes the combination stable.
        wpar_c = vpar - ub
        vprp_e = self.vperp_edge[None, None, :]
        _, dqp_pe, dqq_pe = self._tensor(wpar_c, vprp_e, Tb)
        M_pe = jnp.sqrt(M[:, :, 1:] * M[:, :, :-1])
        vcq = vprp_e * scale * dqq_pe * M_pe / self.dvperp**2  # metric-weighted edge coeff

        cq_lo = jnp.pad(vcq, ((0, 0), (0, 0), (1, 0)))
        cq_hi = jnp.pad(vcq, ((0, 0), (0, 0), (0, 1)))
        diag_q = 1.0 + nudt * (cq_lo + cq_hi) / (vprp * M)
        lower_q = -nudt * vcq / (vprp[:, :, 1:] * M[:, :, :-1])
        upper_q = -nudt * vcq / (vprp[:, :, :-1] * M[:, :, 1:])
        dl_q = jnp.pad(lower_q, ((0, 0), (0, 0), (1, 0)))
        du_q = jnp.pad(upper_q, ((0, 0), (0, 0), (0, 1)))
        f = jax.lax.linalg.tridiagonal_solve(dl_q, diag_q, du_q, f[..., None])[..., 0]

        # ---------------- explicit cross fluxes (substepped) ----------------
        _, dpq_e, _ = self._tensor(wpar_e, vprp, Tb)

        def explicit_step(_, f_cur):
            g = f_cur / M
            dperp_g = self._dperp_g(g)

            # parallel flux (cross term) at interior v_par edges
            dperp_g_e = 0.5 * (dperp_g[:, 1:, :] + dperp_g[:, :-1, :])
            gam_par = scale * dpq_e * M_e * dperp_g_e
            div_par = (
                jnp.pad(gam_par, ((0, 0), (0, 1), (0, 0))) - jnp.pad(gam_par, ((0, 0), (1, 0), (0, 0)))
            ) / self.dv

            # perpendicular flux (cross term) at interior v_perp edges
            dpar_gpad = jnp.concatenate([g[:, :1, :], g, g[:, -1:, :]], axis=1)
            dpar_g = (dpar_gpad[:, 2:, :] - dpar_gpad[:, :-2, :]) / (2.0 * self.dv)
            dpar_g_e = 0.5 * (dpar_g[:, :, 1:] + dpar_g[:, :, :-1])
            gam_perp = scale * dqp_pe * dpar_g_e * M_pe

            # metric divergence: (1/vperp) d(vperp Gamma)/dvperp
            vg = vprp_e * gam_perp
            div_perp = (jnp.pad(vg, ((0, 0), (0, 0), (0, 1))) - jnp.pad(vg, ((0, 0), (0, 0), (1, 0)))) / (
                vprp * self.dvperp
            )

            return f_cur + (nudt / self.n_sub) * (div_par + div_perp)

        f = jax.lax.fori_loop(0, self.n_sub, explicit_step, f)

        return f

    def apply_with_restoration(self, nu: Array, f: Array, dt: float) -> Array:
        """One collision step followed by exact P_par / E restoration."""
        f_new = self(nu, f, dt)
        if not self.restore:
            return f_new

        n, u, T = self._moments(f)
        ub = u[:, None, None]
        Tb = T[:, None, None]
        vpar = self.v[None, :, None]
        vprp = self.vperp[None, None, :]
        M = jnp.exp(-((vpar - ub) ** 2 + vprp**2) / (2.0 * Tb))
        Mhat = M / jnp.einsum("xvp,vp->x", M, self.w)[:, None, None]

        # restoration modes, discretely orthogonalized against density
        phi1 = (vpar - ub) * Mhat
        phi1 = phi1 - jnp.einsum("xvp,vp->x", phi1, self.w)[:, None, None] * Mhat
        s2 = (vpar - ub) ** 2 + vprp**2
        c2 = jnp.einsum("xvp,vp->x", s2 * Mhat, self.w)[:, None, None]
        phi2 = (s2 - c2) * Mhat

        df = f_new - f
        mom_v = vpar * self.w[None, ...]
        mom_e = (vpar**2 + vprp**2) * self.w[None, ...]
        dj = jnp.einsum("xvp,xvp->x", df, mom_v)
        de = jnp.einsum("xvp,xvp->x", df, mom_e)

        a11 = jnp.einsum("xvp,xvp->x", phi1, mom_v)
        a12 = jnp.einsum("xvp,xvp->x", phi2, mom_v)
        a21 = jnp.einsum("xvp,xvp->x", phi1, mom_e)
        a22 = jnp.einsum("xvp,xvp->x", phi2, mom_e)
        det = a11 * a22 - a12 * a21
        b = (-dj * a22 + de * a12) / det
        c = (-de * a11 + dj * a21) / det

        return f_new + b[:, None, None] * phi1 + c[:, None, None] * phi2
