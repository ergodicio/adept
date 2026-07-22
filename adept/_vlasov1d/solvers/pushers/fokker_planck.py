#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
"""
Fokker-Planck collision operators for the Vlasov-1D solver.

This module provides collision operators that can be used with the Vlasov-1D solver.
The operators are built on shared abstractions from adept.driftdiffusion.
"""

from collections.abc import Mapping
from typing import Any

import jax
import lineax as lx
import numpy as np
import optimistix as optx
from jax import Array, shard_map, vmap
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from adept.driftdiffusion import (
    AbstractBetaBasedModel,
    CentralDifferencing,
    ChangCooper,
    analytic_supergaussian_temperature_ratio,
    chang_cooper_delta,
)


class LenardBernstein(AbstractBetaBasedModel):
    """
    Lenard-Bernstein collision model using Buet notation.

    Physics:
        C = v (drift toward v=0)
        D = 1/(2β) = T where β = 1/(2T) is provided EXTERNALLY

    Uses Buet notation: β = 1/(2T), so the Maxwellian is f = exp(-β·v²).
    NOT compatible with Buet weak form scheme.

    The equilibrium solution is a Maxwellian centered at v=0.
    """

    def compute_D(self, f: Array, beta: Array) -> Array:
        """
        Compute Lenard-Bernstein diffusion coefficient from beta.

        Uses Buet notation: β = 1/(2T), so D = 1/(2β) = T.

        Args:
            f: Distribution function, shape (..., nv) - used only to determine batch shape
            beta: Inverse temperature in Buet notation β = 1/(2T), shape (...).

        Returns:
            D: Diffusion coefficient = 1/(2β) = T, shape (...)
        """
        return 1.0 / (2.0 * beta)


class Dougherty(AbstractBetaBasedModel):
    """
    Dougherty collision model using Buet notation.

    Physics:
        C = v - <v> (drift toward mean velocity)
        D = 1/(2β) = T where β = 1/(2T) is provided EXTERNALLY

    Uses Buet notation: β = 1/(2T), so the Maxwellian is f = exp(-β·(v-vbar)²).

    The equilibrium solution is a Maxwellian centered at the mean velocity <v>.
    This model conserves momentum in addition to density and energy.

    Note: vbar is still computed from f (momentum must come from the distribution),
    but D comes from the external β parameter.
    """

    def compute_vbar(self, f: Array) -> Array:
        """
        Compute mean velocity from distribution.

        Args:
            f: Distribution function, shape (..., nv)

        Returns:
            vbar: Mean velocity <v> = ∫v f dv / ∫f dv, shape (...)
        """
        return jnp.sum(f * self.v, axis=-1) / jnp.sum(f, axis=-1)

    def compute_D(self, f: Array, beta: Array) -> Array:
        """
        Compute Dougherty diffusion coefficient from beta.

        Uses Buet notation: β = 1/(2T), so D = 1/(2β) = T.

        Args:
            f: Distribution function, shape (..., nv)
            beta: Inverse temperature in Buet notation β = 1/(2T), shape (...).

        Returns:
            D: Diffusion coefficient = 1/(2β) = T, shape (...)
        """
        return 1.0 / (2.0 * beta)


class SuperGaussianDougherty(Dougherty):
    """
    Dougherty-type collision model whose equilibrium is a super-Gaussian.

    Physics:
        Equilibrium: f₀ ∝ exp(-β·|v - vbar|^m)   (m = 2 recovers Dougherty)
        C = D · Δφ/dv with φ = β·|v - vbar|^m = -ln f₀   (exact finite
            difference of the potential across each edge; the continuum limit
            of C/D = -∂v ln f₀)
        D = β^(-2/m)·Γ(3/m)/Γ(1/m)   (the analytic temperature of the target
            super-Gaussian; reduces to D = 1/(2β) = T at m = 2)

    Here β is the super-Gaussian shape parameter (units of v^-m), computed by
    compute_beta from the energy-conservation closure β = n/(m·⟨|v-vbar|^m⟩)
    (the generalization of the Lenard-Bernstein/Dougherty β = 1/(2T), which it
    equals at m = 2). Paired with the Chang-Cooper scheme and the exact-Δφ
    drift below, the sampled discrete super-Gaussian is the exact fixed point.

    This maintains a prescribed super-Gaussian order m — e.g. a Langdon/DLM
    inverse-bremsstrahlung-heated distribution — against collisional
    relaxation to a Maxwellian. Conserves density exactly (zero-flux BC) and
    energy up to discretization error (via the β closure). Momentum is exact
    only for f symmetric about vbar; skewed transients exchange momentum at
    O(skewness) — unlike m = 2, where C ∝ (v - vbar) makes it exact.
    """

    m: float

    def __init__(self, v: Array, dv: float, m: float):
        """Initialize model with velocity grid and super-Gaussian exponent m."""
        super().__init__(v=v, dv=dv)
        self.m = m

    def compute_beta(self, f: Array, rtol: float = 1e-8, atol: float = 1e-12, max_steps: int = 3) -> Array:
        """
        Compute the energy-conserving shape parameter β.

        The continuum operator conserves energy instantaneously iff
        m·β·⟨|v-vbar|^m⟩ = ⟨1⟩ (from ∫v²·∂v[D·(φ'f + ∂vf)]dv = 0), giving the
        closure β = n/(m·⟨|v-vbar|^m⟩). At m = 2 this reduces to
        β = n/(2·⟨(v-vbar)²⟩) = 1/(2T), the Buet beta computed from the
        discrete temperature — where it coincides with temperature matching,
        which is why the Maxwellian operators get both for free.

        With max_steps > 0, β is refined by a Newton solve of the DISCRETE
        energy-flux condition consistent with the Chang-Cooper weighting,

            h(β) = Σ_edges v_e·[w·f̃(w) + Δf] = 0,   w = β·Δψ,
            f̃(w) = δ(w)·f_i + (1-δ(w))·f_{i+1},     ψ = |v - vbar|^m,

        which zeroes the discrete energy flux of the operator exactly for the
        current f. At a sampled super-Gaussian every edge term vanishes
        identically (the Chang-Cooper equilibrium property), so β₀ is an
        exact root: the discrete equilibrium is exactly stationary and there
        is no secular drift. With max_steps = 0 (self_consistent_beta
        disabled) the continuum closure is used directly; its O(dv²)
        quadrature residual causes a slow secular temperature drift
        (measured ~5e-7 per collision time at nv=128 for m=3). The continuum
        closure lands within O(dv²) of the root, so a SINGLE Newton step
        (max_steps: 1) already reduces the drift to machine level; use it for
        long collisional runs.

        Transient (off-equilibrium) energy conservation is additionally
        limited to O(nu·dt) by the operator splitting: β is frozen from f^n
        while the implicit step advances to f^{n+1}. This is a one-time
        offset proportional to the shape change, negligible for the small
        nu·dt of production runs.

        Args:
            f: Distribution function, shape (..., nv)
            rtol: Relative tolerance for the Newton solve
            atol: Absolute tolerance for the Newton solve
            max_steps: Maximum Newton iterations (0 = continuum closure only)

        Returns:
            beta: Shape parameter (units of v^-m), shape (...)
        """
        m = self.m
        v = self.v
        v_edge = 0.5 * (v[1:] + v[:-1])

        def beta_single(f_v: Array) -> Array:
            vbar = jnp.sum(f_v * v) / jnp.sum(f_v)
            psi = jnp.abs(v - vbar) ** m
            # Continuum closure: β = n / (m·⟨ψ⟩); dv cancels in the ratio
            beta_init = jnp.sum(f_v) / (m * jnp.sum(f_v * psi))
            if max_steps == 0:
                return beta_init

            d_psi = psi[1:] - psi[:-1]
            d_f = f_v[1:] - f_v[:-1]

            def residual(beta, args):
                del args
                w = beta * d_psi
                delta = chang_cooper_delta(w)
                f_tilde = delta * f_v[:-1] + (1.0 - delta) * f_v[1:]
                return jnp.sum(v_edge * (w * f_tilde + d_f))

            solver = optx.Newton(rtol=rtol, atol=atol)
            sol = optx.root_find(fn=residual, solver=solver, y0=beta_init, args=None, max_steps=max_steps, throw=False)
            return sol.value

        return vmap(beta_single)(f)

    def compute_D(self, f: Array, beta: Array) -> Array:
        """
        Compute diffusion coefficient from the super-Gaussian shape parameter.

        D = β^(-2/m)·Γ(3/m)/Γ(1/m) is the analytic temperature of the target
        shape, so that the m=2 case reduces to the Dougherty D = 1/(2β) = T.

        Args:
            f: Distribution function, shape (..., nv) - unused
            beta: Super-Gaussian shape parameter, shape (...)

        Returns:
            D: Diffusion coefficient, shape (...)
        """
        return beta ** (-2.0 / self.m) * analytic_supergaussian_temperature_ratio(self.m)

    def compute_C_and_D(self, f: Array, beta: Array) -> tuple[Array, Array]:
        """
        Compute C and D from the super-Gaussian equilibrium condition.

        C is the exact finite difference of the equilibrium potential
        φ(v) = β·|v - vbar|^m = -ln f₀ across each cell edge:

            C_edge = D · (φ(v_{i+1}) - φ(v_i)) / dv

        rather than the midpoint derivative m·β·|v_eff|^(m-1)·sgn(v_eff).
        The Chang-Cooper fixed point satisfies f_{i+1}/f_i = exp(-C·dv/D),
        so this choice makes the SAMPLED super-Gaussian the exact discrete
        equilibrium (the midpoint form is only exact for m=2 and causes
        secular temperature drift for m≠2). For m=2 the two coincide:
        Δ(v²)/dv = 2·v_edge exactly.

        Args:
            f: Distribution function, shape (..., nv)
            beta: Super-Gaussian shape parameter, shape (...)

        Returns:
            C_edge: Drift coefficient at cell edges, shape (..., nv-1)
            D: Diffusion coefficient, shape (...)
        """
        D = self.compute_D(f, beta)
        vbar = self.compute_vbar(f)
        phi = beta[..., None] * jnp.abs(self.v - vbar[..., None]) ** self.m
        C_edge = D[..., None] * (phi[..., 1:] - phi[..., :-1]) / self.dv
        return C_edge, D


class Collisions:
    """High-level collision operator that wraps Fokker-Planck and Krook terms."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Build collision pushers from configuration.

        :param cfg: Simulation configuration containing term toggles and grid parameters.
        """
        self.cfg = cfg
        self.fp_model, self.fp_scheme = self.__init_fp_operator__()
        self.krook = Krook(self.cfg)

        v = cfg["grid"]["species_grids"]["electron"]["v"]
        self.v_edge = 0.5 * (v[1:] + v[:-1])

        parallel = cfg["grid"].get("parallel", False)
        self.x_parallel = bool(parallel) and "x" in parallel
        if self.x_parallel:
            self._mesh = Mesh(np.array(jax.devices()), ("device",))

        # Self-consistent beta config (defaults to disabled)
        fp_cfg = cfg["terms"]["fokker_planck"]
        sc_cfg = fp_cfg.get("self_consistent_beta", {})
        sc_enabled = sc_cfg.get("enabled", False)
        self._sc_max_steps = sc_cfg.get("max_steps", 3) if sc_enabled else 0
        self._sc_rtol = sc_cfg.get("rtol", 1e-8)
        self._sc_atol = sc_cfg.get("atol", 1e-12)

    def __init_fp_operator__(self):
        """
        Instantiate the configured Fokker-Planck model and scheme.

        :raises NotImplementedError: When the configured operator type is unknown.
        :returns: Tuple of (model, scheme)
        """
        # TODO(gh-173): For multi-species, use electron grid for FP for now
        v = self.cfg["grid"]["species_grids"]["electron"]["v"]
        dv = self.cfg["grid"]["species_grids"]["electron"]["dv"]

        fp_type = self.cfg["terms"]["fokker_planck"]["type"].casefold()

        if fp_type == "lenard_bernstein":
            model = LenardBernstein(v=v, dv=dv)
            return model, CentralDifferencing(dv=dv)
        elif fp_type in ("chang_cooper", "lenard_bernstein_chang_cooper"):
            model = LenardBernstein(v=v, dv=dv)
            return model, ChangCooper(dv=dv)
        elif fp_type in ("chang_cooper_dougherty", "dougherty_chang_cooper"):
            model = Dougherty(v=v, dv=dv)
            return model, ChangCooper(dv=dv)
        elif fp_type == "dougherty":
            model = Dougherty(v=v, dv=dv)
            return model, CentralDifferencing(dv=dv)
        elif fp_type in ("super_gaussian", "super_gaussian_chang_cooper"):
            m = float(self.cfg["terms"]["fokker_planck"].get("m", 2.0))
            model = SuperGaussianDougherty(v=v, dv=dv, m=m)
            return model, ChangCooper(dv=dv)
        else:
            raise NotImplementedError(f"Unknown Fokker-Planck type: {fp_type}")

    def __call__(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f, dt: jnp.float64):
        """
        Apply configured collision operators to the distribution function.

        :param nu_fp: Collision frequencies for the Fokker-Planck operator (shape: nx).
        :param nu_K: Krook collision frequencies (shape: nx).
        :param f: Distribution function (dict or array).
        :param dt: Time step size.
        :return: Updated distribution function after collisions.
        """
        # TODO(gh-173): Properly handle multi-species collisions
        # For now, only apply to electron distribution for backward compatibility
        if isinstance(f, dict):
            result = {}
            for species_name, f_species in f.items():
                if species_name == "electron":
                    result[species_name] = self._apply_collisions(nu_fp, nu_K, f_species, dt)
                else:
                    # For non-electron species, just pass through unchanged for now
                    result[species_name] = f_species
            return result
        else:
            return self._apply_collisions(nu_fp, nu_K, f, dt)

    def _solve_one_x(self, C_edge: Array, D_scalar: Array, nu: Array, f_v: Array, dt: float) -> Array:
        """Solve the collision operator at a single location in space."""
        op = self.fp_scheme.get_operator(C_edge=C_edge, D=D_scalar, nu=nu, dt=dt)
        dl_padded = jnp.pad(op.lower_diagonal, (1, 0))
        du_padded = jnp.pad(op.upper_diagonal, (0, 1))
        # Delta formulation: solve for increment to reduce floating-point error
        rhs = f_v - op.mv(f_v)
        delta = jax.lax.linalg.tridiagonal_solve(dl_padded, op.diagonal, du_padded, rhs[..., None])[..., 0]
        return f_v + delta

    def _apply_collisions(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """Apply collision operators to a single species distribution."""
        # Substitute dummy zeros for disabled operators so shard_map always receives valid arrays.
        nu_fp_in = nu_fp if nu_fp is not None else jnp.zeros(f.shape[0])
        nu_K_in = nu_K if nu_K is not None else jnp.zeros(f.shape[0])

        v = self.cfg["grid"]["species_grids"]["electron"]["v"]
        dv = self.cfg["grid"]["species_grids"]["electron"]["dv"]

        from adept.driftdiffusion import find_self_consistent_beta

        def _collide(f_shard, nu_fp_shard, nu_K_shard):
            if self.cfg["terms"]["fokker_planck"]["is_on"]:
                if hasattr(self.fp_model, "compute_beta"):
                    # Model defines its own beta closure (e.g. the energy-conserving
                    # super-Gaussian beta); the self_consistent_beta knobs control
                    # its Newton refinement of the discrete energy-flux condition.
                    beta = self.fp_model.compute_beta(
                        f_shard, rtol=self._sc_rtol, atol=self._sc_atol, max_steps=self._sc_max_steps
                    )
                else:
                    vbar = self.fp_model.compute_vbar(f_shard)
                    beta = find_self_consistent_beta(
                        f_shard,
                        v,
                        dv,
                        spherical=False,
                        vbar=vbar,
                        rtol=self._sc_rtol,
                        atol=self._sc_atol,
                        max_steps=self._sc_max_steps,
                    )
                C_edge, D = self.fp_model.compute_C_and_D(f_shard, beta)
                f_shard = vmap(self._solve_one_x, in_axes=(0, 0, 0, 0, None))(C_edge, D, nu_fp_shard, f_shard, dt)

            if self.cfg["terms"]["krook"]["is_on"]:
                f_shard = self.krook(nu_K_shard, f_shard, dt)

            return f_shard

        if self.x_parallel:
            return shard_map(
                _collide,
                mesh=self._mesh,
                in_specs=(P("device", None), P("device"), P("device")),
                out_specs=P("device", None),
            )(f, nu_fp_in, nu_K_in)
        else:
            return _collide(f, nu_fp_in, nu_K_in)


class Krook:
    """Krook relaxation operator that damps toward a Maxwellian profile."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Precompute Maxwellian profile used for Krook relaxation.

        :param cfg: Simulation configuration containing grid spacing and velocity grid.
        """
        self.cfg = cfg
        v = cfg["grid"]["species_grids"]["electron"]["v"]
        dv = cfg["grid"]["species_grids"]["electron"]["dv"]
        params = cfg["grid"].get("species_params", {}).get("electron", {})
        T0 = params.get("T0", 1.0)
        mass = params.get("mass", 1.0)
        # Maxwellian with variance T0/m (the species' bulk thermal width)
        f_mx = np.exp(-(v[None, :] ** 2.0) / (2.0 * T0 / mass))
        self.f_mx = f_mx / np.sum(f_mx, axis=1)[:, None] / dv
        self.dv = dv

    def vx_moment(self, f_xv: jnp.ndarray) -> jnp.ndarray:
        """Compute density n(x) by integrating over velocity."""
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(self, nu_K: jnp.ndarray, f_xv: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """
        Relax distribution toward a Maxwellian using Krook collisions.

        :param nu_K: Krook collision frequency profile (shape: nx).
        :param f_xv: Distribution function f(x, v) (shape: nx x nv).
        :param dt: Time step size.
        :return: Updated distribution after Krook relaxation.
        """
        nu_Kxdt = dt * nu_K[:, None]
        exp_nuKxdt = jnp.exp(-nu_Kxdt)
        n_prof = self.vx_moment(f_xv)

        return f_xv * exp_nuKxdt + n_prof[:, None] * self.f_mx * (1.0 - exp_nuKxdt)
