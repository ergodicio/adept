#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Shared Fokker-Planck operator abstractions for 1V drift-diffusion systems.

All Fokker-Planck operators in this module have the form:

    df/dt = nu * d/dv[F]   where F = C*f + D*df/dv (flux without nu)

where:
- nu: collision frequency at cell centers, scalar or shape (nv,) for velocity-dependent
- C: drift coefficient at cell edges, shape (nv-1,)
- D: diffusion coefficient at cell edges, scalar or shape (nv-1,) for velocity-dependent

Key design: nu is applied as row scaling to the cell-centered flux divergence,
NOT interpolated to edges. Each matrix row i gets multiplied by nu[i].

Buet notation is used throughout: β = 1/(2T) so the Maxwellian is f = exp(-β·v²).
This means D = 1/(2β) = T for beta-based models.

Note on conventions: Buet & Le Thanh (2007) define ε = v², β = 1/T_Buet, and
T_Buet = 2⟨v⁴⟩/(3⟨v²⟩). Their T_Buet is exactly 2x the standard kinetic temperature
T = ⟨v²⟩ (cartesian) or ⟨v⁴⟩/(3⟨v²⟩) (spherical) used in ADEPT. The extra factor
of 2 in our β = 1/(2T) accounts for this difference; the Maxwellian exp(-v²/(2T))
is identical in both conventions.

This module provides:
- Model classes that define the physics (D and vbar)
- Differencing scheme classes that define the discretization
- find_self_consistent_beta: finds β* such that a Maxwellian has a target discrete T

Class hierarchy:

AbstractMaxwellianPreservingModel          (base: v, dv, compute_D, compute_vbar)
└── AbstractKernelBasedModel               (D via kernel ∫g·f·dε)

Concrete models live alongside their solvers:
- LenardBernstein, Dougherty → adept._vlasov1d.solvers.pushers.fokker_planck
- FastVFP → adept.vfp1d.fokker_planck

Type 1 (Beta-based): Use Buet β = 1/(2T) to compute D = 1/(2β) = T.
    NOT compatible with Buet weak form scheme.

Type 2 (Kernel-based): Compute D via linear kernel ∫g·f·dε.
    Compatible with Buet weak form scheme via apply_kernel method.

All schemes use true zero-flux boundary conditions:
- At boundaries, only the interior flux contributes to the diagonal
- This conserves density exactly
- Equivalent to Maxwellian continuation beyond the grid

Composition pattern:
    model = LenardBernstein(v=v_grid, dv=dv)
    scheme = ChangCooper(dv=dv)
    D = model.compute_D(f, beta)
    vbar = model.compute_vbar(f)
    C_edge = v_edge if vbar is None else v_edge - vbar[..., None]
    op = scheme.get_operator(C_edge, D, nu, dt)
    f_new = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value

This decoupled design allows modifying C_edge or D before passing to the scheme,
enabling heating/cooling operators or composition of multiple physics effects.
"""

from abc import abstractmethod

import equinox as eqx
import lineax as lx
import optimistix as optx
from jax import Array
from jax import numpy as jnp


def chang_cooper_delta(w: Array) -> Array:
    """
    Compute Chang-Cooper weighting factor delta(w).

    The Chang-Cooper scheme uses a weighted interpolation for positivity preservation.
    The weighting factor delta is defined such that:

        delta(w) = 1/w - 1/(exp(w) - 1)

    For small |w|, a Taylor expansion is used for numerical stability:

        delta(w) ≈ 0.5 - w/12 + w³/720 + O(w⁵)

    Args:
        w: Cell Peclet number array, w = -C * dv / D

    Returns:
        Delta weighting factors, same shape as w
    """
    small = jnp.abs(w) < 1.0e-8
    delta_small = 0.5 - w / 12.0 + w**3 / 720.0
    delta_full = 1.0 / w - 1.0 / jnp.expm1(w)
    return jnp.where(small, delta_small, delta_full)


def discrete_temperature(
    f: Array, v: Array, dv: float | Array, spherical: bool = False, vbar: Array | None = None
) -> Array:
    """
    Compute the discrete temperature from a distribution function.

    For a symmetric grid: T = ⟨(v-vbar)²⟩ = ∫f·(v-vbar)²·dv / ∫f·dv
    For a spherical (positive-only) grid: T = ⟨v⁴⟩/(3⟨v²⟩)

    Args:
        f: Distribution function, shape (..., nv) where last axis is velocity.
        v: Velocity grid (cell centers), shape (nv,)
        dv: Velocity grid spacing, scalar or shape (nv,) for nonuniform grids
        spherical: If True, use spherical moment ⟨v⁴⟩/(3⟨v²⟩) instead of ⟨v²⟩
        vbar: Mean velocity, shape (...) or None. If None, assumes vbar=0.

    Returns:
        T: Discrete temperature, shape (...) matching f's batch dimensions
    """
    v_shifted = v if vbar is None else (v - vbar[..., None])
    vsq = v_shifted**2
    if spherical:
        # For spherical (positive-only grid): T = ⟨v⁴⟩/(3⟨v²⟩)
        # Note: spherical grids assume vbar=0, so v_shifted = v
        v4_moment = jnp.sum(f * vsq**2 * dv, axis=-1)
        v2_moment = jnp.sum(f * vsq * dv, axis=-1)
        return v4_moment / (3.0 * v2_moment)
    else:
        # For symmetric grid: T = ⟨(v-vbar)²⟩
        v2_moment = jnp.sum(f * vsq * dv, axis=-1)
        norm = jnp.sum(f * dv, axis=-1)
        return v2_moment / norm


def _find_self_consistent_beta_single(
    f: Array,
    v: Array,
    dv: float | Array,
    rtol: float,
    atol: float,
    max_steps: int,
    spherical: bool,
    vbar: Array | None = None,
) -> Array:
    """
    Find beta for a single spatial point (internal, not vmapped).

    Given a distribution f, finds β* such that a Maxwellian with that β*
    has the same discrete temperature as f.

    Uses Buet notation: beta = 1/(2T) so f = exp(-beta * (v-vbar)²).

    Args:
        f: Distribution function, shape (nv,)
        v: Velocity grid (cell centers), shape (nv,)
        dv: Velocity grid spacing, scalar or shape (nv,) for nonuniform grids
        rtol: Relative tolerance for Newton solver
        atol: Absolute tolerance for Newton solver
        max_steps: Maximum Newton iterations
        spherical: If True, use spherical moment ⟨v⁴⟩/(3⟨v²⟩) instead of ⟨v²⟩
        vbar: Mean velocity, scalar or None. If None, assumes vbar=0.

    Returns:
        beta_star: Beta value where Maxwellian has T_discrete = T_f, scalar
    """
    # Compute target temperature from f (using vbar if provided)
    T_target = discrete_temperature(f, v, dv, spherical, vbar)

    # Initial guess: Buet beta = 1/(2*T_f)
    beta_init = 1.0 / (2.0 * T_target)

    # Short-circuit: no SC iterations means just use the discrete-T beta
    if max_steps == 0:
        return beta_init

    def residual(beta, args):
        v, dv, T_target, spherical, vbar = args
        # Maxwellian: f = exp(-beta * (v-vbar)²)
        v_shifted = v if vbar is None else (v - vbar)
        f_maxwellian = jnp.exp(-beta * v_shifted**2)
        T_maxwellian = discrete_temperature(f_maxwellian, v, dv, spherical, vbar)
        return T_maxwellian - T_target

    solver = optx.Newton(rtol=rtol, atol=atol)
    sol = optx.root_find(
        fn=residual,
        solver=solver,
        y0=beta_init,
        args=(v, dv, T_target, spherical, vbar),
        max_steps=max_steps,
        throw=False,
    )
    return sol.value


# Vmapped version for batching over spatial points.
# eqx.filter_vmap treats None leaves as static, so vbar=None passes through
# without needing a separate vmap definition.
_find_self_consistent_beta_vmapped = eqx.filter_vmap(
    _find_self_consistent_beta_single,
    in_axes=(0, None, None, None, None, None, None, 0),
)


def find_self_consistent_beta(
    f: Array,
    v: Array,
    dv: float | Array,
    rtol: float,
    atol: float,
    max_steps: int,
    spherical: bool,
    vbar: Array | None = None,
) -> Array:
    """
    Find beta such that a Maxwellian has the same discrete temperature as f.

    Given a distribution f, finds β* such that a Maxwellian with that β*
    has the same discrete temperature as f. This eliminates equilibrium drift
    in Chang-Cooper schemes caused by the mismatch between analytical and
    discrete temperatures.

    Uses Buet notation: beta = 1/(2T) so f = exp(-beta * (v-vbar)²).

    Vmapped over f (and vbar if provided) for batching over spatial points.

    Args:
        f: Distribution function, shape (nx, nv)
        v: Velocity grid (cell centers), shape (nv,)
        dv: Velocity grid spacing, scalar or shape (nv,) for nonuniform grids
        rtol: Relative tolerance for Newton solver
        atol: Absolute tolerance for Newton solver
        max_steps: Maximum Newton iterations (default 3)
        spherical: If True, use spherical moment ⟨v⁴⟩/(3⟨v²⟩) instead of ⟨v²⟩
        vbar: Mean velocity, shape (nx,) or None. If None, assumes vbar=0.

    Returns:
        beta_star: Beta values where Maxwellian has T_discrete = T_f, shape (nx,)
    """
    return _find_self_consistent_beta_vmapped(f, v, dv, rtol, atol, max_steps, spherical, vbar)


class AbstractMaxwellianPreservingModel(eqx.Module):
    """
    Base class for all Fokker-Planck collision models.

    All models share a common interface:
    - compute_D(f, beta): diffusion coefficient
    - compute_vbar(f): mean velocity (or None)

    Type 1 (Beta-based): D = 1/(2β) = T from an external β parameter.
        NOT compatible with Buet weak form scheme.
        Examples: LenardBernstein, Dougherty

    Type 2 (Kernel-based): D via linear kernel ∫g·f·dε (β ignored).
        Compatible with Buet weak form scheme via apply_kernel method.
        Examples: FastVFP, Coulombian

    Attributes:
        v: Velocity grid (cell centers), shape (nv,)
        dv: Velocity grid spacing
    """

    v: Array
    dv: float

    def __init__(self, v: Array, dv: float):
        """Initialize model with velocity grid."""
        self.v = v
        self.dv = dv

    def compute_vbar(self, f: Array) -> Array | None:
        """
        Compute mean velocity from distribution (if model uses it).

        Args:
            f: Distribution function, shape (..., nv)

        Returns:
            vbar: Mean velocity, shape (...), or None if model doesn't use vbar.
        """
        return None  # Default for models that don't use vbar (e.g., LenardBernstein)

    @abstractmethod
    def compute_D(self, f: Array, beta: Array) -> Array:
        """
        Compute diffusion coefficient from f and beta.

        Uses Buet notation: β = 1/(2T), so D = 1/(2β) = T.

        Args:
            f: Distribution function, shape (..., nv) where nv matches len(self.v).
               Leading dimensions are batch dimensions (e.g., spatial points).
            beta: Inverse temperature in Buet notation β = 1/(2T), shape (...).

        Returns:
            D: Diffusion coefficient = 1/(2β) = T, shape (...)
        """
        ...


class AbstractKernelBasedModel(AbstractMaxwellianPreservingModel):
    """
    Type 2: Kernel-based collision model.

    Computes D via linear kernel: D = ∫g(ε, ε')·f(ε')·dε'.
    Compatible with Buet weak form scheme via the apply_kernel method.

    For Buet's weak form (eq 3.9):
    - D = apply_kernel(f_edge)
    - E = apply_kernel(Df_edge)

    Attributes:
        v: Velocity grid (cell centers), shape (nv,)
        dv: Velocity grid spacing
        sqrt_eps_edge: √ε = √(v²) at cell edges, shape (nv-1,)
        d_eps_edge: Energy spacing Δε at cell edges, shape (nv-1,)
    """

    sqrt_eps_edge: Array
    d_eps_edge: Array

    def __init__(self, v: Array, dv: float):
        """Initialize model with velocity grid."""
        super().__init__(v, dv)
        self.sqrt_eps_edge = jnp.sqrt(0.5 * (v[1:] ** 2 + v[:-1] ** 2))
        # d_eps = d(v²) = (v[i+1] + v[i]) * dv
        self.d_eps_edge = (v[1:] + v[:-1]) * dv

    def compute_D(self, f: Array, beta: Array | None = None) -> Array:
        """
        Compute D from kernel. beta is IGNORED.

        Args:
            f: Distribution at cell centers, shape (..., nv)
            beta: Ignored (can be None)

        Returns:
            D at cell edges, shape (..., nv-1)
        """
        f_edge = self._log_mean_interp(f)
        return self.apply_kernel(f_edge)

    @abstractmethod
    def apply_kernel(self, edge_values: Array) -> Array:
        """
        Apply kernel g(ε, ε') to values at cell edges.

        For Buet weak form (eq 3.9):
        - D = apply_kernel(f_edge)
        - E = apply_kernel(Df_edge)

        Args:
            edge_values: Values at cell edges, shape (..., nv-1)

        Returns:
            Kernel integral at cell edges, shape (..., nv-1)
        """
        ...

    def _log_mean_interp(self, f: Array) -> Array:
        """Interpolate cell-centered f to cell edges using log mean.

        Falls back to arithmetic mean when values are too small for log mean.
        """
        f_left = f[..., :-1]
        f_right = f[..., 1:]

        # Floor to prevent log(-inf) for very small values
        floor = 1e-300
        f_left_safe = jnp.maximum(f_left, floor)
        f_right_safe = jnp.maximum(f_right, floor)

        # Log mean: (f_R - f_L) / (log(f_R) - log(f_L))
        # Use safe version to handle f_L ≈ f_R
        log_diff = jnp.log(f_right_safe) - jnp.log(f_left_safe)
        safe_log_diff = jnp.where(jnp.abs(log_diff) < 1e-8, 1.0, log_diff)

        # Use arithmetic mean when values are very small or similar
        use_arithmetic = (jnp.abs(log_diff) < 1e-8) | (f_left < floor) | (f_right < floor)
        return jnp.where(
            use_arithmetic,
            0.5 * (f_left + f_right),  # Arithmetic mean
            (f_right - f_left) / safe_log_diff,
        )


class AbstractDriftDiffusionDifferencingScheme(eqx.Module):
    """
    Base class for Fokker-Planck differencing schemes.

    Subclasses implement specific discretizations (e.g., central, Chang-Cooper)
    of the drift-diffusion operator.

    All schemes use true zero-flux boundary conditions, which conserve
    density exactly by only including interior flux contributions at boundaries.

    The operator structure is: df/dt = nu * d/dv[F] where F = C*f + D*df/dv (flux without nu).
    nu is applied as row scaling to the cell-centered flux divergence, not interpolated to edges.

    Attributes:
        dv: Velocity grid spacing
    """

    dv: float

    @abstractmethod
    def get_operator(self, C_edge: Array, D: Array, nu: Array, dt: float) -> lx.TridiagonalLinearOperator:
        """
        Assemble a lineax TridiagonalLinearOperator for implicit solve.

        The implicit system is: (I - dt * diag(nu) * L_bare) f^{n+1} = f^n
        where L_bare is the bare flux divergence operator (without nu).

        Uses true zero-flux BC: at boundaries, only the interior flux contributes
        to the diagonal (the boundary flux is zero, so its contribution is removed).

        Args:
            C_edge: Drift coefficient at cell edges, shape (nv-1,)
            D: Diffusion coefficient at cell edges, scalar or shape (nv-1,)
            nu: Collision frequency at cell centers, scalar or shape (nv,) for velocity-dependent
            dt: Time step

        Returns:
            A lineax.TridiagonalLinearOperator ready for lx.linear_solve()
        """
        ...


class CentralDifferencing(AbstractDriftDiffusionDifferencingScheme):
    """
    Standard conservative central differencing for Fokker-Planck.

    Uses central differences for spatial derivatives:
        df/dv ≈ (f[i+1] - f[i-1]) / (2*dv)
        d²f/dv² ≈ (f[i+1] - 2*f[i] + f[i-1]) / dv²

    This scheme does not conserve density and may produce negative values.
    """

    def get_operator(self, C_edge: Array, D: Array, nu: Array, dt: float) -> lx.TridiagonalLinearOperator:
        """
        Assemble a lineax TridiagonalLinearOperator for central differencing.

        Uses true zero-flux BC: boundary diagonals only include the interior
        flux contribution (the missing boundary flux is not added).

        The operator structure is: df/dt = nu * d/dv[F] where F = C*f + D*df/dv.
        nu is applied as row scaling (each row i multiplied by nu[i]).

        Args:
            C_edge: Drift coefficient at cell edges, shape (nv-1,)
            D: Diffusion coefficient at cell edges, scalar or shape (nv-1,)
            nu: Collision frequency at cell centers, scalar or shape (nv,)
            dt: Time step

        Returns:
            A lineax.TridiagonalLinearOperator ready for lx.linear_solve()
        """
        nv = C_edge.shape[-1] + 1
        nu_full = jnp.broadcast_to(nu, (nv,)) if nu.ndim == 0 else nu

        # Flux F_{i+1/2} = (C/2 - D/dv) * f_i + (C/2 + D/dv) * f_{i+1}
        bare_diag = jnp.zeros(nv)
        bare_diag = bare_diag.at[:-1].add((C_edge / 2.0 - D / self.dv) / self.dv)
        bare_diag = bare_diag.at[1:].add(-(C_edge / 2.0 + D / self.dv) / self.dv)
        bare_upper = (C_edge / 2.0 + D / self.dv) / self.dv
        bare_lower = (-C_edge / 2.0 + D / self.dv) / self.dv

        diag = 1.0 - dt * nu_full * bare_diag
        upper_diag = -dt * nu_full[:-1] * bare_upper
        lower_diag = -dt * nu_full[1:] * bare_lower

        return lx.TridiagonalLinearOperator(
            diagonal=diag,
            upper_diagonal=upper_diag,
            lower_diagonal=lower_diag,
        )


class ChangCooper(AbstractDriftDiffusionDifferencingScheme):
    """
    Chang-Cooper positivity-preserving differencing scheme.

    Uses weighted interpolation between upwind and downwind to ensure
    the discrete solution remains non-negative. The weighting is determined
    by the local Peclet number.

    Reference: Chang & Cooper (1970), J. Comp. Phys. 6, 1-16
    """

    def get_operator(self, C_edge: Array, D: Array, nu: Array, dt: float) -> lx.TridiagonalLinearOperator:
        """
        Assemble a lineax TridiagonalLinearOperator for Chang-Cooper scheme.

        Uses true zero-flux BC: boundary diagonals only include the interior
        flux contribution (naturally handled by only adding edge contributions
        that exist).

        The operator structure is: df/dt = nu * d/dv[F] where F = C*f + D*df/dv.
        nu is applied as row scaling (each row i multiplied by nu[i]).

        Args:
            C_edge: Drift coefficient at cell edges, shape (nv-1,)
            D: Diffusion coefficient at cell edges, scalar or shape (nv-1,)
            nu: Collision frequency at cell centers, scalar or shape (nv,)
            dt: Time step

        Returns:
            A lineax.TridiagonalLinearOperator ready for lx.linear_solve()
        """
        nv = C_edge.shape[-1] + 1
        nu_full = jnp.broadcast_to(nu, (nv,)) if nu.ndim == 0 else nu

        safe_D = jnp.maximum(D, 1.0e-30)
        w = C_edge * self.dv / safe_D
        delta = chang_cooper_delta(w)

        alpha = -C_edge * delta + safe_D / self.dv
        beta = -C_edge * (1.0 - delta) - safe_D / self.dv

        bare_diag = jnp.zeros(nv)
        bare_diag = bare_diag.at[:-1].add(-alpha / self.dv)
        bare_diag = bare_diag.at[1:].add(beta / self.dv)
        bare_upper = -beta / self.dv
        bare_lower = alpha / self.dv

        diag = 1.0 - dt * nu_full * bare_diag
        upper_diag = -dt * nu_full[:-1] * bare_upper
        lower_diag = -dt * nu_full[1:] * bare_lower

        return lx.TridiagonalLinearOperator(
            diagonal=diag,
            upper_diagonal=upper_diag,
            lower_diagonal=lower_diag,
        )
