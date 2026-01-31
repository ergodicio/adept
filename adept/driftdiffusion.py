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

This module provides:
- Model classes that define the physics (C_edge and D)
- Differencing scheme classes that define the discretization

All schemes use true zero-flux boundary conditions:
- At boundaries, only the interior flux contributes to the diagonal
- This conserves density exactly
- Equivalent to Maxwellian continuation beyond the grid

Composition pattern:
    model = LenardBernstein(v=v_grid, dv=dv)
    scheme = ChangCooper(dv=dv)
    C_edge, D = model(f)
    op = scheme.get_operator(C_edge, D, nu, dt)
    f_new = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value

This decoupled design allows modifying C_edge or D before passing to the scheme,
enabling heating/cooling operators or composition of multiple physics effects.
"""

from abc import abstractmethod

import equinox as eqx
import lineax as lx
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


class AbstractDriftDiffusionModel(eqx.Module):
    """
    Base class for drift-diffusion physics models.

    Subclasses define the drift coefficient C and diffusion coefficient D
    for Fokker-Planck operators of the form: df/dt = nu * d/dv (C*f + D*df/dv)

    Attributes:
        v: Velocity grid (cell centers), shape (nv,)
        dv: Velocity grid spacing
        v_edge: Velocity grid (cell edges), shape (nv-1,)
    """

    v: Array
    dv: float
    v_edge: Array

    def __init__(self, v: Array, dv: float):
        """Initialize model with velocity grid."""
        self.v = v
        self.dv = dv
        self.v_edge = 0.5 * (v[1:] + v[:-1])

    @abstractmethod
    def __call__(self, f: Array) -> tuple[Array, Array]:
        """
        Compute drift and diffusion coefficients from distribution function.

        Args:
            f: Distribution function, shape (..., nv) where nv matches len(self.v).
               Leading dimensions are batch dimensions (e.g., spatial points).

        Returns:
            Tuple of (drift C_edge, diffusion D):
            - C_edge: Drift coefficient at cell edges, shape (..., nv-1)
            - D: Diffusion coefficient, scalar shape (...) or edge-valued shape (..., nv-1)
        """
        ...


class LenardBernstein(AbstractDriftDiffusionModel):
    """
    Lenard-Bernstein collision model.

    Physics:
        C = v (drift toward v=0)
        D = <v²> = integral(f * v² dv) (diffusion proportional to energy)

    The equilibrium solution is a Maxwellian centered at v=0.
    """

    def __call__(self, f: Array) -> tuple[Array, Array]:
        """
        Compute Lenard-Bernstein drift and diffusion coefficients.

        Args:
            f: Distribution function, shape (..., nv)

        Returns:
            Tuple (drift C_edge, diffusion D):
            - C_edge: velocity at cell edges broadcast to f shape, shape (..., nv-1)
            - D: energy <v²>, shape (...)
        """
        energy = jnp.sum(f * self.v**2, axis=-1) * self.dv
        C_edge = jnp.broadcast_to(self.v_edge, f.shape[:-1] + (len(self.v_edge),))
        return C_edge, energy


class SphericalLenardBernstein(AbstractDriftDiffusionModel):
    """
    Lenard-Bernstein model for spherical harmonic (f₀₀) equations.

    Physics:
        C = v (drift toward v=0)
        D = <v⁴>/<v²>/3 (thermal velocity for spherical case)

    This model is used in the VFP1D solver for the isotropic part of the
    distribution function. The diffusion coefficient accounts for the
    velocity space Jacobian in spherical coordinates.

    For a Maxwellian, D = <v⁴>/<v²>/3 = T, same as <v²>.
    """

    def __call__(self, f: Array) -> tuple[Array, Array]:
        """
        Compute spherical Lenard-Bernstein drift and diffusion coefficients.

        Args:
            f: Distribution function, shape (..., nv)

        Returns:
            Tuple (drift C_edge, diffusion D):
            - C_edge: velocity at cell edges broadcast to f shape, shape (..., nv-1)
            - D: <v⁴>/<v²>/3, shape (...)
        """
        v4_moment = jnp.sum(f * self.v**4, axis=-1) * self.dv
        v2_moment = jnp.sum(f * self.v**2, axis=-1) * self.dv
        safe_v2 = jnp.maximum(v2_moment, 1.0e-30)
        diffusion = v4_moment / safe_v2 / 3.0
        C_edge = jnp.broadcast_to(self.v_edge, f.shape[:-1] + (len(self.v_edge),))
        return C_edge, diffusion


class Dougherty(AbstractDriftDiffusionModel):
    """
    Dougherty collision model.

    Physics:
        C = v - <v> (drift toward mean velocity)
        D = <(v - <v>)²> (thermal velocity squared)

    The equilibrium solution is a Maxwellian centered at the mean velocity <v>.
    This model conserves momentum in addition to density and energy.
    """

    def __call__(self, f: Array) -> tuple[Array, Array]:
        """
        Compute Dougherty drift and diffusion coefficients.

        Args:
            f: Distribution function, shape (..., nv)

        Returns:
            Tuple (drift C_edge, diffusion D):
            - C_edge: v_edge - <v>, shape (..., nv-1)
            - D: thermal velocity squared <(v-<v>)²>, shape (...)
        """
        vbar = jnp.sum(f * self.v, axis=-1) * self.dv
        v0t_sq = jnp.sum(f * (self.v - vbar[..., None]) ** 2, axis=-1) * self.dv
        C_edge = self.v_edge - vbar[..., None]
        return C_edge, v0t_sq


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
