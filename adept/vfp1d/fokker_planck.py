#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
"""
Fokker-Planck collision operators for the VFP-1D solver.

The VFP-1D solver uses a positive-only velocity grid (0 to vmax) with reflecting
boundary conditions at v=0. This is different from the symmetric grid used in
Vlasov-1D.

This module provides:
- F0Collisions: Collision operator for isotropic (f₀₀) component
- FLMCollisions: Full FLM collision operator for higher-order harmonics
"""

import equinox as eqx
import lineax as lx
import numpy as np
from jax import Array, vmap
from jax import numpy as jnp

from adept.driftdiffusion import (
    AbstractDriftDiffusionDifferencingScheme,
    AbstractKernelBasedModel,
    AbstractMaxwellianPreservingModel,
    CentralDifferencing,
    ChangCooper,
)
from adept.vfp1d.grid import Grid


class FastVFP(AbstractMaxwellianPreservingModel):
    """
    FastVFP model: D = 1/(2β·v).

    This model computes D directly from the self-consistent β parameter,
    giving a constant drift coefficient C = 1.

    Does not conserve energy for non-Maxwellian distributions with the
    current Chang-Cooper scheme. A proper Buet weak-form scheme is needed.

    Reference:
        Bell, A. R. & Sherlock, M. (2024). "The fastVFP code for solution
        of the Vlasov–Fokker–Planck equation." Plasma Phys. Control. Fusion 66 035014.
    """

    v_edge: Array

    def __init__(self, v: Array, dv: float):
        """Initialize model with velocity grid."""
        super().__init__(v, dv)
        self.v_edge = 0.5 * (v[1:] + v[:-1])

    def compute_D(self, f: Array, beta: Array) -> Array:
        """
        Compute diffusion coefficient D = 1/(2β·v_edge).

        Args:
            f: Distribution function, shape (..., nv) - unused for this model
            beta: Inverse temperature β = 1/(2T), shape (...)

        Returns:
            D at cell edges, shape (..., nv-1)
        """
        del f  # Unused
        return 1.0 / (2.0 * beta[..., None] * self.v_edge)


class AsymptoticLocal(AbstractKernelBasedModel):
    """
    Asymptotic local kernel: g(ε, ε') = √(ε·ε').

    This rank-1 kernel is derived from the high-velocity limit of the
    linearized collision operator, neglecting integral terms. It enables
    O(N) computation instead of O(N²).

    Does not conserve energy for non-Maxwellian distributions with the
    current Chang-Cooper scheme. A proper Buet weak-form scheme is needed.
    """

    def apply_kernel(self, edge_values: Array) -> Array:
        """
        Apply √(ε·ε') kernel: D(ε) = √ε · ⟨√ε' · values⟩

        For rank-1 kernel, this is O(N) not O(N²):
            D[i] = √ε[i] · Σⱼ √ε[j] · edge_values[j] · Δε[j]

        Args:
            edge_values: Values at cell edges, shape (..., nv-1)

        Returns:
            Kernel integral at cell edges, shape (..., nv-1)
        """
        inner = 4.0 * jnp.pi * jnp.sum(self.sqrt_eps_edge * edge_values * self.d_eps_edge, axis=-1, keepdims=True)
        return self.sqrt_eps_edge * inner


class CoulombianKernel(AbstractKernelBasedModel):
    """
    Coulombian kernel: g(ε, ε') = min(ε^(3/2), ε'^(3/2)).

    This kernel corresponds to the standard Landau/Rosenbluth collision operator.
    It is NOT separable but can be computed efficiently in O(N) via two cumulative sums.

    Compatible with Buet weak form scheme.

    Reference:
        Buet, C. & Le Thanh, K. C. (2007). "A fully discretized scheme for
        kinetic equations with Fokker-Planck collision operator."
    """

    def apply_kernel(self, edge_values: Array) -> Array:
        """
        Apply min(ε^(3/2), ε'^(3/2)) kernel with symmetric discretization.

        D(ε_i) = Σ_j min(ε_i^(3/2), ε_j^(3/2)) · f_j · dε_j
               = Σ_{j<i} ε_j^(3/2) · f_j · dε_j + ε_i^(3/2) · Σ_{j>=i} f_j · dε_j
               = lower(ε_i) + ε_i^(3/2) · upper(ε_i)

        Uses inclusive upper (j>=i) to preserve bilinear symmetry for Buet weak form:
            Σ_i K[A]_i · B_i = Σ_i K[B]_i · A_i

        Args:
            edge_values: Values at cell edges, shape (..., nv-1)

        Returns:
            Kernel integral at cell edges, shape (..., nv-1)
        """
        eps_edge_3_2 = self.sqrt_eps_edge**3  # ε^(3/2) at edges

        # Weighted values for lower integral: ε'^(3/2) · f · dε
        weighted_vals = eps_edge_3_2 * edge_values * self.d_eps_edge
        # Unweighted values for upper integral: f · dε
        unweighted_vals = edge_values * self.d_eps_edge

        # lower[i] = Σ_{j<i} ε_j^(3/2) · values[j] · dε[j]  (exclusive cumsum from below)
        lower_cumsum = jnp.cumsum(weighted_vals, axis=-1)
        batch_shape = edge_values.shape[:-1]
        lower = jnp.concatenate([jnp.zeros((*batch_shape, 1)), lower_cumsum[..., :-1]], axis=-1)

        # upper[i] = Σ_{j>=i} values[j] · dε[j]  (inclusive reverse cumsum for symmetry)
        upper = jnp.cumsum(unweighted_vals[..., ::-1], axis=-1)[..., ::-1]

        return 4.0 * jnp.pi * (lower + eps_edge_3_2 * upper) / 3.0


def _get_model(
    model_name: str,
    v: Array,
    dv: float,
) -> AbstractMaxwellianPreservingModel:
    """Create a collision model from config name.

    Args:
        model_name: Name of the model ("CoulombianKernel", "AsymptoticLocal", "BetaBased")
        v: Velocity grid (cell centers)
        dv: Velocity grid spacing

    Returns:
        A collision model instance.
    """
    model_name = model_name.casefold()
    if model_name in ("coulombian", "coulombiankernel", "coulombian_kernel", "coulombian-kernel", "landau"):
        return CoulombianKernel(v=v, dv=dv)
    elif model_name in ("asymptoticlocal", "asymptotic_local", "asymptotic-local", "sqrt"):
        return AsymptoticLocal(v=v, dv=dv)
    elif model_name in ("fastvfp", "fast_vfp", "fast-vfp"):
        return FastVFP(v=v, dv=dv)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_scheme(
    scheme_name: str,
    dv: float,
) -> AbstractDriftDiffusionDifferencingScheme:
    """Create a differencing scheme from config name.

    Args:
        scheme_name: Name of the differencing scheme ("central", "chang_cooper", etc.)
        dv: Velocity grid spacing

    Returns:
        A differencing scheme instance (all schemes use zero-flux BC).
    """
    scheme_name = scheme_name.casefold()
    if scheme_name in ("central", "central_differencing", "central-differencing"):
        return CentralDifferencing(dv=dv)
    elif scheme_name in ("chang_cooper", "chang-cooper", "cc"):
        return ChangCooper(dv=dv)
    else:
        raise ValueError(f"Unknown scheme: {scheme_name}")


class SelfConsistentBetaConfig(eqx.Module):
    """Configuration for self-consistent beta iteration in F0Collisions."""

    max_steps: int = 0
    rtol: float = 1e-8
    atol: float = 1e-12


class F0Collisions(eqx.Module):
    """
    Collision operator for the isotropic (f₀₀) component.

    Uses a positive-only velocity grid (0 to vmax) with zero-flux boundary conditions.
    At v=0, where the drift coefficient C=v=0, zero-flux is equivalent to a reflective
    boundary condition, correctly representing the physics of the isotropic distribution.
    """

    nuee_coeff: float
    grid: Grid
    model: AbstractKernelBasedModel
    scheme: AbstractDriftDiffusionDifferencingScheme
    sc_beta: SelfConsistentBetaConfig = SelfConsistentBetaConfig()

    def _solve_one_vslice_(self, nu: float, f0: Array, dt: float) -> Array:
        """
        Solve the collision operator at a single location in space.

        :param nu: collision frequency (unused, we use nuee_coeff)
        :param f0: distribution function at a single location (nv,)
        :param dt: time step

        :return: updated distribution function (nv,)
        """
        from adept.driftdiffusion import _find_self_consistent_beta_single

        beta = _find_self_consistent_beta_single(
            f0,
            self.grid.v,
            self.grid.dv,
            spherical=True,  # spherical=True for positive-only grid
            rtol=self.sc_beta.rtol,
            atol=self.sc_beta.atol,
            max_steps=self.sc_beta.max_steps,
        )

        # Get D from model (D = 1/(2β) = T in Buet notation)
        D = self.model.compute_D(f0[None, :], beta[None])
        # Remove batch dimension for get_operator
        D = D[0] if D.ndim > 0 else D

        # Compute C_edge using the general formula: C = 2*beta*D*v
        # This ensures Chang-Cooper achieves Maxwellian equilibrium
        C_edge = 2.0 * beta * D * self.grid.v_edge

        # For spherical geometry, nu ~ 1/v² to account for the Jacobian
        nu_arr = self.nuee_coeff / self.grid.v**2
        op = self.scheme.get_operator(C_edge=C_edge, D=D, nu=nu_arr, dt=dt)

        return lx.linear_solve(op, f0, solver=lx.AutoLinearSolver(well_posed=True)).value

    def __call__(self, nu: float, f0x: Array, dt: float) -> Array:
        """
        Solve the collision operator at all locations in space.

        :param nu: collision frequency (unused, we use nuee_coeff)
        :param f0x: distribution function at all locations (nx, nv)
        :param dt: time step

        :return: updated distribution function (nx, nv)
        """
        return vmap(self._solve_one_vslice_, in_axes=(None, 0, None))(nu, f0x, dt)


class FLMCollisions:
    """
    The FLM collision operator is as described in Tzoufras2014.

    It also has an implementation of electron-electron hack
    where the off-diagonal terms in the electron-electron collision
    operator are ignored and a contribution along the diagonal is scaled by a factor depending on Z.
    """

    def __init__(self, Z: float, nuee_coeff: float, grid, logLam_ratio: float = 1.0, full_aniso_ee: bool = True):
        self.grid = grid

        self.Z = Z

        self.nuee_coeff = nuee_coeff
        self.nuei_coeff = self.nuee_coeff * self.Z**2.0 * logLam_ratio  # will be multiplied by ni = ne / Z
        self.full_aniso_ee = full_aniso_ee

        self.Z_nuei_scaling = (Z + 4.2) / (Z + 0.24)

        nl = grid.nl
        il = np.arange(1, nl + 1)
        denom_plus = (2 * il + 1) * (2 * il + 3)
        denom_minus = (2 * il + 1) * (2 * il - 1)
        ll = il * (il + 1) / 2

        a_coeffs = np.stack([(il + 1) * (il + 2) / denom_plus, -(il - 1) * il / denom_minus])
        self.a1, self.a2 = np.pad(a_coeffs, ((0, 0), (1, 0)))

        b_coeffs = np.stack(
            [
                (-ll - (il + 1)) / denom_plus,
                (ll + (il + 2)) / denom_plus,
                (ll + (il - 1)) / denom_minus,
                (ll - il) / denom_minus,
            ]
        )
        self.b1, self.b2, self.b3, self.b4 = np.pad(b_coeffs, ((0, 0), (1, 0)))

    def calc_ros_i(self, flm: Array, power: int) -> Array:
        r"""
        Calculates the Rosenbluth I integral.

        $$4 \pi v^{-i} \int_0^v' [   f(v') v'^(2+i)  ] dv'$$

        :param flm: the distribution function
        :param power: the power of v in the Rosenbluth integral

        :return: the Rosenbluth integral
        """
        return (
            4
            * jnp.pi
            * self.grid.v**-power
            * jnp.cumsum(self.grid.v[None, :] ** (2.0 + power) * flm, axis=1)
            * self.grid.dv
        )

    def calc_ros_j(self, flm: Array, power: int) -> Array:
        r"""
        Calculates the Rosenbluth J integral.

        $$4 \pi v^{-j} \int_v^\infty [  f(v') v'^(2+j)  ] dv'$$

        """
        return (
            4
            * jnp.pi
            * self.grid.v[None, :] ** -power
            * jnp.cumsum((self.grid.v[None, :] ** (2.0 + power) * flm)[:, ::-1], axis=1)[:, ::-1]
            * self.grid.dv
        )

    def get_ee_offdiagonal_contrib(self, t, y: Array, args: dict) -> Array:
        """
        The off-diagonal terms in the electron-electron collision operator are calculated explicitly.

        :param t: time
        :param y: the distribution function (nx, nv)
        :param args: the dictionary of arguments

        :return: the off-diagonal contribution to the electron-electron collision operator (nx, nv)
        """
        ddv = args["ddvf0"]
        d2dv2 = args["d2dv2f0"]
        il = args["il"]
        flm = y

        contrib = (self.a1[il] * d2dv2 + self.b1[il] * ddv) * self.calc_ros_i(flm, power=2.0 + il)
        contrib += (self.a1[il] * d2dv2 + self.b2[il] * ddv) * self.calc_ros_j(flm, power=-il - 1.0)
        contrib += (self.a2[il] * d2dv2 + self.b3[il] * ddv) * self.calc_ros_i(flm, power=il)
        if il > 1:
            contrib += (self.a2[il] * d2dv2 + self.b4[il] * ddv) * self.calc_ros_j(flm, power=1.0 - il)

        return contrib

    def get_ee_diagonal_contrib(self, f0: Array) -> Array:
        """
        Returns the tridiagonal operator for the electron-electron collision operator.

        :param f0: the distribution function (nx, nv)

        :return: tuple(diagonal, lower diagonal, upper diagonal) of shape (nx, nv), (nx, nv-1), (nx, nv-1)
        """
        i0 = self.calc_ros_i(f0, power=0.0)
        jm1 = self.calc_ros_j(f0, power=-1.0)
        i2 = self.calc_ros_i(f0, power=2.0)

        diag_term1 = 8 * jnp.pi * f0

        v = self.grid.v[None, :]
        dv = self.grid.dv

        lower_d2dv2 = (i2 + jm1) / (3.0 * v) / dv**2.0
        diag_d2dv2 = (i2 + jm1) / (3.0 * v) / dv**2.0
        upper_d2dv2 = (i2 + jm1) / (3.0 * v) / dv**2.0

        diag_angular = -(-i2 + 2 * jm1 + 3 * i0) / (3.0 * v**3.0)

        lower_ddv = (-i2 + 2 * jm1 + 3 * i0) / (3.0 * v**2.0) / 2 / dv
        upper_ddv = (-i2 + 2 * jm1 + 3 * i0) / (3.0 * v**2.0) / 2 / dv

        # adding spatial differencing coefficients here
        # 1  -2  1  for d2dv2
        # 1  -1     for ddv
        lower = lower_d2dv2 - lower_ddv
        diag = diag_term1 - 2.0 * diag_d2dv2 + diag_angular
        upper = upper_d2dv2 + upper_ddv

        diag = diag.at[:, 0].add(lower[:, 0])

        return diag, lower[:, :-1], upper[:, 1:]

    def _solve_one_x_tridiag_(self, diag: Array, upper: Array, lower: Array, f10: Array) -> Array:
        """
        Solves a tridiagonal system of equations.
        """
        op = lx.TridiagonalLinearOperator(diagonal=diag, upper_diagonal=upper, lower_diagonal=lower)
        return lx.linear_solve(op, f10, solver=lx.AutoLinearSolver(well_posed=True)).value

    def __call__(self, Z, ni, f0, f10, dt, include_ee_offdiag_explicitly=True):
        """
        Solves the FLM collision operator for all l and m.

        The solve has two options:

        1. The full ee + ei collision operator is used.
        This is done by solving the tridiagonal ee + ei implicitly and calculating the
        off-diagonal terms in the ee collision operator explicitly
        2. The ee collision operator is ignored and the Z* scaling is used instead

        """
        v = self.grid.v[None, :]
        dv = self.grid.dv
        for il in range(1, self.grid.nl + 1):
            ei_diag = -il * (il + 1) / 2.0 * (Z[:, None] ** 2.0) * ni[:, None] / v**3.0

            if self.full_aniso_ee:
                ee_diag, ee_lower, ee_upper = self.get_ee_diagonal_contrib(f0)
                pad_f0 = jnp.concatenate([f0[:, 1::-1], f0], axis=1)
                #
                d2dv2 = 0.5 / v * jnp.gradient(jnp.gradient(pad_f0, dv, axis=1), dv, axis=1)[:, 2:]

                ddv = v**-2.0 * jnp.gradient(pad_f0, dv, axis=1)[:, 2:]

                diag = 1 - dt * (self.nuei_coeff * ei_diag + self.nuee_coeff * ee_diag)
                lower = -dt * self.nuee_coeff * ee_lower
                upper = -dt * self.nuee_coeff * ee_upper

                new_f10 = vmap(self._solve_one_x_tridiag_, in_axes=(0, 0, 0, 0))(diag, upper, lower, f10)

                if include_ee_offdiag_explicitly:
                    new_f10 = new_f10 + dt * self.nuee_coeff * self.get_ee_offdiagonal_contrib(
                        None, f10, {"ddvf0": ddv, "d2dv2f0": d2dv2, "il": il}
                    )

            else:
                # only uses the Z* epperlein haines scaling instead of solving the ee collisions
                new_f10 = f10 / (1 - dt * self.nuei_coeff * self.Z_nuei_scaling * ei_diag)

        return new_f10
