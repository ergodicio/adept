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


class FastVFP(AbstractKernelBasedModel):
    """
    Fast VFP kernel (Bell & Sherlock): g(ε, ε') = √(ε·ε').

    This kernel is equivalent to using the discrete temperature ⟨v²⟩.
    It is a rank-1 kernel, enabling O(N) computation instead of O(N²).

    Compatible with Buet weak form scheme.

    Reference:
        Bell, A. R. & Sherlock, M. (2006). "Fast electron transport in laser
        produced plasmas." Plasma Physics and Controlled Fusion.
    """

    def apply_kernel(self, edge_values: Array) -> Array:
        """
        Apply √(ε·ε') kernel: D(ε) = 4π · √ε · ⟨√ε' · values⟩

        For rank-1 kernel, this is O(N) not O(N²):
            D[i] = 4π · √ε[i] · Σⱼ √ε[j] · edge_values[j] · Δε[j]

        Args:
            edge_values: Values at cell edges, shape (..., nv-1)

        Returns:
            Kernel integral at cell edges, shape (..., nv-1)
        """
        # Inner product: 4π · ⟨√ε · values · Δε⟩
        inner = 2.0 * jnp.pi * jnp.sum(self.sqrt_eps_edge * edge_values * self.d_eps_edge, axis=-1, keepdims=True)
        return self.sqrt_eps_edge * inner


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


class F0Collisions(eqx.Module):
    """
    Collision operator for the isotropic (f₀₀) component.

    Uses a positive-only velocity grid (0 to vmax) with zero-flux boundary conditions.
    At v=0, where the drift coefficient C=v=0, zero-flux is equivalent to a reflective
    boundary condition, correctly representing the physics of the isotropic distribution.

    The model and scheme are configurable via config["terms"]["fokker_planck"]["f00"].
    """

    v: Array
    dv: float
    nv: int
    nuee_coeff: float
    model: AbstractMaxwellianPreservingModel
    scheme: AbstractDriftDiffusionDifferencingScheme
    _sc_max_steps: int
    _sc_rtol: float
    _sc_atol: float

    def __init__(self, cfg: dict):
        """
        Initialize F0Collisions from config.

        Config should have:
        - grid.v, grid.dv, grid.nv
        - units.derived.n0, units.derived.logLambda_ee
        - terms.fokker_planck.f00.model (e.g., "spherical_lenard_bernstein")
        - terms.fokker_planck.f00.scheme (e.g., "central" or "chang_cooper")
        - terms.fokker_planck.self_consistent_beta (optional): dict with enabled, max_steps, rtol, atol
        """
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.nv = cfg["grid"]["nv"]
        self.v_edge = 0.5 * (self.v[1:] + self.v[:-1])

        # Collision coefficient
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        self.nuee_coeff = 4.0 * np.pi / 3 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

        # Create model and scheme from config
        # Use original v grid with zero_flux BC (= reflective at v=0 where C=0)
        f00_cfg = cfg["terms"]["fokker_planck"]["f00"]
        scheme_name = f00_cfg.get("scheme", "central")

        self.model = FastVFP(v=self.v, dv=self.dv)
        self.scheme = _get_scheme(scheme_name, self.dv)

        # Self-consistent beta config (defaults to disabled)
        fp_cfg = cfg["terms"]["fokker_planck"]
        sc_cfg = fp_cfg.get("self_consistent_beta", {})
        sc_enabled = sc_cfg.get("enabled", False)
        self._sc_max_steps = sc_cfg.get("max_steps", 3) if sc_enabled else 0
        self._sc_rtol = sc_cfg.get("rtol", 1e-8)
        self._sc_atol = sc_cfg.get("atol", 1e-12)

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
            self.v,
            self.dv,
            self._sc_rtol,
            self._sc_atol,
            self._sc_max_steps,
            True,  # spherical=True for positive-only grid
        )

        # Get D from model (D = 1/(2β) = T in Buet notation)
        D = self.model.compute_D(f0[None, :], beta[None])
        # Remove batch dimension for get_operator
        D = D[0] if D.ndim > 0 else D

        # Compute C_edge: v_edge for spherical LB (vbar is always None)
        C_edge = self.v_edge

        nu_scalar = jnp.array(self.nuee_coeff)
        op = self.scheme.get_operator(C_edge=C_edge, D=D, nu=nu_scalar, dt=dt)

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

    def __init__(self, cfg: dict):
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.Z = cfg["units"]["Z"]

        r_e = 2.8179402894e-13
        kp = np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        kpre = r_e * kp
        self.nuee_coeff = kpre * cfg["units"]["derived"]["logLambda_ee"]
        self.nuei_coeff = (
            kpre * self.Z**2.0 * cfg["units"]["derived"]["logLambda_ei"]
        )  # will be multiplied by ni = ne / Z

        self.nl = cfg["grid"]["nl"]
        self.ee = cfg["terms"]["fokker_planck"]["flm"]["ee"]

        self.Z_nuei_scaling = (cfg["units"]["Z"] + 4.2) / (cfg["units"]["Z"] + 0.24)

        self.a1, self.a2, self.b1, self.b2, self.b3, self.b4 = (
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
            np.zeros(self.nl + 1),
        )

        for il in range(1, self.nl + 1):
            self.a1[il] = (il + 1) * (il + 2) / (2 * il + 1) / (2 * il + 3)
            self.a2[il] = -(il - 1) * il / (2 * il + 1) / (2 * il - 1)
            self.b1[il] = (-il * (il + 1) / 2 - (il + 1)) / (2 * il + 1) / (2 * il + 3)
            self.b2[il] = (il * (il + 1) / 2 + (il + 2)) / (2 * il + 1) / (2 * il + 3)
            self.b3[il] = (il * (il + 1) / 2 + (il - 1)) / (2 * il + 1) / (2 * il - 1)
            self.b4[il] = (il * (il + 1) / 2 - il) / (2 * il + 1) / (2 * il - 1)

    def calc_ros_i(self, flm: Array, power: int) -> Array:
        r"""
        Calculates the Rosenbluth I integral.

        $$4 \pi v^{-i} \int_0^v' [   f(v') v'^(2+i)  ] dv'$$

        :param flm: the distribution function
        :param power: the power of v in the Rosenbluth integral

        :return: the Rosenbluth integral
        """
        return 4 * jnp.pi * self.v**-power * jnp.cumsum(self.v[None, :] ** (2.0 + power) * flm, axis=1) * self.dv

    def calc_ros_j(self, flm: Array, power: int) -> Array:
        r"""
        Calculates the Rosenbluth J integral.

        $$4 \pi v^{-j} \int_v^\infty [  f(v') v'^(2+j)  ] dv'$$

        """
        return (
            4
            * jnp.pi
            * self.v[None, :] ** -power
            * jnp.cumsum((self.v[None, :] ** (2.0 + power) * flm)[:, ::-1], axis=1)[:, ::-1]
            * self.dv
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

        lower_d2dv2 = (i2 + jm1) / (3.0 * self.v[None, :]) / self.dv**2.0
        diag_d2dv2 = (i2 + jm1) / (3.0 * self.v[None, :]) / self.dv**2.0
        upper_d2dv2 = (i2 + jm1) / (3.0 * self.v[None, :]) / self.dv**2.0

        diag_angular = -(-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v[None, :] ** 3.0)

        lower_ddv = (-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v[None, :] ** 2.0) / 2 / self.dv
        upper_ddv = (-i2 + 2 * jm1 + 3 * i0) / (3.0 * self.v[None, :] ** 2.0) / 2 / self.dv

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
        for il in range(1, self.nl + 1):
            ei_diag = -il * (il + 1) / 2.0 * (Z[:, None] ** 2.0) * ni[:, None] / self.v[None, :] ** 3.0

            if self.ee:
                ee_diag, ee_lower, ee_upper = self.get_ee_diagonal_contrib(f0)
                pad_f0 = jnp.concatenate([f0[:, 1::-1], f0], axis=1)
                #
                d2dv2 = (
                    0.5 / self.v[None, :] * jnp.gradient(jnp.gradient(pad_f0, self.dv, axis=1), self.dv, axis=1)[:, 2:]
                )

                ddv = self.v[None, :] ** -2.0 * jnp.gradient(pad_f0, self.dv, axis=1)[:, 2:]

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
