from typing import Dict

from jax import numpy as jnp, Array
from jax import vmap
import numpy as np
import lineax as lx


class LenardBernstein:
    """
    The Lenard-Bernstein operator serves as the collision operator for the f00 equation

    It would be better to have a Chang-Cooper or Epperlein implementation here. That would
    also support inverse bremsstrahlung

    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]

        # these are twice as large for the solver
        self.ones = jnp.ones(2 * self.cfg["grid"]["nv"])
        self.refl_v = jnp.concatenate([-self.v[::-1], self.v])
        self.midpt = self.cfg["grid"]["nv"]
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        self.nuee_coeff = 4.0 * np.pi / 3 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

        # For debugging/experimentation, boosting/reducing isotropic ee collisions has no impact on LOCAL fields
        # i.e. this is a way to artificially tune dgree of nonlocality without affecting anything else
        # collision_boost = 50
        # print('isotropic ee collisions boosted by factor of {}, '.format(collision_boost)
        #     + 'note this has no impact on the effect on nonlocality from electron inertia')
        # self.nu_ee_coeff = collision_boost*self.nu_ee_coeff

    def _solve_one_vslice_(self, nu: float, f0: Array, dt: float) -> Array:
        """
        Solves the Lenard-Bernstein collision operator at a single location in space

        the shape of f0 is (nv,)

        :param nu: collision frequency
        :param f0: the distribution function at a single location in space (nv, )
        :param dt: the time step

        :return: the distribution function after the collision operator has been applied (nv,)
        """
        operator = self._get_operator_(nu, f0, dt)
        refl_f0 = jnp.concatenate([f0[::-1], f0])
        return lx.linear_solve(operator, refl_f0, solver=lx.Tridiagonal()).value[self.midpt :]

    def _get_operator_(self, nu: float, f0: Array, dt: float) -> lx.TridiagonalLinearOperator:
        """
        Returns the tridiagonal operator for the Lenard-Bernstein collision operator
        Cee0 = - nuee0 (kb Te d2fdv + v dfdv)

        This is called at each location in space because the f0 is different

        :param nu: collision frequency
        :param f0: the distribution function at a single location in space (nv, )
        :param dt: the time step

        :return: the tridiagonal operator for the Lenard-Bernstein collision operator for use with `lineax`

        """
        half_vth_sq = (
            jnp.sum(f0 * (self.v[None, :]) ** 4.0, axis=1) / jnp.sum(f0 * (self.v[None, :]) ** 2.0, axis=1) / 3.0
        )

        lower_diagonal = self.nuee_coeff * dt * (-half_vth_sq / self.dv**2.0 + (self.refl_v[:-1]) / 2.0 / self.dv)
        diagonal = 1.0 + self.nuee_coeff * dt * self.ones * (2.0 * half_vth_sq / self.dv**2.0)
        upper_diagonal = self.nuee_coeff * dt * (-half_vth_sq / self.dv**2.0 - (self.refl_v[1:]) / 2.0 / self.dv)
        return lx.TridiagonalLinearOperator(
            diagonal=diagonal, upper_diagonal=upper_diagonal, lower_diagonal=lower_diagonal
        )

    def __call__(self, nu: float, f0x: Array, dt: float) -> Array:
        """
        Solves the Lenard-Bernstein collision operator at all locations in space

        :param nu: collision frequency
        :param f_xv: the distribution function at all locations in space (nx, nv)
        :param dt: the time step

        :return: the distribution function after the collision operator has been applied (nx, nv)
        """

        return vmap(self._solve_one_vslice_, in_axes=(None, 0, None))(nu, f0x, dt)


class FLMCollisions:
    """
    The FLM collision operator is as described in Tzoufras2014

    It also has an implementation of electron-electron hack where the off-diagonal terms in the electron-electron collision
    operator are ignored and a contribution along the diagonal is scaled by a factor depending on Z
    """

    def __init__(self, cfg: Dict):
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
        """
        Calculates the Rosenbluth I integral

        $$4 \pi v^{-i} \int_0^v' [   f(v') v'^(2+i)  ] dv'$$

        :param flm: the distribution function
        :param power: the power of v in the Rosenbluth integral

        :return: the Rosenbluth integral
        """
        return 4 * jnp.pi * self.v**-power * jnp.cumsum(self.v[None, :] ** (2.0 + power) * flm, axis=1) * self.dv

    def calc_ros_j(self, flm: Array, power: int) -> Array:
        """
        Calculates the Rosenbluth J integral

        $$4 \pi v^{-j} \int_v^\infty [  f(v') v'^(2+j)  ] dv'$$

        """
        return (
            4
            * jnp.pi
            * self.v[None, :] ** -power
            * jnp.cumsum((self.v[None, :] ** (2.0 + power) * flm)[:, ::-1], axis=1)[:, ::-1]
            * self.dv
        )

    def get_ee_offdiagonal_contrib(self, t, y: Array, args: Dict) -> Array:
        """
        The off-diagonal terms in the electron-electron collision operator are calculated explicitly

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
        Returns the tridiagonal operator for the electron-electron collision operator

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

        return diag, lower[:, :-1], upper[:, 1:]

    def _solve_one_x_tridiag_(self, diag: Array, upper: Array, lower: Array, f10: Array) -> Array:
        """
        Solves a tridiagonal system of equations
        """
        op = lx.TridiagonalLinearOperator(diagonal=diag, upper_diagonal=upper, lower_diagonal=lower)
        return lx.linear_solve(op, f10, solver=lx.Tridiagonal()).value

    def __call__(self, Z, ni, f0, f10, dt):
        """
        Solves the FLM collision operator for all l and m

        The solve has two options

        1. The full ee + ei collision operator is used. This is done by solving the tridiagonal ee + ei implicitly and calculating the
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

                new_f10 = new_f10 + dt * self.nuee_coeff * self.get_ee_offdiagonal_contrib(
                    None, f10, {"ddvf0": ddv, "d2dv2f0": d2dv2, "il": il}
                )

            else:
                # only uses the Z* epperlein haines scaling instead of solving the ee collisions
                new_f10 = f10 / (1 - dt * self.nuei_coeff * self.Z_nuei_scaling * ei_diag)

        return new_f10
