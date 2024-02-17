#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from functools import partial
from typing import Dict, Tuple

import numpy as np
from jax import numpy as jnp, vmap
import lineax as lx

from adept.vlasov2d.solver.tridiagonal import TridiagonalSolver


class Collisions:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ee_fp = Dougherty(self.cfg)  # self.__init_ee_operator__()
        # self.krook = Krook(self.cfg)
        self.ei_fp = Banks(self.cfg)
        self.td_solver = TridiagonalSolver(self.cfg)
        self.coll_ee_over_x = vmap(self._single_x_ee_, in_axes=(0, 0, None))
        self.coll_ei_over_x = vmap(self._single_x_ei_, in_axes=(0, 0, None))

    def _single_x_ei_(self, nuei: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        f_vxvy = self.ei_fp.explicit_vy(nuei, f_vxvy, 0.5 * dt)
        f_vxvy = self.ei_fp.implicit_vx(nuei, f_vxvy, 0.5 * dt)
        f_vxvy = self.ei_fp.explicit_vx(nuei, f_vxvy, 0.5 * dt)
        f_vxvy = self.ei_fp.implicit_vy(nuei, f_vxvy, 0.5 * dt)

        return f_vxvy

    def _single_x_ee_(self, nu_ee: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        f_vxvy = self.ee_fp.explicit_vy(nu_ee, f_vxvy, 0.5 * dt)
        f_vxvy = self.ee_fp.implicit_vx(nu_ee, f_vxvy, 0.5 * dt)
        f_vxvy = self.ee_fp.explicit_vx(nu_ee, f_vxvy, 0.5 * dt)
        f_vxvy = self.ee_fp.implicit_vy(nu_ee, f_vxvy, 0.5 * dt)

        return f_vxvy

    def __call__(
        self, nu_ee: jnp.ndarray, nu_ei: jnp.ndarray, nu_K: jnp.ndarray, f: jnp.ndarray, dt: jnp.float64
    ) -> jnp.ndarray:
        if self.cfg["terms"]["fokker_planck"]["nu_ee"]["is_on"]:
            f = self.coll_ee_over_x(nu_ee, f, dt)

        if self.cfg["terms"]["fokker_planck"]["nu_ei"]["is_on"]:
            f = self.coll_ei_over_x(nu_ei, f, dt)

        # if self.cfg["terms"]["krook"]["is_on"]:
        #     f = self.krook(nu_K, f, dt)

        return f


class Krook:
    def __init__(self, cfg):
        self.cfg = cfg
        f_mx = np.exp(-self.cfg["grid"]["v"][None, :] ** 2.0 / 2.0)
        self.f_mx = f_mx / np.trapz(f_mx, dx=self.cfg["grid"]["dv"], axis=1)[:, None]
        self.dv = self.cfg["grid"]["dv"]
        self.vx_moment = partial(jnp.trapz, axis=1, dx=self.dv)

    def __call__(self, nu_K, f_xv, dt) -> jnp.ndarray:
        nu_Kxdt = dt * nu_K[:, None]
        exp_nuKxdt = jnp.exp(-nu_Kxdt)
        n_prof = self.vx_moment(f_xv)

        return f_xv * exp_nuKxdt + n_prof[:, None] * self.f_mx * (1.0 - exp_nuKxdt)


class Dougherty:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones(self.cfg["grid"]["nv"])
        self.scan_over_vy = vmap(self._solve_one_vslice_, in_axes=(None, 1), out_axes=1)
        self.scan_over_vx = vmap(self._solve_one_vslice_, in_axes=(None, 0), out_axes=0)

    def _solve_one_vslice_(self, operator, f_vxvy):
        return lx.linear_solve(operator, f_vxvy, solver=lx.Tridiagonal()).value

    def ddx(self, f_vxvy: jnp.ndarray):
        return jnp.gradient(f_vxvy, self.dv, axis=0)

    def ddy(self, f_vxvy: jnp.ndarray):
        return jnp.gradient(f_vxvy, self.dv, axis=1)

    def get_init_quants_x(self, f_vxvy: jnp.ndarray):
        vxbar = jnp.trapz(jnp.trapz(f_vxvy * self.v[:, None], dx=self.dv, axis=1), dx=self.dv, axis=0)
        v0t_sq = jnp.trapz(jnp.trapz(f_vxvy * (self.v[:, None] - vxbar) ** 2.0, dx=self.dv, axis=1), dx=self.dv, axis=0)
        return vxbar, v0t_sq

    def get_init_quants_y(self, f_vxvy: jnp.ndarray):
        vybar = jnp.trapz(jnp.trapz(f_vxvy * self.v[None, :], dx=self.dv, axis=1), dx=self.dv, axis=0)
        v0t_sq = jnp.trapz(jnp.trapz(f_vxvy * (self.v[None, :] - vybar) ** 2.0, dx=self.dv, axis=1), dx=self.dv, axis=0)

        return vybar, v0t_sq

    def _get_operator_(self, nu, vbar, v0t_sq, dt):
        # TODO
        lower_diagonal = nu * dt * (-v0t_sq / self.dv**2.0 + (self.v[:-1] - vbar) / 2.0 / self.dv)
        diagonal = 1.0 + nu * dt * self.ones * (2.0 * v0t_sq / self.dv**2.0)
        upper_diagonal = nu * dt * (-v0t_sq / self.dv**2.0 - (self.v[1:] - vbar) / 2.0 / self.dv)
        return lx.TridiagonalLinearOperator(
            diagonal=diagonal, upper_diagonal=upper_diagonal, lower_diagonal=lower_diagonal
        )

    def implicit_vx(self, nu: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """

        :param nu:
        :param f_vxvy:
        :param dt:
        :return:
        """

        vxbar, v0t_sq = self.get_init_quants_x(f_vxvy)
        operator = self._get_operator_(nu, vxbar, v0t_sq, dt)
        return self.scan_over_vy(operator, f_vxvy)

    def implicit_vy(self, nu: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """

        :param nu:
        :param f_vxvy:
        :param dt:
        :return:
        """

        vybar, v0t_sq = self.get_init_quants_y(f_vxvy)
        operator = self._get_operator_(nu, vybar, v0t_sq, dt)
        return self.scan_over_vx(operator, f_vxvy)

    def explicit_vx(self, nu, f_vxvy, dt: jnp.float64):
        vxbar, v0t_sq = self.get_init_quants_x(f_vxvy)
        dfdvx = self.ddx(f_vxvy)
        dfdt = nu * self.ddx((self.v[:, None] - vxbar) * f_vxvy + v0t_sq * dfdvx)
        new_fvxvy = f_vxvy + dt * nu * dfdt
        return new_fvxvy

    def explicit_vy(self, nu, f_vxvy, dt: jnp.float64):
        vybar, v0t_sq = self.get_init_quants_y(f_vxvy)
        dfdvy = self.ddy(f_vxvy)
        dfdt = nu * self.ddy((self.v[None, :] - vybar) * f_vxvy + v0t_sq * dfdvy)
        new_fvxvy = f_vxvy + dt * nu * dfdt
        return new_fvxvy


class Banks:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones(self.cfg["grid"]["nv"])
        self.scan_over_vy = vmap(self._solve_one_vy_, in_axes=(None, 0, 1, None, 1), out_axes=1)
        self.scan_over_vx = vmap(self._solve_one_vx_, in_axes=(None, 0, 0, None, 0), out_axes=0)

    def ddx(self, f_vxvy: jnp.ndarray):
        return jnp.gradient(f_vxvy, self.dv, axis=0)

    def ddy(self, f_vxvy: jnp.ndarray):
        return jnp.gradient(f_vxvy, self.dv, axis=1)

    def _get_operator_(self, nu, vxy, dfdvxy, dt):
        a = (
            nu
            * dt
            * (-(vxy**2.0) / self.dv**2.0 + (self.v[:-1] + 0.5 * self.v[:-1] * vxy * dfdvxy[:-1]) / 2.0 / self.dv)
        )
        b = 1.0 + nu * dt * self.ones * (2.0 * vxy**2.0 / self.dv**2.0)
        c = nu * dt * (-(vxy**2.0) / self.dv**2.0 - (self.v[1:] + 0.5 * self.v[1:] * vxy * dfdvxy[1:]) / 2.0 / self.dv)
        return lx.TridiagonalLinearOperator(b, a, c)

    def _solve_one_vy_(self, nu, this_vy, dfvxdvy, dt, f_vx):
        operator = self._get_operator_(nu, this_vy, dfvxdvy, dt)
        return lx.linear_solve(operator, f_vx, solver=lx.Tridiagonal()).value

    def _solve_one_vx_(self, nu, this_vx, dfvydvx, dt, f_vy):
        operator = self._get_operator_(nu, this_vx, dfvydvx, dt)
        return lx.linear_solve(operator, f_vy, solver=lx.Tridiagonal()).value

    def implicit_vx(self, nu: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """

        :param nu:
        :param f_vxvy:
        :param dt:
        :return:
        """
        dfdvy = self.ddy(f_vxvy)
        return self.scan_over_vy(nu, self.v, dfdvy, dt, f_vxvy)

    def implicit_vy(self, nu: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """

        :param nu:
        :param f_vxvy:
        :param dt:
        :return:
        """
        dfdvx = self.ddx(f_vxvy)
        return self.scan_over_vx(nu, self.v, dfdvx, dt, f_vxvy)

    def explicit_vx(self, nu: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64):
        dfdvx, dfdvy = self.ddx(f_vxvy), self.ddy(f_vxvy)
        dfdt = nu * (
            self.v[:, None] * dfdvx
            - self.v[None, :] ** 2.0 * self.ddx(dfdvx)
            + 0.5 * self.v[None, :] * self.v[:, None] * self.ddx(dfdvy)
        )
        new_fvxvy = f_vxvy + dt * nu * dfdt
        return new_fvxvy

    def explicit_vy(self, nu: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64):
        dfdvx, dfdvy = self.ddx(f_vxvy), self.ddy(f_vxvy)
        dfdt = nu * (
            self.v[None, :] * dfdvy
            - self.v[:, None] ** 2.0 * self.ddy(dfdvy)
            + 0.5 * self.v[:, None] * self.v[None, :] * self.ddy(dfdvx)
        )
        new_fvxvy = f_vxvy + dt * nu * dfdt
        return new_fvxvy
