#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from functools import partial
from typing import Dict, Tuple

import numpy as np
from jax import numpy as jnp

from adept.vlasov2d.solver.tridiagonal import TridiagonalSolver


class Collisions:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fp = self.__init_fp_operator__()
        self.krook = Krook(self.cfg)
        self.td_solver = TridiagonalSolver(self.cfg)

    def __init_fp_operator__(self):
        if self.cfg["terms"]["fokker_planck"]["type"].casefold() == "lenard_bernstein":
            return LenardBernstein(self.cfg)
        elif self.cfg["terms"]["fokker_planck"]["type"].casefold() == "dougherty":
            return Dougherty(self.cfg)
        else:
            raise NotImplementedError

    def __call__(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            # The three diagonals representing collision operator for all x
            cee_a, cee_b, cee_c = self.fp(nu=nu_fp, f_xv=f, dt=dt)
            # Solve over all x
            f = self.td_solver(cee_a, cee_b, cee_c, f)

        if self.cfg["terms"]["krook"]["is_on"]:
            f = self.krook(nu_K, f, dt)

        return f


class Krook:
    def __init__(self, cfg):
        self.cfg = cfg
        f_mx = np.exp(-self.cfg["grid"]["v"][None, :] ** 2.0 / 2.0)
        self.f_mx = f_mx / np.sum(f_mx, axis=1)[:, None] / self.cfg["grid"]["dv"]
        self.dv = self.cfg["grid"]["dv"]

    def moment_x(self, f):
        return jnp.sum(f, axis=1) * self.dv

    def __call__(self, nu_K, f_xv, dt) -> jnp.ndarray:
        nu_Kxdt = dt * nu_K[:, None]
        exp_nuKxdt = jnp.exp(-nu_Kxdt)
        n_prof = self.moment_x(f_xv)

        return f_xv * exp_nuKxdt + n_prof[:, None] * self.f_mx * (1.0 - exp_nuKxdt)


class LenardBernstein:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        self.nuee_coeff = 4.0 * np.pi / 3 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

    def moment_x(self, f):
        return jnp.sum(f, axis=1) * self.dv

    def __call__(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """
        nu_eff = nu * self.nuee
        v0t_sq = self.moment_x(f_xv * self.v[None, :] ** 2.0)
        a = nu_eff[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 + jnp.roll(self.v, 1)[None, :] / 2 / self.dv)
        b = 1.0 + nu_eff[:, None] * dt * self.ones * (2.0 * v0t_sq[:, None] / self.dv**2.0)
        c = nu_eff[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 - jnp.roll(self.v, -1)[None, :] / 2 / self.dv)
        return a, b, c


class Dougherty:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        self.nuee_coeff = 4.0 * np.pi / 3 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

    def moment_x(self, f):
        return jnp.sum(f, axis=1) * self.dv

    def __call__(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        nu_eff = nu * self.nuee_coeff
        vbar = self.moment_x(f_xv * self.v[None, :])
        v0t_sq = self.moment_x(f_xv * (self.v[None, :] - vbar[:, None]) ** 2.0)

        a = (
            nu_eff[:, None]
            * dt
            * (-v0t_sq[:, None] / self.dv**2.0 + (jnp.roll(self.v, 1)[None, :] - vbar[:, None]) / 2.0 / self.dv)
        )
        b = 1.0 + nu_eff[:, None] * dt * self.ones * (2.0 * v0t_sq[:, None] / self.dv**2.0)
        c = (
            nu_eff[:, None]
            * dt
            * (-v0t_sq[:, None] / self.dv**2.0 - (jnp.roll(self.v, -1)[None, :] - vbar[:, None]) / 2.0 / self.dv)
        )
        return a, b, c
