#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from functools import partial
from typing import Dict, Tuple

import numpy as np
from jax import numpy as jnp
import equinox as eqx

from adept.vlasov2d.solver.tridiagonal import TridiagonalSolver


class Collisions(eqx.Module):
    def __init__(self, cfg):
        super(Collisions, self).__init__()
        self.cfg = cfg
        self.fp = self.__init_fp_operator__()
        self.krook = Krook(self.cfg)
        self.td_solver = TridiagonalSolver(self.cfg)

    def __init_fp_operator__(self):
        if self.cfg["solver"]["fp_operator"] == "lenard_bernstein":
            return LenardBernstein(self.cfg)
        elif self.cfg["solver"]["fp_operator"] == "dougherty":
            return Dougherty(self.cfg)
        else:
            raise NotImplementedError

    def step_vx(self, nu_fp: jnp.float64, f: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """
        Perform vy linearized FP timestep

        :param nu_fp:
        :param f:
        :param dt:
        :return:
        """
        # get LB or DG operator
        cee_a, cee_b, cee_c = self.fp.get_vx_operator(nu=nu_fp, f_xv=f, dt=dt)
        # make it the right shape
        cee_a = jnp.transpose(cee_a, axes=(0, 1, 3, 2)).reshape(-1, self.cfg["grid"]["nvx"])
        cee_b = jnp.transpose(cee_b, axes=(0, 1, 3, 2)).reshape(-1, self.cfg["grid"]["nvx"])
        cee_c = jnp.transpose(cee_c, axes=(0, 1, 3, 2)).reshape(-1, self.cfg["grid"]["nvx"])
        f = jnp.transpose(f, axes=(0, 1, 3, 2)).reshape(-1, self.cfg["grid"]["nvx"])

        # solve it
        f = self.td_solver(cee_a, cee_b, cee_c, f)

        # shift back to x, y, vx, vy.
        # first reshape
        f = f.reshape(self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], self.cfg["grid"]["nvy"], self.cfg["grid"]["nvx"])
        # then transpose
        # NB - Remember that the new axis 3, the one used here, was the old 2. So this is actually putting it back
        # together
        f = jnp.transpose(f, axes=(0, 1, 3, 2))
        return f

    def step_vy(self, nu_fp: jnp.float64, f: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """
        Perform vy linearized FP timestep

        :param nu_fp:
        :param f:
        :param dt:
        :return:
        """
        # get LB or DG operator

        cee_a, cee_b, cee_c = self.fp.get_vy_operator(nu=nu_fp, f_xv=f, dt=dt)
        cee_a = cee_a.reshape(-1, self.cfg["grid"]["nvy"])
        cee_b = cee_b.reshape(-1, self.cfg["grid"]["nvy"])
        cee_c = cee_c.reshape(-1, self.cfg["grid"]["nvy"])
        f = f.reshape(-1, self.cfg["grid"]["nvy"])
        # Solve over all x, y, vx
        f = self.td_solver(cee_a, cee_b, cee_c, f).reshape(
            self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], self.cfg["grid"]["nvx"], self.cfg["grid"]["nvy"]
        )

        return f

    def __call__(self, nu_fp: jnp.float64, nu_K: jnp.float64, f: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        if np.any(self.cfg["grid"]["nu_prof"] > 0.0):
            # Solve x * y * vy independent linear systems for f(vx)
            f = self.step_vx(nu_fp, f, dt)

            # Solve x * y * vx independent linear systems for f(vy)
            f = self.step_vy(nu_fp, f, dt)

        if (np.any(self.cfg["grid"]["kr_prof"] > 0.0)) and (np.any(self.cfg["grid"]["kt_prof"] > 0.0)):
            f = self.krook(nu_K, f, dt)
        return f


class Krook(eqx.Module):
    def __init__(self, cfg):
        super(Krook, self).__init__()
        self.cfg = cfg
        self.dvx = self.cfg["grid"]["dvx"]
        self.dvy = self.cfg["grid"]["dvy"]
        self.f_mx = jnp.copy(self.cfg["grid"]["f"])
        self.dv = self.cfg["grid"]["dv"]

    def __call__(self, nu_K, f_xv, dt) -> jnp.ndarray:
        nu_Kxdt = dt * nu_K[:, :, None, None]
        exp_nuKxdt = jnp.exp(-nu_Kxdt)
        n_prof = jnp.sum(jnp.sum(f_xv, axis=3), axis=2) * self.dvy * self.dvx

        return f_xv * exp_nuKxdt + n_prof[:, None] * self.f_mx * (1.0 - exp_nuKxdt)


class LenardBernstein(eqx.Module):
    def __init__(self, cfg):
        super(LenardBernstein, self).__init__()
        self.cfg = cfg
        self.vx = self.cfg["grid"]["vx"]
        self.dvx = self.cfg["grid"]["dvx"]
        self.vy = self.cfg["grid"]["vy"]
        self.dvy = self.cfg["grid"]["dvy"]
        self.ones = jnp.ones(
            (self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], self.cfg["grid"]["nvx"], self.cfg["grid"]["nvy"])
        )

    def vx_moment(self, f):
        return jnp.sum(f, axis=2) * self.dvx

    def vy_moment(self, f):
        return jnp.sum(f, axis=3) * self.dvy

    def get_vx_operator(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        v0t_sq = self.vx_moment(f_xv * self.vx[None, None, :, None] ** 2.0)
        a = (
            nu
            * dt
            * (-v0t_sq[:, :, None, :] / self.dvx**2.0 + jnp.roll(self.vx, 1)[None, None, :, None] / 2 / self.dvx)
        )
        b = 1.0 + nu * dt * self.ones * (2.0 * v0t_sq[:, :, None, :] / self.dvx**2.0)
        c = (
            nu
            * dt
            * (-v0t_sq[:, :, None, :] / self.dvx**2.0 - jnp.roll(self.vx, -1)[None, None, :, None] / 2 / self.dvx)
        )
        return a, b, c

    def get_vy_operator(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        v0t_sq = self.vy_moment(f_xv * self.vy[None, None, None, :] ** 2.0)
        a = (
            nu
            * dt
            * (-v0t_sq[:, :, :, None] / self.dvy**2.0 + jnp.roll(self.vy, 1)[None, None, None, :] / 2.0 / self.dvy)
        )
        b = 1.0 + nu * dt * self.ones * (2.0 * v0t_sq[:, :, None, :] / self.dvy**2.0)
        c = (
            nu
            * dt
            * (-v0t_sq[:, :, :, None] / self.dvy**2.0 - jnp.roll(self.vy, -1)[None, None, None, :] / 2.0 / self.dvy)
        )
        return a, b, c


class Dougherty(eqx.Module):
    def __init__(self, cfg):
        super(Dougherty, self).__init__()
        self.cfg = cfg
        self.vx = self.cfg["grid"]["vx"]
        self.dvx = self.cfg["grid"]["dvx"]
        self.vy = self.cfg["grid"]["vy"]
        self.dvy = self.cfg["grid"]["dvy"]
        self.ones = jnp.ones(
            (self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], self.cfg["grid"]["nvx"], self.cfg["grid"]["nvy"])
        )

    def vx_moment(self, f):
        return jnp.sum(f, axis=2) * self.dvx

    def vy_moment(self, f):
        return jnp.sum(f, axis=3) * self.dvy

    def get_vx_operator(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        vbar = self.vx_moment(f_xv * self.vx[None, None, :, None])
        v0t_sq = self.vx_moment(f_xv * (self.vx[None, None, :, None] - vbar[:, :, None, :]) ** 2.0)

        a = (
            nu
            * dt
            * (
                -v0t_sq[:, :, None, :] / self.dvx**2.0
                + (jnp.roll(self.vx, 1)[None, None, :, None] - vbar[:, :, None, :]) / 2.0 / self.dvx
            )
        )
        b = 1.0 + nu * dt * self.ones * (2.0 * v0t_sq[:, :, None, :] / self.dvx**2.0)
        c = (
            nu
            * dt
            * (
                -v0t_sq[:, :, None, :] / self.dvx**2.0
                - (jnp.roll(self.vx, -1)[None, None, :, None] - vbar[:, :, None, :]) / 2.0 / self.dvx
            )
        )
        return a, b, c

    def get_vy_operator(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        vbar = self.vy_moment(f_xv * self.vy[None, None, None, :])
        v0t_sq = self.vy_moment(f_xv * (self.vy[None, None, None, :] - vbar[:, :, :, None]) ** 2.0)

        a = (
            nu
            * dt
            * (
                -v0t_sq[:, :, :, None] / self.dvy**2.0
                + (jnp.roll(self.vy, 1)[None, None, None, :] - vbar[:, :, :, None]) / 2.0 / self.dvy
            )
        )
        b = 1.0 + nu * dt * self.ones * (2.0 * v0t_sq[:, :, :, None] / self.dvy**2.0)
        c = (
            nu
            * dt
            * (
                -v0t_sq[:, :, :, None] / self.dvy**2.0
                - (jnp.roll(self.vy, -1)[None, None, None, :] - vbar[:, :, :, None]) / 2.0 / self.dvy
            )
        )
        return a, b, c
