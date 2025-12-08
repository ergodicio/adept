#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

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
        fp_type = self.cfg["terms"]["fokker_planck"]["type"].casefold()
        if fp_type == "lenard_bernstein":
            return LenardBernstein(self.cfg)
        elif fp_type in ("chang_cooper", "lenard_bernstein_chang_cooper"):
            return ChangCooperLenardBernstein(self.cfg)
        elif fp_type in ("chang_cooper_dougherty", "dougherty_chang_cooper"):
            return ChangCooperDougherty(self.cfg)
        elif fp_type == "dougherty":
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
        f_mx = np.exp(-(self.cfg["grid"]["v"][None, :] ** 2.0) / 2.0)
        self.f_mx = f_mx / np.trapz(f_mx, dx=self.cfg["grid"]["dv"], axis=1)[:, None]
        self.dv = self.cfg["grid"]["dv"]

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(self, nu_K, f_xv, dt) -> jnp.ndarray:
        nu_Kxdt = dt * nu_K[:, None]
        exp_nuKxdt = jnp.exp(-nu_Kxdt)
        n_prof = self.vx_moment(f_xv)

        return f_xv * exp_nuKxdt + n_prof[:, None] * self.f_mx * (1.0 - exp_nuKxdt)


class LenardBernstein:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        v0t_sq = self.vx_moment(f_xv * self.v[None, :] ** 2.0)

        # Build coefficients without periodic boundaries
        # a = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 + self.v[None, :] / 2 / self.dv)
        a = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 + jnp.roll(self.v, 1)[None, :] / 2 / self.dv)

        b = 1.0 + nu[:, None] * dt * self.ones * (2.0 * v0t_sq[:, None] / self.dv**2.0)
        # c = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 - self.v[None, :] / 2 / self.dv)
        c = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 - jnp.roll(self.v, -1)[None, :] / 2 / self.dv)


        # Enforce zero-flux boundary conditions
        # a = a.at[:, 0].set(0.0)
        # c = c.at[:, -1].set(0.0)

        return a, b, c


class ChangCooperLenardBernstein:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.v_edge = 0.5 * (self.v[1:] + self.v[:-1])
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    @staticmethod
    def _chang_cooper_delta(w):
        small = jnp.abs(w) < 1.0e-8
        delta_small = 0.5 - w / 12.0 + w**3 / 720.0
        delta_full = 1.0 / w - 1.0 / jnp.expm1(w)
        return jnp.where(small, delta_small, delta_full)

    def __call__(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        energy = self.vx_moment(f_xv * self.v[None, :] ** 2.0)
        safe_energy = jnp.maximum(energy, 1.0e-30)

        # Chang-Cooper weights are independent of nu because drift/diffusion share it.
        w = -self.v_edge[None, :] * self.dv / safe_energy[:, None]
        delta = self._chang_cooper_delta(w)

        nu = nu[:, None]
        diff = nu * energy[:, None]
        drift = -nu * self.v_edge[None, :]
        alpha = drift * delta + diff / self.dv
        beta = drift * (1.0 - delta) - diff / self.dv

        lam = dt / self.dv
        a = jnp.zeros_like(self.ones)
        a = a.at[:, 1:].set(-lam * alpha)
        c = jnp.zeros_like(self.ones)
        c = c.at[:, :-1].set(lam * beta)

        beta_l = jnp.concatenate([jnp.zeros((self.cfg["grid"]["nx"], 1)), beta], axis=1)
        alpha_r = jnp.concatenate([alpha, jnp.zeros((self.cfg["grid"]["nx"], 1))], axis=1)
        diag = self.ones + lam * (alpha_r - beta_l)

        return a, diag, c


class ChangCooperDougherty:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.v_edge = 0.5 * (self.v[1:] + self.v[:-1])
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    @staticmethod
    def _chang_cooper_delta(w):
        small = jnp.abs(w) < 1.0e-8
        delta_small = 0.5 - w / 12.0 + w**3 / 720.0
        delta_full = 1.0 / w - 1.0 / jnp.expm1(w)
        return jnp.where(small, delta_small, delta_full)

    def __call__(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        vbar = self.vx_moment(f_xv * self.v[None, :])
        v0t_sq = self.vx_moment(f_xv * (self.v[None, :] - vbar[:, None]) ** 2.0)
        safe_v0t_sq = jnp.maximum(v0t_sq, 1.0e-30)

        w = -(self.v_edge[None, :] - vbar[:, None]) * self.dv / safe_v0t_sq[:, None]
        delta = self._chang_cooper_delta(w)

        nu = nu[:, None]
        diff = nu * safe_v0t_sq[:, None]
        drift = -nu * (self.v_edge[None, :] - vbar[:, None])
        alpha = drift * delta + diff / self.dv
        beta = drift * (1.0 - delta) - diff / self.dv

        lam = dt / self.dv
        a = jnp.zeros_like(self.ones)
        a = a.at[:, 1:].set(-lam * alpha)
        c = jnp.zeros_like(self.ones)
        c = c.at[:, :-1].set(lam * beta)

        beta_l = jnp.concatenate([jnp.zeros((self.cfg["grid"]["nx"], 1)), beta], axis=1)
        alpha_r = jnp.concatenate([alpha, jnp.zeros((self.cfg["grid"]["nx"], 1))], axis=1)
        diag = self.ones + lam * (alpha_r - beta_l)

        return a, diag, c


class Dougherty:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(
        self, nu: jnp.float64, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        vbar = self.vx_moment(f_xv * self.v[None, :])
        v0t_sq = self.vx_moment(f_xv * (self.v[None, :] - vbar[:, None]) ** 2.0)

        # Build coefficients without periodic boundaries
        a = (
            nu[:, None]
            * dt
            * (-v0t_sq[:, None] / self.dv**2.0 + (jnp.roll(self.v, 1)[None, :] - vbar[:, None]) / 2.0 / self.dv)

        )
        b = 1.0 + nu[:, None] * dt * self.ones * (2.0 * v0t_sq[:, None] / self.dv**2.0)
        c = (
            nu[:, None]
            * dt
            * (-v0t_sq[:, None] / self.dv**2.0 - (jnp.roll(self.v, -1)[None, :] - vbar[:, None]) / 2.0 / self.dv)

        )
        
        return a, b, c
