import numpy as np
from typing import Dict
from jax import numpy as jnp, vmap
import lineax as lx


class IMPACT:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.m_frac = 1.0 / 1836.0 / 10.0

        self.Z = cfg["units"]["Z"]
        self.dx = cfg["grid"]["dx"]
        self.dt = cfg["grid"]["dt"]
        self.nx = cfg["grid"]["nx"]
        # self.c_squared = cfg["units"]["derived"]["c_light"].magnitude ** 2.0
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)

        self.nuei_coeff = c_kpre * self.Z**2.0 * cfg["units"]["derived"]["logLambda_ei"]
        self.nuee_coeff = 4.0 * np.pi / 3.0 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

    def ddv(self, f):
        temp = jnp.concatenate([f[:, :1], f], axis=1)
        return jnp.gradient(temp, self.dv, axis=1)[:, 1:]

    def ddv_f1(self, f):
        temp = jnp.concatenate([-f[:, :1], f], axis=1)
        return jnp.gradient(temp, self.dv, axis=1)[:, 1:]

    def calc_Ij(self, this_f, j=0):
        return 4.0 * jnp.pi / self.v**j * jnp.cumsum(this_f * self.v ** (2.0 + j), axis=1) * self.dv

    def calc_Jmj(self, this_f, j=0):
        temp = this_f * self.v ** (2.0 + j)
        return 4.0 * jnp.pi / self.v**j * jnp.cumsum(temp[::-1], axis=1)[::-1] * self.dv

    def calc_Cee0(self, f0, prev_f0, I00, I20, Jm10):

        # term1 = 1 / 3 / self.v * self.ddv(self.ddv(f0)) * (I20 + Jm10)
        # term2 = 1 / 3 / self.v**2 * self.ddv(f0) * (2 * Jm10 - I20 + 3 * I00 * self.m_frac)
        # term3 = 4 * jnp.pi * f0 * prev_f0 * self.m_frac

        # Cee0 = 0 * self.Y / 3.0 / self.v**2.0 * self.ddv(3 * f0 * I00 + self.v * (I20 + Jm10) * self.ddv(f0))

        # Cee0 = self.Y * (term1 + term2 + term3)

        prev_n_over_4pi = jnp.sum(self.v**2.0 * prev_f0, axis=1)
        prev_T = jnp.sum(self.v**4.0 * prev_f0, axis=1) / 3.0 / prev_n_over_4pi

        Cee0 = self.nuee_coeff * self.ddv(self.v * f0 + prev_T[:, None] * self.ddv(f0))

        # Cf0 = 4 * jnp.pi * jnp.cumsum(f0 * self.v**2.0, axis=1) * self.dv
        # Df0 = (
        #     4
        #     * jnp.pi
        #     / 3.0
        #     * (
        #         1 / self.v * jnp.cumsum(f0 * self.v[None, :] ** 4, axis=1) * self.dv
        #         + self.v**2 * jnp.cumsum((f0 * self.v[None, :])[::-1], axis=1)[::-1] * self.dv
        #     )
        # )
        # return 1 / self.Z / self.v**2.0 * self.ddv(Cf0 * f0 + Df0 * self.ddv(f0)), None, None, None

        return Cee0

    def calc_C1(self, f1, f0, I00, I20, Jm10, I31, Jm21, I11):

        term1 = 1.0 / 3.0 / self.v * self.ddv(self.ddv(f1)) * (I20 + Jm10)
        term2 = 1.0 / 3.0 / self.v**2.0 * self.ddv(f1) * (3 * I00 * self.m_frac - I20 + Jm10)
        term3 = -1.0 / 3.0 / self.v[None, :] ** 3.0 * f1 * (3 + 0 * (-I20 + 3 * I00 + 2 * Jm10))
        term4 = 8 * jnp.pi * (f1) * self.m_frac
        term5 = 1.0 / 5.0 / self.v * self.ddv(self.ddv(f0)) * (I31 + Jm21)
        term6 = (
            1.0
            / 5.0
            / self.v
            * self.ddv(f0)
            * (-3 * I31 + (7 - 5 * self.m_frac) * Jm21 + (-5 + 10 * self.m_frac) * I11)
        )

        return self.nuei_coeff * (0 * term1 + 0 * term2 + term3 + 0 * term4 + 0 * term5 + 0 * term6)

    def ddx(self, f):
        aug_f = jnp.concatenate([f[-1:], f, f[:1:]], axis=0)
        return jnp.gradient(aug_f, self.dx, axis=0)[1:-1]

    def get_step(self, prev_f0, prev_f1):
        """
        This is for the linear solver

        """

        I00 = self.calc_Ij(prev_f0, j=0)
        I20 = self.calc_Ij(prev_f0, j=2)
        Jm10 = self.calc_Jmj(prev_f0, j=1)

        I31 = self.calc_Ij(prev_f1, j=3)
        Jm21 = self.calc_Jmj(prev_f1, j=2)
        I11 = self.calc_Ij(prev_f1, j=1)

        def _step_(this_y):
            f0, f1, e = this_y["f0"], this_y["f1"], this_y["e"]
            Cee0 = self.calc_Cee0(f0, prev_f0, I00, I20, Jm10)

            prev_f0_approx = f0 + self.dt * (
                self.v[None, :] / 3.0 * self.ddx(f1)
                - 1.0 / 3.0 / self.v[None, :] ** 2.0 * self.ddv_f1(self.v[None, :] ** 2.0 * (e[:, None] * f1))
                - Cee0
            )
            C_f1 = self.calc_C1(f1, f0, I00, I20, Jm10, I31, Jm21, I11)
            prev_f1_approx = f1 + self.dt * (self.v[None, :] * self.ddx(f0) - e[:, None] * self.ddv(f0) - C_f1)

            j = -4 * jnp.pi / 3.0 * jnp.sum(f1 * self.v[None, :] ** 3.0, axis=1) * self.dv
            prev_e_approx = e + self.dt * j

            return {"f0": prev_f0_approx, "f1": prev_f1_approx, "e": prev_e_approx}

        return _step_

    def __call__(self, t, y, args) -> Dict:
        # return y
        f0, f1, e = y["f0"], y["f1"], y["e"]
        operator = lx.FunctionLinearOperator(self.get_step(f0, f1), input_structure={"f0": f0, "f1": f1, "e": e})
        rhs = {"f0": f0, "f1": f1, "e": e}
        solver = lx.BiCGStab(
            rtol=1e-3, atol=1e-4, max_steps=16384, norm=lx.internal.max_norm  # restart=128, stagnation_iters=128
        )
        sol = lx.linear_solve(operator, rhs, solver=solver, options={"y0": {"f0": f0, "f1": f1, "e": e}})

        return {"f0": sol.value["f0"], "f1": sol.value["f1"], "e": sol.value["e"], "b": y["b"]}


class LenardBernstein:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]

        # these are twice as large for the solver
        self.ones = jnp.ones(2 * self.cfg["grid"]["nv"])
        self.refl_v = jnp.concatenate([-self.v[::-1], self.v])
        self.midpt = self.cfg["grid"]["nv"]

    def _solve_one_vslice_(self, nu: jnp.float64, f0: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        operator = self._get_operator_(nu, f0, dt)
        refl_f0 = jnp.concatenate([f0[::-1], f0])
        return lx.linear_solve(operator, refl_f0, solver=lx.Tridiagonal()).value[self.midpt :]

    def _get_operator_(self, nu, f0, dt):
        half_vth_sq = (
            jnp.sum(f0 * (self.v[None, :]) ** 4.0, axis=1) / jnp.sum(f0 * (self.v[None, :]) ** 2.0, axis=1) / 3.0
        )

        lower_diagonal = nu * dt * (-half_vth_sq / self.dv**2.0 + (self.refl_v[:-1]) / 2.0 / self.dv)
        diagonal = 1.0 + nu * dt * self.ones * (2.0 * half_vth_sq / self.dv**2.0)
        upper_diagonal = nu * dt * (-half_vth_sq / self.dv**2.0 - (self.refl_v[1:]) / 2.0 / self.dv)
        return lx.TridiagonalLinearOperator(
            diagonal=diagonal, upper_diagonal=upper_diagonal, lower_diagonal=lower_diagonal
        )

    def __call__(self, nu: jnp.float64, f0x: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """

        :param nu:
        :param f_vxvy:
        :param dt:
        :return:
        """

        return vmap(self._solve_one_vslice_, in_axes=(None, 0, None))(nu, f0x, dt)


class EleectronIonCollisions:
    def __init__(self, cfg):
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]

    def __call__(self, nuei_coeff, f10, dt):
        return f10 / (1.0 + nuei_coeff * dt / self.v[None, :] ** 3.0)


class OSHUN1D:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.m_frac = 1.0 / 1836.0 / 10.0

        self.Z = cfg["units"]["Z"]
        self.dx = cfg["grid"]["dx"]
        self.dt = cfg["grid"]["dt"]
        self.nx = cfg["grid"]["nx"]
        # self.c_squared = cfg["units"]["derived"]["c_light"].magnitude ** 2.0
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)

        self.nuei_coeff = c_kpre * self.Z**2.0 * cfg["units"]["derived"]["logLambda_ei"]
        self.nuee_coeff = 4.0 * np.pi / 3.0 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

        self.lb = LenardBernstein(cfg)
        self.ei = EleectronIonCollisions(cfg)

        self.large_eps = 1e-7
        self.eps = 1e-14

    def ddv(self, f):
        temp = jnp.concatenate([f[:, :1], f], axis=1)
        return jnp.gradient(temp, self.dv, axis=1)[:, 1:]

    def ddv_f1(self, f):
        temp = jnp.concatenate([-f[:, :1], f], axis=1)
        return jnp.gradient(temp, self.dv, axis=1)[:, 1:]

    def ddx(self, f):
        periodic_f = jnp.concatenate([f[-1:], f, f[:1:]], axis=0)
        return jnp.gradient(periodic_f, self.dx, axis=0)[1:-1]

    def step_f0_coll(self, f0):
        return self.lb(self.nuee_coeff, f0, self.dt)

    def step_f10_coll(self, f10):
        return self.ei(self.nuei_coeff, f10, self.dt)

    def calc_j(self, f1):
        return -4 * jnp.pi / 3.0 * jnp.sum(f1 * self.v[None, :] ** 3.0, axis=1) * self.dv

    def implicit_e_solve(self, f0, f10, e):

        # calculate j without any e field
        f10_after_coll = self.step_f10_coll(f10)
        j0 = self.calc_j(f10_after_coll)

        # get perturbation
        de = e**2.0 * self.large_eps + self.eps

        # calculate effect of dex
        g00 = self.ddv(f0)
        df10dt = de[:, None] * g00
        f10_after_dex = f10 + self.dt * df10dt
        f10_after_dex = self.step_f10_coll(f10_after_dex)
        jx_dx = self.calc_j(f10_after_dex)
        # jy_dx = 0.0
        # jz_dx = 0.0

        # f10_after_dey = self.step_f10_coll(self.apply_dey(f10))
        # jx_dy = -4 * jnp.pi / 3.0 * jnp.sum(f10_after_dey * self.v[None, :] ** 3.0, axis=1) * self.dv
        # jy_dy = 0.0
        # jz_dy = 0.0

        # f10_after_dez = self.step_f10_coll(self.apply_dez(f10))
        # jx_dz = -4 * jnp.pi / 3.0 * jnp.sum(f10_after_dez * self.v[None, :] ** 3.0, axis=1) * self.dv
        # jy_dz = 0.0
        # jz_dz = 0.0

        # directly solve for ex
        new_e = -j0 * de / (jx_dx - j0)

        return new_e

    def __call__(self, t, y, args) -> Dict:

        f0 = y["f0"]
        f10 = y["f10"]

        df0dt_sa = -self.v[None, :] / 3.0 * self.ddx(f10)
        df10dt_sa = -self.v[None, :] * self.ddx(y["f0"])

        # push advection
        f0_star = f0 + self.dt * df0dt_sa
        f10_star = f10 + self.dt * df10dt_sa

        # push f00 coll
        f0_star = self.step_f0_coll(f0_star)

        # solve for Enp1
        new_e = self.implicit_e_solve(f0_star, f10_star, y["e"])

        # push e
        g00 = self.ddv(f0)
        h10 = 2.0 / self.v * f10 + self.ddv_f1(f10)

        df0dt_e = new_e[:, None] / 3.0 * h10
        df10dt_e = new_e[:, None] * g00

        new_f0 = f0_star + self.dt * df0dt_e
        new_f10 = f10_star + self.dt * df10dt_e

        # push f10 coll
        # new_f0 = self.step_f0_coll(new_f0)
        new_f10 = self.step_f10_coll(new_f10)

        return {"f0": new_f0, "f10": new_f10, "e": new_e, "b": y["b"]}
