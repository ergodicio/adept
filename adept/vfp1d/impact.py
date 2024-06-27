from typing import Dict

import numpy as np
from jax import numpy as jnp
import lineax as lx


class IMPACT:
    """
    UNUSED

    """

    def __init__(self, cfg):
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
        return jnp.real(jnp.ifft(jnp.fft(f, axis=0) * 1j * self.kx), axis=0)

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
