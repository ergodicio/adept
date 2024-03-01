from jax import numpy as jnp
from jax import vmap
import numpy as np
import lineax as lx


class LenardBernstein:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]

        # these are twice as large for the solver
        self.ones = jnp.ones(2 * self.cfg["grid"]["nv"])
        self.refl_v = jnp.concatenate([-self.v[::-1], self.v])
        self.midpt = self.cfg["grid"]["nv"]
        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)

        self.nuee_coeff = 4.0 * np.pi / 3.0 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

    def _solve_one_vslice_(self, nu: jnp.float64, f0: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        operator = self._get_operator_(nu, f0, dt)
        refl_f0 = jnp.concatenate([f0[::-1], f0])
        return lx.linear_solve(operator, refl_f0, solver=lx.Tridiagonal()).value[self.midpt :]

    def _get_operator_(self, nu, f0, dt):
        half_vth_sq = (
            jnp.sum(f0 * (self.v[None, :]) ** 4.0, axis=1) / jnp.sum(f0 * (self.v[None, :]) ** 2.0, axis=1) / 3.0
        )

        lower_diagonal = self.nuee_coeff * dt * (-half_vth_sq / self.dv**2.0 + (self.refl_v[:-1]) / 2.0 / self.dv)
        diagonal = 1.0 + self.nuee_coeff * dt * self.ones * (2.0 * half_vth_sq / self.dv**2.0)
        upper_diagonal = self.nuee_coeff * dt * (-half_vth_sq / self.dv**2.0 - (self.refl_v[1:]) / 2.0 / self.dv)
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


class ElectronIonCollisions:
    def __init__(self, cfg):
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.Z = cfg["units"]["Z"]

        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        self.nuei_coeff = c_kpre * self.Z * cfg["units"]["derived"]["logLambda_ei"]

    def __call__(self, nuei_profile, f10, dt):
        return f10 / (1.0 + self.nuei_coeff * dt / self.v[None, :] ** 3.0)
