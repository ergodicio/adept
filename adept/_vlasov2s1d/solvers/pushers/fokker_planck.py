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
        if self.cfg["terms"]["fokker_planck"]["type"].casefold() == "lenard_bernstein":
            return LenardBernstein(self.cfg)
        elif self.cfg["terms"]["fokker_planck"]["type"].casefold() == "dougherty":
            return Dougherty(self.cfg)
        else:
            raise NotImplementedError

    def __call__(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f: jnp.ndarray, dt: float) -> jnp.ndarray:
        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            # The three diagonals representing collision operator for all x
            cee_a, cee_b, cee_c = self.fp(nu=nu_fp, f_xv=f, dt=dt)
            # Solve over all x
            f = self.td_solver(cee_a, cee_b, cee_c, f)

        if self.cfg["terms"]["krook"]["is_on"]:
            f = self.krook(nu_K, f, dt)

        return f


class IonCollisions:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fp = self.__init_ion_fp_operator__()
        self.krook = IonKrook(self.cfg)
        self.td_solver = IonTridiagonalSolver(self.cfg)

    def __init_ion_fp_operator__(self):
        if self.cfg["terms"]["fokker_planck"]["type"].casefold() == "lenard_bernstein":
            return IonLenardBernstein(self.cfg)
        elif self.cfg["terms"]["fokker_planck"]["type"].casefold() == "dougherty":
            return IonDougherty(self.cfg)
        else:
            raise NotImplementedError

    def __call__(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f: jnp.ndarray, dt: float) -> jnp.ndarray:
        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            # The three diagonals representing collision operator for all x
            cii_a, cii_b, cii_c = self.fp(nu=nu_fp, f_xv=f, dt=dt)
            # Solve over all x
            f = self.td_solver(cii_a, cii_b, cii_c, f)

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


class IonKrook:
    def __init__(self, cfg):
        self.cfg = cfg
        # Get ion temperature from species-ion config or use default
        ion_T0 = 0.1  # Default ion temperature
        for name, species_params in self.cfg["density"].items():
            if name.startswith("species-") and ("ion" in name.lower() or name == "species-ion"):
                ion_T0 = species_params.get("T0", 0.1)
                break
        
        f_mx = np.exp(-(self.cfg["grid"]["v_i"][None, :] ** 2.0) / (2.0 * ion_T0))
        self.f_mx = f_mx / np.trapz(f_mx, dx=self.cfg["grid"]["dv_i"], axis=1)[:, None]
        self.dv = self.cfg["grid"]["dv_i"]

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
        self, nu: float, f_xv: jnp.ndarray, dt: float
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        v0t_sq = self.vx_moment(f_xv * self.v[None, :] ** 2.0)
        a = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 + jnp.roll(self.v, 1)[None, :] / 2 / self.dv)
        b = 1.0 + nu[:, None] * dt * self.ones * (2.0 * v0t_sq[:, None] / self.dv**2.0)
        c = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 - jnp.roll(self.v, -1)[None, :] / 2 / self.dv)
        return a, b, c


class IonLenardBernstein:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v_i"]
        self.dv = self.cfg["grid"]["dv_i"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(
        self, nu: float, f_xv: jnp.ndarray, dt: float
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        v0t_sq = self.vx_moment(f_xv * self.v[None, :] ** 2.0)
        a = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 + jnp.roll(self.v, 1)[None, :] / 2 / self.dv)
        b = 1.0 + nu[:, None] * dt * self.ones * (2.0 * v0t_sq[:, None] / self.dv**2.0)
        c = nu[:, None] * dt * (-v0t_sq[:, None] / self.dv**2.0 - jnp.roll(self.v, -1)[None, :] / 2 / self.dv)
        return a, b, c


class Dougherty:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(
        self, nu: float, f_xv: jnp.ndarray, dt: float
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        vbar = self.vx_moment(f_xv * self.v[None, :])
        v0t_sq = self.vx_moment(f_xv * (self.v[None, :] - vbar[:, None]) ** 2.0)

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


class IonDougherty:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v_i"]
        self.dv = self.cfg["grid"]["dv_i"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv):
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(
        self, nu: float, f_xv: jnp.ndarray, dt: float
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """

        :param nu:
        :param f_xv:
        :param dt:
        :return:
        """

        vbar = self.vx_moment(f_xv * self.v[None, :])
        v0t_sq = self.vx_moment(f_xv * (self.v[None, :] - vbar[:, None]) ** 2.0)

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


# Ion-specific tridiagonal solver using ion velocity grid
class IonTridiagonalSolver:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, a, b, c, f):
        # This should be adapted for ion grid if different from electron grid
        # For now, using the same tridiagonal algorithm but could be extended
        # to handle different boundary conditions or grid spacing for ions
        from adept.vlasov2d.solver.tridiagonal import TridiagonalSolver
        solver = TridiagonalSolver(self.cfg)
        return solver(a, b, c, f)
