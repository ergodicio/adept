#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from functools import partial
from typing import Dict, Tuple

import numpy as np
from jax import numpy as jnp, vmap
from jax.scipy.ndimage import map_coordinates as mp
import lineax as lx
from interpax import interp2d


class Collisions:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ee_fp = Dougherty(self.cfg)  # self.__init_ee_operator__()
        self.krook = Krook(self.cfg)
        self.ei_fp = Banks(self.cfg)
        self.coll_ee_over_x = vmap(self._single_x_ee_, in_axes=(0, 0, None))
        self.coll_ei_over_x = vmap(self._single_x_ei_, in_axes=(0, 0, None))

    def _single_x_ei_(self, Zsq_ni_nuei: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        nu_ei_eff = Zsq_ni_nuei * self.ei_fp.nuei_coeff
        return self.ei_fp.solve_azimuthal(f_vxvy, nu_ei_eff, dt)

    def _single_x_ee_(self, nu_ee: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        nu_ee_eff = nu_ee * self.ee_fp.nuee_coeff
        f_vxvy = self.ee_fp.explicit_vy(nu_ee_eff, f_vxvy, 0.5 * dt)
        f_vxvy = self.ee_fp.implicit_vx(nu_ee_eff, f_vxvy, 0.5 * dt)
        f_vxvy = self.ee_fp.explicit_vx(nu_ee_eff, f_vxvy, 0.5 * dt)
        f_vxvy = self.ee_fp.implicit_vy(nu_ee_eff, f_vxvy, 0.5 * dt)

        return f_vxvy

    def __call__(
        self,
        nu_ee: jnp.ndarray,
        nu_ei: jnp.ndarray,
        nu_K: jnp.ndarray,
        f: jnp.ndarray,
        Z: jnp.ndarray,
        ni: jnp.ndarray,
        dt: jnp.float64,
    ) -> jnp.ndarray:

        if self.cfg["terms"]["fokker_planck"]["nu_ee"]["is_on"]:
            f = self.coll_ee_over_x(nu_ee, f, dt)

        if self.cfg["terms"]["fokker_planck"]["nu_ei"]["is_on"]:
            Zsq_ni_nuei = Z**2.0 * ni * self.ei_fp.nuei_coeff * nu_ei
            f = self.coll_ei_over_x(Zsq_ni_nuei, f, dt)

        if self.cfg["terms"]["fokker_planck"]["krook"]["is_on"]:
            f = self.krook(nu_K, f, dt)

        return f


class Krook:
    def __init__(self, cfg):
        self.cfg = cfg
        self.f_mx = self.cfg["grid"]["f_mx"]
        self.dv = self.cfg["grid"]["dv"]

    def moment_x(self, f):
        return jnp.sum(f, axis=1) * self.dv

    def __call__(self, nu_K, f_xv, dt) -> jnp.ndarray:
        nu_Kxdt = dt * nu_K[:, None, None]
        exp_nuKxdt = jnp.exp(-nu_Kxdt)

        return f_xv * exp_nuKxdt + self.f_mx * (1.0 - exp_nuKxdt)


class Dougherty:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones(self.cfg["grid"]["nv"])
        self.scan_over_vy = vmap(self._solve_one_vslice_, in_axes=(None, 0, 0, 1, None), out_axes=1)
        self.scan_over_vx = vmap(self._solve_one_vslice_, in_axes=(None, 0, 0, 0, None), out_axes=0)

        r_e = 2.8179402894e-13
        c_kpre = r_e * np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        self.nuee_coeff = 4.0 * np.pi / 3 * c_kpre * cfg["units"]["derived"]["logLambda_ee"]

    def _solve_one_vslice_(self, nu, vbar, v0t_sq, f_vxvy, dt):
        operator = self._get_operator_(nu, vbar, v0t_sq, dt)
        return lx.linear_solve(operator, f_vxvy, solver=lx.Tridiagonal()).value

    def ddx(self, f_vxvy: jnp.ndarray):
        return jnp.gradient(f_vxvy, self.dv, axis=0)

    def ddy(self, f_vxvy: jnp.ndarray):
        return jnp.gradient(f_vxvy, self.dv, axis=1)

    def get_init_quants_x(self, f_vxvy: jnp.ndarray):
        n = jnp.sum(f_vxvy, axis=0)
        vxbar = jnp.sum(f_vxvy * self.v[:, None], axis=0) / n
        v0t_sq = jnp.sum(f_vxvy * (self.v[:, None] - vxbar) ** 2.0, axis=0) / n
        return vxbar, v0t_sq

    def get_init_quants_y(self, f_vxvy: jnp.ndarray):
        n = jnp.sum(f_vxvy, axis=1)
        vybar = jnp.sum(f_vxvy * self.v[None, :], axis=1) / n
        v0t_sq = jnp.sum(f_vxvy * (self.v[None, :] - vybar) ** 2.0, axis=1) / n
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
        return self.scan_over_vy(nu, vxbar, v0t_sq, f_vxvy, dt)

    def implicit_vy(self, nu: jnp.float64, f_vxvy: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """

        :param nu:
        :param f_vxvy:
        :param dt:
        :return:
        """

        vybar, v0t_sq = self.get_init_quants_y(f_vxvy)
        return self.scan_over_vx(nu, vybar, v0t_sq, f_vxvy, dt)

    def explicit_vx(self, nu, f_vxvy, dt: jnp.float64):
        vxbar, v0t_sq = self.get_init_quants_x(f_vxvy)
        dfdvx = self.ddx(f_vxvy)
        dfdt = f_vxvy + (self.v[:, None] - vxbar[None, :]) * self.ddx(f_vxvy) + v0t_sq[None, :] * self.ddx(dfdvx)
        new_fvxvy = f_vxvy + dt * nu * dfdt
        return new_fvxvy

    def explicit_vy(self, nu, f_vxvy, dt: jnp.float64):
        vybar, v0t_sq = self.get_init_quants_y(f_vxvy)
        dfdvy = self.ddy(f_vxvy)
        dfdt = f_vxvy + (self.v[None, :] - vybar[:, None]) * self.ddy(f_vxvy) + v0t_sq[:, None] * self.ddy(dfdvy)
        new_fvxvy = f_vxvy + dt * nu * dfdt
        return new_fvxvy


class Banks:
    """
    Electron-ion collision operator from Banks et al. [1].

    1. Banks, J. W., Brunner, S., Berger, R. L. & Tran, T. M. Vlasov simulations of electron-ion collision effects on damping of electron plasma waves.
    Physics of Plasmas 23, 032108 (2016).

    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        # self.dv = self.cfg["grid"]["dv"]

        # self.scan_over_vy = vmap(self._solve_one_vy_, in_axes=(1, 0, 1, None, 1), out_axes=1)
        # self.scan_over_vx = vmap(self._solve_one_vx_, in_axes=(0, 0, 0, None, 0), out_axes=0)
        # self.vr = np.sqrt(self.v[:, None] ** 2.0 + self.v[None, :] ** 2.0)

        self.nu_ei_solver = self.cfg["terms"]["fokker_planck"]["nu_ei"]["solver"]

        Nth = self.cfg["terms"]["fokker_planck"]["nu_ei"]["nth"]
        Nr = self.cfg["terms"]["fokker_planck"]["nu_ei"]["nr"]
        self.ones = jnp.ones(Nth)

        self.dth = dth = 2 * np.pi / Nth
        rmax = cfg["grid"]["vmax"]
        dr = rmax / Nr

        self.th = np.linspace(dth / 2.0, 2 * np.pi - dth / 2.0, Nth)
        self.vr = np.linspace(dr / 2.0, rmax - dr / 2.0, Nr)

        vr_pad = np.linspace(-dr / 2.0, rmax - dr / 2.0, Nr + 1)
        th_pad = np.linspace(-dth / 2.0, 2 * np.pi + dth / 2.0, Nth + 2)

        vr, vth = np.meshgrid(self.vr, self.th, indexing="ij")

        # cart2pol coordinates
        vx_interp = -vr.flatten() * np.cos(vth.flatten())
        vy_interp = vr.flatten() * np.sin(vth.flatten())

        # pol2cart coordinates
        vmx, vmy = np.meshgrid(self.v, self.v, indexing="ij")
        r_interp = np.sqrt(vmx.flatten() ** 2.0 + vmy.flatten() ** 2.0)
        th_interp = np.arctan2(vmy.flatten(), -vmx.flatten())

        # sin th / -cos th = vy / vx
        # vx = -r cos th
        # vy = r sin th

        self.flat_oob_mask = np.where(r_interp > rmax - 2 * dr, 0, 1)
        self.reshaped_oob_mask = self.flat_oob_mask.reshape(self.v.size, self.v.size, order="C")

        # Negative angles are corrected
        th_interp = np.where(th_interp < 0, 2 * np.pi + th_interp, th_interp)

        self.cart2pol = partial(interp2d, xq=vx_interp, yq=vy_interp, x=self.v, y=self.v, extrap=False, method="cubic")
        self._pol2cart_ = partial(interp2d, xq=r_interp, yq=th_interp, x=vr_pad, y=th_pad, extrap=False, method="cubic")

        thk_sq = (np.fft.rfftfreq(Nth, dth) * 2 * np.pi) ** 2.0

        # collision operator envelope
        nu_envelope = np.zeros(Nr)
        vmax = cfg["grid"]["vmax"]
        # vbar = 0.464 *
        vc = vmax - 1.0
        # nu_envelope[self.vr < vbar] = vbar**-3.0
        locs = self.vr < vc
        nu_envelope[locs] = self.vr[locs] ** -3.0
        locs = (self.vr >= vc) & (self.vr < vmax)
        nu_envelope[locs] = self.vr[locs] ** -3.0 * (
            1 - np.sin(0.5 * np.pi * (self.vr[locs] - vc) / (vmax - vc)) ** 2.0
        )
        self.nu_envelope = nu_envelope
        self.thksq_nu = jnp.array(nu_envelope[:, None] * thk_sq[None, :])

        r_e = 2.8179402894e-13
        kp = np.sqrt(4 * np.pi * cfg["units"]["derived"]["n0"].to("1/cm^3").value * r_e)
        kpre = r_e * kp

        self.nuei_coeff = kpre * cfg["units"]["derived"]["logLambda_ei"]  # will be multiplied by Z^2 ni = Z ne

    def pol2cart(self, f):
        f_pad = jnp.concatenate([f[:1, :], f])
        f_pad = jnp.concatenate([f_pad[:, -1:], f_pad, f_pad[:, :1]], axis=1)
        return self._pol2cart_(f=f_pad)

    def solve_implicit(self, fth, nu, dt):
        c = self.nu_envelope[:, None] * nu * dt / self.dth**2.0
        rhs = jnp.concatenate([fth[:, :1] + c * fth[:, -1:], fth[:, 1:-1], fth[:, -1:] + c * fth[:, :1]], axis=1)
        return vmap(self._solve_one_vr_, in_axes=(0, 0))(c, rhs)

    def _solve_one_vr_(self, nu_coeff, rhs):
        lower_diagonal = -nu_coeff * self.ones[:-1]
        diagonal = 1.0 + 2.0 * nu_coeff * self.ones
        upper_diagonal = -nu_coeff * self.ones[:-1]
        op = lx.TridiagonalLinearOperator(
            diagonal=diagonal, upper_diagonal=upper_diagonal, lower_diagonal=lower_diagonal
        )

        return lx.linear_solve(op, rhs, lx.Tridiagonal()).value

    def solve_azimuthal(self, f_vxvy, nu, dt):

        # interpolate to polar axes
        fth = self.cart2pol(f=f_vxvy).reshape(self.vr.size, self.th.size, order="C")

        # step collisions
        if self.nu_ei_solver == "exact-fft":
            temp = jnp.fft.rfft(fth, axis=-1) * jnp.exp(-self.thksq_nu * nu * dt)
            fnew = jnp.real(jnp.fft.irfft(temp, axis=-1))
        elif self.nu_ei_solver == "implicit":
            fnew = self.solve_implicit(fth, nu, dt)

        # The data is mapped to the new coordinates
        cart_data = self.pol2cart(f=fnew) * self.flat_oob_mask

        # The data is returned with the mask applied
        return jnp.abs(cart_data.reshape(self.v.size, self.v.size, order="C")) + f_vxvy * (1 - self.reshaped_oob_mask)
