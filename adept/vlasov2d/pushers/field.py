#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from typing import Dict, Tuple
from functools import partial

from jax import numpy as jnp


class FieldSolver:
    def __init__(self, cfg):
        self.ion_charge = cfg["grid"]["ion_charge"]
        self.one_over_kx = cfg["grid"]["one_over_kx"]
        self.one_over_ky = cfg["grid"]["one_over_ky"]
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]
        self.vx_mom = partial(jnp.trapz, dx=cfg["grid"]["dvx"], axis=2)
        self.vy_mom = partial(jnp.trapz, dx=cfg["grid"]["dvy"], axis=3)
        self.dvx = cfg["grid"]["dvx"]
        self.dvy = cfg["grid"]["dvy"]
        self.vx = cfg["grid"]["vx"]
        self.vy = cfg["grid"]["vy"]
        self.zero = jnp.concatenate([jnp.array([0.0]), jnp.ones(cfg["grid"]["nx"] - 1)])

    def compute_charge(self, f):
        return self.vx_mom(self.vy_mom(f))

    def compute_jx(self, f):
        return jnp.trapz(self.vx[None, None, :] * jnp.trapz(f, dx=self.dvy, axis=3), dx=self.dvx, axis=2)

    def compute_jy(self, f):
        return jnp.trapz(self.vy[None, None, :] * jnp.trapz(f, dx=self.dvx, axis=2), dx=self.dvy, axis=2)

    def poisson(self, f: jnp.ndarray):
        net_charge = self.compute_charge(f)  # * self.zero[:, None] * self.zero[None, :]
        ex = 1j * self.one_over_kx[:, None] * net_charge  # the background gets zerod out here anyway ...
        ey = 1j * self.one_over_ky[None, :] * net_charge  # it acts as the ion charge
        return ex, ey

    def ampere(self, exk, eyk, bzk, jxk, jyk, dt):
        exkp = exk + dt * (1j * self.ky[None, :] * bzk - jxk)
        eykp = eyk + dt * (-1j * self.kx[:, None] * bzk - jyk)
        return exkp, eykp

    def faraday(self, bzk, exk, eyk, dt):
        print(exk.shape, eyk.shape, bzk.shape, self.ky.shape, self.kx.shape)

        bzkp = bzk + dt * 1j * (self.ky[None, :] * exk - self.kx[:, None] * eyk)

        return bzkp


class Driver:
    xax: jnp.ndarray
    yax: jnp.ndarray

    def __init__(self, xax, yax):
        self.xax = xax
        self.yax = yax

    def get_this_pulse(self, this_pulse: Dict, current_time: jnp.float64):
        kk = this_pulse["k0"]
        ww = this_pulse["w0"]
        dw = this_pulse["dw0"]
        t_L = this_pulse["t_center"] - this_pulse["t_width"] * 0.5
        t_R = this_pulse["t_center"] + this_pulse["t_width"] * 0.5
        t_wL = this_pulse["t_rise"]
        t_wR = this_pulse["t_rise"]
        x_L = this_pulse["x_center"] - this_pulse["x_width"] * 0.5
        x_R = this_pulse["x_center"] + this_pulse["x_width"] * 0.5
        x_wL = this_pulse["x_rise"]
        x_wR = this_pulse["x_rise"]

        y_L = this_pulse["y_center"] - this_pulse["y_width"] * 0.5
        y_R = this_pulse["y_center"] + this_pulse["y_width"] * 0.5
        y_wL = this_pulse["y_rise"]
        y_wR = this_pulse["y_rise"]

        envelope_t = get_envelope(t_wL, t_wR, t_L, t_R, current_time)
        envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, self.xax)
        envelope_y = get_envelope(y_wL, y_wR, y_L, y_R, self.yax)

        return (
            envelope_t
            * envelope_x[:, None]
            * envelope_y[None, :]
            * jnp.abs(kk)
            * this_pulse["a0"]
            * jnp.sin(kk * self.xax[:, None] - (ww + dw) * current_time)
        )

    def __call__(self, current_time, args):
        """
        Applies the driver function

        P.S. This needs names because its going through a jitted JAX call

        :param current_time:
        :param pulses:
        :return:
        """

        total_dex = jnp.zeros(current_time.shape + self.xax.shape + self.yax.shape)
        # total_djy = np.zeros(current_time.shape + xs[0].shape + xs[1].shape)

        for key, pulse in args["drivers"]["ex"].items():
            total_dex += self.get_this_pulse(pulse, current_time)

        # for key, pulse in pulses["ey"].items():
        #     total_djy += get_this_pulse(pulse, current_time)

        total_dey = jnp.zeros((self.xax.size, self.yax.size), dtype=jnp.complex128)
        total_dex = jnp.fft.fft2(total_dex)

        return total_dex, total_dey


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))
