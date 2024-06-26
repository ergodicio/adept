#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from typing import Dict, Tuple
from functools import partial

from jax import numpy as jnp


class FieldSolver:
    def __init__(self, cfg):
        self.ion_charge = cfg["grid"]["ion_charge"]
        self.one_over_ikx = cfg["grid"]["one_over_kx"][:, None] / 1j
        self.one_over_iky = cfg["grid"]["one_over_ky"][None, :] / 1j
        self.kx = cfg["grid"]["kx"][:, None]
        self.kx_mask = jnp.where(jnp.abs(self.kx) > 0, 1, 0)[:, None]
        self.ky = cfg["grid"]["ky"][None, :]
        self.ky_mask = jnp.where(jnp.abs(self.ky) > 0, 1, 0)[None, :]
        self.dvx = cfg["grid"]["dvx"]
        self.dvy = cfg["grid"]["dvy"]
        self.vx = cfg["grid"]["vx"][None, None, :, None]
        self.vy = cfg["grid"]["vy"][None, None, None, :]
        self.zero = jnp.concatenate([jnp.array([0.0]), jnp.ones(cfg["grid"]["nx"] - 1)])

    def compute_charge(self, f):
        return self.vx_mom(self.vy_mom(f))

    def vx_mom(self, f):
        return jnp.sum(f, axis=2) * self.dvx

    def vy_mom(self, f):
        return jnp.sum(f, axis=3) * self.dvy

    def compute_jx(self, f):
        return jnp.sum(jnp.sum(self.vx * f, axis=3), axis=2) * self.kx_mask * self.dvx * self.dvy

    def compute_jy(self, f):
        return jnp.sum(jnp.sum(self.vy * f, axis=3), axis=2) * self.ky_mask * self.dvx * self.dvy

    def ampere(self, exk, eyk, bzk, dt):
        exkp = exk  # - dt * (1j * self.ky * bzk)
        eykp = eyk  # - dt * (-1j * self.kx * bzk)
        return exkp, eykp

    def faraday(self, bzk, exk, eyk, dt):
        return bzk + dt * 1j * (self.ky * exk - self.kx * eyk)

    def hampere_e1(self, exk, fxy, dt):
        fk = jnp.fft.fft2(fxy, axes=(0, 1))
        return (
            exk
            + self.one_over_ikx
            * jnp.sum(jnp.sum(fk * (jnp.exp(-1j * self.kx[..., None, None] * dt * self.vx) - 1), axis=3), axis=2)
            * self.dvx
            * self.dvy
        )

    def hampere_e2(self, eyk, fxy, dt):
        fk = jnp.fft.fft2(fxy, axes=(0, 1))
        return (
            eyk
            + self.one_over_iky
            * jnp.sum(jnp.sum(fk * (jnp.exp(-1j * self.ky[..., None, None] * dt * self.vy) - 1), axis=3), axis=2)
            * self.dvx
            * self.dvy
        )


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

        :param current_time:
        :param pulses:
        :return:
        """

        total_dex = jnp.zeros((self.xax.size, self.yax.size))
        # total_djy = np.zeros(current_time.shape + xs[0].shape + xs[1].shape)

        for key, pulse in args["drivers"]["ex"].items():
            total_dex += self.get_this_pulse(pulse, current_time)

        # for key, pulse in pulses["ey"].items():
        #     total_djy += get_this_pulse(pulse, current_time)

        total_dey = jnp.zeros((self.xax.size, self.yax.size))  # , dtype=jnp.complex128)
        # total_dex = jnp.fft.fft2(total_dex, axes=(0, 1))

        return total_dex, total_dey


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))
