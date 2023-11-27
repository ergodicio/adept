from typing import Dict
import jax
import equinox as eqx
from jax import numpy as jnp


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))


class Driver(eqx.Module):
    xax: jax.Array
    yax: jax.Array

    def __init__(self, cfg):
        self.xax = cfg["grid"]["x"]
        self.yax = cfg["grid"]["y"]

    def __call__(self, this_pulse: Dict, current_time: jnp.float64):
        kk = this_pulse["k0"]
        ww = this_pulse["w0"]
        # dw = 0.this_pulse["dw0"]
        t_L = this_pulse["t_c"] - this_pulse["t_w"] * 0.5
        t_R = this_pulse["t_c"] + this_pulse["t_w"] * 0.5
        t_wL = this_pulse["t_r"]
        t_wR = this_pulse["t_r"]
        x_L = this_pulse["x_c"] - this_pulse["x_w"] * 0.5
        x_R = this_pulse["x_c"] + this_pulse["x_w"] * 0.5
        x_wL = this_pulse["x_r"]
        x_wR = this_pulse["x_r"]

        y_L = this_pulse["y_c"] - this_pulse["y_w"] * 0.5
        y_R = this_pulse["y_c"] + this_pulse["y_w"] * 0.5
        y_wL = this_pulse["y_r"]
        y_wR = this_pulse["y_r"]

        envelope_t = get_envelope(t_wL, t_wR, t_L, t_R, current_time)
        envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, self.xax)
        envelope_y = get_envelope(y_wL, y_wR, y_L, y_R, self.yax)

        return (
            envelope_t
            * envelope_x[:, None]
            * envelope_y[None, :]
            * jnp.abs(kk)
            * this_pulse["a0"]
            * jnp.exp(-1j * (kk * self.xax[:, None] - ww * current_time))
        )
