from typing import Dict
from jax import numpy as jnp
from adept._base_ import get_envelope


class Driver:
    def __init__(self, cfg):
        self.xax = cfg["grid"]["x"]
        self.yax = cfg["grid"]["y"]

    def __call__(self, this_pulse: Dict, current_time: jnp.float64):
        kk = this_pulse["k0"]
        ww = this_pulse["w0"]
        t_L = this_pulse["tc"] - this_pulse["tw"] * 0.5
        t_R = this_pulse["tc"] + this_pulse["tw"] * 0.5
        t_wL = this_pulse["tr"]
        t_wR = this_pulse["tr"]
        x_L = this_pulse["xc"] - this_pulse["xw"] * 0.5
        x_R = this_pulse["xc"] + this_pulse["xw"] * 0.5
        x_wL = this_pulse["xr"]
        x_wR = this_pulse["xr"]

        y_L = this_pulse["yc"] - this_pulse["yw"] * 0.5
        y_R = this_pulse["yc"] + this_pulse["yw"] * 0.5
        y_wL = this_pulse["yr"]
        y_wR = this_pulse["yr"]

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
