from typing import Dict, List

from jax import numpy as jnp
import equinox as eqx
import diffrax

from adept.lpse2d.core import epw


class Stepper(diffrax.Euler):
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful


class VectorField(eqx.Module):
    """
    This class contains the function that updates the state

    All the pushers are chosen and initialized here and a single time-step is defined here.

    :param cfg:
    :return:
    """

    cfg: Dict
    epw: eqx.Module
    complex_state_vars: List

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.epw = epw.EPW2D(cfg)
        self.complex_state_vars = ["e0", "phi"]

    def unpack_y(self, y):
        new_y = {}
        for k in y.keys():
            if k in self.complex_state_vars:
                new_y[k] = y[k].view(jnp.complex128)
            else:
                new_y[k] = y[k].view(jnp.float64)
        return new_y

    def __call__(self, t, y, args):
        new_y = self.epw(t, self.unpack_y(y), args)

        for k in y.keys():
            y[k] = y[k].view(jnp.float64)
            new_y[k] = new_y[k].view(jnp.float64)

        return new_y
