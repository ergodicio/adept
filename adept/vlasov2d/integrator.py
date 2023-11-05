from typing import Dict, List

from jax import numpy as jnp
import equinox as eqx
import diffrax

from adept.vlasov2d.pushers.time import LeapfrogIntegrator


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

    epw: eqx.Module

    def __init__(self, cfg):
        super().__init__()
        self.epw = LeapfrogIntegrator(cfg)

    def __call__(self, t, y, args):
        return self.epw(t, y, args)
