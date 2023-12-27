from typing import Dict, Tuple

import diffrax
import jax
from jax import numpy as jnp
import equinox as eqx
import numpy as np
from theory import electrostatic
from adept.lpse2d.core.driver import Driver


class Pump2D(eqx.Module):
    cfg: Dict

    def __init__(self, cfg):
        self.cfg = cfg

    def calc_damping(self, electron_density: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return omega_pe**2 / self.omega_0**2.0 * self.nuei / 2.0

    def calc_oscillation(self, electron_density):
        coeff = self.omega_0**2.0 - omega_pe**2 - omega_p0**2 * electron_density / self.n0
        return 1j / 2 / self.omega_0 * coeff

    def calc_nabla(self, e0):
        nabla2 = jnp.fft.fft2(e0) * (-self.kax_sq)
        div = self.calc_div(e0)
        grad_x, grad_y = self.kx[:, None] * div, self.ky[None, :] * div
        term = nabla2 - 1j * jnp.concatenate([grad_x[..., None], grad_y[..., None]], axis=-1)
        term *= 1j * self.c_light**2.0 / 2 / self.omega_0
        return term

    def calc_epw_term(self):
        pass

