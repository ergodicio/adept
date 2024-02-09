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
        self.wp0 = cfg["plasma"]["wp0"]
        self.w0 = cfg["drivers"]["E0"]["w0"]
        self.n0 = np.sqrt(self.wp0)
        self.nb = cfg["plasma"]["nb"]
        self.nuei = -cfg["units"]["derived"]["nuei_norm"]
        self.kax_sq = cfg["grid"]["kx"][:, None] ** 2.0 + cfg["grid"]["ky"][None, :] ** 2.0
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]

    def _calc_div_(self, arr: jax.Array) -> jax.Array:
        arrk = jnp.fft.fft2(arr)
        divk = self.kx[:, None] * arrk[..., 0] + self.ky[None, :] * arrk[..., 1]
        return jnp.fft.ifft2(divk)

    def calc_damping(self, nb) -> Tuple[jax.Array, jax.Array]:
        return nb / self.w0**2.0 * self.nuei / 2.0

    def calc_oscillation(self, nb):
        coeff = self.w0**2.0 - nb - self.wp0**2 * nb / self.n0
        return 1j / 2 / self.w0 * coeff

    def calc_nabla(self, e0):
        nabla2 = jnp.fft.fft2(e0) * (-self.kax_sq)
        div = self._calc_div_(e0)
        grad_x, grad_y = self.kx[:, None] * div, self.ky[None, :] * div
        term = nabla2 - 1j * jnp.concatenate([grad_x[..., None], grad_y[..., None]], axis=-1)
        term *= 1j * self.c_light**2.0 / 2.0 / self.w0
        return term

    def calc_epw_term(self, t, eh, nb):
        div_eh = self._calc_div_(eh)
        coeff = 1j / 2.0 / self.w0
        term = nb / self.n0 * (eh * div_eh)
        time_coeff = jnp.exp(1j * (self.w0 - 2 * self.wp0) * t)

        return coeff * term * time_coeff

    def get_eh_x(self, phi: jax.Array) -> jax.Array:
        ehx = -jnp.fft.ifft2(1j * self.kx[:, None] * phi)
        ehy = -jnp.fft.ifft2(1j * self.ky[None, :] * phi)

        return jnp.concatenate([ehx[..., None], ehy[..., None]], axis=-1) * self.kx.size * self.ky.size / 4

    def __call__(self, t, y, args):
        e0 = y["e0"]
        phi = y["phi"]
        nb = y["nb"]

        eh = self.get_eh_x(phi)

        e0 = e0 * jnp.exp(self.calc_damping(nb=nb))
        e0 = e0 + self.dt * self.calc_nabla(e0)
        e0 = e0 + self.dt * self.calc_oscillation(nb)
        y["e0"] = e0 + self.dt * self.calc_epw_term(t, eh, nb)

        return y
