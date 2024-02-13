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
    dt: float
    wp0: float
    w0: float
    n0: float
    nuei: float
    kax_sq: jax.Array
    kx: jax.Array
    ky: jax.Array
    transverse_mask: jax.Array

    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg["grid"]["dt"]
        self.wp0 = cfg["plasma"]["wp0"]
        self.w0 = cfg["drivers"]["E0"]["w0"]
        self.n0 = np.sqrt(self.wp0)
        self.nuei = -cfg["units"]["derived"]["nuei_norm"]
        self.kax_sq = cfg["grid"]["kx"][:, None] ** 2.0 + cfg["grid"]["ky"][None, :] ** 2.0
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]
        self.transverse_mask = None

    def _calc_div_(self, arr: jax.Array) -> jax.Array:
        arrk = jnp.fft.fft2(arr)
        divk = self.kx[:, None] * arrk[..., 0] + self.ky[None, :] * arrk[..., 1]
        return jnp.fft.ifft2(divk)

    def calc_damping(self, nb) -> Tuple[jax.Array, jax.Array]:
        return nb / self.w0**2.0 * self.nuei / 2.0

    def calc_oscillation(self, nb: jax.Array) -> jax.Array:
        """

        calculates 1j / (2 w0) * w0^2 - nb - wp0^2 * nb / n0

        """
        coeff = self.w0**2.0 - nb - self.wp0**2 * nb / self.n0
        return 1j / 2 / self.w0 * coeff

    def calc_nabla(self, e0: jax.Array) -> jax.Array:
        """
        Calculates the spatial advection term

        """
        nabla2 = jnp.fft.fft2(e0, axis=(0, 1)) * (-self.kax_sq)[:, :, None]  # (ikx^2 + iky^2) * E0(kx, ky)
        div = self._calc_div_(e0)  # div(E0)
        grad_x, grad_y = self.kx[:, None] * div, self.ky[None, :] * div  # kx * div(E0), ky * div(E0)
        term = nabla2 - 1j * jnp.concatenate(
            [grad_x[..., None], grad_y[..., None]], axis=-1
        )  # (ikx^2 + iky^2) * E0(kx, ky) - i * (kx * div(E0), ky * div(E0))
        term *= 1j * self.c_light**2.0 / 2.0 / self.w0  # * i * c^2 / 2 / w0
        return jnp.fft.ifft2(term)

    def calc_epw_term(self, t: float, eh: jax.Array, nb: jax.Array) -> jax.Array:
        """
        Calculates the pump depletion term

        """
        coeff = 1j / 2.0 / self.w0 * jnp.exp(1j * (self.w0 - 2 * self.wp0) * t)

        div_eh = self._calc_div_(eh)
        term = nb / self.n0 * (eh * div_eh) * self.transverse_mask

        return coeff * term

    def get_eh_x(self, phi: jax.Array) -> jax.Array:
        ehx = -jnp.fft.ifft2(1j * self.kx[:, None] * phi)
        ehy = -jnp.fft.ifft2(1j * self.ky[None, :] * phi)

        return jnp.concatenate([ehx[..., None], ehy[..., None]], axis=-1) * self.kx.size * self.ky.size / 4

    def __call__(self, t, y, args):
        e0 = y["e0"]
        phi = y["phi"]
        nb = y["nb"]

        eh = self.get_eh_x(phi)

        e0 = e0 * jnp.exp(self.calc_damping(nb=nb)[:, :, None])
        e0 = e0 + self.dt * self.calc_nabla(e0)
        e0 = e0 + self.dt * e0 * self.calc_oscillation(nb)[:, :, None]
        y["e0"] = e0 + self.dt * self.calc_epw_term(t, eh, nb)

        return y
