from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp, vmap
from interpax import interp2d, interp1d


class VelocityExponential:
    def __init__(self, cfg):
        self.kv_real = cfg["grid"]["kvr"]

    def __call__(self, f, e, dt):
        return jnp.real(
            jnp.fft.irfft(
                jnp.exp(-1j * self.kv_real[None, :, None] * dt * e[:, None, None]) * jnp.fft.rfft(f, axis=1), axis=1
            )
        )


# class VelocityCubicSpline:
#     def __init__(self, cfg):
#         self.v = jnp.repeat(cfg["grid"]["v"][None, :], repeats=cfg["grid"]["nx"], axis=0)
#         self.interp = vmap(partial(interp1d, extrap=True), in_axes=0)  # {"xq": 0, "f": 0, "x": None})

#     def __call__(self, f, e, dt):
#         vq = self.v - e[:, None, None] * dt
#         return self.interp(xq=vq, x=self.v, f=f)


class SpaceExponential:
    def __init__(self, cfg):
        self.kx_real = cfg["grid"]["kxr"]
        self.v = cfg["grid"]["v"]

    def __call__(self, f, dt):
        return jnp.real(
            jnp.fft.irfft(
                jnp.exp(-1j * self.kx_real[:, None, None] * dt * self.v[None, :, None]) * jnp.fft.rfft(f, axis=0),
                axis=0,
            )
        )
