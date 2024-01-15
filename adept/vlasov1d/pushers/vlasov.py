from functools import partial
from typing import Callable

import equinox as eqx
from jax import numpy as jnp, vmap
from interpax import interp2d, interp1d


class VlasovExternalE(eqx.Module):
    x: jnp.ndarray
    v: jnp.ndarray
    f_interp: Callable
    dt: int
    dummy_x: jnp.ndarray
    dummy_v: jnp.ndarray
    interp_e: Callable

    def __init__(self, cfg, interp_e):
        self.x = cfg["grid"]["x"]
        self.v = cfg["grid"]["v"]
        self.f_interp = partial(interp2d, x=self.x, y=self.v, period=cfg["grid"]["xmax"])
        self.dt = cfg["grid"]["dt"]
        self.dummy_x = jnp.ones_like(self.x)
        self.dummy_v = jnp.ones_like(self.v)[None, :]
        self.interp_e = interp_e

    def step_vdfdx(self, t, f, frac_dt):
        old_x = self.x[:, None] - self.v[None, :] * frac_dt * self.dt
        old_v = self.dummy_x[:, None] * self.v[None, :]
        new_f = self.f_interp(xq=old_x.flatten(), yq=old_v.flatten(), f=f)
        return jnp.reshape(new_f, (self.x.size, self.v.size))

    def step_edfdv(self, t, f, frac_dt):
        interp_e = self.interp_e(self.dummy_x * t, self.x)
        old_v = self.v[None, :] + interp_e[:, None] * frac_dt * self.dt
        old_x = self.dummy_v * self.x[:, None]
        new_f = self.f_interp(xq=old_x.flatten(), yq=old_v.flatten(), f=f)
        return jnp.reshape(new_f, (self.x.size, self.v.size))

    def __call__(self, t, y, args):
        f = y["electron"]

        new_f = self.step_vdfdx(t, f, 0.5)
        new_f = self.step_edfdv(t, new_f, 1.0)
        new_f = self.step_vdfdx(t, new_f, 0.5)

        return {"electron": new_f}


class VelocityExponential:
    def __init__(self, cfg):
        self.kv_real = cfg["grid"]["kvr"]

    def __call__(self, f, e, dt):
        return jnp.real(
            jnp.fft.irfft(jnp.exp(-1j * self.kv_real[None, :] * dt * e[:, None]) * jnp.fft.rfft(f, axis=1), axis=1)
        )


class VelocityCubicSpline:
    def __init__(self, cfg):
        self.v = jnp.repeat(cfg["grid"]["v"][None, :], repeats=cfg["grid"]["nx"], axis=0)
        self.interp = vmap(partial(interp1d, extrap=True), in_axes=0)  # {"xq": 0, "f": 0, "x": None})

    def __call__(self, f, e, dt):
        vq = self.v - e[:, None] * dt
        return self.interp(xq=vq, x=self.v, f=f)


class SpaceExponential:
    def __init__(self, cfg):
        self.kx_real = cfg["grid"]["kxr"]
        self.v = cfg["grid"]["v"]

    def __call__(self, f, dt):
        return jnp.real(
            jnp.fft.irfft(jnp.exp(-1j * self.kx_real[:, None] * dt * self.v[None, :]) * jnp.fft.rfft(f, axis=0), axis=0)
        )
