from collections.abc import Callable
from functools import partial

import equinox as eqx
from interpax import interp1d, interp2d
from jax import numpy as jnp
from jax import vmap


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
    def __init__(self, species_grids, species_params):
        self.species_grids = species_grids
        self.species_params = species_params

    def __call__(self, f_dict, e, dt):
        result = {}
        for species_name, f in f_dict.items():
            kv_real = self.species_grids[species_name]["kvr"]
            qm = self.species_params[species_name]["charge_to_mass"]
            result[species_name] = jnp.real(
                jnp.fft.irfft(jnp.exp(-1j * kv_real[None, :] * dt * qm * e[:, None]) * jnp.fft.rfft(f, axis=1), axis=1)
            )
        return result


class VelocityCubicSpline:
    def __init__(self, species_grids, species_params):
        self.species_grids = species_grids
        self.species_params = species_params
        self.interp = vmap(partial(interp1d, extrap=True), in_axes=0)

    def __call__(self, f_dict, e, dt):
        result = {}
        for species_name, f in f_dict.items():
            v = self.species_grids[species_name]["v"]
            qm = self.species_params[species_name]["charge_to_mass"]
            nx = f.shape[0]
            v_repeated = jnp.repeat(v[None, :], repeats=nx, axis=0)
            vq = v_repeated - qm * e[:, None] * dt
            result[species_name] = self.interp(xq=vq, x=v_repeated, f=f)
        return result


class SpaceExponential:
    def __init__(self, x, species_grids):
        self.kx_real = jnp.fft.rfftfreq(len(x), d=x[1] - x[0]) * 2 * jnp.pi
        self.species_grids = species_grids

    def __call__(self, f_dict, dt):
        result = {}
        for species_name, f in f_dict.items():
            v = self.species_grids[species_name]["v"]
            result[species_name] = jnp.real(
                jnp.fft.irfft(jnp.exp(-1j * self.kx_real[:, None] * dt * v[None, :]) * jnp.fft.rfft(f, axis=0), axis=0)
            )
        return result
