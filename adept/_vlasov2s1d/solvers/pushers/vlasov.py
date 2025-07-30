from functools import partial

import equinox as eqx
from interpax import interp1d
from jax import numpy as jnp
from jax import vmap


# Electron pushers (same as single species)
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
        self.interp = vmap(partial(interp1d, extrap=True), in_axes=0)

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


# Ion pushers (adapted for ion velocity grid and charge-to-mass ratio)
class IonVelocityExponential:
    def __init__(self, cfg):
        self.kv_real = cfg["grid"]["kvr_i"]
        # Ion charge-to-mass ratio (ions have opposite charge sign and different mass)
        me_over_mi = cfg["units"]["A"] * 1.0 / 1836.0  # electron to ion mass ratio
        self.charge_mass_ratio = -1.0 * me_over_mi  # Z=1 for singly charged ions, negative for opposite charge

    def __call__(self, f, e, dt):
        # Scale electric field by charge-to-mass ratio for ions
        effective_e = self.charge_mass_ratio * e
        return jnp.real(
            jnp.fft.irfft(
                jnp.exp(-1j * self.kv_real[None, :] * dt * effective_e[:, None]) * jnp.fft.rfft(f, axis=1), axis=1
            )
        )


class IonVelocityCubicSpline:
    def __init__(self, cfg):
        self.v = jnp.repeat(cfg["grid"]["v_i"][None, :], repeats=cfg["grid"]["nx"], axis=0)
        self.interp = vmap(partial(interp1d, extrap=True), in_axes=0)
        # Ion charge-to-mass ratio
        me_over_mi = cfg["units"]["A"] * 1.0 / 1836.0
        self.charge_mass_ratio = -1.0 * me_over_mi

    def __call__(self, f, e, dt):
        # Scale electric field by charge-to-mass ratio for ions
        effective_e = self.charge_mass_ratio * e
        vq = self.v - effective_e[:, None] * dt
        return self.interp(xq=vq, x=self.v, f=f)


class IonSpaceExponential:
    def __init__(self, cfg):
        self.kx_real = cfg["grid"]["kxr"]
        self.v = cfg["grid"]["v_i"]

    def __call__(self, f, dt):
        return jnp.real(
            jnp.fft.irfft(jnp.exp(-1j * self.kx_real[:, None] * dt * self.v[None, :]) * jnp.fft.rfft(f, axis=0), axis=0)
        )
