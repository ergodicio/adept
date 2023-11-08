import equinox as eqx
from jax import numpy as jnp


class Exponential(eqx.Module):
    kx_real: jnp.ndarray
    ky_real: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray

    def __init__(self, cfg):
        super(Exponential, self).__init__()
        kx = cfg["grid"]["kx"]
        ky = cfg["grid"]["ky"]
        self.kx_real = jnp.array(kx)
        self.ky_real = jnp.array(ky[: int(ky.size // 2) + 1])

        self.vx = jnp.array(cfg["grid"]["vx"])
        self.vy = jnp.array(cfg["grid"]["vy"])

    def __call__(self, f, dt):
        temp = jnp.fft.rfft2(f, axes=(0, 1))
        temp_x = self.vx[None, None, :, None] * self.kx_real[:, None, None, None]
        temp_y = self.vy[None, None, None, :] * self.ky_real[None, :, None, None]
        return jnp.real(jnp.fft.irfft2(temp * jnp.exp(-1j * dt * (temp_x + temp_y)), axes=(0, 1)))
