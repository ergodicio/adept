import equinox as eqx
from jax import numpy as jnp


class Exponential(eqx.Module):
    kvx_real: jnp.ndarray
    kvy_real: jnp.ndarray

    def __init__(self, cfg):
        super(Exponential, self).__init__()
        kvx = cfg["derived"]["kvx"]
        self.kvx_real = jnp.array(kvx)

        kvy = cfg["derived"]["kvy"]
        self.kvy_real = jnp.array(kvy[: int(kvy.size // 2) + 1])

    def __call__(self, f, e, dt):
        temp = jnp.fft.rfft2(f, axes=(2, 3))
        temp_x = e[0][:, :, None, None] * self.kvx_real[None, None, :, None]
        temp_y = e[1][:, :, None, None] * self.kvy_real[None, None, None, :]
        return jnp.real(jnp.fft.irfft2(temp * jnp.exp(-1j * dt * (temp_x + temp_y)), axes=(2, 3)))
