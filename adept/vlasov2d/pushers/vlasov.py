from jax import numpy as jnp


class ExponentialSpatialAdvection:
    def __init__(self, cfg):
        self.i_kx_vx = -1j * cfg["grid"]["kx"][:, None, None, None] * cfg["grid"]["vx"][None, None, :, None]
        self.i_ky_vy = -1j * cfg["grid"]["ky"][None, :, None, None] * cfg["grid"]["vy"][None, None, None, :]

    def step_x(self, f, dt):
        return f * jnp.exp(self.i_kx_vx * dt)

    def step_y(self, f, dt):
        return f * jnp.exp(self.i_ky_vy * dt)

    def __call__(self, f, dt):
        return f * jnp.exp(self.i_kx_vx + self.i_ky_vy)


class ExponentialVelocityAdvection:
    def __init__(self, cfg):
        self.kvx = cfg["grid"]["kvx"][None, None, :, None]
        self.kvy = cfg["grid"]["kvy"][None, None, None, :]
        self.vx = cfg["grid"]["vx"][None, None, :, None]
        self.vy = cfg["grid"]["vy"][None, None, None, :]

    def __call__(self, fk, ex, ey, bz, dt):
        fxy = jnp.fft.ifft2(fk, axes=(0, 1))
        fxykv = jnp.fft.fft2(fxy, axes=(2, 3))

        exxy = jnp.fft.ifft2(ex, axes=(0, 1))[..., None, None]
        eyxy = jnp.fft.ifft2(ey, axes=(0, 1))[..., None, None]
        bzxy = jnp.fft.ifft2(bz, axes=(0, 1))[..., None, None]

        temp_x = exxy + self.vy * bzxy
        temp_y = eyxy - self.vx * bzxy

        new_fxykv = fxykv * jnp.exp(-1j * dt * (temp_x * self.kvx + temp_y * self.kvy))

        new_fk = jnp.fft.fft2(jnp.fft.ifft2(new_fxykv, axes=(2, 3)), axes=(0, 1))

        return new_fk
