from jax import numpy as jnp


class ExponentialSpatialAdvection:
    def __init__(self, cfg):
        self.kx_real = cfg["grid"]["kxr"]
        self.ky_real = cfg["grid"]["kyr"]

        self.vx = jnp.array(cfg["grid"]["vx"])
        self.vy = jnp.array(cfg["grid"]["vy"])

    def step_x(self, f, dt):
        temp_x = self.vx[None, None, :, None] * self.kx_real[:, None, None, None]
        return f * jnp.exp(-1j * dt * temp_x)

    def step_y(self, f, dt):
        temp_y = self.vy[None, None, None, :] * self.ky_real[None, :, None, None]
        return f * jnp.exp(-1j * dt * temp_y)

    def __call__(self, f, dt):
        temp = jnp.fft.fft(jnp.fft.fft(f, axis=0), axis=1)
        temp_x = self.vx[None, None, :, None] * self.kx_real[:, None, None, None]
        temp_y = self.vy[None, None, None, :] * self.ky_real[None, :, None, None]
        return temp * jnp.exp(-1j * dt * (temp_x + temp_y))


class ExponentialVelocityAdvection:
    def __init__(self, cfg):
        self.kvxr = cfg["grid"]["kvxr"]
        self.kvyr = cfg["grid"]["kvyr"]
        self.vx = cfg["grid"]["vx"]
        self.vy = cfg["grid"]["vy"]

    def __call__(self, fk, ex, ey, bz, dt):
        fxy = jnp.fft.ifft(jnp.fft.ifft(fk, axis=0), axis=1)
        fxykv = jnp.fft.fft(jnp.fft.fft(fxy, axis=2), axis=3)

        exxy = jnp.real(jnp.fft.ifft(jnp.fft.ifft(ex, axis=0), axis=1))
        eyxy = jnp.real(jnp.fft.ifft(jnp.fft.ifft(ey, axis=0), axis=1))
        bzxy = jnp.real(jnp.fft.ifft(jnp.fft.ifft(bz, axis=0), axis=1))

        temp_x = exxy[..., None, None] + self.vy[None, None, None, :] * bzxy[..., None, None]
        temp_y = eyxy[..., None, None] - self.vx[None, None, :, None] * bzxy[..., None, None]

        new_fxykv = fxykv * jnp.exp(
            -1j * dt * (temp_x * self.kvxr[None, None, :, None] + temp_y * self.kvyr[None, None, None, :])
        )

        new_fk = jnp.fft.fft(jnp.fft.fft(jnp.fft.ifft(jnp.fft.ifft(new_fxykv, axis=2), axis=3), axis=0), axis=1)

        return new_fk
