from jax import numpy as jnp


class ExponentialSpatialAdvection:
    def __init__(self, cfg):
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]
        self.kx_mask = jnp.where(jnp.abs(self.kx) > 0, 1, 0)[:, None, None, None]
        self.ky_mask = jnp.where(jnp.abs(self.ky) > 0, 1, 0)[None, :, None, None]
        self.i_kx_vx = -1j * cfg["grid"]["kx"][:, None, None, None] * cfg["grid"]["vx"][None, None, :, None]
        self.i_ky_vy = -1j * cfg["grid"]["ky"][None, :, None, None] * cfg["grid"]["vy"][None, None, None, :]
        self.one_over_ikx = cfg["grid"]["one_over_kx"][:, None, None, None] / 1j
        self.one_over_iky = cfg["grid"]["one_over_ky"][None, :, None, None] / 1j

    def fxh(self, f, dt):
        return f * (1.0 - jnp.exp(self.i_kx_vx * dt) * self.one_over_ikx / dt) * self.kx_mask

    def fyh(self, f, dt):
        return f * (1.0 - jnp.exp(self.i_ky_vy * dt) * self.one_over_iky / dt) * self.ky_mask


class ExponentialVelocityAdvection:
    def __init__(self, cfg):
        self.kvx = cfg["grid"]["kvx"][None, None, :, None]
        self.kvy = cfg["grid"]["kvy"][None, None, None, :]
        self.vx = cfg["grid"]["vx"][None, None, :, None]
        self.vy = cfg["grid"]["vy"][None, None, None, :]

    def __call__(self, fk, ex, ey, bz, dt):
        fxy = jnp.real(jnp.fft.ifft2(fk, axes=(0, 1)))

        # vx update
        fxykvx = jnp.fft.fft(fxy, axis=2)
        exxy = jnp.real(jnp.fft.ifft2(ex, axes=(0, 1))[..., None, None])
        bzxy = jnp.real(jnp.fft.ifft2(bz, axes=(0, 1))[..., None, None])
        temp_x = exxy + self.vy * bzxy
        new_fxykvx = fxykvx * jnp.exp(-1j * dt * temp_x * self.kvx)
        new_fxy = jnp.real(jnp.fft.ifft(new_fxykvx, axis=2))

        # vy update
        fxykvy = jnp.fft.fft(new_fxy, axis=3)
        eyxy = jnp.real(jnp.fft.ifft2(ey, axes=(0, 1))[..., None, None])
        temp_y = eyxy - self.vx * bzxy
        new_fxykvy = fxykvy * jnp.exp(-1j * dt * temp_y * self.kvy)
        new_fxy = jnp.real(jnp.fft.ifft(new_fxykvy, axis=3))

        new_fk = jnp.fft.fft2(new_fxy, axes=(0, 1))

        return new_fk


class CD2VelocityAdvection:
    def __init__(self, cfg):
        # self.kvx = cfg["grid"]["kvx"][None, None, :, None]
        # self.kvy = cfg["grid"]["kvy"][None, None, None, :]
        self.vx = cfg["grid"]["vx"][None, None, :, None]
        self.vy = cfg["grid"]["vy"][None, None, None, :]
        self.dvx = cfg["grid"]["dvx"]
        self.dvy = cfg["grid"]["dvy"]

    def __call__(self, fk, ex, ey, bz, dt):
        fxy = jnp.fft.ifft2(fk, axes=(0, 1))
        exxy = jnp.fft.ifft2(ex, axes=(0, 1))[..., None, None]
        eyxy = jnp.fft.ifft2(ey, axes=(0, 1))[..., None, None]
        bzxy = jnp.fft.ifft2(bz, axes=(0, 1))[..., None, None]

        temp_x = exxy + self.vy * bzxy
        temp_y = eyxy - self.vx * bzxy

        dfx, dfy = jnp.gradient(fxy, axis=2) / self.dvx, jnp.gradient(fxy, axis=3) / self.dvy

        dfxy = dt * (temp_x * dfx + temp_y * dfy)

        new_fk = jnp.fft.fft2(fxy + 1e-12 * dfxy, axes=(0, 1))

        return new_fk
