from functools import partial
import numpy as np
from jax import numpy as jnp, vmap

from interpax import interp1d


class ExponentialSpatialAdvection:
    """
    Performs v df/dx using an exponential integrator in time and spectral in space
    """

    def __init__(self, cfg):
        self.kx = cfg["grid"]["kx"]
        self.ky = cfg["grid"]["ky"]
        self.kx_mask = jnp.where(jnp.abs(self.kx) > 0, 1, 0)[:, None, None, None]
        self.ky_mask = jnp.where(jnp.abs(self.ky) > 0, 1, 0)[None, :, None, None]
        self.i_kx_vx = -1j * cfg["grid"]["kx"][:, None, None, None] * cfg["grid"]["vx"][None, None, :, None]
        self.i_ky_vy = -1j * cfg["grid"]["ky"][None, :, None, None] * cfg["grid"]["vy"][None, None, None, :]
        self.one_over_ikx = cfg["grid"]["one_over_kx"][:, None, None, None] / 1j
        self.one_over_iky = cfg["grid"]["one_over_ky"][None, :, None, None] / 1j

    def step_x(self, f, dt):
        return jnp.real(jnp.fft.ifft(jnp.fft.fft(f, axis=0) * jnp.exp(self.i_kx_vx * dt), axis=0))

    def step_y(self, f, dt):
        return jnp.real(jnp.fft.ifft(jnp.fft.fft(f, axis=1) * jnp.exp(self.i_ky_vy * dt), axis=1))


class PSMVelocityAdvection:
    def __init__(self, cfg):
        A = (
            np.diag(4 * np.ones(cfg["grid"]["nv"]), k=0)
            + np.diag(np.ones(cfg["grid"]["nv"]), k=1)
            + np.diag(np.ones(cfg["grid"]["nv"]), k=-1)
        )
        A[0, -1] = 1
        A[-1, 0] = 1
        self.A = jnp.array(A)

        B = np.diag(2 * np.ones(cfg["grid"]["nv"]), k=0) + np.diag(np.ones(cfg["grid"]["nv"]), k=1)
        # A[0, -1] = 1
        B[-1, 0] = 1
        self.B = jnp.array(B)

    def __call__(self, force, f, dt):
        # construct polynomial
        # get gamma
        _f_ = 3.0 * jnp.concatenate([[f[:, :, -1:, :] + f[:, :, 0:1, :]], f[:, :, 1:, :] + f[:, :, :-1, :]], axis=2)
        gamma = lx.LinearSolve(self.A, _f_)

        # get beta
        _g_ = 6.0 * (f - gamma)
        beta = lx.LinearSolve(self.B, _g_)

        # get alpha
        alpha = 0.5 * (beta[1:] - beta[:-1])
        alpha = jnp.concatenate([beta[0:1] - beta[-1:], alpha])

        new_vs = self.vx - force * dt

        # perform interpolation at new locations

        # return new f
        pass


class ExponentialVelocityAdvection:
    """
    Performs v df/dx using an exponential integrator in time and spectral in velocity space

    It is a bit suspect because velocity space is not exactly periodic but it is `approximately` periodic...

    """

    def __init__(self, cfg):
        self.kvx = cfg["grid"]["kvx"][None, None, :, None]
        self.kvy = cfg["grid"]["kvy"][None, None, None, :]
        self.vx = cfg["grid"]["vx"][None, None, :, None]
        self.vy = cfg["grid"]["vy"][None, None, None, :]

    def edfdv(self, fxy, ex, ey, dt):
        """
        e df/dv spectrally

        :param fxy:
        :param ex:
        :param ey:
        :param dt:
        :return:
        """
        fxykvxkvy = jnp.fft.fft2(fxy, axes=(2, 3))
        new_fxykvxkvy = fxykvxkvy * jnp.exp(
            -1j * dt * (ex[..., None, None] * self.kvx + ey[..., None, None] * self.kvy)
        )
        return jnp.real(jnp.fft.ifft2(new_fxykvxkvy, axes=(2, 3)))


class VelocityCubicSpline:
    def __init__(self, cfg):
        nx, ny, nvx, nvy = cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["nvx"], cfg["grid"]["nvy"]
        self.vx = cfg["grid"]["vy"][None, None, :, None] * jnp.ones((nx, ny, nvx, nvy))
        self.vy = cfg["grid"]["vy"][None, None, None, :] * jnp.ones((nx, ny, nvx, nvy))
        self.interp_vx = vmap(partial(interp1d, extrap=True), in_axes=(0, 1, 3))
        self.interp_vy = vmap(partial(interp1d, extrap=True), in_axes=(0, 1, 2))

    def step_x(self, f, force_x, dt):
        vq = self.vx - force_x * dt
        return self.interp_vx(xq=vq, x=self.vx, f=f)

    def step_y(self, f, force_y, dt):
        vq = self.vy - force_y * dt
        return self.interp_vy(xq=vq, x=self.vy, f=f)

    def step_edfdv_x(self, f, ex, dt):
        force_x = ex[:, :, None, None]
        return self.step_x(f, force_x, dt)

    def step_vxBdfdv_x(self, f, bz, dt):
        force_x = self.vy * bz[:, :, None, None]
        return self.step_x(f, force_x, dt)

    def step_edfdv_y(self, f, ey, dt):
        force_y = ey[:, :, None, None]
        return self.step_y(f, force_y, dt)

    def step_vxBdfdv_y(self, f, bz, dt):
        force_y = -self.vx * bz[:, :, None, None]
        return self.step_y(f, force_y, dt)
