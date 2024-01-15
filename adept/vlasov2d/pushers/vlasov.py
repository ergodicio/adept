import numpy as np
from jax import numpy as jnp


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
