"""Vlasov pushers for the 2D2V solver.

f has shape (nx, ny, nvx, nvy). Streaming and electric-field acceleration are
exponential (spectral shifts). The magnetic rotation in (vx, vy) is a 2D
semi-Lagrangian (cubic) interpolation.
"""

import jax
import jax.numpy as jnp
from interpax import interp2d
from jax import vmap


def _shift_axis(f, phase, axis: int, n: int):
    """rfft along `axis`, multiply by `phase`, irfft back to length n."""
    fk = jnp.fft.rfft(f, axis=axis)
    out = jnp.real(jnp.fft.irfft(phase * fk, axis=axis, n=n))
    return out


class SpaceExponentialX:
    """Spectral streaming in x: f <- exp(-i kx vx dt) f."""

    def __init__(self, kxr, species_grids):
        self.kxr = kxr  # (nxr,)
        self.species_grids = species_grids

    def __call__(self, f_dict, dt: float):
        out = {}
        for name, f in f_dict.items():
            vx = self.species_grids[name]["vx"]  # (nvx,)
            # broadcast: (nxr, 1, nvx, 1)
            phase = jnp.exp(-1j * self.kxr[:, None, None, None] * vx[None, None, :, None] * dt)
            out[name] = _shift_axis(f, phase, axis=0, n=f.shape[0])
        return out


class SpaceExponentialY:
    """Spectral streaming in y: f <- exp(-i ky vy dt) f."""

    def __init__(self, kyr, species_grids):
        self.kyr = kyr  # (nyr,)
        self.species_grids = species_grids

    def __call__(self, f_dict, dt: float):
        out = {}
        for name, f in f_dict.items():
            vy = self.species_grids[name]["vy"]  # (nvy,)
            # broadcast: (1, nyr, 1, nvy)
            phase = jnp.exp(-1j * self.kyr[None, :, None, None] * vy[None, None, None, :] * dt)
            out[name] = _shift_axis(f, phase, axis=1, n=f.shape[1])
        return out


class VelocityExponentialE:
    """Spectral velocity push from electric field only (vx and vy independent).

    For each species: f <- exp(-i kvx*(qEx/m)*dt) f, then same for vy with Ey.
    The two shifts commute exactly so there is no splitting error from this pair.
    """

    def __init__(self, species_grids, species_params):
        self.species_grids = species_grids
        self.species_params = species_params

    def push_vx(self, f_dict, ex, dt: float):
        out = {}
        for name, f in f_dict.items():
            q = self.species_params[name]["charge"]
            m = self.species_params[name]["mass"]
            kvxr = self.species_grids[name]["kvxr"]  # (nvxr,)
            ax = (q / m) * ex  # (nx, ny)
            # phase: (nx, ny, nvxr, 1)
            phase = jnp.exp(-1j * kvxr[None, None, :, None] * ax[:, :, None, None] * dt)
            out[name] = _shift_axis(f, phase, axis=2, n=f.shape[2])
        return out

    def push_vy(self, f_dict, ey, dt: float):
        out = {}
        for name, f in f_dict.items():
            q = self.species_params[name]["charge"]
            m = self.species_params[name]["mass"]
            kvyr = self.species_grids[name]["kvyr"]  # (nvyr,)
            ay = (q / m) * ey
            phase = jnp.exp(-1j * kvyr[None, None, None, :] * ay[:, :, None, None] * dt)
            out[name] = _shift_axis(f, phase, axis=3, n=f.shape[3])
        return out


class VelocitySL2DE:
    """Optional: 2D semi-Lagrangian E push (cubic). Same physics as VelocityExponentialE."""

    def __init__(self, species_grids, species_params):
        self.species_grids = species_grids
        self.species_params = species_params

    def _interp_at(self, name, f, vx_q, vy_q):
        vx = self.species_grids[name]["vx"]
        vy = self.species_grids[name]["vy"]

        # vmap over (x, y)
        def _one(f_xy, qx, qy):
            return interp2d(qx.ravel(), qy.ravel(), vx, vy, f_xy, method="cubic", extrap=0.0).reshape(qx.shape)

        return vmap(vmap(_one))(f, vx_q, vy_q)

    def push_vx(self, f_dict, ex, dt: float):
        out = {}
        for name, f in f_dict.items():
            q = self.species_params[name]["charge"]
            m = self.species_params[name]["mass"]
            vx = self.species_grids[name]["vx"]
            vy = self.species_grids[name]["vy"]
            ax = (q / m) * ex  # (nx, ny)
            nvx, nvy = vx.size, vy.size
            vx_q = vx[None, None, :, None] - ax[:, :, None, None] * dt + 0.0 * vy[None, None, None, :]
            vy_q = jnp.broadcast_to(vy[None, None, None, :], vx_q.shape)
            out[name] = self._interp_at(name, f, vx_q, vy_q)
        return out

    def push_vy(self, f_dict, ey, dt: float):
        out = {}
        for name, f in f_dict.items():
            q = self.species_params[name]["charge"]
            m = self.species_params[name]["mass"]
            vx = self.species_grids[name]["vx"]
            vy = self.species_grids[name]["vy"]
            ay = (q / m) * ey
            vx_q = jnp.broadcast_to(vx[None, None, :, None], f.shape)
            vy_q = vy[None, None, None, :] - ay[:, :, None, None] * dt + 0.0 * vx[None, None, :, None]
            out[name] = self._interp_at(name, f, vx_q, vy_q)
        return out


class VelocityRotateB:
    """2D rotation of f in (vx, vy) under the magnetic force qv x Bz.

    Exact (along characteristics) for constant Bz over dt at each (x, y):
        v(t+dt) = R(theta) v(t)   with   theta = -(q/m) Bz dt
    Backward characteristic (semi-Lagrangian): the new value at (vx, vy) was
    located at R(-theta) (vx, vy) at the start of the step.
    """

    def __init__(self, species_grids, species_params):
        self.species_grids = species_grids
        self.species_params = species_params

    def __call__(self, f_dict, bz: jax.Array, dt: float):
        out = {}
        for name, f in f_dict.items():
            q = self.species_params[name]["charge"]
            m = self.species_params[name]["mass"]
            vx = self.species_grids[name]["vx"]
            vy = self.species_grids[name]["vy"]

            theta = -(q / m) * bz * dt  # (nx, ny)
            cos_t = jnp.cos(theta)[:, :, None, None]  # (nx, ny, 1, 1)
            sin_t = jnp.sin(theta)[:, :, None, None]

            vx2d = vx[None, None, :, None]  # (1, 1, nvx, 1)
            vy2d = vy[None, None, None, :]  # (1, 1, 1, nvy)

            # Backward characteristic: query points are R(-theta) (vx, vy)
            vx_q = cos_t * vx2d + sin_t * vy2d
            vy_q = -sin_t * vx2d + cos_t * vy2d
            vx_q = jnp.broadcast_to(vx_q, f.shape)
            vy_q = jnp.broadcast_to(vy_q, f.shape)

            def _one(f_xy, qx, qy, _vx=vx, _vy=vy):
                return interp2d(qx.ravel(), qy.ravel(), _vx, _vy, f_xy, method="cubic", extrap=0.0).reshape(qx.shape)

            out[name] = vmap(vmap(_one))(f, vx_q, vy_q)
        return out


class HouLiFilter:
    """Separable Hou-Li spectral filter in configuration space {x, y}.

    Velocity-space (vx, vy) filtering was removed: the FFT-based filter is periodic in
    velocity, so it wraps the forward tail onto the -v edge and corrupts f(v). It must
    never be applied in velocity space.
    """

    def __init__(self, nx: int, ny: int, alpha: float, order: int, dimensions: list[str]):
        self.dimensions = dimensions

        def _f1d(n):
            j = jnp.arange(n // 2 + 1)
            eta = j / (n // 2)
            return jnp.exp(-alpha * eta ** (2 * order))

        self.filter_x = _f1d(nx) if "x" in dimensions else None
        self.filter_y = _f1d(ny) if "y" in dimensions else None

    @staticmethod
    def _apply(f, sigma, axis):
        if sigma is None:
            return f
        fk = jnp.fft.rfft(f, axis=axis)
        shape = [1] * f.ndim
        shape[axis] = sigma.shape[0]
        return jnp.real(jnp.fft.irfft(fk * sigma.reshape(shape), axis=axis, n=f.shape[axis]))

    def __call__(self, f_dict):
        out = {}
        for name, f in f_dict.items():
            ff = self._apply(f, self.filter_x, 0)
            ff = self._apply(ff, self.filter_y, 1)
            out[name] = ff
        return out
