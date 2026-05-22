"""Field solvers and EM-driver sources for the Vlasov-2D solver.

We solve TE-mode Maxwell on a periodic 2D box, with all spatial derivatives
taken spectrally:

    dEx/dt =  c^2 dBz/dy - Jx
    dEy/dt = -c^2 dBz/dx - Jy
    dBz/dt = -(dEy/dx - dEx/dy)

Time integration is Strang-split with respect to (B-half, E-full, B-half):

    B^{n+1/2} = B^n     + (dt/2) * (dEx/dy - dEy/dx)^n
    E^{n+1}   = E^n     + dt    * ( c^2 curl(B^{n+1/2}) - J^{n+1/2} )
    B^{n+1}   = B^{n+1/2} + (dt/2) * (dEx/dy - dEy/dx)^{n+1}

Drivers are externally prescribed current sources Jx_driver, Jy_driver(t, x, y).
"""

import jax.numpy as jnp

from adept._vlasov2d.simulation import EMDriver


class MaxwellSpectral:
    """Strang-split spectral Maxwell solver on a periodic 2D box."""

    def __init__(self, kx, ky, c: float):
        # broadcast helpers: kx -> (nx, 1), ky -> (1, ny)
        self.kx = kx[:, None]
        self.ky = ky[None, :]
        self.c2 = c * c

    @staticmethod
    def _ddx(field, kx):
        # spectral d/dx
        return jnp.real(jnp.fft.ifft(1j * kx * jnp.fft.fft(field, axis=0), axis=0))

    @staticmethod
    def _ddy(field, ky):
        return jnp.real(jnp.fft.ifft(1j * ky * jnp.fft.fft(field, axis=1), axis=1))

    def b_half_step(self, ex, ey, bz, dt: float):
        dExdy = self._ddy(ex, self.ky)
        dEydx = self._ddx(ey, self.kx)
        return bz - 0.5 * dt * (dEydx - dExdy)

    def e_full_step(self, ex, ey, bz, jx, jy, dt: float):
        dBzdy = self._ddy(bz, self.ky)
        dBzdx = self._ddx(bz, self.kx)
        ex_new = ex + dt * (self.c2 * dBzdy - jx)
        ey_new = ey + dt * (-self.c2 * dBzdx - jy)
        return ex_new, ey_new

    def __call__(self, ex, ey, bz, jx, jy, dt: float):
        bz = self.b_half_step(ex, ey, bz, dt)
        ex, ey = self.e_full_step(ex, ey, bz, jx, jy, dt)
        bz = self.b_half_step(ex, ey, bz, dt)
        return ex, ey, bz


class ChargeMoment:
    """Compute total charge density rho = sum_s q_s * integral f_s dvx dvy."""

    def __init__(self, species_grids, species_params, static_charge_density=None):
        self.species_grids = species_grids
        self.species_params = species_params
        self.static_charge_density = static_charge_density

    def __call__(self, f_dict):
        rho = None
        for name, f in f_dict.items():
            q = self.species_params[name]["charge"]
            dvx = self.species_grids[name]["dvx"]
            dvy = self.species_grids[name]["dvy"]
            n = jnp.sum(f, axis=(2, 3)) * (dvx * dvy)
            rho = q * n if rho is None else rho + q * n
        if self.static_charge_density is not None:
            rho = rho + self.static_charge_density
        return rho


class CurrentMoments:
    """Compute total current density (Jx, Jy) from all species."""

    def __init__(self, species_grids, species_params):
        self.species_grids = species_grids
        self.species_params = species_params

    def __call__(self, f_dict):
        jx_total = None
        jy_total = None
        for name, f in f_dict.items():
            q = self.species_params[name]["charge"]
            vx = self.species_grids[name]["vx"]  # (nvx,)
            vy = self.species_grids[name]["vy"]
            dvx = self.species_grids[name]["dvx"]
            dvy = self.species_grids[name]["dvy"]
            jx_s = q * jnp.sum(vx[None, None, :, None] * f, axis=(2, 3)) * (dvx * dvy)
            jy_s = q * jnp.sum(vy[None, None, None, :] * f, axis=(2, 3)) * (dvx * dvy)
            jx_total = jx_s if jx_total is None else jx_total + jx_s
            jy_total = jy_s if jy_total is None else jy_total + jy_s
        return jx_total, jy_total


class InitialPoissonSolver:
    """Solve nabla^2 phi = -rho once at t=0 to seed (Ex, Ey).

    After t=0 we rely on Ampere's law (dE/dt = c^2 curl B - J) to keep E
    consistent; on a periodic box this preserves the longitudinal constraint
    up to discretization error.
    """

    def __init__(self, kx, ky):
        self.kx = kx[:, None]
        self.ky = ky[None, :]
        k2 = (kx[:, None] ** 2) + (ky[None, :] ** 2)
        # avoid div-by-zero at (kx, ky) = (0, 0)
        inv_k2 = jnp.where(k2 > 0, 1.0 / k2, 0.0)
        self.inv_k2 = inv_k2

    def __call__(self, rho):
        rho_k = jnp.fft.fft2(rho)
        phi_k = rho_k * self.inv_k2
        ex = jnp.real(jnp.fft.ifft2(-1j * self.kx * phi_k))
        ey = jnp.real(jnp.fft.ifft2(-1j * self.ky * phi_k))
        return ex, ey


class EMDriverFieldSource:
    """Sum of prescribed Jx / Jy from driver pulses.

    A driver J = -w^2 * a0 * env(x,y,t) * sin(kx*x + ky*y - w*t) emulates a
    plane-wave current source. For purely longitudinal forcing (no Bz), this
    drives an Ex(kx) oscillation at frequency w with amplitude ~ a0; after the
    envelope turns off, Ex decays at the Landau rate.
    """

    def __init__(self, x, y, drivers: list[EMDriver]):
        self.x = x
        self.y = y
        self.drivers = drivers

    def __call__(self, t: float):
        out = jnp.zeros((self.x.shape[0], self.y.shape[0]))
        for d in self.drivers:
            w = d.w0 + d.dw0
            env = d.envelope(self.x, self.y, t)
            phase = d.k0x * self.x[:, None] + d.k0y * self.y[None, :] - w * t
            out = out + -(w**2) * d.a0 * env * jnp.sin(phase)
        return out
