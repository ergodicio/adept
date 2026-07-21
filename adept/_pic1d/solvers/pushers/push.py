"""Particle pushers + symplectic compositions.

Both integrators operate on a state dict::

    {"x_<species>": (Np,), "v_<species>": (Np,)}

The "kick" step uses the self-consistent E plus the longitudinal external
driver evaluated at the kick time. Field solves and gathers happen inside the
``kick`` helper so we can compose any symplectic timesplit.
"""

from jax import numpy as jnp

from adept._pic1d.solvers.pushers.field import (
    LongitudinalElectricFieldDriver,
    ParticleChargeDensity,
    SpectralPoissonSolver,
)
from adept._pic1d.solvers.pushers.shape import gather


def _drift(particles: dict, dt: float, xmin: float, xmax: float) -> dict:
    """Free-streaming drift: x += dt * v, with periodic wrap on [xmin, xmax)."""
    L = xmax - xmin
    out = {}
    for name, p in particles.items():
        x_new = p["x"] + dt * p["v"]
        # Wrap modulo L while keeping particle weight & velocity untouched.
        x_new = jnp.mod(x_new - xmin, L) + xmin
        out[name] = {"x": x_new, "v": p["v"], "w": p["w"]}
    return out


class _Kicker:
    """Helper that builds E(x_p, t) and applies v += dt * (q/m) * E_p.

    Evaluating the field requires (i) deposit, (ii) Poisson solve, (iii) add
    external driver, (iv) gather. We separate this so the integrators can call
    it any number of times per step.

    When a transverse vector potential ``a`` is supplied to :meth:`apply`,
    we additionally gather the ponderomotive force ``F_pond = -½ ∂_x(a²)``
    onto each particle. The full per-species kick is then

        Δv = dt · (q/m) · ( E + (q/m) · F_pond )

    matching the Vlasov-1D convention in ``vlasov.VelocityExponential``.
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        xmin: float,
        species_params: dict,
        one_over_kx: jnp.ndarray,
        xax: jnp.ndarray,
        ex_drivers,
        static_charge_density: jnp.ndarray | None,
        shape: str,
    ):
        self.species_params = species_params
        self.shape = shape
        self.dx = dx
        self.xmin = xmin
        self.charge_density = ParticleChargeDensity(nx, dx, xmin, species_params, shape)
        self.poisson = SpectralPoissonSolver(one_over_kx, static_charge_density=static_charge_density)
        self.driver = LongitudinalElectricFieldDriver(xax, ex_drivers)

    def field(self, particles: dict, t: float) -> jnp.ndarray:
        rho = self.charge_density(particles)
        e_sc = self.poisson(rho)
        e_drv = self.driver(t)
        return e_sc, e_drv

    def apply(self, particles: dict, dt: float, t: float, a: jnp.ndarray | None = None):
        e_sc, e_drv = self.field(particles, t)
        e_total = e_sc + e_drv
        # Ponderomotive force from transverse vector potential.
        # ``a`` is sized (nx+2,) — interior slice of ``gradient(a²)`` is (nx,).
        if a is not None:
            pond_grid = -0.5 * jnp.gradient(a**2, self.dx)[1:-1]
        else:
            pond_grid = None
        out = {}
        for name, p in particles.items():
            qm = self.species_params[name]["charge_to_mass"]
            Ep = gather(e_total, p["x"], self.dx, self.xmin, self.shape)
            if pond_grid is not None:
                pond_p = gather(pond_grid, p["x"], self.dx, self.xmin, self.shape)
                accel = qm * Ep + qm * qm * pond_p
            else:
                accel = qm * Ep
            v_new = p["v"] + dt * accel
            out[name] = {"x": p["x"], "v": v_new, "w": p["w"]}
        return out, e_sc, e_drv


class LeapfrogIntegrator:
    """Standard kick-drift-kick (KDK) leapfrog.

    With ``x, v`` co-located at integer time levels, one step is::

        v_{n+1/2} = v_n + (dt/2) (q/m) E(x_n, t_n)
        x_{n+1}   = x_n + dt v_{n+1/2}
        v_{n+1}   = v_{n+1/2} + (dt/2) (q/m) E(x_{n+1}, t_{n+1})
    """

    def __init__(self, kicker: _Kicker, xmin: float, xmax: float, dt: float):
        self.kicker = kicker
        self.xmin = xmin
        self.xmax = xmax
        self.dt = dt

    def __call__(self, particles: dict, t: float, a: jnp.ndarray | None = None):
        particles, _, _ = self.kicker.apply(particles, 0.5 * self.dt, t, a)
        particles = _drift(particles, self.dt, self.xmin, self.xmax)
        particles, e_sc, e_drv = self.kicker.apply(particles, 0.5 * self.dt, t + self.dt, a)
        return particles, e_sc, e_drv


class Yoshida4Integrator:
    """4th-order symplectic composition (Yoshida 1990) of three leapfrog steps.

    ``L(dt) = L(dt1) ∘ L(dt2) ∘ L(dt1)`` with ``dt1 = dt/(2 - 2^{1/3})`` and
    ``dt2 = -2^{1/3} dt1``. Each inner ``L`` is a KDK leapfrog of duration
    ``sub_dt`` evaluated at the appropriate substep start time.
    """

    def __init__(self, kicker: _Kicker, xmin: float, xmax: float, dt: float):
        self.kicker = kicker
        self.xmin = xmin
        self.xmax = xmax
        self.dt = dt

        cbrt2 = 2.0 ** (1.0 / 3.0)
        self.dt1 = dt / (2.0 - cbrt2)
        self.dt2 = -cbrt2 * self.dt1
        # Substep start times for each inner leapfrog (relative to step start)
        self.t_offsets = (0.0, self.dt1, self.dt1 + self.dt2)
        self.sub_dts = (self.dt1, self.dt2, self.dt1)

    def _inner_leapfrog(self, particles, t_start, sub_dt, a):
        particles, _, _ = self.kicker.apply(particles, 0.5 * sub_dt, t_start, a)
        particles = _drift(particles, sub_dt, self.xmin, self.xmax)
        particles, e_sc, e_drv = self.kicker.apply(particles, 0.5 * sub_dt, t_start + sub_dt, a)
        return particles, e_sc, e_drv

    def __call__(self, particles: dict, t: float, a: jnp.ndarray | None = None):
        e_sc = None
        e_drv = None
        for off, sd in zip(self.t_offsets, self.sub_dts, strict=True):
            particles, e_sc, e_drv = self._inner_leapfrog(particles, t + off, sd, a)
        return particles, e_sc, e_drv
