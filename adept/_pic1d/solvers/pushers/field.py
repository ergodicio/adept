"""Field solvers for PIC-1D: spectral Poisson + longitudinal driver.

The Poisson solver here is identical in form to the one used by Vlasov-1D
(:class:`adept._vlasov1d.solvers.pushers.field.SpectralPoissonSolver`), but
takes the charge density already deposited from particles so that the same
spectral solve produces the self-consistent ``E`` for 1:1 comparison.
"""

from jax import numpy as jnp

from adept._pic1d.solvers.pushers.shape import deposit
from adept._vlasov1d.simulation import EMDriver


class SpectralPoissonSolver:
    """Spectral Poisson solve from a pre-computed charge density.

    ``E_k = -i / k * rho_k``, real-valued ``E(x)`` recovered via inverse FFT.
    The k=0 mode is set to zero (no DC field), consistent with periodic BCs and
    a neutralizing static background.
    """

    def __init__(self, one_over_kx: jnp.ndarray, static_charge_density: jnp.ndarray | None = None):
        self.one_over_kx = one_over_kx
        self.static_charge_density = static_charge_density

    def __call__(self, rho: jnp.ndarray) -> jnp.ndarray:
        if self.static_charge_density is not None:
            rho = rho + self.static_charge_density
        return jnp.real(jnp.fft.ifft(-1j * self.one_over_kx * jnp.fft.fft(rho)))


class ParticleChargeDensity:
    """Sum charge density contributions from all species via shape deposition."""

    def __init__(
        self,
        nx: int,
        dx: float,
        xmin: float,
        species_params: dict,
        shape: str = "tsc",
    ):
        self.nx = nx
        self.dx = dx
        self.xmin = xmin
        self.species_params = species_params
        self.shape = shape

    def __call__(self, particles: dict) -> jnp.ndarray:
        """particles: {species_name: {"x": (Np,), "w": (Np,)}} -> rho(nx)."""
        rho = jnp.zeros(self.nx)
        for name, p in particles.items():
            q = self.species_params[name]["charge"]
            n_s = deposit(p["x"], p["w"], self.nx, self.dx, self.xmin, self.shape)
            rho = rho + q * n_s
        return rho


class ElectronChargeDensity:
    """Electron-only charge density q_e * n_e for the transverse wave equation.

    Matches Vlasov-1D's ``compute_electron_charge_density``: deposits only the
    ``electron`` species so that the same ``-0.5 * (cd_n + cd_np1)`` convention
    used by ``WaveSolver`` reproduces the ``n_e * a`` (= ω_pe² * a) term.
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        xmin: float,
        species_params: dict,
        shape: str = "tsc",
    ):
        self.nx = nx
        self.dx = dx
        self.xmin = xmin
        self.shape = shape
        self.electron_charge = species_params.get("electron", {}).get("charge", 0.0)

    def __call__(self, particles: dict) -> jnp.ndarray:
        if "electron" not in particles:
            return jnp.zeros(self.nx)
        p = particles["electron"]
        n_e = deposit(p["x"], p["w"], self.nx, self.dx, self.xmin, self.shape)
        return self.electron_charge * n_e


class LongitudinalElectricFieldDriver:
    """E_drive(x, t) = Σ_d ω * a0 * envelope * sin(k x − ω t). Identical to
    Vlasov-1D's :class:`LongitudinalElectricFieldDriver`, redefined here so we
    don't depend on Vlasov-1D's grid object directly."""

    def __init__(self, xax: jnp.ndarray, drivers: list[EMDriver]):
        self.xax = xax
        self.drivers = drivers

    def _single(self, driver: EMDriver, t):
        kk = driver.k0
        ww = driver.w0
        dw = driver.dw0
        factor = driver.envelope(self.xax, t)
        return factor * (ww + dw) * driver.a0 * jnp.sin(kk * self.xax - (ww + dw) * t)

    def __call__(self, t, args=None) -> jnp.ndarray:
        total = jnp.zeros_like(self.xax)
        for d in self.drivers:
            total = total + self._single(d, t)
        return total
