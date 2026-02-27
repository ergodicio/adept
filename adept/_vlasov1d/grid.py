"""Configuration space grid for Vlasov-1D simulations."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from adept._vlasov1d.normalization import UREG, PlasmaNormalization


class Grid(eqx.Module):
    """Configuration space grid (x, t, and their Fourier duals).

    Only the minimal set of input parameters are passed to the constructor.
    All derived quantities (dx, nt, arrays, etc.) are computed in __init__.
    """

    # Stored fields (all final values, no "requested" intermediates)
    xmin: float
    xmax: float
    nx: int
    tmin: float
    tmax: float  # Actual tmax (aligned to dt)
    dt: float  # Actual dt (possibly overridden for EM stability)
    dx: float
    nt: int
    max_steps: int

    x: jnp.ndarray
    t: jnp.ndarray
    kx: jnp.ndarray
    kxr: jnp.ndarray
    one_over_kx: jnp.ndarray
    one_over_kxr: jnp.ndarray
    x_a: jnp.ndarray

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        tmin: float,
        tmax_requested: float,
        dt_requested: float,
        should_override_dt_for_em_waves: bool,
        beta: float,
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.tmin = tmin

        # Compute dx
        self.dx = xmax / nx

        # Override dt for EM wave stability if needed
        if should_override_dt_for_em_waves:
            c_light = 1.0 / beta
            self.dt = float(0.95 * self.dx / c_light)
        else:
            self.dt = dt_requested

        # Compute nt and adjust tmax
        self.nt = int(tmax_requested / self.dt + 1)
        self.tmax = self.dt * self.nt
        self.max_steps = (self.nt + 4, int(1e8))

        # Build arrays
        self.x = jnp.linspace(xmin + self.dx / 2, xmax - self.dx / 2, nx)
        self.t = jnp.linspace(0, self.tmax, self.nt)

        self.kx = jnp.fft.fftfreq(nx, d=self.dx) * 2.0 * np.pi
        self.kxr = jnp.fft.rfftfreq(nx, d=self.dx) * 2.0 * np.pi

        one_over_kx = np.zeros(nx)
        one_over_kx[1:] = 1.0 / np.array(self.kx)[1:]
        self.one_over_kx = jnp.array(one_over_kx)

        one_over_kxr = np.zeros(len(self.kxr))
        one_over_kxr[1:] = 1.0 / np.array(self.kxr)[1:]
        self.one_over_kxr = jnp.array(one_over_kxr)

        self.x_a = jnp.concatenate([jnp.array([self.x[0] - self.dx]), self.x, jnp.array([self.x[-1] + self.dx])])


def grid_from_dimensionless_cfg(cfg_grid: dict, beta: float, should_override_dt_for_em_waves: bool) -> Grid:
    """Construct Grid from config dict.

    Args:
        cfg_grid: Dictionary of grid configuration
        beta: Speed of light normalization (1/c_norm), needed for EM dt override
        should_override_dt_for_em_waves: Whether dt should be limited to ensure stability of EM waves.
    """

    return Grid(
        xmin=cfg_grid["xmin"],
        xmax=cfg_grid["xmax"],
        nx=cfg_grid["nx"],
        tmin=cfg_grid.get("tmin", 0.0),
        tmax_requested=cfg_grid["tmax"],
        dt_requested=cfg_grid["dt"],
        should_override_dt_for_em_waves=should_override_dt_for_em_waves,
        beta=beta,
    )


def grid_from_physical_units_cfg(
    cfg_grid: dict, should_override_dt_for_em_waves: bool, plasma_norm: PlasmaNormalization
) -> Grid:
    """Construct Grid from config dict with physical units.

    Args:
        cfg_grid: Dictionary of grid configuration
        should_override_dt_for_em_waves: Whether dt should be limited to ensure stability of EM waves.
        plasma_norm: The PlasmaNormalization to use to convert dimensional quantities to simulation quantities.
    """
    dimensionless = lambda q: q.to("").magnitude
    length = lambda s: dimensionless(UREG.Quantity(s) / plasma_norm.L0)
    time = lambda s: dimensionless(UREG.Quantity(s) / plasma_norm.tau)

    beta = 1.0 / plasma_norm.speed_of_light_norm()

    return Grid(
        xmin=length(cfg_grid["xmin"]),
        xmax=length(cfg_grid["xmax"]),
        nx=cfg_grid["nx"],
        tmin=time(cfg_grid.get("tmin", "0.0 s")),
        tmax_requested=time(cfg_grid["tmax"]),
        dt_requested=time(cfg_grid["dt"]),
        should_override_dt_for_em_waves=should_override_dt_for_em_waves,
        beta=beta,
    )
