"""Configuration space grid for Vlasov-1D simulations."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from adept.normalization import UREG, PlasmaNormalization, normalize


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
            self.dt = min(dt_requested, float(0.95 * self.dx / c_light))
        else:
            self.dt = dt_requested

        # Compute nt and adjust tmax
        self.nt = int(tmax_requested / self.dt + 1)
        self.tmax = self.dt * self.nt

        max_steps = 1e8
        if self.nt > max_steps:
            print(f"Requested {self.nt} steps, only running {int(max_steps)} steps")
        self.max_steps = min(self.nt + 4, int(max_steps))

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

    @staticmethod
    def from_config(
        cfg_grid: dict, beta: float, should_override_dt_for_em_waves: bool, norm: PlasmaNormalization | None
    ) -> "Grid":
        """Construct Grid from config dict.

        Args:
            cfg_grid: Dictionary of grid configuration
            beta: Speed of light normalization (1/c_norm), needed for EM dt override
            should_override_dt_for_em_waves: Whether dt should be limited to ensure stability of EM waves.
        """

        return Grid(
            xmin=normalize(cfg_grid["xmin"], norm, dim="x"),
            xmax=normalize(cfg_grid["xmax"], norm, dim="x"),
            nx=cfg_grid["nx"],
            tmin=normalize(cfg_grid.get("tmin", 0.0), norm, dim="t"),
            tmax_requested=normalize(cfg_grid["tmax"], norm, dim="t"),
            dt_requested=normalize(cfg_grid["dt"], norm, dim="t"),
            should_override_dt_for_em_waves=should_override_dt_for_em_waves,
            beta=beta,
        )
