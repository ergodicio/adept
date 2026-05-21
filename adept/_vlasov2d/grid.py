"""Configuration-space grid for the Vlasov-2D solver."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from adept._vlasov2d.datamodel import GridConfig
from adept.normalization import PlasmaNormalization, normalize


class Grid(eqx.Module):
    """Periodic 2D configuration grid + Fourier duals."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    nx: int
    ny: int
    tmin: float
    tmax: float
    dt: float
    dx: float
    dy: float
    nt: int
    max_steps: int

    x: jnp.ndarray
    y: jnp.ndarray
    t: jnp.ndarray
    kx: jnp.ndarray
    ky: jnp.ndarray
    kxr: jnp.ndarray
    kyr: jnp.ndarray
    one_over_kx: jnp.ndarray
    one_over_ky: jnp.ndarray

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        tmin: float,
        tmax_requested: float,
        dt_requested: float,
        should_override_dt_for_em_waves: bool,
        beta: float,
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.tmin = tmin

        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny

        if should_override_dt_for_em_waves:
            c_light = 1.0 / beta
            # Spectral Maxwell: stability requires c dt * k_max < 1; use 0.5 of the
            # min cell to be safe given splitting overhead.
            dt_cap = 0.5 * min(self.dx, self.dy) / c_light
            self.dt = min(dt_requested, float(dt_cap))
        else:
            self.dt = dt_requested

        self.nt = int(tmax_requested / self.dt + 1)
        self.tmax = self.dt * self.nt

        max_steps = int(1e8)
        if self.nt > max_steps:
            print(f"Requested {self.nt} steps, only running {max_steps} steps")
        self.max_steps = min(self.nt + 4, max_steps)

        self.x = jnp.linspace(xmin + self.dx / 2, xmax - self.dx / 2, nx)
        self.y = jnp.linspace(ymin + self.dy / 2, ymax - self.dy / 2, ny)
        self.t = jnp.linspace(0, self.tmax, self.nt)

        self.kx = jnp.fft.fftfreq(nx, d=self.dx) * 2.0 * np.pi
        self.ky = jnp.fft.fftfreq(ny, d=self.dy) * 2.0 * np.pi
        self.kxr = jnp.fft.rfftfreq(nx, d=self.dx) * 2.0 * np.pi
        self.kyr = jnp.fft.rfftfreq(ny, d=self.dy) * 2.0 * np.pi

        one_over_kx = np.zeros(nx)
        one_over_kx[1:] = 1.0 / np.array(self.kx)[1:]
        self.one_over_kx = jnp.array(one_over_kx)

        one_over_ky = np.zeros(ny)
        one_over_ky[1:] = 1.0 / np.array(self.ky)[1:]
        self.one_over_ky = jnp.array(one_over_ky)

    @staticmethod
    def from_config(
        cfg: GridConfig, beta: float, should_override_dt_for_em_waves: bool, norm: PlasmaNormalization | None
    ) -> "Grid":
        return Grid(
            xmin=normalize(cfg.xmin, norm, dim="x"),
            xmax=normalize(cfg.xmax, norm, dim="x"),
            nx=cfg.nx,
            ymin=normalize(cfg.ymin, norm, dim="x"),
            ymax=normalize(cfg.ymax, norm, dim="x"),
            ny=cfg.ny,
            tmin=normalize(cfg.tmin, norm, dim="t"),
            tmax_requested=normalize(cfg.tmax, norm, dim="t"),
            dt_requested=normalize(cfg.dt, norm, dim="t"),
            should_override_dt_for_em_waves=should_override_dt_for_em_waves,
            beta=beta,
        )
