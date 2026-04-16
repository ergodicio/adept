"""Configuration-space and velocity-space grid for VFP-1D simulations."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from adept.normalization import PlasmaNormalization


class Grid(eqx.Module):
    """Spatial, velocity, and temporal grid for VFP-1D.

    Simpler than the Vlasov-1D grid: no FFT / Fourier-dual arrays are needed
    because VFP-1D does not use spectral solvers.

    All derived quantities (dx, dv, nt, arrays, etc.) are computed in __init__
    from the minimal set of input parameters.
    """

    # Spatial grid
    xmin: float
    xmax: float
    nx: int
    dx: float
    x: jnp.ndarray  # cell centres (nx,)
    x_edge: jnp.ndarray  # cell edges  (nx+1,)

    # Velocity grid (positive-only, 0 to vmax)
    nv: int
    vmax: float
    dv: float
    v: jnp.ndarray  # cell centres (nv,)
    v_edge: jnp.ndarray  # cell edges (nv-1,)

    # Temporal grid
    tmin: float
    tmax: float  # actual tmax (aligned to dt)
    dt: float
    nt: int
    max_steps: int
    t: jnp.ndarray

    # Physics / mode parameters stored on the grid for convenience
    nl: int  # number of Legendre harmonics
    boundary: str  # "periodic" or "reflective"

    def __init__(
        self,
        *,
        xmin: float,
        xmax: float,
        nx: int,
        tmin: float,
        tmax: float,
        dt: float,
        nv: int,
        vmax: float,
        nl: int,
        boundary: str = "periodic",
    ):
        # -- Spatial ----------------------------------------------------------
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.dx = xmax / nx

        self.x = jnp.linspace(xmin + self.dx / 2, xmax - self.dx / 2, nx)
        self.x_edge = jnp.linspace(xmin, xmax, nx + 1)

        # -- Velocity ---------------------------------------------------------
        self.nv = nv
        self.vmax = vmax
        self.dv = vmax / nv

        self.v = jnp.linspace(self.dv / 2, vmax - self.dv / 2, nv)
        self.v_edge = 0.5 * (self.v[1:] + self.v[:-1])

        # -- Temporal ---------------------------------------------------------
        self.tmin = tmin
        self.dt = dt
        self.nt = int(tmax / dt) + 1

        max_steps = 1e6
        if self.nt > max_steps:
            print(f"Requested {self.nt} steps, only running {int(max_steps)} steps")
            self.max_steps = int(max_steps)
        else:
            self.max_steps = self.nt + 4

        self.tmax = self.dt * self.nt
        self.t = jnp.linspace(0, self.tmax, self.nt)

        # -- Physics ----------------------------------------------------------
        self.nl = nl
        self.boundary = boundary

    @staticmethod
    def from_config(cfg_grid: dict, norm: PlasmaNormalization) -> Grid:
        """Construct Grid from config dict and plasma normalization.

        Args:
            cfg_grid: The ``cfg["grid"]`` dict (raw, with unit strings).
            norm: Plasma normalization (provides L0, tau, T0 for unit conversion).
        """
        from adept.normalization import normalize

        xmax = normalize(cfg_grid["xmax"], norm, dim="x")
        xmin = normalize(cfg_grid["xmin"], norm, dim="x")
        tmax = normalize(cfg_grid["tmax"], norm, dim="t")
        dt = normalize(cfg_grid["dt"], norm, dim="t")

        vmax = 8.0 * norm.vth_norm() / np.sqrt(2.0)

        return Grid(
            xmin=xmin,
            xmax=xmax,
            nx=cfg_grid["nx"],
            tmin=0.0,
            tmax=tmax,
            dt=dt,
            nv=cfg_grid["nv"],
            vmax=vmax,
            nl=cfg_grid["nl"],
            boundary=cfg_grid.get("boundary", "periodic"),
        )

    def as_dict(self) -> dict:
        """Return all grid fields as a plain dict (delegates to ``dataclasses.asdict``)."""
        return asdict(self)
