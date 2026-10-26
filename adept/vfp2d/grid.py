"""Periodic configuration-space and radial-momentum grid for VFP-2D."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass(frozen=True)
class Grid:
    xmin: float
    xmax: float
    nx: int
    ymin: float
    ymax: float
    ny: int
    vmax: float
    nv: int
    dt: float
    l_max: int
    m_max: int
    dx: float
    dy: float
    dv: float
    x: Array
    y: Array
    v: Array
    kx: Array
    ky: Array

    def __init__(
        self,
        *,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        vmax: float,
        nv: int,
        dt: float,
        l_max: int,
        m_max: int | None = None,
    ):
        if nx < 1 or ny < 1 or nv < 2:
            raise ValueError("nx and ny must be positive and nv must be at least 2")
        if xmax <= xmin or ymax <= ymin or vmax <= 0 or dt <= 0:
            raise ValueError("grid extents, vmax, and dt must be positive")
        if m_max is None:
            m_max = l_max
        if not 0 <= m_max <= l_max:
            raise ValueError("m_max must satisfy 0 <= m_max <= l_max")

        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        dv = vmax / nv
        object.__setattr__(self, "xmin", xmin)
        object.__setattr__(self, "xmax", xmax)
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ymin", ymin)
        object.__setattr__(self, "ymax", ymax)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "vmax", vmax)
        object.__setattr__(self, "nv", nv)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "l_max", l_max)
        object.__setattr__(self, "m_max", m_max)
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        object.__setattr__(self, "dv", dv)
        object.__setattr__(self, "x", jnp.linspace(xmin + dx / 2, xmax - dx / 2, nx))
        object.__setattr__(self, "y", jnp.linspace(ymin + dy / 2, ymax - dy / 2, ny))
        object.__setattr__(self, "v", jnp.linspace(dv / 2, vmax - dv / 2, nv))
        object.__setattr__(self, "kx", jnp.fft.fftfreq(nx, d=dx) * 2.0 * np.pi)
        object.__setattr__(self, "ky", jnp.fft.fftfreq(ny, d=dy) * 2.0 * np.pi)
