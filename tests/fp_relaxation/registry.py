#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Pytest fixtures for Fokker-Planck relaxation tests.

Provides grid setup, model/scheme instantiation, and MLflow configuration.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from adept._vlasov1d.solvers.pushers.fokker_planck import Dougherty, LenardBernstein
from adept.driftdiffusion import CentralDifferencing, ChangCooper
from adept.vfp1d.fokker_planck import FastVFP


class VelocityGrid(eqx.Module):
    """Velocity grid configuration for Fokker-Planck tests.

    Args:
        nv: Number of velocity cells
        vmax: Maximum velocity (positive-only for spherical, symmetric for cartesian)
        spherical: If True, grid is [0, vmax]; if False, grid is [-vmax, vmax]
    """

    v: Array  # Cell centers, shape (nv,)
    dv: Array = eqx.field(converter=jnp.asarray)  # Grid spacing, 0-d
    vmax: Array = eqx.field(converter=jnp.asarray)  # Domain extent, 0-d
    nv: int = eqx.field(static=True)  # Number of cells
    spherical: bool = eqx.field(static=True)  # True = positive-only [0, vmax]

    def __init__(self, nv: int = 128, vmax: float = 8.0, spherical: bool = False):
        if spherical:
            dv = vmax / nv
            v = jnp.linspace(dv / 2, vmax - dv / 2, nv)
        else:
            dv = 2 * vmax / nv
            v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        self.v = v
        self.dv = dv
        self.vmax = vmax
        self.nv = nv
        self.spherical = spherical

    @property
    def v_edge(self) -> Array:
        """Cell edges, shape (nv-1,)."""
        return 0.5 * (self.v[1:] + self.v[:-1])


# Model/scheme registries: name -> class (all models take (v, dv), all schemes take (dv,))
MODELS = {
    "LenardBernstein": LenardBernstein,
    "Dougherty": Dougherty,
    "FastVFP": FastVFP,
}
SCHEMES = {
    "ChangCooper": ChangCooper,
    "CentralDifferencing": CentralDifferencing,
}


def get_git_info() -> dict[str, str]:
    """Get git repository information for MLflow tagging."""
    import subprocess

    info = {}
    try:
        info["git_branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_branch"] = "unknown"

    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = "unknown"

    try:
        info["git_author"] = (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%an"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_author"] = "unknown"

    return info
