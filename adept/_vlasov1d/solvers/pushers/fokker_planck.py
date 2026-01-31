#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
"""
Fokker-Planck collision operators for the Vlasov-1D solver.

This module provides collision operators that can be used with the Vlasov-1D solver.
The operators are built on shared abstractions from adept.driftdiffusion.
"""

from collections.abc import Mapping
from typing import Any

import lineax as lx
import numpy as np
from jax import Array, vmap
from jax import numpy as jnp

from adept.driftdiffusion import (
    CentralDifferencing,
    ChangCooper,
    Dougherty,
    LenardBernstein,
)


class Collisions:
    """High-level collision operator that wraps Fokker-Planck and Krook terms."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Build collision pushers from configuration.

        :param cfg: Simulation configuration containing term toggles and grid parameters.
        """
        self.cfg = cfg
        self.fp_model, self.fp_scheme = self.__init_fp_operator__()
        self.krook = Krook(self.cfg)

    def __init_fp_operator__(self):
        """
        Instantiate the configured Fokker-Planck model and scheme.

        :raises NotImplementedError: When the configured operator type is unknown.
        :returns: Tuple of (model, scheme)
        """
        # TODO(gh-173): For multi-species, use electron grid for FP for now
        v = self.cfg["grid"]["species_grids"]["electron"]["v"]
        dv = self.cfg["grid"]["species_grids"]["electron"]["dv"]

        fp_type = self.cfg["terms"]["fokker_planck"]["type"].casefold()

        if fp_type == "lenard_bernstein":
            model = LenardBernstein(v=v, dv=dv)
            return model, CentralDifferencing(dv=dv)
        elif fp_type in ("chang_cooper", "lenard_bernstein_chang_cooper"):
            model = LenardBernstein(v=v, dv=dv)
            return model, ChangCooper(dv=dv)
        elif fp_type in ("chang_cooper_dougherty", "dougherty_chang_cooper"):
            model = Dougherty(v=v, dv=dv)
            return model, ChangCooper(dv=dv)
        elif fp_type == "dougherty":
            model = Dougherty(v=v, dv=dv)
            return model, CentralDifferencing(dv=dv)
        else:
            raise NotImplementedError(f"Unknown Fokker-Planck type: {fp_type}")

    def __call__(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f, dt: jnp.float64):
        """
        Apply configured collision operators to the distribution function.

        :param nu_fp: Collision frequencies for the Fokker-Planck operator (shape: nx).
        :param nu_K: Krook collision frequencies (shape: nx).
        :param f: Distribution function (dict or array).
        :param dt: Time step size.
        :return: Updated distribution function after collisions.
        """
        # TODO(gh-173): Properly handle multi-species collisions
        # For now, only apply to electron distribution for backward compatibility
        if isinstance(f, dict):
            result = {}
            for species_name, f_species in f.items():
                if species_name == "electron":
                    result[species_name] = self._apply_collisions(nu_fp, nu_K, f_species, dt)
                else:
                    # For non-electron species, just pass through unchanged for now
                    result[species_name] = f_species
            return result
        else:
            return self._apply_collisions(nu_fp, nu_K, f, dt)

    def _solve_one_x(self, C_edge: Array, D_scalar: Array, nu: Array, f_v: Array, dt: float) -> Array:
        """Solve the collision operator at a single location in space."""
        op = self.fp_scheme.get_operator(C_edge=C_edge, D=D_scalar, nu=nu, dt=dt)
        return lx.linear_solve(op, f_v, solver=lx.AutoLinearSolver(well_posed=True)).value

    def _apply_collisions(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """Apply collision operators to a single species distribution."""
        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            C, D = self.fp_model(f)
            f = vmap(self._solve_one_x, in_axes=(0, 0, 0, 0, None))(C, D, nu_fp, f, dt)

        if self.cfg["terms"]["krook"]["is_on"]:
            f = self.krook(nu_K, f, dt)

        return f


class Krook:
    """Krook relaxation operator that damps toward a Maxwellian profile."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Precompute Maxwellian profile used for Krook relaxation.

        :param cfg: Simulation configuration containing grid spacing and velocity grid.
        """
        self.cfg = cfg
        v = cfg["grid"]["species_grids"]["electron"]["v"]
        dv = cfg["grid"]["species_grids"]["electron"]["dv"]
        f_mx = np.exp(-(v[None, :] ** 2.0) / 2.0)
        self.f_mx = f_mx / np.trapz(f_mx, dx=dv, axis=1)[:, None]
        self.dv = dv

    def vx_moment(self, f_xv: jnp.ndarray) -> jnp.ndarray:
        """Compute density n(x) by integrating over velocity."""
        return jnp.sum(f_xv, axis=1) * self.dv

    def __call__(self, nu_K: jnp.ndarray, f_xv: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """
        Relax distribution toward a Maxwellian using Krook collisions.

        :param nu_K: Krook collision frequency profile (shape: nx).
        :param f_xv: Distribution function f(x, v) (shape: nx x nv).
        :param dt: Time step size.
        :return: Updated distribution after Krook relaxation.
        """
        nu_Kxdt = dt * nu_K[:, None]
        exp_nuKxdt = jnp.exp(-nu_Kxdt)
        n_prof = self.vx_moment(f_xv)

        return f_xv * exp_nuKxdt + n_prof[:, None] * self.f_mx * (1.0 - exp_nuKxdt)
