#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from collections.abc import Mapping
from typing import Any

import numpy as np
from jax import numpy as jnp

from adept.vlasov2d.solver.tridiagonal import TridiagonalSolver


class Collisions:
    """High-level collision operator that wraps Fokker-Planck and Krook terms."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Build collision pushers from configuration.

        :param cfg: Simulation configuration containing term toggles and grid parameters.
        """
        self.cfg = cfg
        self.fp = self.__init_fp_operator__()
        self.krook = Krook(self.cfg)
        self.td_solver = TridiagonalSolver(self.cfg)

    def __init_fp_operator__(self) -> "LenardBernstein | ChangCooperLenardBernstein | ChangCooperDougherty | Dougherty":
        """
        Instantiate the configured Fokker-Planck operator.

        :raises NotImplementedError: When the configured operator type is unknown.
        """
        fp_type = self.cfg["terms"]["fokker_planck"]["type"].casefold()
        if fp_type == "lenard_bernstein":
            return LenardBernstein(self.cfg)
        elif fp_type in ("chang_cooper", "lenard_bernstein_chang_cooper"):
            return ChangCooperLenardBernstein(self.cfg)
        elif fp_type in ("chang_cooper_dougherty", "dougherty_chang_cooper"):
            return ChangCooperDougherty(self.cfg)
        elif fp_type == "dougherty":
            return Dougherty(self.cfg)
        else:
            raise NotImplementedError

    def __call__(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f, dt: jnp.float64):
        """
        Apply configured collision operators to the distribution function.

        :param nu_fp: Collision frequencies for the Fokker-Planck operator (shape: nx).
        :param nu_K: Krook collision frequencies (shape: nx).
        :param f: Distribution function (dict or array).
        :param dt: Time step size.
        :return: Updated distribution function after collisions.
        """
        # TODO: Properly handle multi-species collisions
        # CR: make a ticket for this TODO
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

    def _apply_collisions(self, nu_fp: jnp.ndarray, nu_K: jnp.ndarray, f: jnp.ndarray, dt: jnp.float64) -> jnp.ndarray:
        """Apply collision operators to a single species distribution."""
        if self.cfg["terms"]["fokker_planck"]["is_on"]:
            # The three diagonals representing collision operator for all x
            cee_a, cee_b, cee_c = self.fp(nu=nu_fp, f_xv=f, dt=dt)
            # Solve over all x
            f = self.td_solver(cee_a, cee_b, cee_c, f)

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
        f_mx = np.exp(-(self.cfg["grid"]["v"][None, :] ** 2.0) / 2.0)
        self.f_mx = f_mx / np.trapz(f_mx, dx=self.cfg["grid"]["dv"], axis=1)[:, None]
        self.dv = self.cfg["grid"]["dv"]

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


class _DriftDiffusionBase:
    """Shared utilities for drift-diffusion-style Fokker-Planck operators."""

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv: jnp.ndarray) -> jnp.ndarray:
        """Compute density n(x) by integrating over velocity."""
        return jnp.sum(f_xv, axis=1) * self.dv

    def _build_tridiagonal(
        self, nu: jnp.ndarray, f_xv: jnp.ndarray, dt: jnp.float64, vbar: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Assemble tridiagonal coefficients given a drift velocity vbar."""
        v0t_sq = self.vx_moment(f_xv * (self.v[None, :] - vbar[:, None]) ** 2.0)

        a = (
            nu[:, None]
            * dt
            * (-v0t_sq[:, None] / self.dv**2.0 + (jnp.roll(self.v, 1)[None, :] - vbar[:, None]) / 2.0 / self.dv)
        )
        b = 1.0 + nu[:, None] * dt * self.ones * (2.0 * v0t_sq[:, None] / self.dv**2.0)
        c = (
            nu[:, None]
            * dt
            * (-v0t_sq[:, None] / self.dv**2.0 - (jnp.roll(self.v, -1)[None, :] - vbar[:, None]) / 2.0 / self.dv)
        )
        return a, b, c


class LenardBernstein(_DriftDiffusionBase):
    """Classic Lenard-Bernstein Fokker-Planck operator."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Initialize Lenard-Bernstein coefficients.

        :param cfg: Simulation configuration providing grid metadata.
        """
        super().__init__(cfg)

    def __call__(
        self, nu: jnp.ndarray, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Assemble tridiagonal coefficients for the Lenard-Bernstein operator.

        NB: The equilbrium solution has a mean velocity of 0.

        :param nu: Collision frequency profile (shape: nx).
        :param f_xv: Distribution function f(x, v) (shape: nx x nv).
        :param dt: Time step size.
        :return: tuple of (lower, diagonal, upper) tridiagonal diagonals.
        """
        vbar = jnp.zeros_like(nu)
        return self._build_tridiagonal(nu, f_xv, dt, vbar)


class ChangCooperLenardBernstein:
    """Chang-Cooper discretization of the Lenard-Bernstein operator."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Precompute velocity grid helpers for Chang-Cooper coefficients.

        :param cfg: Simulation configuration providing grid metadata.
        """
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.v_edge = 0.5 * (self.v[1:] + self.v[:-1])
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv: jnp.ndarray) -> jnp.ndarray:
        """Compute density n(x) by integrating over velocity."""
        return jnp.sum(f_xv, axis=1) * self.dv

    @staticmethod
    def _chang_cooper_delta(w: jnp.ndarray) -> jnp.ndarray:
        """Compute Chang-Cooper weighting factor delta(w)."""
        small = jnp.abs(w) < 1.0e-8
        delta_small = 0.5 - w / 12.0 + w**3 / 720.0
        delta_full = 1.0 / w - 1.0 / jnp.expm1(w)
        return jnp.where(small, delta_small, delta_full)

    def __call__(
        self, nu: jnp.ndarray, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Assemble Chang-Cooper tridiagonal coefficients for Lenard-Bernstein.

        B = v f
        C = <v^2> df/dv

        :param nu: Collision frequency profile (shape: nx).
        :param f_xv: Distribution function f(x, v) (shape: nx x nv).
        :param dt: Time step size.
        :return: tuple of (lower, diagonal, upper) tridiagonal diagonals.
        """
        energy = self.vx_moment(f_xv * self.v[None, :] ** 2.0)
        safe_energy = jnp.maximum(energy, 1.0e-30)

        # Chang-Cooper weights are independent of nu because drift/diffusion share it.
        w = -self.v_edge[None, :] * self.dv / safe_energy[:, None]
        delta = self._chang_cooper_delta(w)

        nu = nu[:, None]
        diff = nu * energy[:, None]
        drift = -nu * self.v_edge[None, :]
        alpha = drift * delta + diff / self.dv
        beta = drift * (1.0 - delta) - diff / self.dv

        lam = dt / self.dv
        a = jnp.zeros_like(self.ones)
        a = a.at[:, 1:].set(-lam * alpha)
        c = jnp.zeros_like(self.ones)
        c = c.at[:, :-1].set(lam * beta)

        beta_l = jnp.concatenate([jnp.zeros((self.cfg["grid"]["nx"], 1)), beta], axis=1)
        alpha_r = jnp.concatenate([alpha, jnp.zeros((self.cfg["grid"]["nx"], 1))], axis=1)
        diag = self.ones + lam * (alpha_r - beta_l)

        return a, diag, c


class ChangCooperDougherty:
    """Chang-Cooper discretization of the Dougherty operator."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Precompute velocity grid helpers for Chang-Cooper coefficients.

        :param cfg: Simulation configuration providing grid metadata.
        """
        self.cfg = cfg
        self.v = self.cfg["grid"]["v"]
        self.dv = self.cfg["grid"]["dv"]
        self.v_edge = 0.5 * (self.v[1:] + self.v[:-1])
        self.ones = jnp.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["nv"]))

    def vx_moment(self, f_xv: jnp.ndarray) -> jnp.ndarray:
        """Compute density n(x) by integrating over velocity."""
        return jnp.sum(f_xv, axis=1) * self.dv

    @staticmethod
    def _chang_cooper_delta(w: jnp.ndarray) -> jnp.ndarray:
        """Compute Chang-Cooper weighting factor delta(w)."""
        small = jnp.abs(w) < 1.0e-8
        delta_small = 0.5 - w / 12.0 + w**3 / 720.0
        delta_full = 1.0 / w - 1.0 / jnp.expm1(w)
        return jnp.where(small, delta_small, delta_full)

    def __call__(
        self, nu: jnp.ndarray, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Assemble Chang-Cooper tridiagonal coefficients for the Dougherty operator.

        B = (v - <v>) f
        C = <(v-<v>)^2> df/dv

        :param nu: Collision frequency profile (shape: nx).
        :param f_xv: Distribution function f(x, v) (shape: nx x nv).
        :param dt: Time step size.
        :return: tuple of (lower, diagonal, upper) tridiagonal diagonals.
        """
        vbar = self.vx_moment(f_xv * self.v[None, :])
        v0t_sq = self.vx_moment(f_xv * (self.v[None, :] - vbar[:, None]) ** 2.0)
        safe_v0t_sq = jnp.maximum(v0t_sq, 1.0e-30)

        w = -(self.v_edge[None, :] - vbar[:, None]) * self.dv / safe_v0t_sq[:, None]
        delta = self._chang_cooper_delta(w)

        nu = nu[:, None]
        diff = nu * safe_v0t_sq[:, None]
        drift = -nu * (self.v_edge[None, :] - vbar[:, None])
        alpha = drift * delta + diff / self.dv
        beta = drift * (1.0 - delta) - diff / self.dv

        lam = dt / self.dv
        a = jnp.zeros_like(self.ones)
        a = a.at[:, 1:].set(-lam * alpha)
        c = jnp.zeros_like(self.ones)
        c = c.at[:, :-1].set(lam * beta)

        beta_l = jnp.concatenate([jnp.zeros((self.cfg["grid"]["nx"], 1)), beta], axis=1)
        alpha_r = jnp.concatenate([alpha, jnp.zeros((self.cfg["grid"]["nx"], 1))], axis=1)
        diag = self.ones + lam * (alpha_r - beta_l)

        return a, diag, c


class Dougherty(_DriftDiffusionBase):
    """Dougherty collision operator using a thermalized drift term."""

    def __init__(self, cfg: Mapping[str, Any]):
        """
        Initialize Dougherty coefficients.

        :param cfg: Simulation configuration providing grid metadata.
        """
        super().__init__(cfg)

    def __call__(
        self, nu: jnp.ndarray, f_xv: jnp.ndarray, dt: jnp.float64
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Assemble tridiagonal coefficients for the Dougherty operator.

        NB: The equilbrium solution has a non-zero and self-consistent mean velocity.

        :param nu: Collision frequency profile (shape: nx).
        :param f_xv: Distribution function f(x, v) (shape: nx x nv).
        :param dt: Time step size.
        :return: tuple of (lower, diagonal, upper) tridiagonal diagonals.
        """
        vbar = self.vx_moment(f_xv * self.v[None, :])
        return self._build_tridiagonal(nu, f_xv, dt, vbar)
