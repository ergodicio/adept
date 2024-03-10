from typing import Dict, Callable
from collections import defaultdict
import functools

import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np
from adept.sh2d.solvers.tridiagonal import TridiagonalSolver


class IsotropicCollisions(eqx.Module):
    coll_opp: eqx.Module

    def __init__(self, cfg):
        # self.v = cfg["grid"]["v"]
        if cfg["terms"]["fokker-planck"]["f00"] == "lenard_bernstein":
            self.coll_opp = LenardBernstein()
        elif cfg["terms"]["fokker-planck"]["f00"] == "chang_cooper":
            self.coll_opp = ChangCooper(cfg)
        elif cfg["terms"]["fokker-planck"]["f00"] == "shkarofsky":
            self.coll_opp = Shkarofsky(cfg)
        else:
            raise NotImplementedError

    def __call__(self, t, y, args):
        new_f00 = self.coll_opp(t, y, args)
        return new_f00


def calc_i(v, flm, j):
    return 4 * jnp.pi / v[None, None, :] * jnp.cumsum(flm * v[None, None, :] ** (j + 2), axis=2)


def calc_j(v, flm, j):
    temp = jnp.cumsum(flm * v[None, None, :] ** (j + 2), axis=2)
    return 4 * jnp.pi / v[None, None, :] * (jnp.sum(flm * v[None, None, :] ** (j + 2), axis=2)[..., None] - temp)


class Shkarofsky(eqx.Module):
    def __init__(self, cfg):
        self.v = cfg["grid"]["v"]
        self.nv = cfg["grid"]["nv"]
        self.nuee_dt = cfg["grid"]["dt"] * cfg["units"]["nuee_norm"]
        self.td_solve = TridiagonalSolver(cfg)
        self.calc_i2 = functools.partial(calc_i, v=self.v, j=2)
        self.calc_i0 = functools.partial(calc_i, v=self.v, j=0)
        self.calc_jm1 = functools.partial(calc_j, v=self.v, j=-1)

    def __call__(self, t, y, args):
        f00 = y["flm"][0][0]
        i20 = self.calc_i2(f00)
        jm10 = self.calc_jm1(f00)
        i00 = self.calc_i0(f00)

        v = self.v
        subdiag = (1 / 3 / v) * (i20 + jm10) - (1 / 3 / v**2) * (2 * jm10 - i20 + 3 * i00)
        diag = 4 * np.pi * f00 - 2 / 3 / v * (i20 + jm10)
        supdiag = (1 / 3 / v) * (i20 + jm10) + (1 / 3 / v**2) * (2 * jm10 - i20 + 3 * i00)
        subdiag *= -self.nuee_dt
        diag *= -self.nuee_dt
        supdiag *= -self.nuee_dt
        diag = 1 - diag

        subdiag = subdiag.reshape(-1, self.nv)
        diag = diag.reshape(-1, self.nv)
        supdiag = supdiag.reshape(-1, self.nv)

        return self.td_solve(subdiag, diag, supdiag, f00)


class ChangCooper(eqx.Module):
    v: jax.Array
    nx: int
    ny: int
    dv: float
    nuee_dt: float
    td_solve: eqx.Module
    zeros: jax.Array

    def __init__(self, cfg):
        self.nx = cfg["grid"]["nx"]
        self.ny = cfg["grid"]["ny"]
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.nuee_dt = cfg["grid"]["dt"] * cfg["units"]["derived"]["nuee_norm"].magnitude
        self.td_solve = TridiagonalSolver(cfg)
        self.zeros = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"], 1))

    def calc_c(self, f00):
        return 4.0 * jnp.pi * self.dv * jnp.cumsum(f00 * self.v[None, None, :] ** 2.0, axis=2)

    def calc_d(self, f00):
        inner_integral = (
            jnp.cumsum((f00 * (self.v + self.dv / 2.0)[None, None, :])[..., ::-1], axis=-1)[..., ::-1] * self.dv
        )
        return (
            4.0
            * jnp.pi
            * self.dv
            / self.v[None, None, :]
            * jnp.cumsum(self.v[None, None, :] ** 2.0 * inner_integral, axis=-1)
        )

    def __call__(self, t, y, args):
        f00 = y
        dv = self.dv
        nuee_dt = self.nuee_dt

        ck = self.calc_c(f00)
        dk = self.calc_d(f00)
        w = dv * ck / dk
        dlt = 1.0 / w - 1.0 / (jnp.exp(w) - 1.0)
        supdiag = ck[..., :-1] * (1 - dlt[..., :-1]) + dlt[..., :-1] / dv
        subdiag = -ck[..., :-1] * dlt[..., :-1] + dlt[..., :-1] / dv

        diag = (
            -ck[..., :-2] * (1 - dlt[..., :-2]) + ck[..., 1:-1] * dlt[..., 1:-1] - (dlt[..., 1:-1] + dlt[..., :-2]) / dv
        )
        diag = jnp.concatenate(
            [
                (ck[..., 0] * dlt[..., 0] - dk[..., 0] / dv)[..., None],
                diag,
                (-ck[..., -2] * (1 - dlt[..., -2]) - dk[..., -2] / dv)[..., None],
            ],
            axis=-1,
        )

        supdiag /= self.v[None, None, 1:] ** 2.0 * dv
        subdiag /= self.v[None, None, :-1] ** 2.0 * dv
        diag /= self.v[None, None, :] ** 2.0 * dv

        supdiag *= -nuee_dt
        subdiag *= -nuee_dt
        diag = 1 + nuee_dt * diag

        subdiag = jnp.concatenate([self.zeros, subdiag], axis=-1)
        supdiag = jnp.concatenate([supdiag, self.zeros], axis=-1)

        subdiag = subdiag.reshape((-1, self.v.size))
        diag = diag.reshape((-1, self.v.size))
        supdiag = supdiag.reshape((-1, self.v.size))
        f00 = f00.reshape((-1, self.v.size))

        return self.td_solve(subdiag, diag, supdiag, f00).reshape((self.nx, self.ny, self.v.size))


class AnisotropicCollisions(eqx.Module):
    td: bool
    num_batches: int
    v: jax.Array
    nl: int
    dv: float
    dv_sq: float
    lms: Dict
    td_solve: eqx.Module
    Y_dt: float
    calc_i2: Callable
    calc_i0: Callable
    calc_jm1: Callable

    def __init__(self, cfg):
        self.td = cfg["terms"]["fokker-planck"]["flm"] == "tridiagonal"
        self.num_batches = cfg["grid"]["nx"] * cfg["grid"]["ny"]
        self.v = cfg["grid"]["v"]
        self.nl = cfg["grid"]["nl"]
        self.dv = self.v[1] - self.v[0]
        self.dv_sq = self.dv**2.0
        self.lms = defaultdict(dict)
        self.td_solve = TridiagonalSolver(cfg)
        self.Y_dt = (
            4
            * np.pi
            * (cfg["units"]["Z"] * cfg["units"]["Zp"]) ** 2.0
            * cfg["units"]["derived"]["logLambda_ee"]
            * cfg["units"]["derived"]["nuee_norm"].magnitude
            * cfg["grid"]["dt"]
        )
        for il in range(0, cfg["grid"]["nl"] + 1):
            for im in range(0, il + 1):
                self.lms[il][im] = (il, im)

        self.calc_i2 = functools.partial(calc_i, v=self.v, j=2)
        self.calc_i0 = functools.partial(calc_i, v=self.v, j=0)
        self.calc_jm1 = functools.partial(calc_j, v=self.v, j=-1)

    def tridiagonal_flm(self, flm, il, im):
        i_2 = self.calc_i2(flm=flm)
        j_minus_1 = self.calc_jm1(flm=flm)
        i_0 = self.calc_i0(flm=flm)

        coeff1 = (-i_2 + 2 * j_minus_1 + 3 * i_0) / 3 / self.v[None, None, :] ** 2.0
        coeff2 = (i_2 + j_minus_1) / 3 / self.v[None, None, :]

        subdiagonal = -coeff1 / 2 / self.dv + coeff2 / self.dv_sq
        diagonal = 8 * jnp.pi * flm + coeff2 * 0.5 / self.dv_sq - il * (il + 1) / 2 * coeff1 / self.v[None, None, :]
        superdiagonal = coeff1 / 2 / self.dv + coeff2 / self.dv_sq

        subdiagonal = subdiagonal.reshape((self.num_batches, -1)) * self.Y_dt
        diagonal = diagonal.reshape((self.num_batches, -1)) * self.Y_dt + 1
        superdiagonal = superdiagonal.reshape((self.num_batches, -1)) * self.Y_dt

        flm = flm.reshape((self.num_batches, -1))

        return self.td_solve(subdiagonal, diagonal, superdiagonal, flm)

    def __call__(self, t, y, args):
        for il in range(1, self.nl + 1):
            for im in range(0, il + 1):
                y[il][im] = self.tridiagonal_flm(y[il][im], il, im)

        # if self.td:
        #     new_f = jax.tree_util.tree_map_with_path(self.tridiagonal_flm, y, (self.lms,))
        # else:
        #     new_f = jax.tree_util.tree_map_with_path(self.full_flm, y, (self.lms,))

        return y
