import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np
from adept.sh2d.tridiagonal import TridiagonalSolver


class IsotropicCollisions(eqx.Module):
    def __init__(self, cfg):
        self.v = cfg["grid"]["v"]
        if self.cfg["coll"]["isotropic"] == "lenard_bernstein":
            self.coll_opp = LenardBernstein()
        elif self.cfg["coll"]["isotropic"] == "chang_cooper":
            self.coll_opp = ChangCooper(cfg)
        else:
            raise NotImplementedError

    def __call__(self, t, y, args):
        new_f00 = self.coll_opp(t, y, args)
        return new_f00


class ChangCooper(eqx.Module):
    def __init__(self, cfg):
        self.v = cfg["grid"]["v"]
        self.dv = cfg["grid"]["dv"]
        self.nuee_dt = cfg["grid"]["dt"] * cfg["units"]["nuee_norm"]
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
        f00 = y[0][0]
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
                [ck[..., 0] * dlt[..., 0] - dk[..., 0] / dv],
                diag,
                [-ck[..., -2] * (1 - dlt[..., -2]) - dk[..., -2] / dv],
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

        return self.td_solve(subdiag, diag, supdiag, f00)


class AnisotropicCollisions(eqx.Module):
    def __init__(self, cfg):
        self.td = cfg["coll"]["tridiagonal_only"]
        self.num_batches = cfg["grid"]["nx"] * cfg["grid"]["ny"]
        self.v = cfg["grid"]["v"]
        self.dv = self.v[1] - self.v[0]
        self.dv_sq = self.dv**2.0
        self.lms = {}
        self.td_solve = TridiagonalSolver(cfg)
        self.Y_dt = 4 * np.pi * (cfg["units"]["Z"] * ["units"]["Zp"]) ** 2.0 * cfg["units"]["logLambda_ee"]
        for il in range(0, cfg["grid"]["nl"] + 1):
            for im in range(0, il + 1):
                self.lms[il][im] = (il, im)

    def calc_ij(self, flm, ij):
        ilm_j = 4 * jnp.pi / self.v[None, None, :] * jnp.cumsum(flm * self.v[None, None, :] ** (ij + 2), axis=2)
        jlm_j = ilm_j[None, None, ::-1]

        return ilm_j, jlm_j

    def tridiagonal_flm(self, flm, il, im):
        i_2, _ = self.calc_ij(flm, ij=2)
        _, j_minus_1 = self.calc_ij(flm, ij=-1)
        i_0, _ = self.calc_ij(flm, ij=0)

        coeff1 = (-i_2 + 2 * j_minus_1 + 3 * i_0) / 3 / self.v[None, None, :] ** 2.0
        coeff2 = (i_2 + j_minus_1) / 3 / self.v[None, None, :]

        subdiagonal = -coeff1 / 2 / self.dv + coeff2 / self.dv_sq
        diagonal = 8 * jnp.pi * flm + coeff2 * 0.5 / self.dv_sq - il * (il + 1) / 2 * coeff1 / self.v[None, None, :]
        superdiagonal = coeff1 / 2 / self.dv + coeff2 / self.dv_sq

        subdiagonal = subdiagonal.reshape((self.num_batches, -1)) * self.Y_dt
        diagonal = diagonal.reshape((self.num_batches, -1)) * self.Y_dt + 1
        superdiagonal = superdiagonal.reshape((self.num_batches, -1)) * self.Y_dt

        return self.tri_solve(subdiagonal, diagonal, superdiagonal, flm)

    def __call__(self, prev_f):
        if self.td:
            new_f = jax.tree_util.tree_map_with_path(self.tridiagonal_flm, prev_f, (self.lms,))
        else:
            new_f = jax.tree_util.tree_map_with_path(self.full_flm, prev_f, (self.lms,))

        return new_f
