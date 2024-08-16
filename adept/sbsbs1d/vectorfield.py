from typing import Dict

from jax import numpy as jnp, Array
import numpy as np

from adept import get_envelope


class ExponentialLeapfrog:

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.dz = cfg["grid"]["dz"]
        self.zax = cfg["grid"]["z"]
        self.nc = cfg["units"]["derived"]["nc"]

    def calculate_noise(self, args: Dict[str, float]) -> float:
        return np.random.normal(0, np.sqrt(2 * args["nu_coll"] * self.dz))

    def calc_kappa(self, z, args: Dict[str, float]) -> float:
        n_over_nc, TkeV = args["n_over_nc"](z), args["TkeV"](z)
        vth = jnp.sqrt(TkeV)
        ND = n_over_nc * (vth / omegap) ** 3.0
        nuei_over_w0 = jnp.sqrt(n_over_nc) * self.nuei_const * self.logLambda(n, T) / ND

        return kappa_ib

    def calc_imfx0(self, z, args: Dict[str, float]) -> float:
        n, T = args["n"](z), args["T"](z)
        return imfx0

    def __call__(self, t, y, args):
        Ji = y["Ji"]
        Jr = y["Jr"]
        z = t

        noise = self.calculate_noise(args)

        dJidz = self.calculate_dJidz(z, Ji, Jr, args)
        dJrdz = self.calculate_dJrdz(z, Ji, Jr, args)

        new_Ji = jnp.exp(self.dz * dJidz) * Ji
        new_Jr = jnp.exp(self.dz * dJrdz) * Jr - noise

        return {"Ji": new_Ji, "Jr": new_Jr}

    def calculate_dJidz(self, z: float, Ji: float, Jr: float, args: Dict[str, float]) -> float:
        a0 = args["a0"]
        kappaIB = self.calc_kappa(args)
        imfx0 = self.calc_imfx0(args)

        return kappaIB - a0**2.0 * self.kz0 * imfx0 * Jr

    def calculate_dJrdz(self, z: float, Ji: float, Jr: float, args: Dict[str, float]) -> float:
        a0 = args["a0"]
        kappaIB = self.calc_kappa(args)
        imfx0 = self.calc_imfx0(args)

        return -kappaIB - a0**2.0 * self.kz0 * imfx0 * Ji
