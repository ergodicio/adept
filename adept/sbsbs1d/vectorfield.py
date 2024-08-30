from typing import Dict

from jax import numpy as jnp, Array, random
from jax.scipy.special import erfc
import numpy as np

from adept import get_envelope


class ExponentialLeapfrog:

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.PRNGKey = random.PRNGKey(0)
        self.dz = cfg["grid"]["dz"]
        self.zax = cfg["grid"]["z"]
        self.nc = cfg["units"]["derived"]["nc"]
        self.ratio_M_m = cfg["units"]["derived"]["ratio_M_m"]

        # self.logLambda = 5.0

    def zprime(self, z: float) -> float:
        return jnp.exp(-(z**2.0)) * erfc(-1j * z)

    def calculate_noise(self, args: Dict[str, float]) -> float:
        return random.uniform(self.PRNGKey, (1,))

    def calc_kappa(self, z, args: Dict[str, float]) -> float:
        n_over_nc, Te, Zeff = args["n_over_nc"](z), args["Te_keV"](z), args["Zeff"](z)
        # vth = jnp.sqrt(TkeV)
        # ND = n_over_nc * (vth / omegap) ** 3.0
        # nuei_over_w0 = jnp.sqrt(n_over_nc) * self.nuei_const * self.logLambda / ND

        plas_Lambda = self.nclam3 * Te**1.5 / (2 * jnp.pi) ** 3
        nu_ei_bar = 1 / 12 / jnp.pi * jnp.sqrt(2 / jnp.pi) * Zeff * jnp.log(plas_Lambda) / plas_Lambda

        nu_ei = density * nu_ei_bar  # Local electron-ion collision frequency/omega
        nu_IB = density**2 * nu_ei_bar  # Inverse Bremstrahlung absorption rate/omega
        # Electron-ion collision frequency at critical density/omega

        kappa_ib = kz / omega * nu_IB
        return kappa_ib

    def calc_imfx0(self, z, args: Dict[str, float]) -> float:
        n_over_nc, Te, Ti, Zeff, flow, omega_beat = (
            args["n_over_nc"](z),
            args["Te_keV"](z),
            args["Ti_keV"](z),
            args["Zeff"](z),
            args["flow"](z),
            args["omega_beat"](z),
        )

        omega_p2_elec = n_over_nc  # omega_pe^2/omega^2
        omega_p2_ion = Zeff / self.ratio_M_m * omega_p2_elec  # omega_pi^2/omega^2
        omega_beat_plasframe = omega_beat + 2 * kz * flow  # Beat frequency in plasma frame/omega

        v_th_e = jnp.sqrt(Te)  ## This is a placeholder
        v_th_i = jnp.sqrt(Ti)  ## This is a placeholder

        kz = self.kz_vac * jnp.sqrt(1 - n_over_nc)

        # Electron susceptibility
        chi_e = -omega_p2_elec / 8 / kz**2 / v_th_e**2 * self.zprime(omega_beat_plasframe / 2**1.5 / kz / v_th_e)

        # Ion susceptibility
        chi_i = -omega_p2_ion / 8 / self.kz**2 / v_th_i**2 * self.zprime(omega_beat_plasframe / 2**1.5 / kz / v_th_i)

        F_chi = chi_e * (1.0 + chi_i) / (1.0 + chi_e + chi_i)
        return jnp.imag(F_chi)

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
