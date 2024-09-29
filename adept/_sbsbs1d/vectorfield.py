from typing import Dict, Union
from astropy import constants as const

from jax import numpy as jnp, Array, random
from tensorflow_probability.substrates import jax as tfpj


class ExponentialLeapfrog:

    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.PRNGKey = random.PRNGKey(0)
        self.dz = cfg["grid"]["dz"]
        self.zax = cfg["grid"]["z"]
        self.nc = cfg["units"]["derived"]["nc_over_n0"]
        self.ratio_M_m = cfg["units"]["derived"]["ratio_M_m"]
        self.nuei0 = cfg["units"]["derived"]["nuei_epphaines0"].to("Hz").value
        self.logLambda_ei0 = cfg["units"]["derived"]["logLambda_ei0"]
        self.nc_over_n0 = cfg["units"]["derived"]["nc_over_n0"]
        self.vg_const = ((const.c**2.0) / cfg["units"]["derived"]["w0"]).to("m^2/s").value
        self.a0 = cfg["units"]["derived"]["a0"]
        self.kz0 = cfg["units"]["derived"]["kz0"].to("1/m").value
        self.n0 = cfg["units"]["derived"]["n0"].to("cm-3").value
        self.T0 = cfg["units"]["derived"]["T0"].to("eV").value
        self.Z0 = cfg["profiles"]["Zeff"]

    def calc_logLambda(self, n_over_n0: float, Te_over_T0: float, Z_over_Z0: int) -> float:
        log_ne = jnp.log(n_over_n0 * self.n0)
        log_Te = jnp.log(Te_over_T0 * self.T0)
        log_Z = jnp.log(Z_over_Z0 * self.Z0)

        # logLambda_ee = max(2.0, 23.5 - 0.5 * log_ne + 1.25 * log_Te - np.sqrt(1e-5 + 0.0625 * (log_Te - 2.0) ** 2.0))

        # if Te.to("eV").value > 10 * Z**2.0:
        # logLambda_ei = max(2.0, 24.0 - 0.5 * log_ne + log_Te)
        logLambda_ei = 24.0 - 0.5 * log_ne + log_Te
        return logLambda_ei
        # else:
        #     logLambda_ei = max(2.0, 23.0 - 0.5 * log_ne + 1.5 * log_Te - log_Z)

    def zprime(self, x: Union[complex, float]) -> complex:
        Zx = 1j * jnp.sqrt(jnp.pi) * jnp.exp(-(x**2.0)) - 2.0 * tfpj.math.dawsn(x)
        return -2 * (1 + x * Zx)

    def calculate_noise(self, args: Dict[str, float]) -> float:
        return 0.0  # * random.uniform(self.PRNGKey, (1,))

    def calc_kappa(self, z: float, args: Dict[str, float]) -> float:
        n_over_n0, Te_over_T0, Ti_over_T0, Zeff_over_Z0, flow_over_flow0, omegabeat_over_omegabeat0 = (
            args["n_over_n0"](z),
            args["Te_over_T0"](z),
            args["Ti_over_T0"](z),
            args["Zeff_over_Z0"](z),
            args["flow_over_flow0"](z),
            args["omegabeat_over_omegabeat0"](z),
        )

        # nu_ei = density * nu_ei_bar  # Local electron-ion collision frequency/omega
        logLambda_ei = self.calc_logLambda(n_over_n0, Te_over_T0, Zeff_over_Z0)
        factor = 1 / (Te_over_T0**1.5 / (n_over_n0 * Zeff_over_Z0 * (logLambda_ei / self.logLambda_ei0)))
        nuei_epphaines = factor * self.nuei0

        nu_IB = n_over_n0 / self.nc_over_n0 * nuei_epphaines  # Inverse Bremstrahlung absorption rate/omega
        # Electron-ion collision frequency at critical density/omega
        kz = self.kz0 * jnp.sqrt(1 - n_over_n0)
        kappa_ib = nu_IB / (self.vg_const * kz)

        return 1e-5 * kappa_ib

    def calc_imfx0(self, z: float, args: Dict[str, float]) -> float:
        n_over_n0, Te_over_T0, Ti_over_T0, Zeff_over_Z0, flow_over_flow0, omegabeat_over_omegabeat0 = (
            args["n_over_n0"](z),
            args["Te_over_T0"](z),
            args["Ti_over_T0"](z),
            args["Zeff_over_Z0"](z),
            args["flow_over_flow0"](z),
            args["omegabeat_over_omegabeat0"](z),
        )

        kz = self.kz0 * jnp.sqrt(1 - n_over_n0)

        omega_p2_elec = n_over_n0  # omega_pe^2/omega^2
        omega_p2_ion = Zeff_over_Z0 / self.ratio_M_m * omega_p2_elec  # omega_pi^2/omega^2
        omega_beat_plasframe = (
            omegabeat_over_omegabeat0 + 2 * kz * flow_over_flow0
        )  # Beat frequency in plasma frame/omega

        v_th_e = jnp.sqrt(Te_over_T0)  ## This is a placeholder
        v_th_i = jnp.sqrt(Ti_over_T0)  ## This is a placeholder

        # Electron susceptibility
        chi_e = -omega_p2_elec / 8 / kz**2 / v_th_e**2 * self.zprime(omega_beat_plasframe / 2**1.5 / kz / v_th_e)

        # Ion susceptibility
        chi_i = -omega_p2_ion / 8 / kz**2 / v_th_i**2 * self.zprime(omega_beat_plasframe / 2**1.5 / kz / v_th_i)

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
        kappaIB = -self.calc_kappa(z, args)
        imfx0 = self.calc_imfx0(z, args)

        return kappaIB - self.a0**2.0 * self.kz0 * imfx0 * Jr

    def calculate_dJrdz(self, z: float, Ji: float, Jr: float, args: Dict[str, float]) -> float:
        kappaIB = -self.calc_kappa(z, args)
        imfx0 = self.calc_imfx0(z, args)

        return -kappaIB - self.a0**2.0 * self.kz0 * imfx0 * Ji
