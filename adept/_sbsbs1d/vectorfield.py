from typing import Dict, Union
from astropy import constants as const

from jax import numpy as jnp, Array, random
from tensorflow_probability.substrates import jax as tfpj


class BaseLPIVectorField:
    def __init__(self, cfg):
        self.cfg = cfg
        self.PRNGKey = random.PRNGKey(0)
        self.dz = cfg["grid"]["dz"]
        self.zax = cfg["grid"]["z"]
        self.nc = cfg["units"]["derived"]["nc_over_n0"]
        self.ratio_M_m = cfg["units"]["derived"]["ratio_M_m"]
        self.nuei0 = cfg["units"]["derived"]["nuei_epphaines0"].to("Hz").value
        self.logLambda_ei0 = cfg["units"]["derived"]["logLambda_ei0"]
        self.nc_over_n0 = cfg["units"]["derived"]["nc_over_n0"]
        self.vg_const = ((const.c**2.0) / cfg["units"]["derived"]["w0"]).to("um^2/s").value
        self.a0 = cfg["units"]["derived"]["a0"]
        self.kz0 = cfg["units"]["derived"]["kz0"].to("1/um").value
        self.w0 = cfg["units"]["derived"]["w0"].to("Hz").value
        self.n0 = cfg["units"]["derived"]["n0"].to("cm-3").value
        self.T0 = cfg["units"]["derived"]["T0"].to("eV").value
        self.Zeff = cfg["profiles"]["Zeff"]
        self.vth0 = cfg["units"]["derived"]["vth0"].to("um/s").value
        self.cs0 = cfg["units"]["derived"]["cs0"].to("um/s").value

    def zprime(self, x: Union[complex, float]) -> complex:
        Zx = 1j * jnp.sqrt(jnp.pi) * jnp.exp(-(x**2.0)) - 2.0 * tfpj.math.dawsn(x)
        return -2 * (1 + x * Zx)


class ExponentialLeapfrog(BaseLPIVectorField):

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)

    def calc_logLambda(self, n_over_n0: float, Te_over_T0: float, Zeff: int) -> float:
        log_ne = jnp.log(n_over_n0 * self.n0)
        log_Te = jnp.log(Te_over_T0 * self.T0)
        log_Z = jnp.log(Zeff)

        # logLambda_ee = max(2.0, 23.5 - 0.5 * log_ne + 1.25 * log_Te - np.sqrt(1e-5 + 0.0625 * (log_Te - 2.0) ** 2.0))

        # if Te.to("eV").value > 10 * Z**2.0:
        # logLambda_ei = max(2.0, 24.0 - 0.5 * log_ne + log_Te)
        logLambda_ei = 24.0 - 0.5 * log_ne + log_Te
        return logLambda_ei
        # else:
        #     logLambda_ei = max(2.0, 23.0 - 0.5 * log_ne + 1.5 * log_Te - log_Z)

    def calculate_noise(self, args: Dict[str, float]) -> float:
        return 0.0  # * random.uniform(self.PRNGKey, (1,))

    def calc_kappa(self, z: float, profiles) -> float:
        n_over_n0 = profiles["n_over_n0"]
        Te_over_T0 = profiles["Te_over_T0"]
        Zeff = profiles["Zeff"]

        # nu_ei = density * nu_ei_bar  # Local electron-ion collision frequency/omega
        logLambda_ei = self.calc_logLambda(n_over_n0, Te_over_T0, Zeff)
        factor = 1 / (Te_over_T0**1.5 / (n_over_n0 * Zeff * (logLambda_ei / self.logLambda_ei0)))
        nuei_epphaines = factor * self.nuei0

        nu_IB = n_over_n0 / self.nc_over_n0 * nuei_epphaines  # Inverse Bremstrahlung absorption rate/omega
        # Electron-ion collision frequency at critical density/omega
        kz = self.kz0 * jnp.sqrt(1 - n_over_n0 / self.nc_over_n0)
        kappa_ib = nu_IB / (self.vg_const * kz)

        return kappa_ib

    def calc_imfx0_sbs(self, plasma_conditions_at_z: Dict[str, float]) -> tuple[float, float, float, float, float]:

        n_over_n0 = plasma_conditions_at_z["n_over_n0"]
        Te_over_T0 = plasma_conditions_at_z["Te_over_T0"]
        Ti_over_T0 = plasma_conditions_at_z["Ti_over_T0"]
        Zeff = plasma_conditions_at_z["Zeff"]
        flow_over_flow0 = plasma_conditions_at_z["flow_over_flow0"]
        omegabeat = plasma_conditions_at_z["omegabeat"]

        kz = self.kz0 * jnp.sqrt(1 - n_over_n0 / self.nc_over_n0)

        omega_p2_elec = n_over_n0 * self.w0**2.0  # omega_pe^2
        omega_p2_ion = Zeff / self.ratio_M_m * omega_p2_elec  # omega_pi^2
        omega_beat_plasframe = omegabeat + 2 * kz * flow_over_flow0  # Beat frequency in plasma frame/omega

        v_th_e = self.vth0 * jnp.sqrt(Te_over_T0)
        v_th_i = self.vth0 * jnp.sqrt(Ti_over_T0)

        # Electron susceptibility
        chi_e = -omega_p2_elec / 8 / kz**2 / v_th_e**2 * self.zprime(omega_beat_plasframe / 2**1.5 / kz / v_th_e)

        # Ion susceptibility
        chi_i = -omega_p2_ion / 8 / kz**2 / v_th_i**2 * self.zprime(omega_beat_plasframe / 2**1.5 / kz / v_th_i)

        F_chi = chi_e * (1.0 + chi_i) / (1.0 + chi_e + chi_i)
        cs = jnp.sqrt(Zeff * Te_over_T0) * self.cs0

        return jnp.imag(F_chi), omega_beat_plasframe, cs, kz, jnp.abs(omega_beat_plasframe) - 2 * kz * cs

    def __call__(self, t, y, args):
        Ji = y["Ji"]
        Jr = y["Jr"]
        z = t

        plasma_conditions_at_z = {k: func(z) for k, func in args.items()}

        noise = self.calculate_noise(args)
        kappaIB = self.calc_kappa(z, plasma_conditions_at_z)
        imfx0, omega_beat_plasframe, cs, kz, res_cond = self.calc_imfx0_sbs(plasma_conditions_at_z)

        dlogJidz = -kappaIB - self.a0**2.0 * self.kz0 * imfx0 * Jr
        dlogJrdz = kappaIB - self.a0**2.0 * self.kz0 * imfx0 * Ji

        new_Ji = jnp.exp(self.dz * dlogJidz) * Ji
        new_Jr = jnp.exp(self.dz * dlogJrdz) * Jr - noise

        return {
            "Ji": new_Ji,
            "Jr": new_Jr,
            "imfx0": imfx0,
            "kappaIB": kappaIB,
            "omega_beat_plasframe": omega_beat_plasframe,
            "cs": cs,
            "kz": kz,
            "res_cond": res_cond,
        } | plasma_conditions_at_z


class CBET(BaseLPIVectorField):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_beams = None
        self.delta_psi_ij = None
        self.delta_k_ij = (
            ki[:, None] ** 2.0 + ki[None, :] ** 2.0 - 2.0 * ki[:, None] * ki[None, :] * jnp.cos(self.delta_psi_ij)
        )
        self.delta_omega_ij = None
        self.th_i = None

    def calc_gamma(self, args):

        plasma_conditions_at_LEH = {k: func(0.0) for k, func in args.items()}
        imfx0, omega_beat_plasframe, cs, kz, res_cond = self.calc_imfx0(plasma_conditions_at_LEH)
        gammaij = -imfx0 * self.delta_k_ij**2.0 / 4.0 * (1 + jnp.cos(self.delta_psi_ij) ** 2.0) / 2

    def __call__(self, t, y, args):
        """

        t: independent coordinate of integration - physically, it is z in this case
        y: dependent variables of the system. For this problem, the dependent variables are the interacting beams
        args: parameters of the system. For this problem, the parameters are the profiles of the plasma

        """
        z = t

        gamma_ij = self.calc_gamma(args)

        dIidz = 1 / jnp.cos(th_i) * (gamma_ij @ y["I"]) * y["I"]

        pass
