from typing import Dict, Union
from astropy import constants as const

from jax import numpy as jnp, Array, random
from tensorflow_probability.substrates import jax as tfpj


class BaseLPIVectorField:
    def __init__(self, grid, units):
        # self.cfg = cfg
        self.PRNGKey = random.PRNGKey(0)
        self.dz = grid["dz"]
        self.zax = grid["z"]
        self.nc = units["derived"]["nc_over_n0"]
        self.ratio_M_m = units["derived"]["ratio_M_m"]
        self.nuei0 = units["derived"]["nuei_epphaines0"].to("Hz").value
        self.logLambda_ei0 = units["derived"]["logLambda_ei0"]
        self.nc_over_n0 = units["derived"]["nc_over_n0"]
        self.vg_const = ((const.c**2.0) / units["derived"]["w0"]).to("um^2/s").value
        self.a0 = units["derived"]["a0"]
        self.kz0 = units["derived"]["kz0"].to("1/um").value
        self.w0 = units["derived"]["w0"].to("Hz").value
        self.n0 = units["derived"]["n0"].to("cm-3").value
        self.T0 = units["derived"]["T0"].to("eV").value
        self.vth0 = units["derived"]["vth0"].to("um/s").value
        self.cs0 = units["derived"]["cs0"].to("um/s").value

    def zprime(self, x: Union[complex, float]) -> complex:
        Zx = 1j * jnp.sqrt(jnp.pi) * jnp.exp(-(x**2.0)) - 2.0 * tfpj.math.dawsn(x)
        return -2 * (1 + x * Zx)


class SBSVectorField(BaseLPIVectorField):
    def __init__(self, grid, units) -> None:
        super().__init__(grid, units)

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

        dlogJidz = -kappaIB - self.a0**2 * self.kz0 * imfx0 * Jr
        dlogJrdz = kappaIB - self.a0**2 * self.kz0 * imfx0 * Ji

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


class CBETVectorField(BaseLPIVectorField):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dz = 1 / self.cfg["cbet"]["grid"]["n_steps"]
        self.a0_sq = self.cfg["laser"]["a0_sq"]
        self.beam_params = self.cfg["laser"]

        kxs = -jnp.sin(self.cfg["laser"]["thetas"]) * jnp.cos(self.cfg["laser"]["phis"])  # k*c/omega
        kys = -jnp.sin(self.cfg["laser"]["thetas"]) * jnp.sin(self.cfg["laser"]["phis"])
        kzs = -jnp.cos(self.cfg["laser"]["thetas"])
        self.kb_x = kxs.reshape(-1, 1) - kxs.reshape(1, -1)  # k_b*c/omega
        self.kb_y = kys.reshape(-1, 1) - kys.reshape(1, -1)
        self.kb_z = kzs.reshape(-1, 1) - kzs.reshape(1, -1)
        self.kb_sq = self.kb_x**2 + self.kb_y**2 + self.kb_z**2
        self.one_over_kbsq = jnp.where(self.kb_sq == 0, 0.0, 1.0 / self.kb_sq)

        self.mass_ratio = (const.m_p / const.m_e).to("").value  # Mass ratio of proton to electron
        self.m_ec2 = (
            (const.m_e.to("kg") * const.c.to("m/s") ** 2).to("keV").value
        )  # Electron rest mass energy in Joules

    def get_coeffs(self, plasma_params):
        """
        Calculate the C-BET effect for given plasma and beam parameters.

        Parameters:
        plasma_params (dict): Plasma parameters including electron density, temperature, etc.
        constants (dict): Physical constants.

        Returns:
        None: Plots the results of the C-BET effect.
        """

        # Derived quantities
        m_pc2 = self.mass_ratio * self.m_ec2  # proton mass
        m_ic2 = plasma_params["A_ion"] * m_pc2  # ion mass

        v_th_e_sq = plasma_params["T_e"] / self.m_ec2  # T_e/(m_ec^2)=v_th_e^2/c^2
        v_th_i_sq = plasma_params["T_i"] / m_ic2  # T_i/(m_ic^2)=v_th_i^2/c^2

        omega_beat = -(
            self.beam_params["dls"].reshape(-1, 1) - self.beam_params["dls"].reshape(1, -1)
        )  # omega_beat/omega

        chi_e_coeff = -plasma_params["ne_over_nc"] / (2 * v_th_e_sq) * self.one_over_kbsq

        chi_i_coeff = (
            -plasma_params["ne_over_nc"]
            * plasma_params["Z_eff"]
            / (2 * self.mass_ratio * v_th_i_sq)
            * self.one_over_kbsq
        )

        kb_dot_u = plasma_params["flow_magnitude"] * (
            self.kb_x * jnp.sin(plasma_params["flow_theta"]) * jnp.cos(plasma_params["flow_phi"])
            + self.kb_y * jnp.sin(plasma_params["flow_theta"]) * jnp.sin(plasma_params["flow_phi"])
            + self.kb_z * jnp.cos(plasma_params["flow_theta"])
        )

        chi_e = chi_e_coeff * self.zprime(
            (omega_beat - kb_dot_u) / jnp.sqrt(2 * v_th_e_sq) * jnp.sqrt(self.one_over_kbsq)
        )
        chi_i = chi_i_coeff * self.zprime(
            (omega_beat - kb_dot_u) / jnp.sqrt(2 * v_th_i_sq) * jnp.sqrt(self.one_over_kbsq)
        )
        F_chi = chi_e * (1 + chi_i) / (1 + chi_e + chi_i)

        coeffs = (
            -jnp.imag(F_chi)
            * self.kb_sq
            * self.a0_sq
            * plasma_params["L_p"]
            / (4 * jnp.abs(jnp.cos(self.beam_params["thetas"]).reshape(-1, 1)))
        )

        return coeffs

    def __call__(self, t, y, args):
        """
        Update step for the ODE system.

        :param t: Time variable (not used in this case).
        :param y: Current state of the system (intensities).
        :param args: Additional arguments (coefficients).
        :return: Derivative of the state.
        """
        plasma_params = args["plasma"]
        coeffs = self.get_coeffs(plasma_params)

        return y * jnp.exp(coeffs @ y * self.dz)


# class SBSBS_CBET_VectorField:
#     def __init__(self, cfg):
#         self.sbs_solvers = [
#             SBSVectorField(cfg["sbs"]["grid"][i], cfg["units"]) for i in range(cfg["laser"]["num_beams"])
#         ]
#         self.cbet = CBETVectorField(cfg)

#     def __call__(self, t, y, args):
#         """
#         Update step for the ODE system.

#         :param t: Time variable (not used in this case).
#         :param y: Current state of the system (intensities).
#         :param args: Additional arguments (coefficients).
#         :return: Derivative of the state.
#         """
#         cbet_result = self.cbet(t, y, args)
#         reflected_light = []
#         for this_beam, sbs_solver in zip(cbet_result, self.sbs_solvers):
#             y["Ji"] = this_beam
#             sbs_result = sbs_solver(t, y, args)
#             reflected_light.append(sbs_result["Jr"])

#         return reflected_light
