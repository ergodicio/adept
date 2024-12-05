from typing import Dict, Tuple

import numpy as np
from astropy import constants as csts, units as u
from astropy.units import Quantity as _Q
from diffrax import diffeqsolve, SaveAt, ODETerm, SubSaveAt
from jax import numpy as jnp, tree_util as jtu

from adept._base_ import ADEPTModule, Stepper
from adept.vfp1d.vector_field import OSHUN1D
from adept.vfp1d.helpers import _initialize_total_distribution_, calc_logLambda
from adept.vfp1d.storage import get_save_quantities, post_process


class BaseVFP1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def post_process(self, solver_result: Dict, td: str) -> Dict:
        return post_process(solver_result["solver result"], cfg=self.cfg, td=td, args=self.args)

    def write_units(self) -> Dict:
        ne = u.Quantity(self.cfg["units"]["reference electron density"]).to("1/cm^3")
        ni = ne / self.cfg["units"]["Z"]
        Te = u.Quantity(self.cfg["units"]["reference electron temperature"]).to("eV")
        Ti = u.Quantity(self.cfg["units"]["reference ion temperature"]).to("eV")
        Z = self.cfg["units"]["Z"]
        # Should we change this to reference electron density? or allow it to be user set?
        n0 = u.Quantity("9.0663e21/cm^3")
        ion_species = self.cfg["units"]["Ion"]

        wp0 = np.sqrt(n0 * csts.e.to("C") ** 2.0 / (csts.m_e * csts.eps0)).to("Hz")
        tp0 = (1 / wp0).to("fs")

        vth = np.sqrt(2 * Te / csts.m_e).to("m/s")  # mean square velocity eq 4-51a in Shkarofsky

        x0 = (csts.c / wp0).to("nm")

        beta = vth / csts.c

        logLambda_ei, logLambda_ee = calc_logLambda(self.cfg, ne, Te, Z, ion_species)
        logLambda_ee = logLambda_ei

        nD_NRL = 1.72e9 * Te.value**1.5 / np.sqrt(ne.value)
        nD_Shkarofsky = np.exp(logLambda_ei) * Z / 9

        nuei_shk = np.sqrt(2.0 / np.pi) * wp0 * logLambda_ei / np.exp(logLambda_ei)
        # JPB - Maybe comment which page/eq this is from? There are lots of collision times in NRL
        # For example, nu_ei on page 32 does include Z^2
        nuei_nrl = np.sqrt(2.0 / np.pi) * wp0 * logLambda_ei / nD_NRL

        lambda_mfp_shk = (vth / nuei_shk).to("micron")
        lambda_mfp_nrl = (vth / nuei_nrl).to("micron")

        nuei_epphaines = (
            1
            / (
                0.75
                * np.sqrt(csts.m_e)
                * Te**1.5
                / (np.sqrt(2 * np.pi) * ni * Z**2.0 * csts.e.gauss**4.0 * logLambda_ei)
            )
        ).to("Hz")

        all_quantities = {
            "wp0": wp0,
            "n0": n0,
            "tp0": tp0,
            "ne": ne,
            "vth": vth,
            "Te": Te,
            "Ti": Ti,
            "logLambda_ei": logLambda_ei,
            "logLambda_ee": logLambda_ee,
            "beta": beta,
            "x0": x0,
            "nuei_shk": nuei_shk,
            "nuei_nrl": nuei_nrl,
            "nuei_epphaines": nuei_epphaines,
            "nuei_shk_norm": nuei_shk / wp0,
            "nuei_nrl_norm": nuei_nrl / wp0,
            "nuei_epphaines_norm": nuei_epphaines / wp0,
            "lambda_mfp_shk": lambda_mfp_shk,
            "lambda_mfp_nrl": lambda_mfp_nrl,
            "lambda_mfp_epphaines": (vth / nuei_epphaines).to("micron"),
            "nD_NRL": nD_NRL,
            "nD_Shkarofsky": nD_Shkarofsky,
            # "box_length": box_length,
            # "box_width": box_width,
            # "sim_duration": sim_duration,
        }

        self.cfg["units"]["derived"] = all_quantities
        self.cfg["grid"]["beta"] = beta.value

        return {k: str(v) for k, v in all_quantities.items()}

    def get_derived_quantities(self):
        """
        This function just updates the config with the derived quantities that are only integers or strings.

        This is run prior to the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = self.cfg["grid"]
        cfg_grid["xmax"] = (_Q(cfg_grid["xmax"]) / _Q(self.cfg["units"]["derived"]["x0"])).to("").value
        cfg_grid["xmin"] = (_Q(cfg_grid["xmin"]) / _Q(self.cfg["units"]["derived"]["x0"])).to("").value
        cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]

        # sqrt(2 * k * T / m)
        cfg_grid["vmax"] = (
            8
            * np.sqrt((_Q(self.cfg["units"]["reference electron temperature"]) / (csts.m_e * csts.c**2.0)).to("")).value
        )

        cfg_grid["dv"] = cfg_grid["vmax"] / cfg_grid["nv"]

        cfg_grid["tmax"] = (_Q(cfg_grid["tmax"]) / self.cfg["units"]["derived"]["tp0"]).to("").value
        cfg_grid["dt"] = (_Q(cfg_grid["dt"]) / self.cfg["units"]["derived"]["tp0"]).to("").value

        cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"]) + 1

        if cfg_grid["nt"] > 1e6:
            cfg_grid["max_steps"] = int(1e6)
            print(r"Only running $10^6$ steps")
        else:
            cfg_grid["max_steps"] = cfg_grid["nt"] + 4

        cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

        print("tmax", cfg_grid["tmax"], "dt", cfg_grid["dt"])
        print("xmax", cfg_grid["xmax"], "dx", cfg_grid["dx"])

        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self):
        """
        This function just updates the config with the derived quantities that are arrays

        This is run after the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = self.cfg["grid"]

        cfg_grid = {
            **cfg_grid,
            **{
                "x": jnp.linspace(
                    cfg_grid["xmin"] + cfg_grid["dx"] / 2, cfg_grid["xmax"] - cfg_grid["dx"] / 2, cfg_grid["nx"]
                ),
                "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
                "v": jnp.linspace(cfg_grid["dv"] / 2, cfg_grid["vmax"] - cfg_grid["dv"] / 2, cfg_grid["nv"]),
                "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
                "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            },
        }

        # config axes
        one_over_kx = np.zeros_like(cfg_grid["kx"])
        one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
        cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

        one_over_kxr = np.zeros_like(cfg_grid["kxr"])
        one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
        cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

        cfg_grid["nuprof"] = 1.0
        # get_profile_with_mask(cfg["nu"]["time-profile"], t, cfg["nu"]["time-profile"]["bump_or_trough"])
        cfg_grid["ktprof"] = 1.0
        # get_profile_with_mask(cfg["krook"]["time-profile"], t, cfg["krook"]["time-profile"]["bump_or_trough"])
        cfg_grid["kprof"] = np.ones_like(cfg_grid["x"])
        # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

        cfg_grid["ion_charge"] = np.zeros_like(cfg_grid["x"]) + cfg_grid["x"]

        cfg_grid["x_a"] = np.concatenate(
            [[cfg_grid["x"][0] - cfg_grid["dx"]], cfg_grid["x"], [cfg_grid["x"][-1] + cfg_grid["dx"]]]
        )

        self.cfg["grid"] = cfg_grid

    def init_state_and_args(self) -> Dict:
        """
        This function initializes the state

        :param cfg:
        :return:
        """
        f0, f10, ne_prof = _initialize_total_distribution_(self.cfg, self.cfg["grid"])

        state = {"f0": f0}
        # not currently necessary but kept for completeness
        for il in range(1, self.cfg["grid"]["nl"] + 1):
            for im in range(0, il + 1):
                state[f"f{il}{im}"] = jnp.zeros_like(f0)

        state["f10"] = f10

        for field in ["e", "b"]:
            state[field] = jnp.zeros(self.cfg["grid"]["nx"])

        state["Z"] = jnp.ones(self.cfg["grid"]["nx"])
        state["ni"] = ne_prof / self.cfg["units"]["Z"]

        self.state = state
        self.args = {"drivers": self.cfg["drivers"]}

    def init_diffeqsolve(self):
        self.cfg = get_save_quantities(self.cfg)
        self.time_quantities = {
            "t0": 0.0,
            "t1": self.cfg["grid"]["tmax"],
            "max_steps": self.cfg["grid"]["max_steps"],
            "save_t0": 0.0,
            "save_t1": self.cfg["grid"]["tmax"],
            "save_nt": self.cfg["grid"]["tmax"],
        }
        self.diffeqsolve_quants = dict(
            terms=ODETerm(OSHUN1D(self.cfg)),
            solver=Stepper(),
            saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in self.cfg["save"].items()}),
        )

    def __call__(self, trainable_modules: Dict, args: Dict):
        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=self.cfg["grid"]["max_steps"],
            dt0=self.cfg["grid"]["dt"],
            y0=self.state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return {"solver result": solver_result}
