from typing import Dict
from astropy import constants as const
from astropy.units import Quantity as _Q
import numpy as np
from diffrax import diffeqsolve, SaveAt

from adept import ADEPTModule


class BaseSteadyStateBackwardStimulatedBrilloiunScattering(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def write_units(self) -> Dict:
        """ """
        m_e = const.m_e
        e = const.e
        c = const.c
        eps_0 = const.eps0
        mi = self.cfg["units"]["atomic mass number"] * const.m_p

        lambda0 = _Q(self.cfg["units"]["laser_wavelength"]).to("um")
        T0 = _Q(self.cfg["units"]["reference temperature"]).to("keV")
        vth0 = np.sqrt(2 * T0 / m_e).to("m/s")
        w0 = (2 * np.pi * c / lambda0).to("Hz")
        cs0 = np.sqrt(T0 / mi).to("m/s")
        I0 = _Q(self.cfg["units"]["reference intensity"]).to("W/cm^2")
        a0 = None
        kz0 = 2 * np.pi / lambda0

        if self.cfg["units"]["reference density"] == "nc":
            n0 = (2 * np.pi * c / self.cfg["units"]["laser_wavelength"]) ** 2.0 * m_e * eps_0 / e**2.0
        else:
            n0 = _Q(self.cfg["units"]["reference density"]).to("cm-3")

        self.cfg["units"]["derived"] = {
            "n0": n0,
            "lambda0": lambda0,
            "T0": T0,
            "vth0": vth0,
            "w0": w0,
            "kz0": kz0,
            "cs0": cs0,
            "I0": I0,
            "a0": a0,
            "mi": mi,
        }

        return {k: str(v) for k, v in self.cfg["units"]["derived"].items()}

    def get_solver_quantities(self, cfg):

        cfg_grid = cfg["grid"]

        cfg_grid = {
            **cfg_grid,
            **{
                "z": np.linspace(
                    cfg_grid["zmin"] + cfg_grid["dz"] / 2,
                    cfg_grid["zmax"] - cfg_grid["dz"] / 2,
                    cfg_grid["nz"],
                ),
            },
        }

        return cfg_grid

    def get_derived_quantities(self, cfg) -> Dict:
        """
        This function just updates the config with the derived quantities that are only integers or strings.

        This is run prior to the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = cfg["grid"]

        Lgrid = _Q(cfg_grid["zmax"]).to("um").value

        cfg_grid["zmax"] = Lgrid
        cfg_grid["zmin"] = 0.0

        cfg_grid["nz"] = int(cfg_grid["zmax"] / cfg_grid["dz"]) + 1

        if cfg_grid["nz"] > 1e6:
            cfg_grid["max_steps"] = int(1e6)
            print(r"Only running $10^6$ steps")
        else:
            cfg_grid["max_steps"] = cfg_grid["nz"] + 4

        cfg["grid"] = cfg_grid

        return cfg

    def init_state_and_args(self):
        nprof = (
            self.cfg["profiles"]["n"]["min"]
            + (self.cfg["profiles"]["n"]["max"] - self.cfg["profiles"]["n"]["min"])
            * self.cfg["grid"]["z"]
            / self.cfg["grid"]["zmax"]
        )
        Teprof = np.ones_like(nprof) * _Q(self.cfg["profiles"]["Te"]).to("keV").value
        Tiprof = np.ones_like(nprof) * _Q(self.cfg["profiles"]["Ti"]).to("keV").value
        Zeffprof = np.ones_like(nprof) * self.cfg["profiles"]["Zeff"]
        flowprof = np.ones_like(nprof) * _Q(self.cfg["profiles"]["flow"]).to("um/ns").value
        omega_beat_prof = (
            np.ones_like(nprof)
            * 2
            * np.pi
            / _Q(self.cfg["profiles"]["omega_beat"]).to("um").value
            * self.cfg["units"]["derived"]["c"]
        )
        self.state = {"Ji": 1.0, "Jr": 0.1}
        self.args = {
            "n_over_nc": nprof,
            "Te_keV": Teprof,
            "Ti_keV": Tiprof,
            "Zeff": Zeffprof,
            "flow": flowprof,
            "omega_beat": omega_beat_prof,
        }

    def __call__(self, trainable_modules: Dict, args: Dict):
        state = self.state

        if args is None:
            args = self.args

        for name, module in trainable_modules.items():
            state, args = module(self.state, args)

        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=self.cfg["grid"]["max_steps"],
            dt0=self.cfg["grid"]["dt"],
            y0=state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return {"solver result": solver_result, "args": args}
