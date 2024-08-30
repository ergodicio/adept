from typing import Dict
from astropy import constants as const
from astropy.units import Quantity as _Q
import numpy as np
from adept import ADEPTModule


class BaseSteadyStateBackwardStimulatedBrilloiunScattering(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def write_units(self, cfg) -> Dict:
        """ """
        m_e = const.m_e
        e = const.e
        c = const.c
        eps_0 = const.eps0
        mi = cfg["units"]["atomic mass number"] * const.m_p

        lambda0 = _Q(cfg["units"]["laser_wavelength"]).to("nm")
        T0 = _Q(cfg["units"]["reference temperature"]).to("keV")
        vth0 = np.sqrt(2 * T0 / m_e).to("m/s")
        w0 = (2 * np.pi * c / lambda0).to("Hz")
        cs0 = np.sqrt(T0 / mi).to("m/s")
        I0 = _Q(cfg["units"]["reference intensity"]).to("W/cm^2")
        a0 = None

        if cfg["units"]["reference density"] == "nc":
            n0 = (2 * np.pi * c / cfg["units"]["laser_wavelength"]) ** 2.0 * m_e * eps_0 / e**2.0
        else:
            n0 = _Q(cfg["units"]["reference density"]).to("cm-3")

        cfg["units"]["derived"] = {
            "n0": n0,
            "lambda0": lambda0,
            "T0": T0,
            "vth0": vth0,
            "w0": w0,
            "cs0": cs0,
            "I0": I0,
            "a0": a0,
            "mi": mi,
        }

        return {}

    def get_derived_quantities(self) -> Dict:
        return super().get_derived_quantities()

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
