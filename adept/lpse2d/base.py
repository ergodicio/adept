from typing import Dict
import numpy as np
from astropy.units import Quantity as _Q
from diffrax import diffeqsolve, SaveAt, ODETerm
from equinox import filter_jit

from adept import ADEPTModule, Stepper
from adept.lpse2d.helpers import (
    post_process,
    calc_threshold_intensity,
    get_derived_quantities,
    get_solver_quantities,
    get_save_quantities,
    get_density_profile,
)
from adept.lpse2d.vector_field import SplitStep
from adept.lpse2d.modules.driver import BandwidthModule


class BaseLPSE2D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def post_process(self, run_output: Dict, td: str) -> Dict:
        return post_process(run_output["solver result"], self.cfg, td, run_output["args"])

    def write_units(self) -> Dict:
        """
        Write the units to a file

        :param cfg:
        :param td:
        :return: cfg
        """
        timeScale = 1e-12  # cgs (ps)
        spatialScale = 1e-4  # cgs (um)
        velocityScale = spatialScale / timeScale
        massScale = 1
        chargeScale = spatialScale ** (3 / 2) * massScale ** (1 / 2) / timeScale
        fieldScale = massScale ** (1 / 2) / spatialScale ** (1 / 2) / timeScale
        # forceScale = massScale * spatialScale/timeScale^2

        Te = _Q(self.cfg["units"]["reference electron temperature"]).to("keV").value
        Ti = _Q(self.cfg["units"]["reference ion temperature"]).to("keV").value
        Z = self.cfg["units"]["ionization state"]
        A = self.cfg["units"]["atomic number"]
        lam0 = _Q(self.cfg["units"]["laser wavelength"]).to("um").value
        I0 = _Q(self.cfg["units"]["laser intensity"]).to("W/cm^2").value
        envelopeDensity = self.cfg["units"]["envelope density"]

        # Scaled constants
        c_cgs = 2.99792458e10
        me_cgs = 9.10938291e-28
        mp_cgs = 1.6726219e-24
        e_cgs = 4.8032068e-10
        c = c_cgs / velocityScale
        me = me_cgs / massScale
        mi = mp_cgs * A / massScale
        e = e_cgs / chargeScale
        w0 = 2 * np.pi * c / lam0  # 1/ps
        wp0 = w0 * np.sqrt(envelopeDensity)
        w1 = w0 - wp0
        # nc = (w0*1e12)^2 * me / (4*pi*e^2) * (1e-4)^3
        vte = c * np.sqrt(Te / 511)
        vte_sq = vte**2
        cs = c * np.sqrt((Z * Te + 3 * Ti) / (A * 511 * 1836))

        # nu_sideloss = 1e-1

        # nu_ei = calc_nuei(ne, Te, Z, ni, Ti)
        # nu_ee = calc_nuee(ne, Te)

        nc = w0**2 * me / (4 * np.pi * e**2)

        E0_source = np.sqrt(8 * np.pi * I0 * 1e7 / c_cgs) / fieldScale

        ne_cc = nc * envelopeDensity * 1e4**3
        Te_eV = Te * 1000

        coulomb_log = (
            23.0 - np.log(np.sqrt(ne_cc) * Z / Te_eV**1.5)
            if Te_eV < 10 * Z**2
            else 24.0 - np.log(np.sqrt(ne_cc) / Te_eV)
        )
        fract = 1
        Zbar = Z * fract
        ni = fract * ne_cc / Zbar

        # logLambda_ei = np.zeros(len(Z))
        # for iZ in range(len(Z)):
        if self.cfg["terms"]["epw"]["damping"]["collisions"]:
            if Te_eV < 0.01 * Z**2:
                logLambda_ei = 22.8487 - np.log(np.sqrt(ne_cc) * Z / (Te * 1000) ** (3 / 2))
            elif Te_eV > 0.01 * Z**2:
                logLambda_ei = 24 - np.log(np.sqrt(ne_cc) / (Te * 1000))

            e_sq = 510.9896 * 2.8179e-13
            this_me = 510.9896 / 2.99792458e10**2
            nu_coll = (
                float(
                    (4 * np.sqrt(2 * np.pi) / 3 * e_sq**2 / np.sqrt(this_me) * Z**2 * ni * logLambda_ei / Te**1.5)
                    / 2
                    * timeScale
                )
                * self.cfg["terms"]["epw"]["damping"]["collisions"]
            )
        else:
            nu_coll = 1e-4  # nu_ee + nu_ei + nu_sideloss

        gradient_scale_length = _Q(self.cfg["density"]["gradient scale length"]).to("um").value
        I_thresh = calc_threshold_intensity(Te, Ln=gradient_scale_length, w0=w0)

        self.cfg["units"]["derived"] = {
            "c": c,
            "me": me,
            "mi": mi,
            "e": e,
            "w0": w0,
            "wp0": wp0,
            "w1": w1,
            "vte": vte,
            "vte_sq": vte_sq,
            "cs": cs,
            "nc": nc,
            "nu_coll": nu_coll,
            "I_thresh": I_thresh,
            "E0_source": E0_source,
            "timeScale": timeScale,
            "spatialScale": spatialScale,
            "velocityScale": velocityScale,
            "massScale": massScale,
            "chargeScale": chargeScale,
            "fieldScale": fieldScale,
        }

        return {}

    def get_derived_quantities(self):
        self.cfg = get_derived_quantities(self.cfg)

    def get_solver_quantities(self):
        self.cfg["grid"] = get_solver_quantities(self.cfg)

    def init_modules(self) -> Dict:
        return {"bandwidth": BandwidthModule(self.cfg)}

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
            terms=ODETerm(SplitStep(self.cfg)),
            solver=Stepper(),
            saveat=dict(ts=self.cfg["save"]["t"]["ax"], fn=self.cfg["save"]["func"]),
        )

    def init_state_and_args(self) -> Dict:
        if self.cfg["density"]["noise"]["type"] == "uniform":
            random_amps = np.random.uniform(
                self.cfg["density"]["noise"]["min"],
                self.cfg["density"]["noise"]["max"],
                (self.cfg["grid"]["nx"], self.cfg["grid"]["ny"]),
            )

        elif self.cfg["density"]["noise"]["type"] == "normal":
            loc = 0.5 * (self.cfg["density"]["noise"]["min"] + self.cfg["density"]["noise"]["max"])
            scale = 1.0
            random_amps = np.random.normal(loc, scale, (self.cfg["grid"]["nx"], self.cfg["grid"]["ny"]))

        else:
            raise NotImplementedError

        random_phases = np.random.uniform(0, 2 * np.pi, (self.cfg["grid"]["nx"], self.cfg["grid"]["ny"]))
        phi_noise = 1 * np.exp(1j * random_phases)
        epw = 0 * phi_noise

        background_density = get_density_profile(self.cfg)
        vte_sq = np.ones((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"])) * self.cfg["units"]["derived"]["vte"] ** 2
        E0 = np.zeros((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], 2), dtype=np.complex128)
        state = {"background_density": background_density, "epw": epw, "E0": E0, "vte_sq": vte_sq}

        # drivers = assemble_bandwidth(self.cfg)
        self.state = {k: v.view(dtype=np.float64) for k, v in state.items()}
        self.args = {"drivers": {"E0": {}}}

    @filter_jit
    def __call__(self, trainable_modules: Dict, args: Dict = None) -> Dict:

        if args is None:
            args = self.args

        for name, module in trainable_modules.items():
            state, args = module(self.state, args)

        # args = {"drivers": self.assemble_driver(params)}

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
