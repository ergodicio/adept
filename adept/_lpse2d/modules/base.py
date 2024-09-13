from typing import Dict
import numpy as np
from diffrax import diffeqsolve, SaveAt, ODETerm, SubSaveAt
from equinox import filter_jit

from adept import ADEPTModule
from adept.base import Stepper
from adept._lpse2d.helpers import (
    write_units,
    post_process,
    get_derived_quantities,
    get_solver_quantities,
    get_save_quantities,
    get_density_profile,
)
from adept._lpse2d.core.vector_field import SplitStep
from adept._lpse2d.modules import driver


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
        return write_units(self.cfg)

    def get_derived_quantities(self):
        self.cfg = get_derived_quantities(self.cfg)

    def get_solver_quantities(self):
        self.cfg["grid"] = get_solver_quantities(self.cfg)

    def init_modules(self) -> Dict:

        modules = {}
        if "E0" in self.cfg["drivers"]:
            DriverModule = driver.choose_driver(self.cfg["drivers"]["E0"]["shape"])
            if "file" in self.cfg["drivers"]["E0"]:
                modules["driver"] = driver.load(self.cfg, DriverModule)
            else:
                modules["driver"] = DriverModule(self.cfg)

        return modules

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
            saveat=dict(
                subs={
                    k: SubSaveAt(ts=subsave["t"]["ax"], fn=subsave["func"]) for k, subsave in self.cfg["save"].items()
                }
            ),
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

        self.state = {k: v.view(dtype=np.float64) for k, v in state.items()}
        self.args = {"drivers": {k: v["derived"] for k, v in self.cfg["drivers"].items()}}

    @filter_jit
    def __call__(self, trainable_modules: Dict, args: Dict = None) -> Dict:
        state = self.state

        if args is None:
            args = self.args

        for name, module in trainable_modules.items():
            state, args = module(state, args)

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
