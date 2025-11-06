import math

import diffrax
import numpy as np
from diffrax import ODETerm, SaveAt, SubSaveAt, diffeqsolve
from equinox import filter_jit

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._lpse2d.core.vector_field import SplitStep
from adept._lpse2d.helpers import (
    get_density_profile,
    get_derived_quantities,
    get_save_quantities,
    get_solver_quantities,
    post_process,
    write_units,
)
from adept._lpse2d.modules import driver


class BaseLPSE2D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def post_process(self, run_output: dict, td: str) -> dict:
        return post_process(run_output["solver result"], self.cfg, td)

    def write_units(self) -> dict:
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

    def init_modules(self) -> dict:
        modules = {}
        if "E0" in self.cfg["drivers"]:
            DriverModule = driver.choose_driver(self.cfg["drivers"]["E0"]["shape"])
            if "file" in self.cfg["drivers"]["E0"]:
                modules["laser"] = driver.load(self.cfg, DriverModule)
            else:
                modules["laser"] = DriverModule(self.cfg)

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

    def init_state_and_args(self) -> dict:
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

        self.cfg["grid"]["background_density"] = get_density_profile(self.cfg)
        E0 = np.zeros((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], 2), dtype=np.complex128)
        E1 = np.zeros((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], 2), dtype=np.complex128)
        state = {"epw": epw, "E0": E0, "E1": E1}

        self.state = {k: v.view(dtype=np.float64) for k, v in state.items()}
        self.args = {"drivers": {k: v["derived"] for k, v in self.cfg["drivers"].items()}}

    @filter_jit
    def __call__(self, trainable_modules: dict, args: dict | None = None) -> dict:
        state = self.state

        if args is not None:
            args = self.args | args
        else:
            args = self.args

        for name, module in trainable_modules.items():
            state, args = module(state, args)

        if self.cfg.get("opt", {}).get("checkpoints_coeff"):
            base_checkpoints = math.floor(-1.5 + math.sqrt(2 * (self.cfg["grid"]["max_steps"] + 2048) + 0.25)) + 1
            checkpoints = int(self.cfg["opt"]["checkpoints_coeff"] * base_checkpoints)
        else:
            checkpoints = None

        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=self.cfg["grid"]["max_steps"] + 2048,
            dt0=self.cfg["grid"]["dt"],
            y0=state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=checkpoints),
        )

        return {"solver result": solver_result, "args": args}
