import math

import diffrax
import numpy as np
from diffrax import ODETerm, SaveAt, SubSaveAt, diffeqsolve
from equinox import filter_jit

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._lpse2d.core.vector_field import SplitStep
from adept._lpse2d.helpers import (
    _Q,
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
        t0 = float(getattr(self, "restart_t0", 0.0))
        self.time_quantities = {
            "t0": t0,
            "t1": self.cfg["grid"]["tmax"],
            "max_steps": self.cfg["grid"]["max_steps"],
            "save_t0": t0,
            "save_t1": self.cfg["grid"]["tmax"],
            "save_nt": self.cfg["grid"]["tmax"],
        }

        subs = {
            k: SubSaveAt(ts=subsave["t"]["ax"], fn=subsave["func"])
            for k, subsave in self.cfg["save"].items()
            if isinstance(subsave, dict) and "func" in subsave
        }
        if self.cfg["save"].get("checkpoint"):
            # the full state at the final time (save.checkpoint: true | path), written by
            # post_process as an .npz that `restart.file` accepts
            subs["checkpoint"] = SubSaveAt(ts=np.array([self.cfg["grid"]["tmax"]]), fn=lambda t, y, args: y)
        self.diffeqsolve_quants = dict(terms=ODETerm(SplitStep(self.cfg)), solver=Stepper(), saveat=dict(subs=subs))

    def init_state_and_args(self) -> dict:
        # The initial EPW is identically zero; noise-seeded runs get their seeding from
        # terms.epw.source.noise (a per-step source in SpectralEPWSolver). The old
        # density.noise draws were dead code, but they consumed the global numpy RNG
        # stream and made "identical" runs differ -- so they are gone.
        epw = np.zeros((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"]), dtype=np.complex128)

        self.cfg["grid"]["background_density"] = get_density_profile(self.cfg)
        # the light fields carry three components (x, y, z) as LPSE's do on any grid (plan 2
        # F.1); the grid is 2-D, so E_z is purely transverse and only an s-polarised beam or
        # seed populates it
        E0 = np.zeros((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], 3), dtype=np.complex128)
        E1 = np.zeros((self.cfg["grid"]["nx"], self.cfg["grid"]["ny"], 3), dtype=np.complex128)
        state = {"epw": epw, "E0": E0, "E1": E1}

        if self.cfg["terms"]["epw"].get("energy_ledger", False):
            from adept._lpse2d.core.epw import LEDGER_CHANNELS, LEDGER_KEY

            state[LEDGER_KEY] = np.zeros(len(LEDGER_CHANNELS), dtype=np.float64)

        if self.cfg["terms"].get("iaw", {}).get("active", False):
            iaw_shape = (self.cfg["grid"]["nx"], self.cfg["grid"]["ny"])
            state["iaw_density"] = np.zeros(iaw_shape, dtype=np.float64)
            state["iaw_velocity_divergence"] = np.zeros(iaw_shape, dtype=np.float64)

        if self.cfg["terms"].get("hpe", {}).get("active", False):
            from adept._lpse2d.core.hpe import load_particles

            # particle positions/momenta and histogram/damping state are real float64,
            # so the .view below is a no-op
            state = state | load_particles(self.cfg)

        self.state = {k: v.view(dtype=np.float64) for k, v in state.items()}
        # ---- restart (LPSE --restart): replace the freshly built state by a checkpoint and
        # continue from its time; the per-step noise / wall keys are folded in from the
        # time index, so a resumed run reproduces the unbroken one to round-off
        self.restart_t0 = 0.0
        restart = self.cfg.get("restart")
        if restart and restart.get("file"):
            loaded = np.load(restart["file"], allow_pickle=False)
            self.restart_t0 = float(loaded["t"])
            # the saved time carries the solver's float precision; snap it to the step grid so the
            # resumed step times coincide with the unbroken run's
            dt = float(self.cfg["grid"]["dt"])
            n_steps = round(self.restart_t0 / dt)
            if abs(self.restart_t0 - n_steps * dt) < 1.0e-5 * dt:
                self.restart_t0 = n_steps * dt
            if self.restart_t0 >= self.cfg["grid"]["tmax"]:
                raise ValueError(f"restart time {self.restart_t0} ps is not before grid.tmax")
            missing = [k for k in self.state if k not in loaded.files]
            if missing:
                raise ValueError(f"checkpoint {restart['file']} lacks state entries {missing}")
            restored = {}
            for k, v in self.state.items():
                arr = np.asarray(loaded[k], dtype=np.float64)
                if k in ("E0", "E1") and arr.shape[:-1] == v.shape[:-1] and arr.shape[-1] == 4:
                    # a two-component (x, y) checkpoint from before plan 2 F.1: pad E_z = 0
                    # (the float64 view of a complex component axis has twice the length)
                    arr = np.concatenate([arr, np.zeros(arr.shape[:-1] + (2,), dtype=np.float64)], axis=-1)
                if arr.shape != v.shape:
                    raise ValueError(f"checkpoint entry {k} has shape {arr.shape}, expected {v.shape}")
                restored[k] = arr
            self.state = restored
            for sub in self.cfg["save"].values():
                if isinstance(sub, dict) and isinstance(sub.get("t"), dict) and "tmin" in sub["t"]:
                    if _Q(sub["t"]["tmin"]).to("ps").value < self.restart_t0:
                        sub["t"]["tmin"] = f"{self.restart_t0:.9g}ps"
            # the default series samples the grid's time axis: keep only the resumed interval
            t_axis = np.asarray(self.cfg["grid"]["t"])
            kept = t_axis[t_axis >= self.restart_t0 - 1.0e-12]
            self.cfg["grid"]["t"] = kept if kept.size else np.asarray([self.cfg["grid"]["tmax"]])
            print(f"restarting from {restart['file']} at t = {self.restart_t0:.6g} ps")
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
