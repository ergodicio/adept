import numpy as np
from astropy import constants as csts
from astropy import units as u
from diffrax import ODETerm, SaveAt, SubSaveAt, diffeqsolve
from jax import numpy as jnp

from adept._base_ import ADEPTModule, Stepper
from adept.utils import filter_scalars
from adept.vfp1d.grid import Grid
from adept.vfp1d.helpers import _initialize_total_distribution_
from adept.vfp1d.normalization import PlasmaNorm
from adept.vfp1d.storage import get_save_quantities, post_process
from adept.vfp1d.vector_field import OSHUN1D


class BaseVFP1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.plasma_norm = PlasmaNorm.from_config(cfg["units"])
        self.grid = Grid.from_config(cfg["grid"], self.plasma_norm)

    def post_process(self, solver_result: dict, td: str) -> dict:
        return post_process(solver_result["solver result"], cfg=self.cfg, td=td, args=self.args)

    def write_units(self) -> dict:
        norm = self.plasma_norm
        Z = self.cfg["units"]["Z"]
        ne = norm.ne
        Te = norm.Te
        logLambda_ei = norm.logLambda_ei

        nD_NRL = 1.72e9 * Te.value**1.5 / np.sqrt(ne.value)

        nuei_shk = np.sqrt(2.0 / np.pi) * norm.wp0 * logLambda_ei / np.exp(logLambda_ei)
        # JPB - Maybe comment which page/eq this is from? There are lots of collision times in NRL
        # For example, nu_ei on page 32 does include Z^2
        nuei_nrl = np.sqrt(2.0 / np.pi) * norm.wp0 * logLambda_ei / nD_NRL

        nuei_epphaines = (
            1
            / (
                0.75
                * np.sqrt(csts.m_e)
                * Te**1.5
                / (np.sqrt(2 * np.pi) * (ne / Z) * Z**2.0 * csts.e.gauss**4.0 * logLambda_ei)
            )
        ).to("Hz")

        all_quantities = {
            "wp0": norm.wp0,
            "n0": norm.n0,
            "tp0": norm.tp0,
            "ne": ne,
            "vth": norm.vth,
            "Te": Te,
            "Ti": u.Quantity(self.cfg["units"]["reference ion temperature"]).to("eV"),
            "logLambda_ei": logLambda_ei,
            "logLambda_ee": norm.logLambda_ee,
            "beta": norm.beta,
            "x0": norm.x0,
            "nuei_shk": nuei_shk,
            "nuei_nrl": nuei_nrl,
            "nuei_epphaines": nuei_epphaines,
            "nuei_shk_norm": nuei_shk / norm.wp0,
            "nuei_nrl_norm": nuei_nrl / norm.wp0,
            "nuei_epphaines_norm": nuei_epphaines / norm.wp0,
            "lambda_mfp_shk": (norm.vth / nuei_shk).to("micron"),
            "lambda_mfp_nrl": (norm.vth / nuei_nrl).to("micron"),
            "lambda_mfp_epphaines": (norm.vth / nuei_epphaines).to("micron"),
            "nD_NRL": nD_NRL,
            "nD_Shkarofsky": np.exp(logLambda_ei) * Z / 9,
        }

        self.cfg["units"]["derived"] = all_quantities
        self.cfg["grid"]["beta"] = norm.beta

        return {k: str(v) for k, v in all_quantities.items()}

    def get_derived_quantities(self):
        """Sync scalar grid values into cfg for logging.

        This is run prior to the log params step.
        """
        cfg_grid = self.cfg["grid"]
        grid = self.grid

        # Default save.*.t.tmin/tmax to computed grid values
        for save_type in self.cfg.get("save", {}).keys():
            if "t" in self.cfg["save"][save_type]:
                t_cfg = self.cfg["save"][save_type]["t"]
                t_cfg.setdefault("tmin", cfg_grid.get("tmin", "0ps"))
                t_cfg.setdefault("tmax", cfg_grid["tmax"])

        # Merge scalar grid values into cfg for logging (arrays come later in get_solver_quantities)
        cfg_grid.update(filter_scalars(grid.as_dict()))

        print("tmax", grid.tmax, "dt", grid.dt)
        print("xmax", grid.xmax, "dx", grid.dx)

        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self):
        """Merge all grid values (including arrays) into cfg['grid'] for backward compatibility."""
        self.cfg["grid"].update(self.grid.as_dict())

    def init_state_and_args(self) -> dict:
        grid = self.grid
        f0, f10, ne_prof = _initialize_total_distribution_(self.cfg, grid, self.plasma_norm)

        state = {"f0": f0}
        # not currently necessary but kept for completeness
        for il in range(1, grid.nl + 1):
            for im in range(0, il + 1):
                state[f"f{il}{im}"] = jnp.zeros((grid.nx + 1, grid.nv))

        state["f10"] = f10

        for field in ["e", "b"]:
            state[field] = jnp.zeros(grid.nx + 1)

        state["Z"] = jnp.ones(grid.nx)
        state["ni"] = ne_prof / self.cfg["units"]["Z"]

        self.state = state
        self.args = {"drivers": self.cfg["drivers"]}

    def init_diffeqsolve(self):
        grid = self.grid
        self.cfg = get_save_quantities(self.cfg)
        self.time_quantities = {
            "t0": 0.0,
            "t1": grid.tmax,
            "max_steps": grid.max_steps,
        }
        self.diffeqsolve_quants = dict(
            terms=ODETerm(OSHUN1D(self.cfg, grid=grid)),
            solver=Stepper(),
            saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in self.cfg["save"].items()}),
        )

    def __call__(self, trainable_modules: dict, args: dict):
        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=self.grid.max_steps,
            dt0=self.grid.dt,
            y0=self.state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return {"solver result": solver_result}
