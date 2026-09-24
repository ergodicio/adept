"""Legacy ADEPT lifecycle sharing preparation with the explicit Vlasov-1D builder."""

#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import sys

from diffrax import NoProgressMeter, ODETerm, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve

from adept._base_ import ADEPTModule, Stepper
from adept._vlasov1d.helpers import post_process
from adept._vlasov1d.preparation import Vlasov1DSetup
from adept._vlasov1d.preparation import sim_from_config as sim_from_config
from adept._vlasov1d.preparation import species_set_from_config as species_set_from_config


class BaseVlasov1D(Vlasov1DSetup, ADEPTModule):
    """ADEPT module wrapper for configuring, running, and post-processing Vlasov-1D."""

    def __init__(self, cfg) -> None:
        """Validate configuration and construct the Vlasov-1D simulation domain."""
        ADEPTModule.__init__(self, cfg)
        Vlasov1DSetup.__init__(self, cfg)

    def post_process(self, run_output: dict, td: str):
        """Post-process a solver result into plots, netCDF files, and MLflow metrics."""
        return post_process(run_output["solver result"], self.cfg, td, self.args)

    def get_derived_quantities(self):
        """Resolve scalar grid quantities and retain the legacy stability notice."""
        super().get_derived_quantities()
        if len(self.cfg["drivers"]["ey"].keys()) > 0:
            print("overriding dt to ensure wave solver stability")

    def init_diffeqsolve(self):
        """Assemble Diffrax terms, solver, save functions, and solve time bounds."""
        saves = self.prepare_save_quantities()
        grid = self.simulation.grid
        self.time_quantities = {"t0": 0.0, "t1": grid.tmax, "max_steps": grid.max_steps}
        self.diffeqsolve_quants = dict(
            terms=ODETerm(self.prepare_step()),
            solver=Stepper(),
            saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in saves.items()}),
        )

    def __call__(self, trainable_modules: dict, args: dict | None = None):
        """Run the configured Vlasov-1D solve and return the raw Diffrax result."""
        if args is None:
            args = self.args
        grid = self.simulation.grid
        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=grid.max_steps,
            dt0=grid.dt,
            y0=self.state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
            progress_meter=TqdmProgressMeter(refresh_steps=grid.max_steps // 100)
            if sys.stdout.isatty()
            else NoProgressMeter(),
        )

        return {"solver result": solver_result}
