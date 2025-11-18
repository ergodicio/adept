import math

import diffrax
import numpy as np
from diffrax import ODETerm, SaveAt, SubSaveAt, diffeqsolve
from equinox import filter_jit

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._lpse1d.core.vector_field import SplitStep1D


class BaseLPSE1D(ADEPTModule):
    """
    1D Laser-Plasma Spectral Envelope solver.

    This module solves the coupled laser-plasma wave equations in 1D:
    - Plasma waves (EPW) via spectral method
    - Laser fields (pump E0 and seed E1) via finite differences

    Designed for SRS (Stimulated Raman Scattering) simulations.

    Matches MATLAB's LPSE solver (m201805_matlabLpse_v11.m) for nDims=1.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def write_units(self) -> dict:
        """
        Calculate and return derived units.

        This uses the same helper as 2D since units are dimension-independent.
        """
        from adept._lpse2d.helpers import write_units

        return write_units(self.cfg)

    def get_derived_quantities(self):
        """
        Calculate derived grid quantities (dx, dt, nx, etc).

        Uses 1D-specific helper.
        """
        from adept._lpse1d.helpers import get_derived_quantities_1d

        self.cfg = get_derived_quantities_1d(self.cfg)

    def get_solver_quantities(self):
        """
        Set up solver-specific quantities (k-space grids, filters, etc).

        This needs 1D-specific implementation.
        """
        from adept._lpse1d.helpers import get_solver_quantities_1d

        self.cfg["grid"] = get_solver_quantities_1d(self.cfg)

    def init_modules(self) -> dict:
        """
        Initialize driver modules.

        For 1D SRS, we typically don't need spatial drivers.
        """
        modules = {}
        # Could add driver support here if needed
        return modules

    def init_diffeqsolve(self):
        """
        Initialize diffeqsolve parameters.
        """
        from adept._lpse1d.helpers import get_save_quantities_1d

        self.cfg = get_save_quantities_1d(self.cfg)

        self.time_quantities = {
            "t0": 0.0,
            "t1": self.cfg["grid"]["tmax"],
            "max_steps": self.cfg["grid"]["max_steps"],
            "save_t0": 0.0,
            "save_t1": self.cfg["grid"]["tmax"],
            "save_nt": self.cfg["grid"]["tmax"],
        }

        self.diffeqsolve_quants = dict(
            terms=ODETerm(SplitStep1D(self.cfg)),
            solver=Stepper(),
            saveat=dict(
                subs={
                    k: SubSaveAt(ts=subsave["t"]["ax"], fn=subsave["func"]) for k, subsave in self.cfg["save"].items()
                }
            ),
        )

    def init_state_and_args(self) -> dict:
        """
        Initialize state vectors and arguments.

        State contains:
        - epw: Plasma wave potential in k-space (complex, stored as float64 view)
        - E0: Pump laser field (complex, stored as float64 view)
        - E1: Seed laser field (complex, stored as float64 view)
        """
        from adept._lpse2d.helpers import get_density_profile

        # Initialize plasma wave with noise (if configured)
        if self.cfg["density"]["noise"]["type"] == "uniform":
            random_amps = np.random.uniform(
                self.cfg["density"]["noise"]["min"],
                self.cfg["density"]["noise"]["max"],
                (self.cfg["grid"]["nx"],),
            )
        elif self.cfg["density"]["noise"]["type"] == "normal":
            loc = 0.5 * (self.cfg["density"]["noise"]["min"] + self.cfg["density"]["noise"]["max"])
            scale = 1.0
            random_amps = np.random.normal(loc, scale, (self.cfg["grid"]["nx"],))
        else:
            random_amps = np.zeros((self.cfg["grid"]["nx"],))

        random_phases = np.random.uniform(0, 2 * np.pi, (self.cfg["grid"]["nx"],))
        phi_noise = random_amps * np.exp(1j * random_phases)
        epw = 0 * phi_noise  # Start with no EPW

        # Get density profile (1D version)
        # The 2D helper returns (nx, ny) but we'll take just the x-profile
        density_2d = get_density_profile(self.cfg)
        self.cfg["grid"]["background_density"] = density_2d[:, 0]  # Take first column

        # Initialize laser fields
        # E0: pump field at frequency w0
        # E1: seed field at frequency w1 (for SRS)
        E0 = np.zeros((self.cfg["grid"]["nx"],), dtype=np.complex128)
        E1 = np.zeros((self.cfg["grid"]["nx"],), dtype=np.complex128)

        # Add initial seed field if specified
        if "initial_seed_amplitude" in self.cfg["terms"].get("srs", {}):
            E1[:] = self.cfg["terms"]["srs"]["initial_seed_amplitude"]

        # Add pump field if specified
        if "pump_amplitude" in self.cfg.get("drivers", {}).get("E0", {}):
            E0[:] = self.cfg["drivers"]["E0"]["pump_amplitude"]

        state = {"epw": epw, "E0": E0, "E1": E1}

        # Convert complex arrays to float64 view for diffrax
        self.state = {k: v.view(dtype=np.float64) for k, v in state.items()}
        self.args = {}

    @filter_jit
    def __call__(self, trainable_modules: dict, args: dict | None = None) -> dict:
        """
        Run the simulation.

        Args:
            trainable_modules: Dictionary of trainable modules (drivers, etc)
            args: Optional additional arguments

        Returns:
            Dictionary containing solver result and args
        """
        state = self.state

        if args is not None:
            args = self.args | args
        else:
            args = self.args

        # Apply trainable modules (if any)
        for name, module in trainable_modules.items():
            state, args = module(state, args)

        # Set up checkpointing
        if self.cfg.get("opt", {}).get("checkpoints_coeff"):
            base_checkpoints = math.floor(-1.5 + math.sqrt(2 * (self.cfg["grid"]["max_steps"] + 2048) + 0.25)) + 1
            checkpoints = int(self.cfg["opt"]["checkpoints_coeff"] * base_checkpoints)
        else:
            checkpoints = None

        # Run ODE solver
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

    def post_process(self, run_output: dict, td: str) -> dict:
        """
        Post-process simulation results.

        Args:
            run_output: Output from __call__
            td: Target directory for output

        Returns:
            Dictionary with processed results
        """
        from adept._lpse1d.helpers import post_process_1d

        return post_process_1d(run_output["solver result"], self.cfg, td)
