#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from dataclasses import asdict

import numpy as np
from diffrax import ODETerm, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import numpy as jnp

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._vlasov1d.grid import grid_from_dimensionless_cfg
from adept._vlasov1d.helpers import _initialize_total_distribution_, post_process
from adept._vlasov1d.normalization import electron_debye_normalization
from adept._vlasov1d.simulation import Vlasov1DSimulation
from adept._vlasov1d.solvers.vector_field import VlasovMaxwell
from adept._vlasov1d.storage import get_save_quantities
from adept.utils import SpaceTimeEnvelopeFunction, filter_scalars


def sim_from_cfg(cfg: dict) -> Vlasov1DSimulation:
    """Construct a Vlasov1DSimulation from a base config dict."""
    plasma_norm = electron_debye_normalization(
        cfg["units"]["normalizing_density"],
        cfg["units"]["normalizing_temperature"],
    )
    beta = 1.0 / plasma_norm.speed_of_light_norm()
    has_ey_driver = len(cfg.get("drivers", {}).get("ey", {}).keys()) > 0
    grid = grid_from_dimensionless_cfg(cfg["grid"], beta, should_override_dt_for_em_waves=has_ey_driver)

    # Construct collision frequency profiles if enabled
    nu_fp_prof = None
    if cfg["terms"]["fokker_planck"]["is_on"]:
        nu_fp_prof = SpaceTimeEnvelopeFunction.from_config(cfg["terms"]["fokker_planck"])

    nu_K_prof = None
    if cfg["terms"]["krook"]["is_on"]:
        nu_K_prof = SpaceTimeEnvelopeFunction.from_config(cfg["terms"]["krook"])

    return Vlasov1DSimulation(plasma_norm, grid, nu_fp_prof, nu_K_prof)


class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.simulation = sim_from_cfg(cfg)

    def post_process(self, run_output: dict, td: str):
        return post_process(run_output["solver result"], self.cfg, td, self.args)

    def write_units(self) -> dict:
        norm = self.simulation.plasma_norm

        box_length = ((self.cfg["grid"]["xmax"] - self.cfg["grid"]["xmin"]) * norm.L0).to("microns")
        if "ymax" in self.cfg["grid"].keys():
            box_width = ((self.cfg["grid"]["ymax"] - self.cfg["grid"]["ymin"]) * norm.L0).to("microns")
        else:
            box_width = "inf"
        sim_duration = (self.cfg["grid"]["tmax"] * norm.tau).to("ps")

        nu_ee = norm.approximate_ee_collision_frequency()

        beta = 1.0 / norm.speed_of_light_norm()

        all_quantities = {
            "wp0": (1 / norm.tau).to("rad/s"),
            "tp0": norm.tau.to("fs"),
            "n0": norm.n0.to("1/cc"),
            "v0": norm.v0.to("m/s"),
            "T0": norm.T0.to("eV"),
            "c_light": norm.speed_of_light_norm(),
            "beta": beta,
            "x0": norm.L0.to("nm"),
            "nuee": nu_ee.to("Hz"),
            "logLambda_ee": norm.logLambda_ee(),
            "box_length": box_length,
            "box_width": box_width,
            "sim_duration": sim_duration,
        }

        self.cfg["units"]["derived"] = all_quantities

        self.cfg["grid"]["beta"] = beta

        return all_quantities

    def get_derived_quantities(self):
        """
        This function just updates the config with the derived quantities that are only integers or strings.

        This is run prior to the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = self.cfg["grid"]
        grid = self.simulation.grid

        # Merge grid scalar values from the Grid object
        cfg_grid.update(filter_scalars(asdict(grid)))

        # Default save.*.t.tmin/tmax to computed grid values
        for save_type in self.cfg.get("save", {}).keys():
            if "t" in self.cfg["save"][save_type]:
                t_cfg = self.cfg["save"][save_type]["t"]
                t_cfg.setdefault("tmin", grid.tmin)
                t_cfg.setdefault("tmax", grid.tmax)

        # Normalize species config: if not provided, generate a default electron species
        if self.cfg["terms"].get("species", None) is None:
            # Collect all density components (keys starting with "species-")
            density_components = [name for name in self.cfg["density"].keys() if name.startswith("species-")]
            if not density_components:
                raise ValueError("No density components found (expected keys starting with 'species-')")

            # Generate default electron species config
            self.cfg["terms"]["species"] = [
                {
                    "name": "electron",
                    "charge": -1.0,
                    "mass": 1.0,
                    "vmax": cfg_grid["vmax"],
                    "nv": cfg_grid["nv"],
                    "density_components": density_components,
                }
            ]

        # Print message if dt was overridden for EM wave stability
        if len(self.cfg["drivers"]["ey"].keys()) > 0:
            print("overriding dt to ensure wave solver stability")

        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self) -> dict:
        """
        This function just updates the config with the derived quantities that are arrays

        This is run after the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = self.cfg["grid"]
        grid = self.simulation.grid

        # Merge all grid values (including arrays) from the Grid object
        cfg_grid.update(asdict(grid))

        # Initialize distributions (always returns dict format)
        dist_result = _initialize_total_distribution_(self.cfg, cfg_grid)
        cfg_grid["species_distributions"] = dist_result

        # Build species_grids and species_params
        cfg_grid["species_grids"] = {}
        cfg_grid["species_params"] = {}
        n_prof_total = np.zeros([cfg_grid["nx"]])

        for species_name, (n_prof, f_s, v_ax) in dist_result.items():
            n_prof_total += n_prof

            # Find the species config (always exists due to normalization in get_derived_quantities)
            species_cfg = next((s for s in self.cfg["terms"]["species"] if s["name"] == species_name), None)
            if species_cfg is None:
                raise ValueError(f"Species '{species_name}' not found in config['terms']['species']")

            nv = species_cfg["nv"]
            vmax = species_cfg["vmax"]

            dv = 2.0 * vmax / nv

            # Build velocity grid parameters for this species
            cfg_grid["species_grids"][species_name] = {
                "v": jnp.array(v_ax),
                "dv": dv,
                "nv": nv,
                "vmax": vmax,
                "kv": jnp.fft.fftfreq(nv, d=dv) * 2.0 * np.pi,
                "kvr": jnp.fft.rfftfreq(nv, d=dv) * 2.0 * np.pi,
            }

            # one_over_kv for this species (size is length of kvr for real FFT)
            kvr_len = len(cfg_grid["species_grids"][species_name]["kvr"])
            one_over_kv = np.zeros(nv)
            one_over_kv[1:] = 1.0 / cfg_grid["species_grids"][species_name]["kv"][1:]
            cfg_grid["species_grids"][species_name]["one_over_kv"] = jnp.array(one_over_kv)

            one_over_kvr = np.zeros(kvr_len)
            one_over_kvr[1:] = 1.0 / cfg_grid["species_grids"][species_name]["kvr"][1:]
            cfg_grid["species_grids"][species_name]["one_over_kvr"] = jnp.array(one_over_kvr)

            # Build species parameters (charge, mass, charge-to-mass ratio)
            cfg_grid["species_params"][species_name] = {
                "charge": species_cfg["charge"],
                "mass": species_cfg["mass"],
                "charge_to_mass": species_cfg["charge"] / species_cfg["mass"],
            }

        cfg_grid["n_prof_total"] = n_prof_total

        # Quasineutrality handling
        # For single-species electron-only sims, assume static ion background
        # For multi-species, quasineutrality is handled by the species themselves
        has_multiple_species = len(self.cfg["terms"]["species"]) > 1
        if has_multiple_species:
            cfg_grid["ion_charge"] = np.zeros_like(n_prof_total)
        else:
            cfg_grid["ion_charge"] = n_prof_total.copy()

        # For single-species configs, also store velocity grid at grid level for backward compatibility
        if not has_multiple_species and "electron" in cfg_grid["species_grids"]:
            cfg_grid["v"] = jnp.array(dist_result["electron"][2])
            cfg_grid["kv"] = cfg_grid["species_grids"]["electron"]["kv"]
            cfg_grid["kvr"] = cfg_grid["species_grids"]["electron"]["kvr"]
            cfg_grid["one_over_kv"] = cfg_grid["species_grids"]["electron"]["one_over_kv"]
            cfg_grid["one_over_kvr"] = cfg_grid["species_grids"]["electron"]["one_over_kvr"]

        self.cfg["grid"] = cfg_grid

    def init_state_and_args(self) -> dict:
        """
        This function initializes the state

        :param cfg:
        :return:
        """
        grid = self.simulation.grid

        # Initialize distributions (always returns dict format)
        dist_result = _initialize_total_distribution_(self.cfg, self.cfg["grid"])

        state = {}

        # Build state dict with all species distributions
        for species_name, (n_prof, f_s, v_ax) in dist_result.items():
            state[species_name] = jnp.array(f_s)

        # Reference distribution for diagnostics (use first species)
        # TODO(gh-174): Store species distributions separately for multi-species diagnostics
        first_species_name = next(iter(dist_result.keys()))
        f_ref = dist_result[first_species_name][1]

        # Field quantities (same for all modes)
        for field in ["e", "de"]:
            state[field] = jnp.zeros(grid.nx)

        for field in ["a", "da", "prev_a"]:
            state[field] = jnp.zeros(grid.nx + 2)  # need boundary cells

        # Diagnostics (use reference distribution shape)
        for k in ["diag-vlasov-dfdt", "diag-fp-dfdt"]:
            if self.cfg["diagnostics"][k]:
                state[k] = jnp.zeros_like(f_ref)

        self.state = state
        self.args = {"drivers": self.cfg["drivers"], "terms": self.cfg["terms"]}

    def init_diffeqsolve(self):
        self.cfg = get_save_quantities(self.cfg)
        grid = self.simulation.grid
        self.time_quantities = {"t0": 0.0, "t1": grid.tmax, "max_steps": grid.max_steps}
        self.diffeqsolve_quants = dict(
            terms=ODETerm(
                VlasovMaxwell(
                    self.cfg,
                    grid,
                    nu_fp_prof=self.simulation.nu_fp_prof,
                    nu_K_prof=self.simulation.nu_K_prof,
                )
            ),
            solver=Stepper(),
            saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in self.cfg["save"].items()}),
        )

    def __call__(self, trainable_modules: dict, args: dict | None = None):
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
            progress_meter=TqdmProgressMeter(refresh_steps=grid.max_steps // 100),
        )

        return {"solver result": solver_result}
