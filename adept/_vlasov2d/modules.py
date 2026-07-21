"""ADEPTModule entrypoint for the Vlasov-2D solver."""

import sys
from dataclasses import asdict

import numpy as np
from diffrax import NoProgressMeter, ODETerm, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import numpy as jnp

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._vlasov2d.datamodel import SpeciesConfig, Vlasov2DConfig
from adept._vlasov2d.grid import Grid
from adept._vlasov2d.helpers import _initialize_total_distribution_, post_process
from adept._vlasov2d.simulation import (
    CollisionProfile2D,
    EMDriverSet,
    Species,
    SubspeciesDistributionSpec,
    Vlasov2DSimulation,
)
from adept._vlasov2d.solvers.pushers.field import InitialPoissonSolver
from adept._vlasov2d.solvers.vector_field import VlasovMaxwell
from adept._vlasov2d.storage import get_save_quantities
from adept.normalization import electron_debye_normalization
from adept.utils import filter_scalars


def species_set_from_config(cfg: Vlasov2DConfig) -> list[Species]:
    if cfg.terms.species:
        return [Species.from_config(s) for s in cfg.terms.species]
    components = [name for name in cfg.density.model_extra.keys() if name.startswith("species-")]
    if not components:
        raise ValueError("No density components found (expected keys starting with 'species-').")
    nvx = cfg.grid.nvx
    nvy = cfg.grid.nvy
    if nvx is None or nvy is None or cfg.grid.vmax is None:
        raise ValueError("For single-species configs, grid.nvx, grid.nvy, and grid.vmax must be set.")
    return [
        Species.from_config(
            SpeciesConfig(
                name="electron",
                charge=-1.0,
                mass=1.0,
                vmax=cfg.grid.vmax,
                nvx=nvx,
                nvy=nvy,
                density_components=components,
            )
        )
    ]


def sim_from_config(cfg: Vlasov2DConfig) -> Vlasov2DSimulation:
    plasma_norm = electron_debye_normalization(
        cfg.units.normalizing_density,
        cfg.units.normalizing_temperature,
    )
    beta = 1.0 / plasma_norm.speed_of_light_norm()
    has_driver = len(cfg.drivers.ex) + len(cfg.drivers.ey) > 0
    grid = Grid.from_config(cfg.grid, beta, should_override_dt_for_em_waves=has_driver, norm=plasma_norm)

    nu_fp_prof = None
    if cfg.terms.fokker_planck.is_on:
        nu_fp_prof = CollisionProfile2D.from_envs(
            cfg.terms.fokker_planck.time,
            cfg.terms.fokker_planck.space_x,
            cfg.terms.fokker_planck.space_y,
            plasma_norm,
        )

    nu_K_prof = None
    if cfg.terms.krook.is_on:
        nu_K_prof = CollisionProfile2D.from_envs(
            cfg.terms.krook.time,
            cfg.terms.krook.space_x,
            cfg.terms.krook.space_y,
            plasma_norm,
        )

    species = species_set_from_config(cfg)
    species_distribution_specs = {
        s.name: [
            SubspeciesDistributionSpec.from_config(cfg.density.get_component(component_name), norm=plasma_norm)
            for component_name in s.density_components
        ]
        for s in species
    }
    drivers = EMDriverSet.from_config(cfg.drivers, plasma_norm)
    return Vlasov2DSimulation(plasma_norm, grid, species, species_distribution_specs, drivers, nu_fp_prof, nu_K_prof)


class BaseVlasov2D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.config_model = Vlasov2DConfig.model_validate(cfg)
        self.simulation = sim_from_config(self.config_model)
        self._initial_distribution = None

    def post_process(self, run_output: dict, td: str):
        return post_process(run_output["solver result"], self.cfg, td, self.args)

    def write_units(self) -> dict:
        norm = self.simulation.plasma_norm
        grid = self.simulation.grid

        box_length = ((grid.xmax - grid.xmin) * norm.L0).to("microns")
        box_width = ((grid.ymax - grid.ymin) * norm.L0).to("microns")
        sim_duration = (grid.tmax * norm.tau).to("ps")
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
        cfg_grid = self.cfg["grid"]
        grid = self.simulation.grid
        cfg_grid.update(filter_scalars(asdict(grid)))

        for save_val in self.cfg.get("save", {}).values():
            if "t" in save_val:
                save_val["t"].setdefault("tmin", grid.tmin)
                save_val["t"].setdefault("tmax", grid.tmax)
            else:
                for label_cfg in save_val.values():
                    if isinstance(label_cfg, dict) and "t" in label_cfg:
                        label_cfg["t"].setdefault("tmin", grid.tmin)
                        label_cfg["t"].setdefault("tmax", grid.tmax)

        if len(self.cfg["drivers"]["ex"]) + len(self.cfg["drivers"]["ey"]) > 0:
            print("overriding dt to ensure Maxwell wave solver stability")
        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self) -> dict:
        cfg_grid = self.cfg["grid"]
        grid = self.simulation.grid
        cfg_grid.update(asdict(grid))

        dist_result = _initialize_total_distribution_(self.cfg, self.simulation)
        self._initial_distribution = dist_result
        cfg_grid["species_distributions"] = dist_result

        cfg_grid["species_grids"] = {}
        cfg_grid["species_params"] = {}
        n_prof_total = np.zeros([grid.nx, grid.ny])

        for species_name, (n_prof, _f, vxax, vyax) in dist_result.items():
            n_prof_total += n_prof
            sp_cfg = self.simulation.species_dict[species_name]

            dvx = 2.0 * sp_cfg.vmax / sp_cfg.nvx
            dvy = 2.0 * sp_cfg.vmax / sp_cfg.nvy

            kvxr = jnp.fft.rfftfreq(sp_cfg.nvx, d=dvx) * 2.0 * np.pi
            kvyr = jnp.fft.rfftfreq(sp_cfg.nvy, d=dvy) * 2.0 * np.pi

            cfg_grid["species_grids"][species_name] = {
                "vx": jnp.array(vxax),
                "vy": jnp.array(vyax),
                "dvx": dvx,
                "dvy": dvy,
                "nvx": sp_cfg.nvx,
                "nvy": sp_cfg.nvy,
                "vmax": sp_cfg.vmax,
                "kvxr": kvxr,
                "kvyr": kvyr,
            }
            cfg_grid["species_params"][species_name] = {
                "charge": sp_cfg.charge,
                "mass": sp_cfg.mass,
                "charge_to_mass": sp_cfg.charge / sp_cfg.mass,
            }

        cfg_grid["n_prof_total"] = n_prof_total

        # static ion background for single-species electron sims
        has_multiple_species = len(self.simulation.species) > 1
        if has_multiple_species:
            cfg_grid["ion_charge"] = np.zeros_like(n_prof_total)
        else:
            cfg_grid["ion_charge"] = n_prof_total.copy()

        self.cfg["grid"] = cfg_grid

    def init_state_and_args(self) -> dict:
        grid = self.simulation.grid
        dist_result = self._initial_distribution
        if dist_result is None:
            dist_result = _initialize_total_distribution_(self.cfg, self.simulation)

        state: dict = {}
        for s_name, (_n, f_s, _vx, _vy) in dist_result.items():
            state[s_name] = jnp.asarray(f_s)

        # initial fields from Poisson on initial charge density
        poisson = InitialPoissonSolver(self.simulation.grid.kx, self.simulation.grid.ky)
        rho0 = jnp.zeros((grid.nx, grid.ny))
        for s_name in dist_result.keys():
            sg = self.cfg["grid"]["species_grids"][s_name]
            q = self.cfg["grid"]["species_params"][s_name]["charge"]
            n_s = jnp.sum(state[s_name], axis=(2, 3)) * (sg["dvx"] * sg["dvy"])
            rho0 = rho0 + q * n_s
        # add static ion background if single species
        ion_charge = self.cfg["grid"].get("ion_charge")
        if ion_charge is not None:
            rho0 = rho0 + jnp.asarray(ion_charge)

        ex0, ey0 = poisson(rho0)
        state["ex"] = ex0
        state["ey"] = ey0
        state["bz"] = jnp.zeros((grid.nx, grid.ny))

        # driver currents for diagnostics
        state["jx_driver"] = jnp.zeros((grid.nx, grid.ny))
        state["jy_driver"] = jnp.zeros((grid.nx, grid.ny))

        self.state = state
        self.args = {"drivers": self.simulation.drivers, "terms": self.cfg["terms"]}

    def init_diffeqsolve(self):
        self.cfg = get_save_quantities(self.cfg)
        grid = self.simulation.grid
        self.time_quantities = {"t0": 0.0, "t1": grid.tmax, "max_steps": grid.max_steps}
        self.diffeqsolve_quants = dict(
            terms=ODETerm(VlasovMaxwell(self.cfg, grid, self.simulation)),
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
            progress_meter=TqdmProgressMeter(refresh_steps=grid.max_steps // 100)
            if sys.stdout.isatty()
            else NoProgressMeter(),
        )
        return {"solver result": solver_result}
