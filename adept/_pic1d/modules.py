"""PIC-1D ADEPT module.

Implements the same lifecycle hooks as :class:`adept._vlasov1d.modules.BaseVlasov1D`
so the ``ergoExo`` harness drives it identically. The only solver differences
versus the Vlasov module are:

- particles instead of f(x, v) — state arrays are particle (x, v, w) per species
- charge density is deposited from particles using a configurable B-spline shape
- no Fokker-Planck, Krook, or transverse light-wave physics
"""

import sys
from dataclasses import asdict

import numpy as np
from diffrax import NoProgressMeter, ODETerm, SaveAt, SubSaveAt, TqdmProgressMeter, diffeqsolve
from jax import numpy as jnp

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._pic1d.datamodel import PIC1DConfig
from adept._pic1d.helpers import _initialize_particles_, post_process
from adept._pic1d.simulation import sim_from_config
from adept._pic1d.solvers.vector_field import PIC1DVectorField
from adept._pic1d.storage import get_save_quantities
from adept.utils import filter_scalars


class BasePIC1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.config_model = PIC1DConfig.model_validate(cfg)
        self.simulation = sim_from_config(self.config_model)

    def post_process(self, run_output: dict, td: str):
        return post_process(run_output["solver result"], self.cfg, td, self.args)

    def write_units(self) -> dict:
        norm = self.simulation.plasma_norm
        grid = self.simulation.grid

        box_length = ((grid.xmax - grid.xmin) * norm.L0).to("microns")
        sim_duration = (grid.tmax * norm.tau).to("ps")

        all_quantities = {
            "wp0": (1 / norm.tau).to("rad/s"),
            "tp0": norm.tau.to("fs"),
            "n0": norm.n0.to("1/cc"),
            "v0": norm.v0.to("m/s"),
            "T0": norm.T0.to("eV"),
            "x0": norm.L0.to("nm"),
            "c_light": norm.speed_of_light_norm(),
            "box_length": box_length,
            "sim_duration": sim_duration,
            "ppc": self.simulation.ppc,
            "particle_shape": self.simulation.particle_shape,
        }
        self.cfg["units"]["derived"] = all_quantities
        # Speed of light is irrelevant for pure ES-PIC, but enters the wave
        # equation when transverse EM drivers (``drivers.ey``) are configured.
        has_ey = len(self.cfg["drivers"].get("ey", {})) > 0
        self.cfg["grid"]["beta"] = 1.0 / norm.speed_of_light_norm() if has_ey else 1.0
        return all_quantities

    def get_derived_quantities(self):
        cfg_grid = self.cfg["grid"]
        grid = self.simulation.grid
        cfg_grid.update(filter_scalars(asdict(grid)))

        # Default save tmin/tmax to grid bounds.
        for save_val in self.cfg.get("save", {}).values():
            if "t" in save_val:
                save_val["t"].setdefault("tmin", grid.tmin)
                save_val["t"].setdefault("tmax", grid.tmax)
            else:
                # Nested species save: {label: {t: {...}}}
                for label_config in save_val.values():
                    if isinstance(label_config, dict) and "t" in label_config:
                        label_config["t"].setdefault("tmin", grid.tmin)
                        label_config["t"].setdefault("tmax", grid.tmax)

        # PIC-specific grid-level params (carry through so storage can read them).
        cfg_grid["ppc"] = int(self.simulation.ppc)
        cfg_grid["particle_shape"] = self.simulation.particle_shape
        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self) -> dict:
        cfg_grid = self.cfg["grid"]
        grid = self.simulation.grid
        cfg_grid.update(asdict(grid))

        loaded = _initialize_particles_(self.cfg, self.simulation)

        # Per-species (charge, mass, charge/mass) — same shape as Vlasov-1D.
        species_params = {}
        n_prof_total = np.zeros(grid.nx)
        for species in self.simulation.species:
            n_prof_total += loaded[species.name][3]
            species_params[species.name] = {
                "charge": species.charge,
                "mass": species.mass,
                "charge_to_mass": species.charge / species.mass,
            }
        cfg_grid["species_params"] = species_params
        cfg_grid["n_prof_total"] = n_prof_total

        # Static neutralizing background.
        # If ``density.quasineutrality`` is true (the default for these
        # configs), we balance the sum of the loaded species charge densities
        # so the initial total charge density is zero by construction. This
        # works for an arbitrary mix of species and reduces to "ion_charge =
        # n_e0" for the single-electron case.
        if self.cfg["density"].get("quasineutrality", True):
            ion_charge = np.zeros_like(n_prof_total)
            for species in self.simulation.species:
                ion_charge -= species.charge * loaded[species.name][3]
            cfg_grid["ion_charge"] = ion_charge
        else:
            cfg_grid["ion_charge"] = np.zeros_like(n_prof_total)

        # Stash initial particle arrays for init_state_and_args.
        cfg_grid["_initial_particles"] = {
            name: {"x": jnp.array(x), "v": jnp.array(v), "w": jnp.array(w)} for name, (x, v, w, _, _) in loaded.items()
        }
        # Synthetic velocity axis per species — for downstream plot helpers
        # only; not used by the solver itself.
        cfg_grid["species_v_axes"] = {name: jnp.array(vax) for name, (_, _, _, _, vax) in loaded.items()}
        self.cfg["grid"] = cfg_grid

    def init_state_and_args(self) -> dict:
        grid = self.simulation.grid
        state = {}
        for name, p in self.cfg["grid"]["_initial_particles"].items():
            state[f"x_{name}"] = p["x"]
            state[f"v_{name}"] = p["v"]
            state[f"w_{name}"] = p["w"]
        state["e"] = jnp.zeros(grid.nx)
        state["de"] = jnp.zeros(grid.nx)
        # Transverse vector potential and its wave-solver shadow / driver
        # source. Sized (nx+2,) to carry the absorbing boundary ghost cells,
        # matching Vlasov-1D.
        for field_name in ("a", "prev_a", "da"):
            state[field_name] = jnp.zeros(grid.nx + 2)
        self.state = state
        self.args = {"drivers": self.simulation.drivers, "terms": self.cfg["terms"]}

    def init_diffeqsolve(self):
        self.cfg = get_save_quantities(self.cfg)
        grid = self.simulation.grid
        self.time_quantities = {"t0": 0.0, "t1": grid.tmax, "max_steps": grid.max_steps}
        self.diffeqsolve_quants = dict(
            terms=ODETerm(PIC1DVectorField(self.cfg, grid, self.simulation.drivers)),
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
