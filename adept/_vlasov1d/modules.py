#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import numpy as np
import pint
from diffrax import ODETerm, SaveAt, SubSaveAt, diffeqsolve
from jax import numpy as jnp

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._vlasov1d.helpers import _initialize_total_distribution_, post_process
from adept._vlasov1d.solvers.vector_field import VlasovMaxwell
from adept._vlasov1d.storage import get_save_quantities


class BaseVlasov1D(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.ureg = pint.UnitRegistry()

    def post_process(self, run_output: dict, td: str):
        return post_process(run_output["solver result"], self.cfg, td, self.args)

    def write_units(self) -> dict:
        _Q = self.ureg.Quantity

        n0 = _Q(self.cfg["units"]["normalizing_density"]).to("1/cc")
        T0 = _Q(self.cfg["units"]["normalizing_temperature"]).to("eV")

        wp0 = np.sqrt(n0 * self.ureg.e**2.0 / (self.ureg.m_e * self.ureg.epsilon_0)).to("rad/s")
        tp0 = (1 / wp0).to("fs")

        v0 = np.sqrt(2.0 * T0 / self.ureg.m_e).to("m/s")
        x0 = (v0 / wp0).to("nm")
        c_light = _Q(1.0 * self.ureg.c).to("m/s") / v0
        beta = (v0 / self.ureg.c).to("dimensionless")

        box_length = ((self.cfg["grid"]["xmax"] - self.cfg["grid"]["xmin"]) * x0).to("microns")
        if "ymax" in self.cfg["grid"].keys():
            box_width = ((self.cfg["grid"]["ymax"] - self.cfg["grid"]["ymin"]) * x0).to("microns")
        else:
            box_width = "inf"
        sim_duration = (self.cfg["grid"]["tmax"] * tp0).to("ps")

        # collisions
        logLambda_ee = 23.5 - np.log(n0.magnitude**0.5 / T0.magnitude**-1.25)
        logLambda_ee -= (1e-5 + (np.log(T0.magnitude) - 2) ** 2.0 / 16) ** 0.5
        nuee = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0.magnitude**1.5, "Hz")
        nuee_norm = nuee / wp0

        all_quantities = {
            "wp0": wp0,
            "tp0": tp0,
            "n0": n0,
            "v0": v0,
            "T0": T0,
            "c_light": c_light,
            "beta": beta,
            "x0": x0,
            "nuee": nuee,
            "logLambda_ee": logLambda_ee,
            "box_length": box_length,
            "box_width": box_width,
            "sim_duration": sim_duration,
        }

        self.cfg["units"]["derived"] = all_quantities

        self.cfg["grid"]["beta"] = beta.magnitude

        return all_quantities

    def get_derived_quantities(self):
        """
        This function just updates the config with the derived quantities that are only integers or strings.

        This is run prior to the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = self.cfg["grid"]

        cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]

        # Normalize species config: if not provided, generate a default electron species
        if self.cfg["terms"].get("species", None) is None:
            # Collect all density components (keys starting with "species-")
            density_components = [
                name for name in self.cfg["density"].keys()
                if name.startswith("species-")
            ]
            if not density_components:
                raise ValueError("No density components found (expected keys starting with 'species-')")

            # Generate default electron species config
            self.cfg["terms"]["species"] = [{
                "name": "electron",
                "charge": -1.0,
                "mass": 1.0,
                "vmax": cfg_grid["vmax"],
                "nv": cfg_grid["nv"],
                "density_components": density_components,
            }]

        if len(self.cfg["drivers"]["ey"].keys()) > 0:
            print("overriding dt to ensure wave solver stability")
            cfg_grid["dt"] = float(0.95 * cfg_grid["dx"] / self.cfg["units"]["derived"]["c_light"].magnitude)

        cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)

        if cfg_grid["nt"] > 1e6:
            cfg_grid["max_steps"] = int(1e6)
            print(r"Only running $10^6$ steps")
        else:
            cfg_grid["max_steps"] = cfg_grid["nt"] + 4

        cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]
        self.cfg["grid"] = cfg_grid

    def get_solver_quantities(self) -> dict:
        """
        This function just updates the config with the derived quantities that are arrays

        This is run after the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = self.cfg["grid"]

        cfg_grid = {
            **cfg_grid,
            **{
                "x": jnp.linspace(
                    cfg_grid["xmin"] + cfg_grid["dx"] / 2, cfg_grid["xmax"] - cfg_grid["dx"] / 2, cfg_grid["nx"]
                ),
                "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
                "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
                "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            },
        }

        # config axes
        one_over_kx = np.zeros_like(cfg_grid["kx"])
        one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
        cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

        one_over_kxr = np.zeros_like(cfg_grid["kxr"])
        one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
        cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

        cfg_grid["nuprof"] = 1.0
        # get_profile_with_mask(cfg["nu"]["time-profile"], t, cfg["nu"]["time-profile"]["bump_or_trough"])
        cfg_grid["ktprof"] = 1.0
        # get_profile_with_mask(cfg["krook"]["time-profile"], t, cfg["krook"]["time-profile"]["bump_or_trough"])

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
            species_cfg = next(
                (s for s in self.cfg["terms"]["species"] if s["name"] == species_name), None
            )
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

        cfg_grid["kprof"] = np.ones_like(cfg_grid["n_prof_total"])
        # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

        cfg_grid["x_a"] = np.concatenate(
            [
                [cfg_grid["x"][0] - cfg_grid["dx"]],
                cfg_grid["x"],
                [cfg_grid["x"][-1] + cfg_grid["dx"]],
            ]
        )

        self.cfg["grid"] = cfg_grid

    def init_state_and_args(self) -> dict:
        """
        This function initializes the state

        :param cfg:
        :return:
        """
        # Initialize distributions (always returns dict format)
        dist_result = _initialize_total_distribution_(self.cfg, self.cfg["grid"])

        state = {}

        # Build state dict with all species distributions
        for species_name, (n_prof, f_s, v_ax) in dist_result.items():
            state[species_name] = jnp.array(f_s)

        # Reference distribution for diagnostics (use first species)
        # TODO(a-6c41): Store species distributions separately for multi-species diagnostics
        first_species_name = list(dist_result.keys())[0]
        f_ref = dist_result[first_species_name][1]

        # Field quantities (same for all modes)
        for field in ["e", "de"]:
            state[field] = jnp.zeros(self.cfg["grid"]["nx"])

        for field in ["a", "da", "prev_a"]:
            state[field] = jnp.zeros(self.cfg["grid"]["nx"] + 2)  # need boundary cells

        # Diagnostics (use reference distribution shape)
        for k in ["diag-vlasov-dfdt", "diag-fp-dfdt"]:
            if self.cfg["diagnostics"][k]:
                state[k] = jnp.zeros_like(f_ref)

        self.state = state
        self.args = {"drivers": self.cfg["drivers"], "terms": self.cfg["terms"]}

    def init_diffeqsolve(self):
        self.cfg = get_save_quantities(self.cfg)
        self.time_quantities = {"t0": 0.0, "t1": self.cfg["grid"]["tmax"], "max_steps": self.cfg["grid"]["max_steps"]}
        self.diffeqsolve_quants = dict(
            terms=ODETerm(VlasovMaxwell(self.cfg)),
            solver=Stepper(),
            saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in self.cfg["save"].items()}),
        )

    def __call__(self, trainable_modules: dict, args: dict | None = None):
        if args is None:
            args = self.args
        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.time_quantities["t0"],
            t1=self.time_quantities["t1"],
            max_steps=self.cfg["grid"]["max_steps"],
            dt0=self.cfg["grid"]["dt"],
            y0=self.state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return {"solver result": solver_result}
