"""Shared, logging-free host preparation for the Vlasov-1D solver."""

from dataclasses import asdict

import numpy as np
from jax import numpy as jnp

from adept._vlasov1d.datamodel import SpeciesConfig, Vlasov1DConfig
from adept._vlasov1d.grid import Grid
from adept._vlasov1d.helpers import _initialize_total_distribution_
from adept._vlasov1d.simulation import EMDriverSet, Species, SubspeciesDistributionSpec, Vlasov1DSimulation
from adept._vlasov1d.solvers.vector_field import VlasovMaxwell
from adept._vlasov1d.storage import get_save_quantities
from adept.functions import SpaceTimeEnvelopeConfig, SpaceTimeEnvelopeFunction


def species_set_from_config(cfg: Vlasov1DConfig) -> list[Species]:
    """Return explicit species or synthesize the legacy single-electron species."""
    if cfg.terms.species:
        return [Species.from_config(species_cfg) for species_cfg in cfg.terms.species]
    else:
        # Collect all density components (keys starting with "species-")
        density_components = [name for name in cfg.density.model_extra.keys() if name.startswith("species-")]
        if not density_components:
            raise ValueError("No density components found (expected keys starting with 'species-')")

        return [
            Species.from_config(
                SpeciesConfig(
                    name="electron",
                    charge=-1.0,
                    mass=1.0,
                    vmax=cfg.grid.vmax,
                    vmin=cfg.grid.vmin,
                    nv=cfg.grid.nv,
                    density_components=density_components,
                )
            )
        ]


def sim_from_config(
    cfg: Vlasov1DConfig,
) -> Vlasov1DSimulation:
    """Construct a Vlasov1DSimulation from a Vlasov1DConfig."""
    plasma_norm = cfg.units.make_normalization()
    beta = 1.0 / plasma_norm.speed_of_light_norm()
    has_ey_driver = len(cfg.drivers.ey) > 0
    grid = Grid.from_config(cfg.grid, beta, should_override_dt_for_em_waves=has_ey_driver, norm=plasma_norm)

    # Construct collision frequency profiles if enabled
    nu_fp_prof = None
    if cfg.terms.fokker_planck.is_on:
        st_cfg = SpaceTimeEnvelopeConfig(time=cfg.terms.fokker_planck.time, space=cfg.terms.fokker_planck.space)
        nu_fp_prof = SpaceTimeEnvelopeFunction.from_config(st_cfg, norm=plasma_norm)

    nu_K_prof = None
    if cfg.terms.krook.is_on:
        st_cfg = SpaceTimeEnvelopeConfig(time=cfg.terms.krook.time, space=cfg.terms.krook.space)
        nu_K_prof = SpaceTimeEnvelopeFunction.from_config(st_cfg, norm=plasma_norm)

    species = species_set_from_config(cfg)
    species_distribution_specs = {
        s.name: [
            SubspeciesDistributionSpec.from_config(cfg.density.get_component(component_name), norm=plasma_norm)
            for component_name in s.density_components
        ]
        for s in species
    }
    drivers = EMDriverSet.from_config(cfg.drivers, norm=plasma_norm, grid=grid)

    return Vlasov1DSimulation(
        plasma_norm,
        grid,
        species,
        species_distribution_specs,
        drivers,
        nu_fp_prof,
        nu_K_prof,
    )


class Vlasov1DSetup:
    """Prepare the shared domain, initial state, numerical map, and observations."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.state = None
        self.args = None
        # The historical config schema includes experiment metadata, but numerical
        # preparation does not require a tracking configuration or modify it.
        validation_cfg = {"mlflow": {"experiment": "", "run": ""}, **cfg}
        self.config_model = Vlasov1DConfig.model_validate(validation_cfg)
        self.simulation = sim_from_config(self.config_model)

    def write_units(self) -> dict:
        """Compute and attach physical normalization quantities to the run config."""
        norm = self.simulation.plasma_norm
        grid = self.simulation.grid

        box_length = ((grid.xmax - grid.xmin) * norm.L0).to("microns")
        if "ymax" in self.cfg["grid"].keys():
            box_width = ((self.cfg["grid"]["ymax"] - self.cfg["grid"]["ymin"]) * norm.L0).to("microns")
        else:
            box_width = "inf"
        sim_duration = (grid.tmax * norm.tau).to("ps")

        beta = 1.0 / norm.speed_of_light_norm()

        # wp0/tp0/v0/x0 are the plasma frequency, thermal speed, and Debye length
        # of the reference species selected by units.reference (electron or ion).
        all_quantities = {
            "wp0": (1 / norm.tau).to("rad/s"),
            "tp0": norm.tau.to("fs"),
            "n0": norm.n0.to("1/cc"),
            "v0": norm.v0.to("m/s"),
            "T0": norm.T0.to("eV"),
            "c_light": norm.speed_of_light_norm(),
            "beta": beta,
            "x0": norm.L0.to("nm"),
            "box_length": box_length,
            "box_width": box_width,
            "sim_duration": sim_duration,
        }

        if self.config_model.units.reference == "ion":
            nu_ii = norm.approximate_ii_collision_frequency()
            all_quantities["reference_species"] = "ion"
            all_quantities["nuii"] = nu_ii.to("Hz")
            # i-i collision rate in code units (1/wpi) — the ion analogue of nuee_norm.
            all_quantities["nuii_norm"] = (nu_ii * norm.tau).to("").magnitude
            all_quantities["logLambda_ii"] = norm.logLambda_ii()
        else:
            nu_ee = norm.approximate_ee_collision_frequency()
            all_quantities["nuee"] = nu_ee.to("Hz")
            # e-e collision rate in code units (1/wp0). This is what the FP/Krook
            # `baseline` rates should be compared against.
            all_quantities["nuee_norm"] = (nu_ee * norm.tau).to("").magnitude
            all_quantities["logLambda_ee"] = norm.logLambda_ee()

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
        cfg_grid.update({name: value for name, value in asdict(grid).items() if np.isscalar(value)})

        # Default save.*.t.tmin/tmax to computed grid values.
        # Species saves are nested ({label: {t: ...}}); fields/diags are flat ({t: ...}).
        def default_time_bounds(time_config):
            for name, value in (("tmin", grid.tmin), ("tmax", grid.tmax)):
                if time_config.get(name) is None:
                    time_config[name] = value

        for save_val in self.cfg.get("save", {}).values():
            if "t" in save_val:
                # Flat save (fields, diags)
                default_time_bounds(save_val["t"])
            else:
                # Nested species save: {label: {t: {...}, ...}}
                for label_config in save_val.values():
                    if isinstance(label_config, dict) and "t" in label_config:
                        default_time_bounds(label_config["t"])

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
        dist_result = _initialize_total_distribution_(self.cfg, self.simulation)
        cfg_grid["species_distributions"] = dist_result

        # Build species_grids and species_params
        cfg_grid["species_grids"] = {}
        cfg_grid["species_params"] = {}
        n_prof_total = np.zeros([grid.nx])

        for species_name, (n_prof, f_s, v_ax) in dist_result.items():
            n_prof_total += n_prof

            # Find the species config (always exists due to normalization in get_derived_quantities)
            species_cfg = self.simulation.species_dict[species_name]
            if species_cfg is None:
                raise ValueError(f"Species '{species_name}' not found in config['terms']['species']")

            nv = species_cfg.nv
            vmax = species_cfg.vmax
            vmin = species_cfg.vmin

            dv = (vmax - vmin) / nv

            # Build velocity grid parameters for this species
            cfg_grid["species_grids"][species_name] = {
                "v": jnp.array(v_ax),
                "dv": dv,
                "nv": nv,
                "vmax": vmax,
                "vmin": vmin,
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
            # T0 is the bulk (first-listed component) temperature, used e.g. by the Krook target
            cfg_grid["species_params"][species_name] = {
                "charge": species_cfg.charge,
                "mass": species_cfg.mass,
                "charge_to_mass": species_cfg.charge / species_cfg.mass,
                "T0": self.simulation.species_distributions[species_name][0].T0,
            }

        cfg_grid["n_prof_total"] = n_prof_total

        # Quasineutrality handling
        # For single-species electron-only sims, assume static ion background
        # For multi-species, quasineutrality is handled by the species themselves
        has_multiple_species = len(self.simulation.species) > 1
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
        dist_result = _initialize_total_distribution_(self.cfg, self.simulation)

        state = {}

        # Build state dict with all species distributions
        for species_name, (n_prof, f_s, v_ax) in dist_result.items():
            state[species_name] = jnp.array(f_s)

        # Reference distribution for diagnostics — must match the reference species
        # used by VlasovPoissonFokkerPlanck for the dfdt diagnostics and the electron
        # grid used by the diag save machinery in storage.py.
        # TODO(gh-174): Store species distributions separately for multi-species diagnostics
        ref_species = "electron" if "electron" in dist_result else next(iter(dist_result.keys()))
        f_ref = dist_result[ref_species][1]

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
        self.args = {"drivers": self.simulation.drivers, "terms": self.cfg["terms"]}

    def prepare_step(self) -> VlasovMaxwell:
        """Construct the numerical timestep without choosing an execution backend."""
        return VlasovMaxwell(
            self.cfg,
            self.simulation.grid,
            self.simulation.drivers,
            nu_fp_prof=self.simulation.nu_fp_prof,
            nu_K_prof=self.simulation.nu_K_prof,
        )

    def prepare_save_quantities(self) -> dict:
        """Resolve observation schedules and callbacks without constructing a solve."""
        self.cfg = get_save_quantities(self.cfg)
        return self.cfg["save"]
