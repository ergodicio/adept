"""ADEPT module class for Vlasov-1D2V (cylindrical velocity geometry).

Consolidated from the retired planar (vx, vy) vlasov1d2v stub (see the D0
audit in the weakly-collisional-nlepw PRL plan): the geometry-agnostic
pieces of vlasov-1d (units, drivers, x-grid, field solvers, storage layout)
are reused; the velocity space is (v_par, v_perp) cylindrical with weight
w_perp = 2*pi*v_perp*dv_perp; collisions act along v_par with
marginal-moment coefficients (the 1D-Dougherty-limit rung).
"""

from dataclasses import asdict

import numpy as np
from diffrax import ODETerm, SubSaveAt
from jax import numpy as jnp

from adept._base_ import Stepper
from adept._vlasov1d.modules import BaseVlasov1D, sim_from_config
from adept._vlasov1d2v.datamodel import Vlasov1D2VConfig
from adept._vlasov1d2v.helpers import _initialize_total_distribution_2v_, perp_grid, post_process
from adept._vlasov1d2v.solvers.vector_field import VlasovMaxwell2V
from adept._vlasov1d2v.storage import get_save_quantities


class BaseVlasov1D2V(BaseVlasov1D):
    """ADEPT module wrapper for configuring, running, and post-processing Vlasov-1D2V."""

    def __init__(self, cfg) -> None:
        """Validate configuration and construct the simulation domain."""
        # Skip BaseVlasov1D.__init__ (it validates the 1D config model); reuse
        # its grandparent initialization and the shared sim builder.
        super(BaseVlasov1D, self).__init__(cfg)
        self.config_model = Vlasov1D2VConfig.model_validate(cfg)
        self.simulation = sim_from_config(self.config_model)
        self.nvperp = self.config_model.grid.nvperp
        self.vperp_max = self.config_model.grid.vperp_max

    def post_process(self, run_output: dict, td: str):
        """Post-process a solver result into plots, netCDF files, and MLflow metrics."""
        return post_process(run_output["solver result"], self.cfg, td, self.args)

    def get_solver_quantities(self) -> dict:
        """Attach array-valued derived quantities (grids, distributions) to the config."""
        cfg_grid = self.cfg["grid"]
        grid = self.simulation.grid

        cfg_grid.update(asdict(grid))

        dist_result = _initialize_total_distribution_2v_(self.cfg, self.simulation, self.nvperp, self.vperp_max)
        cfg_grid["species_distributions"] = dist_result

        vperp, dvperp, wperp = perp_grid(self.nvperp, self.vperp_max)

        cfg_grid["species_grids"] = {}
        cfg_grid["species_params"] = {}
        n_prof_total = np.zeros([grid.nx])

        for species_name, (n_prof, f_s, v_ax, vperp_ax) in dist_result.items():
            n_prof_total += n_prof

            species_cfg = self.simulation.species_dict[species_name]
            nv = species_cfg.nv
            vmax = species_cfg.vmax
            vmin = species_cfg.vmin
            dv = (vmax - vmin) / nv

            kv = jnp.fft.fftfreq(nv, d=dv) * 2.0 * np.pi
            kvr = jnp.fft.rfftfreq(nv, d=dv) * 2.0 * np.pi
            one_over_kv = np.zeros(nv)
            one_over_kv[1:] = 1.0 / np.array(kv)[1:]
            one_over_kvr = np.zeros(len(kvr))
            one_over_kvr[1:] = 1.0 / np.array(kvr)[1:]

            cfg_grid["species_grids"][species_name] = {
                "v": jnp.array(v_ax),
                "dv": dv,
                "nv": nv,
                "vmax": vmax,
                "vmin": vmin,
                "kv": kv,
                "kvr": kvr,
                "one_over_kv": jnp.array(one_over_kv),
                "one_over_kvr": jnp.array(one_over_kvr),
                "vperp": jnp.array(vperp),
                "dvperp": dvperp,
                "nvperp": self.nvperp,
                "vperp_max": self.vperp_max,
                "wperp": jnp.array(wperp),
            }

            cfg_grid["species_params"][species_name] = {
                "charge": species_cfg.charge,
                "mass": species_cfg.mass,
                "charge_to_mass": species_cfg.charge / species_cfg.mass,
                "T0": self.simulation.species_distributions[species_name][0].T0,
            }

        cfg_grid["n_prof_total"] = n_prof_total

        has_multiple_species = len(self.simulation.species) > 1
        if has_multiple_species:
            cfg_grid["ion_charge"] = np.zeros_like(n_prof_total)
        else:
            cfg_grid["ion_charge"] = n_prof_total.copy()

        if not has_multiple_species and "electron" in cfg_grid["species_grids"]:
            cfg_grid["v"] = cfg_grid["species_grids"]["electron"]["v"]
            cfg_grid["kv"] = cfg_grid["species_grids"]["electron"]["kv"]
            cfg_grid["kvr"] = cfg_grid["species_grids"]["electron"]["kvr"]
            cfg_grid["one_over_kv"] = cfg_grid["species_grids"]["electron"]["one_over_kv"]
            cfg_grid["one_over_kvr"] = cfg_grid["species_grids"]["electron"]["one_over_kvr"]

        self.cfg["grid"] = cfg_grid

    def init_state_and_args(self) -> dict:
        """Initialize the solver state pytree and driver/term args."""
        grid = self.simulation.grid

        dist_result = _initialize_total_distribution_2v_(self.cfg, self.simulation, self.nvperp, self.vperp_max)

        state = {}
        for species_name, (n_prof, f_s, v_ax, vperp_ax) in dist_result.items():
            state[species_name] = jnp.array(f_s)

        for field_key in ["e", "de"]:
            state[field_key] = jnp.zeros(grid.nx)
        for field_key in ["a", "da", "prev_a"]:
            state[field_key] = jnp.zeros(grid.nx + 2)

        # Diagnostics are MARGINAL (nx, nv) accumulators holding the running time
        # integral of each term's contribution — see solvers.vector_field for why
        # they are integrated rather than sampled. Start from zero; post-processing
        # differences them to get exact interval-averaged rates.
        ref_species = "electron" if "electron" in dist_result else next(iter(dist_result.keys()))
        nv = dist_result[ref_species][1].shape[1]
        for k in ["diag-vlasov-cumulative", "diag-fp-cumulative"]:
            if self.cfg["diagnostics"][k]:
                state[k] = jnp.zeros((grid.nx, nv))

        self.state = state
        self.args = {"drivers": self.simulation.drivers, "terms": self.cfg["terms"]}

    def init_diffeqsolve(self):
        """Assemble Diffrax terms, solver, save functions, and solve time bounds."""
        self.cfg = get_save_quantities(self.cfg)
        grid = self.simulation.grid
        self.time_quantities = {"t0": 0.0, "t1": grid.tmax, "max_steps": grid.max_steps}
        self.diffeqsolve_quants = {
            "terms": ODETerm(
                VlasovMaxwell2V(
                    self.cfg,
                    grid,
                    self.simulation.drivers,
                    nu_fp_prof=self.simulation.nu_fp_prof,
                    nu_K_prof=self.simulation.nu_K_prof,
                )
            ),
            "solver": Stepper(),
            "saveat": {"subs": {k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in self.cfg["save"].items()}},
        }
