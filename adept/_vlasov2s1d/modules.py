#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import numpy as np
import pint
from diffrax import ODETerm, SaveAt, SubSaveAt, diffeqsolve
from jax import numpy as jnp

from adept import ADEPTModule
from adept._base_ import Stepper
from adept._vlasov2s1d.helpers import _initialize_total_distribution_, post_process
from adept._vlasov2s1d.storage import get_save_quantities


class BaseVlasov2S1D(ADEPTModule):
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
        me_over_mi = 1.0 / (self.cfg["units"]["A"] * 1836.0)  # electron to ion mass ratio

        # Get ion temperature from species-ion config if it exists
        ion_T0 = 0.1  # Default ion temperature
        for name, species_params in self.cfg["density"].items():
            if name.startswith("species-") and ("ion" in name.lower() or name == "species-ion"):
                ion_T0 = species_params.get("T0", 0.1)
                break

        cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
        cfg_grid["dv"] = 2.0 * cfg_grid["vmax"] / cfg_grid["nv"]

        cfg_grid["vmax_i"] = cfg_grid["vmax"] * np.sqrt(ion_T0 * me_over_mi)
        cfg_grid["dv_i"] = 2.0 * cfg_grid["vmax_i"] / cfg_grid["nv"]

        if len(self.cfg["drivers"]["ey"].keys()) > 0:
            print("overriding dt to ensure wave solver stability")
            cfg_grid["dt"] = 0.95 * cfg_grid["dx"] / self.cfg["units"]["derived"]["c_light"]

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
                "v": jnp.linspace(
                    -cfg_grid["vmax"] + cfg_grid["dv"] / 2, cfg_grid["vmax"] - cfg_grid["dv"] / 2, cfg_grid["nv"]
                ),
                "v_i": jnp.linspace(
                    -cfg_grid["vmax_i"] + cfg_grid["dv_i"] / 2,
                    cfg_grid["vmax_i"] - cfg_grid["dv_i"] / 2,
                    cfg_grid["nv"],
                ),
                "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
                "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
                "kv": jnp.fft.fftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
                "kvr": jnp.fft.rfftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
                "kv_i": jnp.fft.fftfreq(cfg_grid["nv"], d=cfg_grid["dv_i"]) * 2.0 * np.pi,
                "kvr_i": jnp.fft.rfftfreq(cfg_grid["nv"], d=cfg_grid["dv_i"]) * 2.0 * np.pi,
            },
        }

        # config axes
        one_over_kx = np.zeros_like(cfg_grid["kx"])
        one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
        cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

        one_over_kxr = np.zeros_like(cfg_grid["kxr"])
        one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
        cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

        # velocity axes
        one_over_kv = np.zeros_like(cfg_grid["kv"])
        one_over_kv[1:] = 1.0 / cfg_grid["kv"][1:]
        cfg_grid["one_over_kv"] = jnp.array(one_over_kv)

        one_over_kvr = np.zeros_like(cfg_grid["kvr"])
        one_over_kvr[1:] = 1.0 / cfg_grid["kvr"][1:]
        cfg_grid["one_over_kvr"] = jnp.array(one_over_kvr)

        one_over_kv_i = np.zeros_like(cfg_grid["kv_i"])
        one_over_kv_i[1:] = 1.0 / cfg_grid["kv_i"][1:]
        cfg_grid["one_over_kv_i"] = jnp.array(one_over_kv_i)

        one_over_kvr_i = np.zeros_like(cfg_grid["kvr_i"])
        one_over_kvr_i[1:] = 1.0 / cfg_grid["kvr_i"][1:]
        cfg_grid["one_over_kvr_i"] = jnp.array(one_over_kvr_i)

        cfg_grid["nuprof"] = 1.0
        # get_profile_with_mask(cfg["nu"]["time-profile"], t, cfg["nu"]["time-profile"]["bump_or_trough"])
        cfg_grid["ktprof"] = 1.0
        # get_profile_with_mask(cfg["krook"]["time-profile"], t, cfg["krook"]["time-profile"]["bump_or_trough"])
        # cfg_grid["n_prof_total"], cfg_grid["starting_f"] = _initialize_total_distribution_(self.cfg, cfg_grid)

        cfg_grid["kprof"] = np.ones(cfg_grid["nx"])
        # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

        # cfg_grid["ion_charge"] = np.zeros_like(cfg_grid["n_prof_total"]) + cfg_grid["n_prof_total"]

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
        This function initializes the state for 2-species solver

        :param cfg:
        :return:
        """
        from adept._vlasov2s1d.helpers import _initialize_distribution_

        # n_prof_total, f_e = _initialize_total_distribution_(self.cfg, self.cfg["grid"])
        f_e, _ = _initialize_distribution_(
            nx=int(self.cfg["grid"]["nx"]),
            nv=int(self.cfg["grid"]["nv"]),
            v0=0.0,
            m=2.0,
            T0=1.0,  # Use ion temperature directly
            vmax=self.cfg["grid"]["vmax"],
            mass=1.0,  # Convert to kg
        )
        # Initialize ion distribution properly using the same framework as electrons
        # Use the same distribution function framework for ions

        # Get ion parameters from config or use defaults
        ion_v0 = 0.0  # Default ion drift velocity
        ion_m = 2.0  # Default shape parameter (Maxwellian)
        ion_T0 = 0.1  # Default ion temperature (cold ions)
        # ion_noise_val = 0.0
        # ion_noise_seed = 42

        # Check if there are ion-specific parameters in the config
        ion_species_found = False
        for name, species_params in self.cfg["density"].items():
            if name.startswith("species-") and ("ion" in name.lower() or name == "species-ion"):
                ion_v0 = species_params.get("v0", 0.0)
                ion_m = species_params.get("m", 2.0)
                ion_T0 = species_params.get("T0", 0.1)  # Use T0 directly from config
                # ion_noise_val = species_params.get("noise_val", 0.0)
                # ion_noise_seed = int(species_params.get("noise_seed", 42))
                ion_species_found = True
                break

        if not ion_species_found:
            print("No explicit ion species found in config, using default ion parameters")

        # Create ion distribution using the proper framework
        f_i, _ = _initialize_distribution_(
            nx=int(self.cfg["grid"]["nx"]),
            nv=int(self.cfg["grid"]["nv"]),
            v0=ion_v0,
            m=ion_m,
            T0=ion_T0,  # Use ion temperature directly
            vmax=self.cfg["grid"]["vmax_i"],
            mass=self.cfg["units"]["A"] * 1836.0,  # Convert to kg
            # n_prof=n_prof_total,
            # noise_val=ion_noise_val,
            # noise_seed=ion_noise_seed,
            # noise_type="gaussian",
        )

        state = {}
        state["electron"] = f_e
        state["ion"] = f_i

        for field in ["e", "de"]:
            state[field] = jnp.zeros(self.cfg["grid"]["nx"])

        for field in ["a", "da", "prev_a"]:
            state[field] = jnp.zeros(self.cfg["grid"]["nx"] + 2)  # need boundary cells

        self.state = state
        self.args = {"drivers": self.cfg["drivers"], "terms": self.cfg["terms"]}

    def init_diffeqsolve(self):
        self.cfg = get_save_quantities(self.cfg)
        self.time_quantities = {"t0": 0.0, "t1": self.cfg["grid"]["tmax"], "max_steps": self.cfg["grid"]["max_steps"]}
        from adept._vlasov2s1d.solvers.vector_field import Vlasov2SMaxwell

        self.diffeqsolve_quants = dict(
            terms=ODETerm(Vlasov2SMaxwell(self.cfg)),
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
