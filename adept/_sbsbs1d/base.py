from typing import Dict
import os

from astropy import constants as const
from astropy.units import Quantity as _Q
import numpy as np
from jax import numpy as jnp
from diffrax import diffeqsolve, SaveAt, ODETerm
import xarray as xr
import matplotlib.pyplot as plt

from adept import ADEPTModule
from adept.vfp1d.helpers import calc_logLambda
from adept._base_ import Stepper
from adept._sbsbs1d.vectorfield import SBSVectorField, CBETVectorField


class BaseSteadyStateBackwardStimulatedBrilloiunScattering(ADEPTModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def write_units(self) -> Dict:
        """ """
        m_e = const.m_e
        e = const.e
        c = const.c
        eps_0 = const.eps0
        Zeff = self.cfg["profiles"]["Zeff"]

        mi = self.cfg["units"]["atomic mass number"] * const.m_p

        lambda0 = _Q(self.cfg["units"]["laser_wavelength"]).to("um")
        T0 = _Q(self.cfg["units"]["reference temperature"]).to("keV")
        vth0 = np.sqrt(2 * T0 / m_e).to("m/s")
        w0 = (2 * np.pi * c / lambda0).to("Hz")
        cs0 = np.sqrt(T0 / mi).to("m/s")
        I0 = _Q(self.cfg["units"]["laser_intensity"]).to("W/cm^2")
        a0 = 0.86 * np.sqrt(I0.to("W/cm^2").value / 1e18) * lambda0.to("um").value
        kz0 = 2 * np.pi / lambda0
        omegabeat0 = kz0 * cs0
        if self.cfg["units"]["reference density"] == "nc":
            n0 = (2 * np.pi * c / self.cfg["units"]["laser_wavelength"]) ** 2.0 * m_e * eps_0 / e.si**2.0
            nc_over_n0 = 1.0
        else:
            n0 = _Q(self.cfg["units"]["reference density"]).to("cm-3")
            nc_over_n0 = (2 * np.pi * c / self.cfg["units"]["laser_wavelength"]) ** 2.0 * m_e * eps_0 / e.si**2.0 / n0

        logLambda_ei, logLambda_ee = calc_logLambda(self.cfg, n0, T0, Zeff)
        # logLambda_ee = logLambda_ei
        ni0 = n0 / Zeff

        nuei_epphaines = (
            1
            / (
                0.75
                * np.sqrt(const.m_e)
                * T0**1.5
                / (np.sqrt(2 * np.pi) * ni0 * Zeff**2.0 * const.e.gauss**4.0 * logLambda_ei)
            )
        ).to("Hz")

        self.cfg["units"]["derived"] = {
            "n0": n0,
            "lambda0": lambda0,
            "T0": T0,
            "vth0": vth0,
            "w0": w0,
            "kz0": kz0,
            "flow0": cs0,
            "cs0": cs0,
            "I0": I0,
            "a0": a0,
            "mi": mi,
            "nuei_epphaines0": nuei_epphaines,
            "logLambda_ee0": logLambda_ee,
            "logLambda_ei0": logLambda_ei,
            "nc_over_n0": nc_over_n0,
            "omegabeat0": omegabeat0,
            "c": c,
            "ratio_M_m": mi / m_e,
        }

        return {k: str(v) for k, v in self.cfg["units"]["derived"].items()}

    def get_solver_quantities(self):
        cfg_grid = self.cfg["grid"]

        cfg_grid = {
            **cfg_grid,
            **{
                "z": np.linspace(
                    cfg_grid["zmin"] + cfg_grid["dz"] / 2,
                    cfg_grid["zmax"] - cfg_grid["dz"] / 2,
                    cfg_grid["nz"],
                ),
            },
        }

        self.cfg["grid"] = cfg_grid

    def get_derived_quantities(self) -> Dict:
        """
        This function just updates the config with the derived quantities that are only integers or strings.

        This is run prior to the log params step

        :param cfg_grid:
        :return:
        """
        cfg_grid = self.cfg["grid"]

        Lgrid = _Q(cfg_grid["zmax"]).to("um").value

        cfg_grid["zmax"] = Lgrid
        cfg_grid["zmin"] = 0.0
        cfg_grid["dz"] = _Q(cfg_grid["dz"]).to("um").value

        cfg_grid["nz"] = int(cfg_grid["zmax"] / cfg_grid["dz"]) + 1

        if cfg_grid["nz"] > 1e6:
            cfg_grid["max_steps"] = int(1e6)
            print(r"Only running $10^6$ steps")
        else:
            cfg_grid["max_steps"] = cfg_grid["nz"] + 4

        self.cfg["grid"] = cfg_grid

        return self.cfg

    def init_state_and_args(self):
        if "nc" in self.cfg["profiles"]["n"]["min"]:
            nmin = (
                float(self.cfg["profiles"]["n"]["min"].strip("nc")) * self.cfg["units"]["derived"]["nc_over_n0"]
                # * self.cfg["units"]["derived"]["n0"].value
            )
            nmax = (
                float(self.cfg["profiles"]["n"]["max"].strip("nc")) * self.cfg["units"]["derived"]["nc_over_n0"]
                # * self.cfg["units"]["derived"]["n0"].value
            )
        else:
            nmin = (_Q(self.cfg["profiles"]["n"]["min"]) / self.cfg["units"]["derived"]["n0"]).to("").value
            nmax = (_Q(self.cfg["profiles"]["n"]["max"]) / self.cfg["units"]["derived"]["n0"]).to("").value

        nprof = nmin + (nmax - nmin) * self.cfg["grid"]["z"] / self.cfg["grid"]["zmax"]
        Teprof = np.ones_like(nprof) * _Q(self.cfg["profiles"]["Te"]).to("keV").value
        Tiprof = np.ones_like(nprof) * _Q(self.cfg["profiles"]["Ti"]).to("keV").value
        Zeffprof = np.ones_like(nprof) * self.cfg["profiles"]["Zeff"]
        flowprof = np.ones_like(nprof) * _Q(self.cfg["profiles"]["flow"]).to("um/s").value
        omega_beat_prof = np.ones_like(nprof) * (
            self.cfg["units"]["derived"]["w0"].to("Hz").value
            - (
                (2 * np.pi * self.cfg["units"]["derived"]["c"])
                / (self.cfg["units"]["derived"]["lambda0"] + _Q(self.cfg["profiles"]["lambdabeat"]))
            )
            .to("Hz")
            .value
        )
        omegabeat = omega_beat_prof  # / self.cfg["units"]["derived"]["omegabeat0"].to("Hz").value

        self.args = {
            "n_over_n0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], nprof),
            "Te_over_T0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], Teprof),
            "Ti_over_T0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], Tiprof),
            "Zeff": lambda z: jnp.interp(z, self.cfg["grid"]["z"], Zeffprof),
            "flow_over_flow0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], flowprof),
            "omegabeat": lambda z: jnp.interp(z, self.cfg["grid"]["z"], omegabeat),
        }
        self.state = {
            "Ji": 1.0,
            "Jr": 0.1,
            "imfx0": 0.0,
            "kappaIB": 0.0,
            "omega_beat_plasframe": 0.0,
            "cs": 0.0,
            "kz": 0.0,
            "res_cond": 0.0,
        } | {k: 0.0 for k in self.args.keys()}

    def init_diffeqsolve(self) -> Dict:
        self.space_quantities = {
            "z0": self.cfg["grid"]["zmin"],
            "z1": self.cfg["grid"]["zmax"],
            "max_steps": self.cfg["grid"]["max_steps"],
            "save_z0": self.cfg["grid"]["zmin"],
            "save_z1": self.cfg["grid"]["zmax"],
            "save_nz": self.cfg["grid"]["nz"],
        }

        self.diffeqsolve_quants = {
            "terms": ODETerm(SBSVectorField(self.cfg)),
            "solver": Stepper(),
            "saveat": {
                "ts": np.linspace(
                    self.space_quantities["z0"], self.space_quantities["z1"], self.space_quantities["save_nz"]
                )
            },
        }

    def __call__(self, trainable_modules: Dict, args: Dict):
        state = self.state

        if args is None:
            args = self.args

        for name, module in trainable_modules.items():
            state, args = module(self.state, args)

        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.space_quantities["z0"],
            t1=self.space_quantities["z1"],
            max_steps=self.cfg["grid"]["max_steps"],
            dt0=self.cfg["grid"]["dz"],
            y0=state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        return {"solver result": solver_result, "args": args}

    def post_process(self, run_output: Dict, td: str) -> Dict:
        result = run_output["solver result"]
        os.makedirs(os.path.join(td, "binary"))
        os.makedirs(os.path.join(td, "plots"))
        Ji, Jr, imfx0, kappaIB = result.ys["Ji"], result.ys["Jr"], result.ys["imfx0"], result.ys["kappaIB"]
        zs = result.ts

        Ji = xr.DataArray(Ji, coords=[("z", zs)], dims=["z (um)"])
        Jr = xr.DataArray(Jr, coords=[("z", zs)], dims=["z (um)"])
        imfx0 = xr.DataArray(imfx0, coords=[("z", zs)], dims=["z (um)"])
        kappaIB = xr.DataArray(kappaIB, coords=[("z", zs)], dims=["z (um)"])
        n_over_n0 = xr.DataArray(result.ys["n_over_n0"], coords=[("z", zs)], dims=["z (um)"])
        Te_over_T0 = xr.DataArray(result.ys["Te_over_T0"], coords=[("z", zs)], dims=["z (um)"])
        Ti_over_T0 = xr.DataArray(result.ys["Ti_over_T0"], coords=[("z", zs)], dims=["z (um)"])
        Zeff = xr.DataArray(result.ys["Zeff"], coords=[("z", zs)], dims=["z (um)"])
        flow_over_flow0 = xr.DataArray(result.ys["flow_over_flow0"], coords=[("z", zs)], dims=["z (um)"])
        omegabeat = xr.DataArray(result.ys["omegabeat"], coords=[("z", zs)], dims=["z (um)"])

        ds = xr.Dataset(
            {
                "Ji": Ji,
                "Jr": Jr,
                "imfx0": imfx0,
                "kappaIB": kappaIB,
                "n_over_n0": n_over_n0,
                "Te_over_T0": Te_over_T0,
                "Ti_over_T0": Ti_over_T0,
                "Zeff": Zeff,
                "flow_over_flow0": flow_over_flow0,
                "omegabeat": omegabeat,
                "omega_beat_plasframe": xr.DataArray(
                    result.ys["omega_beat_plasframe"], coords=[("z", zs)], dims=["z (um)"]
                ),
                "cs": xr.DataArray(result.ys["cs"], coords=[("z", zs)], dims=["z (um)"]),
                "kz": xr.DataArray(result.ys["kz"], coords=[("z", zs)], dims=["z (um)"]),
                "res_cond": xr.DataArray(result.ys["res_cond"], coords=[("z", zs)], dims=["z (um)"]),
            }
        )

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        ds["Ji"][1:].plot(ax=ax[0])
        ds["Jr"][1:].plot(ax=ax[1])
        ax[0].grid()
        ax[1].grid()
        fig.savefig(os.path.join(td, "plots", "Ji_Jr.png"), bbox_inches="tight")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        ds["imfx0"][1:].plot(ax=ax[0])
        ds["kappaIB"][1:].plot(ax=ax[1])
        ax[0].grid()
        ax[1].grid()
        fig.savefig(os.path.join(td, "plots", "imfx0_kappaIB.png"), bbox_inches="tight")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        ds["n_over_n0"][1:].plot(ax=ax[0])
        ds["Te_over_T0"][1:].plot(ax=ax[1])
        ax[0].grid()
        ax[1].grid()
        fig.savefig(os.path.join(td, "plots", "n_Te.png"), bbox_inches="tight")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        ds["Ti_over_T0"][1:].plot(ax=ax[0])
        ds["Zeff"][1:].plot(ax=ax[1])
        ax[0].grid()
        ax[1].grid()
        fig.savefig(os.path.join(td, "plots", "Ti_Zeff.png"), bbox_inches="tight")

        fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
        ds["flow_over_flow0"][1:].plot(ax=ax[0])
        ds["omegabeat"][1:].plot(ax=ax[1])
        ds["omega_beat_plasframe"][1:].plot(ax=ax[2])
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        fig.savefig(os.path.join(td, "plots", "flow_omegabeat.png"), bbox_inches="tight")

        fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
        ds["cs"][1:].plot(ax=ax[0])
        ds["kz"][1:].plot(ax=ax[1])
        ds["res_cond"][1:].plot(ax=ax[2])
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        fig.savefig(os.path.join(td, "plots", "cs_kz_res_cond.png"), bbox_inches="tight")

        ds.to_netcdf(os.path.join(td, "binary", "datasets.nc"))
        return {"ds": ds}


class BaseSteadyStateCBET(ADEPTModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def write_units(self) -> Dict:
        pass

    def get_solver_quantities(self):
        pass

    def get_derived_quantities(self) -> Dict:
        self.cfg["grid"]["dz"] = 1.0 / self.cfg["grid"]["n_steps"]

    def init_state_and_args(self):
        self.state = jnp.ones(self.cfg["laser"]["num_beams"])  # Initial state vector
        self.args = {"plasma": self.cfg["profiles"]}

    def init_diffeqsolve(self) -> Dict:
        self.space_quantities = {
            "z0": 0.0,
            "z1": 1.0,
            "max_steps": self.cfg["grid"]["n_steps"],
            "save_z0": 0.0,
            "save_z1": 1.0,
            "save_nz": self.cfg["grid"]["n_steps"],
        }

        self.diffeqsolve_quants = {
            "terms": ODETerm(CBETVectorField(self.cfg)),
            "solver": Stepper(),
            "saveat": {
                "ts": np.linspace(
                    self.space_quantities["z0"], self.space_quantities["z1"], self.space_quantities["save_nz"]
                )
            },
        }

    def __call__(self, trainable_modules: Dict, args: Dict):
        state = self.state

        if args is None:
            args = self.args

        for name, module in trainable_modules.items():
            state, args = module(self.state, args)

        solver_result = diffeqsolve(
            terms=self.diffeqsolve_quants["terms"],
            solver=self.diffeqsolve_quants["solver"],
            t0=self.space_quantities["z0"],
            t1=self.space_quantities["z1"],
            max_steps=self.cfg["grid"]["max_steps"],
            dt0=self.cfg["grid"]["dz"],
            y0=state,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["saveat"]),
        )

        # result = diffeqsolve(
        # term,
        # solver=Stepper(),
        # t0=0.0,
        # t1=1.0,
        # args=args,
        # y0=jnp.ones(4),
        # dt0=dz,
        # max_steps=integrator_params["n_steps"] + 4,
        # )

        return solver_result, args


class SBSBS_CBET(ADEPTModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cbet_steps = cfg["cbet"]["grid"]["n_steps"]
        self.cbet_dz = 1.0 / self.cbet_steps

    def init_modules(self):
        # if self.cfg["profiles"]["type"] == "user":
        profiles = [
            self.make_sbs_profiles(self.cfg["sbs"]["profiles"][str(i)], self.cfg["sbs"]["grid"][str(i)])
            for i in range(self.cfg["laser"]["num_beams"])
        ]
        sbs_nprof = [p[0] for p in profiles]
        sbs_Teprof = [p[1] for p in profiles]
        sbs_Tiprof = [p[2] for p in profiles]
        sbs_Zeffprof = [p[3] for p in profiles]
        sbs_flowprof = [p[4] for p in profiles]
        sbs_omegabeat = [p[5] for p in profiles]

        sbs_grids = self.cfg["sbs"]["grid"]
        profile_functions = {
            "cbet": {
                "n_over_n0": self.cfg["cbet"]["profiles"]["n_over_n0"],
                "Te_over_T0": self.cfg["cbet"]["profiles"]["Te_over_T0"],
                "Ti_over_T0": self.cfg["cbet"]["profiles"]["Ti_over_T0"],
                "Zeff": self.cfg["units"]["Z_eff"],
                "flow_magnitude": self.cfg["cbet"]["profiles"]["flow_magnitude"],
                "flow_theta": self.cfg["cbet"]["profiles"]["flow_theta"],
                "flow_phi": self.cfg["cbet"]["profiles"]["flow_phi"],
                "A_ion": self.cfg["cbet"]["profiles"]["A_ion"],
                "L_p": self.cfg["cbet"]["profiles"]["L_p"],
            },
            "sbs": [
                {
                    "n_over_n0": lambda z: jnp.interp(z, sbs_grids[str(i)]["z"], sbs_nprof[i]),
                    "Te_over_T0": lambda z: jnp.interp(z, sbs_grids[str(i)]["z"], sbs_Teprof[i]),
                    "Ti_over_T0": lambda z: jnp.interp(z, sbs_grids[str(i)]["z"], sbs_Tiprof[i]),
                    "Zeff": lambda z: jnp.interp(z, sbs_grids[str(i)]["z"], sbs_Zeffprof[i]),
                    "flow_over_flow0": lambda z: jnp.interp(z, sbs_grids[str(i)]["z"], sbs_flowprof[i]),
                    "omegabeat": lambda z: jnp.interp(z, sbs_grids[str(i)]["z"], sbs_omegabeat[i]),
                }
                for i in range(self.cfg["laser"]["num_beams"])
            ],
        }
        # elif self.cfg["profiles"]["type"] == "nn":
        #     raise NotImplementedError("Neural network profiles are not implemented yet.")
        # else:
        #     raise NotImplementedError("Unknown profile type. Please use 'user' or 'nn'.")

        return profile_functions

    def write_units(self) -> Dict:
        """
        Write the units used in the simulation to a dictionary.


        """
        m_e = const.m_e
        e = const.e
        c = const.c
        eps_0 = const.eps0
        Zeff = self.cfg["units"]["Z_eff"]

        mi = self.cfg["units"]["atomic mass number"] * const.m_p

        lambda0 = _Q(self.cfg["units"]["laser_wavelength"]).to("um")
        T0 = _Q(self.cfg["units"]["reference temperature"]).to("keV")
        vth0 = np.sqrt(2 * T0 / m_e).to("m/s")
        w0 = (2 * np.pi * c / lambda0).to("Hz")
        cs0 = np.sqrt(T0 / mi).to("m/s")
        I0 = _Q(self.cfg["units"]["laser_intensity"]).to("W/cm^2")
        a0 = 0.86 * np.sqrt(I0.to("W/cm^2").value / 1e18) * lambda0.to("um").value
        kz0 = 2 * np.pi / lambda0
        omegabeat0 = kz0 * cs0
        if self.cfg["units"]["reference density"] == "nc":
            n0 = (2 * np.pi * c / self.cfg["units"]["laser_wavelength"]) ** 2.0 * m_e * eps_0 / e.si**2.0
            nc_over_n0 = 1.0
        else:
            n0 = _Q(self.cfg["units"]["reference density"]).to("cm-3")
            nc_over_n0 = (2 * np.pi * c / self.cfg["units"]["laser_wavelength"]) ** 2.0 * m_e * eps_0 / e.si**2.0 / n0

        logLambda_ei, logLambda_ee = calc_logLambda(self.cfg, n0, T0, Zeff)
        # logLambda_ee = logLambda_ei
        ni0 = n0 / Zeff

        nuei_epphaines = (
            1
            / (
                0.75
                * np.sqrt(const.m_e)
                * T0**1.5
                / (np.sqrt(2 * np.pi) * ni0 * Zeff**2.0 * const.e.gauss**4.0 * logLambda_ei)
            )
        ).to("Hz")

        self.cfg["units"]["derived"] = {
            "n0": n0,
            "lambda0": lambda0,
            "T0": T0,
            "vth0": vth0,
            "w0": w0,
            "kz0": kz0,
            "flow0": cs0,
            "cs0": cs0,
            "I0": I0,
            "a0": a0,
            "mi": mi,
            "nuei_epphaines0": nuei_epphaines,
            "logLambda_ee0": logLambda_ee,
            "logLambda_ei0": logLambda_ei,
            "nc_over_n0": nc_over_n0,
            "omegabeat0": omegabeat0,
            "c": c,
            "ratio_M_m": mi / m_e,
        }

        return {k: str(v) for k, v in self.cfg["units"]["derived"].items()}

    def get_solver_quantities(self):
        sbs_grids = self.cfg["sbs"]["grid"]

        for i in range(self.cfg["laser"]["num_beams"]):
            i = str(i)
            sbs_grids[i]["z"] = np.linspace(
                sbs_grids[i]["dz"] / 2,
                sbs_grids[i]["zmax"] - sbs_grids[i]["dz"] / 2,
                sbs_grids[i]["nz"],
            )

        self.sbs_solvers = [
            SBSVectorField(self.cfg["sbs"]["grid"][str(i)], self.cfg["units"])
            for i in range(self.cfg["laser"]["num_beams"])
        ]
        self.cbet = CBETVectorField(self.cfg)
        self.cfg["sbs"]["grid"] = sbs_grids

    def get_derived_quantities(self) -> Dict:
        """
        This function just updates the config with the derived quantities that are only integers or strings.

        This is run prior to the log params step

        :param cfg_grid:
        :return:
        """
        sbs_grids = self.cfg["sbs"]["grid"]

        for i in range(self.cfg["laser"]["num_beams"]):
            i = str(i)
            Lgrid = _Q(sbs_grids[str(i)]["zmax"]).to("um").value

            sbs_grids[i]["zmax"] = Lgrid
            sbs_grids[i]["zmin"] = 0.0
            sbs_grids[i]["dz"] = _Q(sbs_grids[i]["dz"]).to("um").value

            sbs_grids[i]["nz"] = int(sbs_grids[i]["zmax"] / sbs_grids[i]["dz"]) + 1

            if sbs_grids[i]["nz"] > 1e6:
                sbs_grids[i]["max_steps"] = int(1e6)
                print(r"Only running $10^6$ steps")
            else:
                sbs_grids[i]["max_steps"] = sbs_grids[i]["nz"] + 4

        self.cfg["sbs"]["grid"] = sbs_grids

        # CBET
        self.cfg["cbet"]["grid"]["dz"] = 1 / self.cfg["cbet"]["grid"]["n_steps"]
        self.cfg["cbet"]["grid"]["z"] = np.linspace(0.0, 1.0, self.cfg["cbet"]["grid"]["n_steps"] + 1)

        return self.cfg

    def init_state_and_args(self):
        # nprof, Teprof, Tiprof, Zeffprof, flowprof, omegabeat = self.make_sbs_profiles(nmin, nmax)  # / self.cfg["units"]["derived"]["omegabeat0"].to("Hz").value

        # self.args = {
        #     "n_over_n0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], nprof),
        #     "Te_over_T0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], Teprof),
        #     "Ti_over_T0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], Tiprof),
        #     "Zeff": lambda z: jnp.interp(z, self.cfg["grid"]["z"], Zeffprof),
        #     "flow_over_flow0": lambda z: jnp.interp(z, self.cfg["grid"]["z"], flowprof),
        #     "omegabeat": lambda z: jnp.interp(z, self.cfg["grid"]["z"], omegabeat),
        # }
        self.args = {}
        self.state = {
            "Ji": jnp.ones(4),
            "Jr": jnp.full(4, 0.1),
            "imfx0": jnp.zeros(4),
            "kappaIB": jnp.zeros(4),
            "omega_beat_plasframe": jnp.zeros(4),
            "cs": jnp.zeros(4),
            "kz": jnp.zeros(4),
            "res_cond": jnp.zeros(4),
        } | {k: 0.0 for k in self.args.keys()}

    def make_sbs_profiles(self, this_profile, this_grid):
        if "nc" in this_profile["n"]["min"]:
            nmin = (
                float(this_profile["n"]["min"].strip("nc")) * self.cfg["units"]["derived"]["nc_over_n0"]
                # * self.cfg["units"]["derived"]["n0"].value
            )
            nmax = (
                float(this_profile["n"]["max"].strip("nc")) * self.cfg["units"]["derived"]["nc_over_n0"]
                # * self.cfg["units"]["derived"]["n0"].value
            )
        else:
            nmin = (_Q(this_profile["n"]["min"]) / self.cfg["units"]["derived"]["n0"]).to("").value
            nmax = (_Q(this_profile["n"]["max"]) / self.cfg["units"]["derived"]["n0"]).to("").value

        nprof = nmin + (nmax - nmin) * this_grid["z"] / this_grid["zmax"]
        Teprof = np.ones_like(nprof) * _Q(this_profile["Te"]).to("keV").value
        Tiprof = np.ones_like(nprof) * _Q(this_profile["Ti"]).to("keV").value
        Zeffprof = np.ones_like(nprof) * self.cfg["units"]["Z_eff"]
        flowprof = np.ones_like(nprof) * _Q(this_profile["flow"]).to("um/s").value
        omegabeat = self.compute_omega_beat(nprof, this_profile["lambdabeat"])

        return nprof, Teprof, Tiprof, Zeffprof, flowprof, omegabeat

    def compute_omega_beat(self, nprof, lambdabeat):
        omega_beat_prof = jnp.ones_like(nprof) * (
            self.cfg["units"]["derived"]["w0"].to("Hz").value
            - (
                (2 * np.pi * self.cfg["units"]["derived"]["c"])
                / (self.cfg["units"]["derived"]["lambda0"] + _Q(lambdabeat))
            )
            .to("Hz")
            .value
        )
        return omega_beat_prof

    def init_diffeqsolve(self) -> Dict:
        self.sbs_space_quantities = [
            {
                "z0": 0.0,
                "z1": self.cfg["sbs"]["grid"][str(i)]["zmax"],
                "max_steps": self.cfg["sbs"]["grid"][str(i)]["max_steps"],
                "save_z0": 0.0,
                "save_z1": self.cfg["sbs"]["grid"][str(i)]["zmax"],
                "save_nz": self.cfg["sbs"]["grid"][str(i)]["nz"],
            }
            for i in range(self.cfg["laser"]["num_beams"])
        ]

        diffeqsolve_quants_cbet = {
            "terms": ODETerm(CBETVectorField(self.cfg)),
            "solver": Stepper(),
            "saveat": {"ts": np.linspace(0, 1, 2)},
        }
        diffeqsolve_quants_sbs = [
            self.init_sbs_diffeqsolve(self.cfg["sbs"]["grid"][str(i)], self.cfg["units"], self.sbs_space_quantities[i])
            for i in range(self.cfg["laser"]["num_beams"])
        ]

        self.diffeqsolve_quants = {"cbet": diffeqsolve_quants_cbet, "sbs": diffeqsolve_quants_sbs}

    def init_sbs_diffeqsolve(self, grid, units, space_quantity):
        return {
            "terms": ODETerm(SBSVectorField(grid, units)),
            "solver": Stepper(),
            "saveat": {
                "ts": np.linspace(
                    space_quantity["z0"],
                    space_quantity["z1"],
                    space_quantity["save_nz"],
                )
            },
        }

    def perform_sbs(self, diffeqsolve_quants, beam_modules, _cbet_light, this_space_quantity, this_grid):
        args = {
            "n_over_n0": beam_modules["n_over_n0"],
            "Te_over_T0": beam_modules["Te_over_T0"],
            "Ti_over_T0": beam_modules["Ti_over_T0"],
            "Zeff": beam_modules["Zeff"],
            "flow_over_flow0": beam_modules["flow_over_flow0"],
            "omegabeat": beam_modules["omegabeat"],
        }

        sbs_states = {
            "Ji": _cbet_light,
            "Jr": 0.1,
            "imfx0": 0.0,
            "kappaIB": 0.0,
            "omega_beat_plasframe": 0.0,
            "cs": 0.0,
            "kz": 0.0,
            "res_cond": 0.0,
        } | {k: 0.0 for k in args.keys()}

        solver_result = diffeqsolve(
            terms=diffeqsolve_quants["terms"],
            solver=diffeqsolve_quants["solver"],
            t0=this_space_quantity["z0"],
            t1=this_space_quantity["z1"],
            max_steps=this_grid["max_steps"],
            dt0=this_grid["dz"],
            y0=sbs_states,
            args=args,
            saveat=SaveAt(**diffeqsolve_quants["saveat"]),
        )

        return solver_result, args

    def perform_cbet(self, cbet_modules, beams, args):
        cbet_nprof = cbet_modules["n_over_n0"]
        cbet_Teprof = cbet_modules["Te_over_T0"]
        cbet_Tiprof = cbet_modules["Ti_over_T0"]
        cbet_Zeffprof = cbet_modules["Zeff"]
        cbet_flowmagnitude = cbet_modules["flow_magnitude"]
        cbet_flowtheta = cbet_modules["flow_theta"]
        cbet_flowphi = cbet_modules["flow_phi"]
        cbet_Aion = cbet_modules["A_ion"]
        cbet_Lp = cbet_modules["L_p"]

        plasma_args = {
            "n_over_n0": cbet_nprof,
            "Te_over_T0": cbet_Teprof,
            "Ti_over_T0": cbet_Tiprof,
            "Zeff": cbet_Zeffprof,
            "flow_magnitude": cbet_flowmagnitude,
            "flow_theta": cbet_flowtheta,
            "flow_phi": cbet_flowphi,
            "A_ion": cbet_Aion,
            "L_p": cbet_Lp,
        }
        if args is None:
            args = {"plasma": plasma_args}
        else:
            args = {"plasma": plasma_args} | args

        cbet_result = diffeqsolve(
            terms=self.diffeqsolve_quants["cbet"]["terms"],
            solver=self.diffeqsolve_quants["cbet"]["solver"],
            t0=0.0,
            t1=1.0,
            max_steps=self.cbet_steps + 4,
            dt0=self.cbet_dz,
            y0=beams,
            args=args,
            saveat=SaveAt(**self.diffeqsolve_quants["cbet"]["saveat"]),
        )

        return cbet_result, args

    def __call__(self, trainable_modules, args):
        beams = self.state["Ji"]

        cbet_result, args = self.perform_cbet(trainable_modules["cbet"], beams, args)
        sbs_solver_results = []

        # loop over beams / quads / rays
        for i in range(self.cfg["laser"]["num_beams"]):
            diffeqsolve_quants = self.diffeqsolve_quants["sbs"][i]
            beam_modules = trainable_modules["sbs"][i]
            this_space_quantity = self.sbs_space_quantities[i]
            _cbet_light = cbet_result.ys[-1, i]
            this_grid = self.cfg["sbs"]["grid"][str(i)]
            beam_result, beam_args = self.perform_sbs(
                diffeqsolve_quants, beam_modules, _cbet_light, this_space_quantity, this_grid
            )
            sbs_solver_results.append(beam_result)

        return {"cbet result": cbet_result, "sbs results": sbs_solver_results, "args": args}
