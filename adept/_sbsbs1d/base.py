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
from adept._sbsbs1d.vectorfield import ExponentialLeapfrog


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
                float(self.cfg["profiles"]["n"]["min"].strip("nc"))
                * self.cfg["units"]["derived"]["nc_over_n0"]
                # * self.cfg["units"]["derived"]["n0"].value
            )
            nmax = (
                float(self.cfg["profiles"]["n"]["max"].strip("nc"))
                * self.cfg["units"]["derived"]["nc_over_n0"]
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
            "terms": ODETerm(ExponentialLeapfrog(self.cfg)),
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
