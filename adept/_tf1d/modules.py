from typing import Dict, Callable, Union
from functools import partial

import os, numpy as np, pint
from diffrax import diffeqsolve, SaveAt, ODETerm, Tsit5
from jax import numpy as jnp, tree_util as jtu

from adept import ADEPTModule
from adept._tf1d.solvers.vector_field import VF
from adept._tf1d.storage import save_arrays, plot_xrs


class BaseTwoFluid1D(ADEPTModule):
    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        self.ureg = pint.UnitRegistry()

    def post_process(self, solver_result: Dict, td: str):
        """
        This function is the post-processing step that is run after the simulation is complete

        It relies on xarray and matplotlib to save the results to disk

        """
        # result = run_output

        os.makedirs(os.path.join(td, "binary"))
        os.makedirs(os.path.join(td, "plots"))

        datasets = {}
        if any(x in ["x", "kx"] for x in self.cfg["save"]):
            if "x" in self.cfg["save"].keys():
                datasets["x"] = save_arrays(solver_result["solver result"], td, self.cfg, label="x")
                plot_xrs("x", td, datasets["x"])
            if "kx" in self.cfg["save"].keys():
                datasets["kx"] = save_arrays(solver_result["solver result"], td, self.cfg, label="kx")
                plot_xrs("kx", td, datasets["kx"])
        else:
            datasets["full"] = save_arrays(solver_result["solver result"], td, self.cfg, label=None)
            plot_xrs("x", td, datasets["full"])

        return datasets

    def write_units(self):
        """
        This function writes the units into the config and stores them in a yaml file
        that gets logged by mlflow


        """
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

        # with open(os.path.join(td, "units.yaml"), "w") as fi:
        #     yaml.dump({k: str(v) for k, v in all_quantities.items()}, fi)

        return {k: str(v) for k, v in all_quantities.items()}

    def get_derived_quantities(self):
        """
        This function just updates the config with the derived quantities that are only integers or strings.

        This is run prior to the log params step

        :param cfg_grid:
        :return:
        """

        cfg_grid = self.cfg["grid"]

        cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
        cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
        cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
        cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

        if cfg_grid["nt"] > 1e6:
            cfg_grid["max_steps"] = int(1e6)
            print(r"Only running $10^6$ steps")
        else:
            cfg_grid["max_steps"] = cfg_grid["nt"] + 4

        self.cfg["grid"] = cfg_grid

        return True

    def get_solver_quantities(self):
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

        one_over_kx = np.zeros_like(cfg_grid["kx"])
        one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
        cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

        one_over_kxr = np.zeros_like(cfg_grid["kxr"])
        one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
        cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

        self.cfg["grid"] = cfg_grid

        return True

    def init_state_and_args(self):
        """
        This function initializes the static state and args. By static we mean that these quantities are not differentiable
        Any modifications that need to be differentiable (e.g. parameterized density/driver profile) need to be done in the
        `__call__` function

        """

        self.state = {}
        self.args = {"drivers": self.cfg["drivers"]}

        for species in ["ion", "electron"]:
            self.state[species] = dict(
                n=jnp.ones(self.cfg["grid"]["nx"]),
                p=jnp.full(self.cfg["grid"]["nx"], self.cfg["physics"][species]["T0"]),
                u=jnp.zeros(self.cfg["grid"]["nx"]),
                delta=jnp.zeros(self.cfg["grid"]["nx"]),
            )

    def get_save_func(self) -> Callable:
        """
        This function returns the function that saves the processed state and args to memory during the diffeqsolve

        """
        if any(x in ["x", "kx"] for x in self.cfg["save"]):
            if "x" in self.cfg["save"].keys():
                dx = (self.cfg["save"]["x"]["xmax"] - self.cfg["save"]["x"]["xmin"]) / self.cfg["save"]["x"]["nx"]
                self.cfg["save"]["x"]["ax"] = jnp.linspace(
                    self.cfg["save"]["x"]["xmin"] + dx / 2.0,
                    self.cfg["save"]["x"]["xmax"] - dx / 2.0,
                    self.cfg["save"]["x"]["nx"],
                )

                save_x = partial(jnp.interp, self.cfg["save"]["x"]["ax"], self.cfg["grid"]["x"])

            if "kx" in self.cfg["save"].keys():
                self.cfg["save"]["kx"]["ax"] = jnp.linspace(
                    self.cfg["save"]["kx"]["kxmin"], self.cfg["save"]["kx"]["kxmax"], self.cfg["save"]["kx"]["nkx"]
                )

                def save_kx(field):
                    complex_field = jnp.fft.rfft(field, axis=0) * 2.0 / self.cfg["grid"]["nx"]
                    interped_field = jnp.interp(self.cfg["save"]["kx"]["ax"], self.cfg["grid"]["kxr"], complex_field)
                    return {"mag": jnp.abs(interped_field), "ang": jnp.angle(interped_field)}

            def save_func(t, y, args):
                save_dict = {}
                if "x" in self.cfg["save"].keys():
                    save_dict["x"] = jtu.tree_map(save_x, y)
                if "kx" in self.cfg["save"].keys():
                    save_dict["kx"] = jtu.tree_map(save_kx, y)

                return save_dict

        else:
            save_func = None

        return save_func

    def init_diffeqsolve(self):
        """
        This function returns the quantities required for the Diffrax solver

        """
        self.time_quantities = {
            "t0": 0.0,
            "t1": self.cfg["grid"]["tmax"],
            "max_steps": self.cfg["grid"]["max_steps"],
            "save_t0": 0.0,
            "save_t1": self.cfg["grid"]["tmax"],
            "save_nt": self.cfg["grid"]["tmax"],
        }
        save_f = self.get_save_func()
        self.cfg["save"]["t"]["ax"] = jnp.linspace(
            self.cfg["save"]["t"]["tmin"], self.cfg["save"]["t"]["tmax"], self.cfg["save"]["t"]["nt"]
        )
        self.diffeqsolve_quants = dict(
            terms=ODETerm(VF(self.cfg)), solver=Tsit5(), saveat=dict(ts=self.cfg["save"]["t"]["ax"], fn=save_f)
        )

    def __call__(self, trainable_modules: Dict, args: Dict) -> Dict:
        """
        This is the time loop solve for a two fluid 1d run
        """
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

    def vg(self, trainable_modules: Dict, args: Dict) -> Dict:
        raise NotImplementedError(
            "This is the base class and does not have a gradient implemented. This is "
            + "likely because there is no metric in place. Subclass this class and implement the gradient"
        )
        # return eqx.filter_value_and_grad(self.__call__)(params)
