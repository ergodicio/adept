from collections import defaultdict
from typing import Callable, Dict
from functools import partial

import os

import jax.random
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr, pint, yaml
from jax import tree_util as jtu
from flatdict import FlatDict
import equinox as eqx
from diffrax import ODETerm, Tsit5

from jax import numpy as jnp
from adept.tf1d import pushers
from equinox import nn


def write_units(cfg, td):
    ureg = pint.UnitRegistry()
    _Q = ureg.Quantity

    n0 = _Q(cfg["units"]["normalizing density"]).to("1/cc")
    T0 = _Q(cfg["units"]["normalizing temperature"]).to("eV")

    wp0 = np.sqrt(n0 * ureg.e**2.0 / (ureg.m_e * ureg.epsilon_0)).to("rad/s")
    tp0 = (1 / wp0).to("fs")

    v0 = np.sqrt(2.0 * T0 / ureg.m_e).to("m/s")
    x0 = (v0 / wp0).to("nm")
    c_light = _Q(1.0 * ureg.c).to("m/s") / v0
    beta = (v0 / ureg.c).to("dimensionless")

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * x0).to("microns")
    if "ymax" in cfg["grid"].keys():
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * x0).to("microns")
    else:
        box_width = "inf"
    sim_duration = (cfg["grid"]["tmax"] * tp0).to("ps")

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

    cfg["units"]["derived"] = all_quantities

    cfg["grid"]["beta"] = beta.magnitude

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump({k: str(v) for k, v in all_quantities.items()}, fi)

    return cfg


def save_arrays(result, td, cfg, label):
    if label is None:
        label = "x"
        flattened_dict = dict(FlatDict(result.ys, delimiter="-"))
        save_ax = cfg["grid"]["x"]
    else:
        flattened_dict = dict(FlatDict(result.ys[label], delimiter="-"))
        save_ax = cfg["save"][label]["ax"]
    data_vars = {
        k: xr.DataArray(v, coords=(("t", cfg["save"]["t"]["ax"]), (label, save_ax))) for k, v in flattened_dict.items()
    }

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"state_vs_{label}.nc"))
    return saved_arrays_xr


def plot_xrs(which, td, xrs):
    os.makedirs(os.path.join(td, "plots", which))
    os.makedirs(os.path.join(td, "plots", which, "ion"))
    os.makedirs(os.path.join(td, "plots", which, "electron"))

    for k, v in xrs.items():
        fname = f"{'-'.join(k.split('-')[1:])}.png"
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
        v.plot(ax=ax, cmap="gist_ncar")
        ax.grid()
        fig.savefig(os.path.join(td, "plots", which, k.split("-")[0], fname), bbox_inches="tight")
        plt.close(fig)

        if which == "kx":
            os.makedirs(os.path.join(td, "plots", which, "ion", "hue"), exist_ok=True)
            os.makedirs(os.path.join(td, "plots", which, "electron", "hue"), exist_ok=True)
            # only plot
            if v.coords["kx"].size > 8:
                hue_skip = v.coords["kx"].size // 8
            else:
                hue_skip = 1

            for log in [True, False]:
                fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
                v[:, ::hue_skip].plot(ax=ax, hue="kx")
                ax.set_yscale("log" if log else "linear")
                ax.grid()
                fig.savefig(
                    os.path.join(
                        td, "plots", which, k.split("-")[0], f"hue", f"{'-'.join(k.split('-')[1:])}-log-{log}.png"
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig)


def post_process(result, cfg: Dict, td: str) -> Dict:
    os.makedirs(os.path.join(td, "binary"))
    os.makedirs(os.path.join(td, "plots"))

    datasets = {}
    if any(x in ["x", "kx"] for x in cfg["save"]):
        if "x" in cfg["save"].keys():
            datasets["x"] = save_arrays(result, td, cfg, label="x")
            plot_xrs("x", td, datasets["x"])
        if "kx" in cfg["save"].keys():
            datasets["kx"] = save_arrays(result, td, cfg, label="kx")
            plot_xrs("kx", td, datasets["kx"])
    else:
        datasets["full"] = save_arrays(result, td, cfg, label=None)
        plot_xrs("x", td, datasets["full"])

    return datasets


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """

    cfg_grid = cfg["grid"]

    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    cfg["grid"] = cfg_grid

    return cfg


def get_solver_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are arrays

    This is run after the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

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

    return cfg_grid


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    cfg["save"]["func"] = {"callable": get_save_func(cfg)}
    cfg["save"]["t"]["ax"] = jnp.linspace(cfg["save"]["t"]["tmin"], cfg["save"]["t"]["tmax"], cfg["save"]["t"]["nt"])

    return cfg


def get_diffeqsolve_quants(cfg):
    return dict(
        terms=ODETerm(VectorField(cfg)),
        solver=Tsit5(),
        saveat=dict(ts=cfg["save"]["t"]["ax"], fn=cfg["save"]["func"]["callable"]),
    )


def init_state(cfg: Dict) -> Dict:
    """
    This function initializes the state

    :param cfg:
    :return:
    """
    state = {}
    for species in ["ion", "electron"]:
        state[species] = dict(
            n=jnp.ones(cfg["grid"]["nx"]),
            p=jnp.full(cfg["grid"]["nx"], cfg["physics"][species]["T0"]),
            u=jnp.zeros(cfg["grid"]["nx"]),
            delta=jnp.zeros(cfg["grid"]["nx"]),
        )

    return state


class VectorField(eqx.Module):
    """
    This function returns the function that defines $d_state / dt$

    All the pushers are chosen and initialized here and a single time-step is defined here.

    We use the time-integrators provided by diffrax, and therefore, only need $d_state / dt$ here

    :param cfg:
    :return:
    """

    cfg: Dict
    pusher_dict: Dict
    push_driver: Callable
    poisson_solver: Callable

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pusher_dict = {"ion": {}, "electron": {}}
        for species_name in ["ion", "electron"]:
            self.pusher_dict[species_name]["push_n"] = pushers.DensityStepper(cfg["grid"]["kx"])
            self.pusher_dict[species_name]["push_u"] = pushers.VelocityStepper(
                cfg["grid"]["kx"], cfg["grid"]["kxr"], cfg["grid"]["one_over_kxr"], cfg["physics"][species_name]
            )
            self.pusher_dict[species_name]["push_e"] = pushers.EnergyStepper(
                cfg["grid"]["kx"], cfg["physics"][species_name]
            )
            if cfg["physics"][species_name]["trapping"]["is_on"]:
                self.pusher_dict[species_name]["particle_trapper"] = pushers.ParticleTrapper(cfg, species_name)

        self.push_driver = pushers.Driver(cfg["grid"]["x"])
        # if "ey" in self.cfg["drivers"]:
        #     self.wave_solver = pushers.WaveSolver(cfg["grid"]["c"], cfg["grid"]["dx"], cfg["grid"]["dt"])
        self.poisson_solver = pushers.PoissonSolver(cfg["grid"]["one_over_kx"])

    def __call__(self, t: float, y: Dict, args: Dict):
        """
        This function is used by the time integrators specified in diffrax

        :param t:
        :param y:
        :param args:
        :return:
        """
        e = self.poisson_solver(
            self.cfg["physics"]["ion"]["charge"] * y["ion"]["n"]
            + self.cfg["physics"]["electron"]["charge"] * y["electron"]["n"]
        )
        ed = 0.0

        for p_ind in self.cfg["drivers"]["ex"].keys():
            ed += self.push_driver(args["drivers"]["ex"][p_ind], t)

        # if "ey" in self.cfg["drivers"]:
        #     ad = 0.0
        #     for p_ind in self.cfg["drivers"]["ey"].keys():
        #         ad += self.push_driver(args["pulse"]["ey"][p_ind], t)
        #     a = self.wave_solver(a, aold, djy_array, charge)
        #     total_a = y["a"] + ad
        #     ponderomotive_force = -0.5 * jnp.gradient(jnp.square(total_a), self.cfg["grid"]["dx"])[1:-1]

        total_e = e + ed  # + ponderomotive_force

        dstate_dt = {"ion": {}, "electron": {}}
        for species_name in ["ion", "electron"]:
            n = y[species_name]["n"]
            u = y[species_name]["u"]
            p = y[species_name]["p"]
            delta = y[species_name]["delta"]
            if self.cfg["physics"][species_name]["is_on"]:
                q_over_m = self.cfg["physics"][species_name]["charge"] / self.cfg["physics"][species_name]["mass"]
                p_over_m = p / self.cfg["physics"][species_name]["mass"]

                dstate_dt[species_name]["n"] = self.pusher_dict[species_name]["push_n"](n, u)
                dstate_dt[species_name]["u"] = self.pusher_dict[species_name]["push_u"](
                    n, u, p_over_m, q_over_m * total_e, delta
                )
                dstate_dt[species_name]["p"] = self.pusher_dict[species_name]["push_e"](n, u, p_over_m, q_over_m * e)
            else:
                dstate_dt[species_name]["n"] = jnp.zeros(self.cfg["grid"]["nx"])
                dstate_dt[species_name]["u"] = jnp.zeros(self.cfg["grid"]["nx"])
                dstate_dt[species_name]["p"] = jnp.zeros(self.cfg["grid"]["nx"])

            if self.cfg["physics"][species_name]["trapping"]["is_on"]:
                dstate_dt[species_name]["delta"] = self.pusher_dict[species_name]["particle_trapper"](e, delta, args)
            else:
                dstate_dt[species_name]["delta"] = jnp.zeros(self.cfg["grid"]["nx"])

        return dstate_dt


def get_save_func(cfg):
    if any(x in ["x", "kx"] for x in cfg["save"]):
        if "x" in cfg["save"].keys():
            dx = (cfg["save"]["x"]["xmax"] - cfg["save"]["x"]["xmin"]) / cfg["save"]["x"]["nx"]
            cfg["save"]["x"]["ax"] = jnp.linspace(
                cfg["save"]["x"]["xmin"] + dx / 2.0, cfg["save"]["x"]["xmax"] - dx / 2.0, cfg["save"]["x"]["nx"]
            )

            save_x = partial(jnp.interp, cfg["save"]["x"]["ax"], cfg["grid"]["x"])

        if "kx" in cfg["save"].keys():
            cfg["save"]["kx"]["ax"] = jnp.linspace(
                cfg["save"]["kx"]["kxmin"], cfg["save"]["kx"]["kxmax"], cfg["save"]["kx"]["nkx"]
            )

            def save_kx(field):
                complex_field = jnp.fft.rfft(field, axis=0) * 2.0 / cfg["grid"]["nx"]
                interped_field = jnp.interp(cfg["save"]["kx"]["ax"], cfg["grid"]["kxr"], complex_field)
                return {"mag": jnp.abs(interped_field), "ang": jnp.angle(interped_field)}

        def save_func(t, y, args):
            save_dict = {}
            if "x" in cfg["save"].keys():
                save_dict["x"] = jtu.tree_map(save_x, y)
            if "kx" in cfg["save"].keys():
                save_dict["kx"] = jtu.tree_map(save_kx, y)

            return save_dict

    else:
        save_func = None

    return save_func


def get_models(model_config: Dict) -> defaultdict[eqx.Module]:
    if model_config:
        model_keys = jax.random.split(jax.random.PRNGKey(420), len(model_config.keys()))
        model_dict = defaultdict(eqx.Module)
        for (term, config), this_key in zip(model_config.items(), model_keys):
            if term == "file":
                pass
            else:
                for act in ["activation", "final_activation"]:
                    if config[act] == "tanh":
                        config[act] = jnp.tanh

                model_dict[term] = nn.MLP(**{**config, "key": this_key})
        if model_config["file"]:
            model_dict = eqx.tree_deserialise_leaves(model_config["file"], model_dict)

        return model_dict
    else:
        return False
