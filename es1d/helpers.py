from typing import Callable, Dict
from functools import partial

import os
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from jax import tree_util as jtu
from flatdict import FlatDict

from jax import numpy as jnp
from es1d import pushers


def save_arrays(result, td, cfg, label):
    flattened_dict = dict(FlatDict(result.ys[label], delimiter="-"))
    data_vars = {
        k: xr.DataArray(v, coords=(("t", cfg["save"]["t_save"]), (label, cfg["save"][label]["ax"])))
        for k, v in flattened_dict.items()
    }

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"state_vs_{label}.nc"))
    os.makedirs(os.path.join(td, "plots", label))

    for k, v in saved_arrays_xr.items():
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
        v.plot(ax=ax, cmap="gist_ncar")
        fig.savefig(os.path.join(td, "plots", label, f"{k}.png"), bbox_inches="tight")
        plt.close(fig)


def post_process(result, cfg: Dict, td: str) -> None:
    os.makedirs(os.path.join(td, "binary"))
    os.makedirs(os.path.join(td, "plots"))

    if cfg["save"]["x"]["is_on"]:
        save_arrays(result, td, cfg, "x")

    if cfg["save"]["kx"]["is_on"]:
        save_arrays(result, td, cfg, "kx")


def get_derived_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dt"] = 0.5 * cfg_grid["dx"] / 10
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    return cfg_grid


def get_solver_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are arrays

    This is run after the log params step

    :param cfg_grid:
    :return:
    """
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

    return cfg_grid


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    cfg["save"] = {
        **cfg["save"],
        **{
            "t_save": jnp.linspace(cfg["save"]["t"]["tmin"], cfg["save"]["t"]["tmax"], cfg["save"]["t"]["nt"]),
            "func": get_save_func(cfg),
        },
    }

    return cfg


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
            p=cfg["physics"]["T0"] * jnp.ones(cfg["grid"]["nx"]),
            u=jnp.zeros(cfg["grid"]["nx"]),
            delta=jnp.zeros(cfg["grid"]["nx"]),
        )

    return state


def get_vector_field(cfg: Dict) -> Callable:
    """
    This function returns the function that defines $d_state / dt$

    All the pushers are chosen and initialized here and a single time-step is defined here.

    We use the time-integrators provided by diffrax, and therefore, only need $d_state / dt$ here

    :param cfg:
    :return:
    """
    pusher_dict = {"ion": {}, "electron": {}}
    for species_name in ["ion", "electron"]:
        pusher_dict[species_name]["push_n"] = pushers.DensityStepper(cfg["grid"]["kx"])
        pusher_dict[species_name]["push_u"] = pushers.VelocityStepper(
            cfg["grid"]["kx"], cfg["grid"]["kxr"], cfg["physics"]
        )
        pusher_dict[species_name]["push_e"] = pushers.EnergyStepper(cfg["grid"]["kx"], cfg["physics"]["gamma"])
        if cfg["physics"][species_name]["trapping"]["is_on"]:
            pusher_dict[species_name]["particle_trapper"] = pushers.ParticleTrapper(
                cfg["physics"][species_name]["trapping"]["kld"],
                cfg["grid"]["kxr"],
                cfg["grid"]["kx"],
                cfg["physics"]["kinetic_real_wepw"],
            )

    push_driver = pushers.Driver(cfg["grid"]["x"])
    poisson_solver = pushers.PoissonSolver(cfg["grid"]["one_over_kx"])

    def push_everything(t: float, y: Dict, args: Dict):
        """
        This function is used by the time integrators specified in diffrax

        :param t:
        :param y:
        :param args:
        :return:
        """
        e = poisson_solver(
            cfg["physics"]["ion"]["charge"] * y["ion"]["n"] + cfg["physics"]["electron"]["charge"] * y["electron"]["n"]
        )
        ed = push_driver(cfg["drivers"]["ex"]["0"], t)
        total_e = e + ed

        dstate_dt = {"ion": {}, "electron": {}}
        for species_name in ["ion", "electron"]:
            n = y[species_name]["n"]
            u = y[species_name]["u"]
            p = y[species_name]["p"]
            delta = y[species_name]["delta"]
            if cfg["physics"][species_name]["is_on"]:
                q_over_m = cfg["physics"][species_name]["charge"] / cfg["physics"][species_name]["mass"]
                p_over_m = p / cfg["physics"][species_name]["mass"]

                dstate_dt[species_name]["n"] = pusher_dict[species_name]["push_n"](n, u)
                dstate_dt[species_name]["u"] = pusher_dict[species_name]["push_u"](
                    n, u, p_over_m, q_over_m * total_e, delta
                )
                dstate_dt[species_name]["p"] = pusher_dict[species_name]["push_e"](n, u, p_over_m, q_over_m * e)
            else:
                dstate_dt[species_name]["n"] = 0.0
                dstate_dt[species_name]["u"] = 0.0
                dstate_dt[species_name]["p"] = 0.0

            if cfg["physics"][species_name]["trapping"]["is_on"]:
                dstate_dt[species_name]["delta"] = pusher_dict[species_name]["particle_trapper"](e, delta)
            else:
                dstate_dt[species_name]["delta"] = 0.0

        return dstate_dt

    return push_everything


def get_save_func(cfg):
    if cfg["save"]["x"]["is_on"]:
        dx = (cfg["save"]["x"]["xmax"] - cfg["save"]["x"]["xmin"]) / cfg["save"]["x"]["nx"]
        cfg["save"]["x"]["ax"] = jnp.linspace(
            cfg["save"]["x"]["xmin"] + dx / 2.0, cfg["save"]["x"]["xmax"] - dx / 2.0, cfg["save"]["x"]["nx"]
        )

        save_x = partial(jnp.interp, cfg["save"]["x"]["ax"], cfg["grid"]["x"])

    if cfg["save"]["kx"]["is_on"]:
        cfg["save"]["kx"]["ax"] = jnp.linspace(
            cfg["save"]["kx"]["kxmin"], cfg["save"]["kx"]["kxmax"], cfg["save"]["kx"]["nkx"]
        )

        def save_kx(field):
            complex_field = 2.0 / cfg["grid"]["nx"] * jnp.fft.rfft(field, axis=0)
            interped_field = jnp.interp(cfg["save"]["kx"]["ax"], cfg["grid"]["kxr"], complex_field)
            return {"mag": jnp.abs(interped_field), "ang": jnp.angle(interped_field)}

    def save_func(t, y, args):
        save_dict = {}
        if cfg["save"]["x"]["is_on"]:
            save_dict["x"] = jtu.tree_map(save_x, y)

        if cfg["save"]["kx"]["is_on"]:
            save_dict["kx"] = jtu.tree_map(save_kx, y)

        return save_dict

    return save_func
