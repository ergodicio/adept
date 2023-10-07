from typing import Callable, Dict
from functools import partial

import os

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from flatdict import FlatDict
import equinox as eqx

from jax import numpy as jnp
from jax import tree_util as jtu
from adept.sh2d import vlasov, field


def save_arrays(result, td, cfg):
    """

    :param result:
    :param td:
    :param cfg:
    :param label:
    :return:
    """
    new_dict = {}
    for k in result.ys["flm"].keys():
        new_dict[str(k)] = {}
        for kk in result.ys["flm"][k].keys():
            new_dict[str(k)][str(kk)] = result.ys["flm"][k][kk]

    flattened_flm_dict = dict(FlatDict(new_dict, delimiter=","))

    data_vars = {
        k: xr.DataArray(
            v.view(dtype=np.complex128),
            coords=(
                ("t", cfg["save"]["t"]["ax"]),
                ("x", cfg["grid"]["x"]),
                ("y", cfg["grid"]["y"]),
                ("v", cfg["grid"]["v"]),
            ),
        )
        for k, v in flattened_flm_dict.items()
    }

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"flm_xyv.nc"), engine="h5netcdf", invalid_netcdf=True)

    data_vars = {
        k: xr.DataArray(
            result.ys[k],
            coords=(
                ("t", cfg["save"]["t"]["ax"]),
                ("x", cfg["grid"]["x"]),
                ("y", cfg["grid"]["y"]),
                ("component", ["x", "y", "z"]),
            ),
        )
        for k in ["e", "b"]
    }

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"fields.nc"))

    return saved_arrays_xr


def get_save_func(cfg):
    if cfg["save"]["func"]["is_on"]:
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
                complex_field = jnp.fft.rfft(field, axis=0) * 2.0 / cfg["grid"]["nx"]
                interped_field = jnp.interp(cfg["save"]["kx"]["ax"], cfg["grid"]["kxr"], complex_field)
                return {"mag": jnp.abs(interped_field), "ang": jnp.angle(interped_field)}

        def save_func(t, y, args):
            save_dict = {}
            if cfg["save"]["x"]["is_on"]:
                save_dict["x"] = jtu.tree_map(save_x, y)
            if cfg["save"]["kx"]["is_on"]:
                save_dict["kx"] = jtu.tree_map(save_kx, y)

            return save_dict

    else:
        cfg["save"]["x"]["ax"] = cfg["grid"]["x"]
        save_func = None

    return save_func


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


def post_process(result, cfg: Dict, td: str) -> None:
    os.makedirs(os.path.join(td, "binary"))
    os.makedirs(os.path.join(td, "plots"))

    # if cfg["save"]["func"]["is_on"]:
    #     if cfg["save"]["x"]["is_on"]:
    #         xrs = save_arrays(result, td, cfg, label="x")
    #         plot_xrs("x", td, xrs)
    #
    #     if cfg["save"]["kx"]["is_on"]:
    #         xrs = save_arrays(result, td, cfg, label="kx")
    #         plot_xrs("kx", td, xrs)
    # else:
    xrs = save_arrays(result, td, cfg)
    # plot_xrs("x", td, xrs)


def get_derived_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dy"] = cfg_grid["ymax"] / cfg_grid["ny"]
    cfg_grid["dv"] = cfg_grid["vmax"] / cfg_grid["nv"]
    cfg_grid["dt"] = cfg_grid["tmax"] / cfg_grid["nt"]
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
            "y": jnp.linspace(
                cfg_grid["ymin"] + cfg_grid["dy"] / 2, cfg_grid["ymax"] - cfg_grid["dy"] / 2, cfg_grid["ny"]
            ),
            "v": jnp.linspace(cfg_grid["dv"] / 2, cfg_grid["vmax"] - cfg_grid["dv"] / 2, cfg_grid["nv"]),
            "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "ky": jnp.fft.fftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
            "kyr": jnp.fft.rfftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
            "kv": jnp.fft.fftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
            "kvr": jnp.fft.rfftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
        },
    }

    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

    one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

    one_over_ky = np.zeros_like(cfg_grid["ky"])
    one_over_ky[1:] = 1.0 / cfg_grid["ky"][1:]
    cfg_grid["one_over_ky"] = jnp.array(one_over_ky)

    one_over_kyr = np.zeros_like(cfg_grid["kyr"])
    one_over_kyr[1:] = 1.0 / cfg_grid["kyr"][1:]
    cfg_grid["one_over_kyr"] = jnp.array(one_over_kyr)

    one_over_kv = np.zeros_like(cfg_grid["kv"])
    one_over_kv[1:] = 1.0 / cfg_grid["kv"][1:]
    cfg_grid["one_over_kv"] = jnp.array(one_over_kv)

    one_over_kvr = np.zeros_like(cfg_grid["kvr"])
    one_over_kvr[1:] = 1.0 / cfg_grid["kvr"][1:]
    cfg_grid["one_over_kvr"] = jnp.array(one_over_kvr)

    return cfg_grid


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    # cfg["save"]["func"] = {**cfg["save"]["func"], **{"callable": get_save_func(cfg)}}
    cfg["save"]["t"]["ax"] = jnp.linspace(cfg["save"]["t"]["tmin"], cfg["save"]["t"]["tmax"], cfg["save"]["t"]["nt"])
    # cfg["save"]["x"]["ax"] = jnp.linspace(cfg["save"]["x"]["xmin"], cfg["save"]["x"]["xmax"], cfg["save"]["x"]["nx"])
    # cfg["save"]["y"]["ax"] = jnp.linspace(cfg["save"]["y"]["ymin"], cfg["save"]["y"]["ymax"], cfg["save"]["y"]["ny"])

    return cfg


def init_state(cfg: Dict) -> Dict:
    """
    This function initializes the state

    :param cfg:
    :return:
    """

    nx = cfg["grid"]["nx"]
    ny = cfg["grid"]["ny"]
    nv = cfg["grid"]["nv"]
    norm = 1.0 / (
        4.0 * jnp.pi * cfg["grid"]["dv"] * jnp.sum(cfg["grid"]["v"] ** 2.0 * jnp.exp(-cfg["grid"]["v"] ** 2.0 / 2.0))
    )

    density_profile = jnp.ones((nx, ny))  # cfg["profile"]["density"]
    temperature_profile = jnp.ones((nx, ny))  # cfg["profile"]["temperature"]
    state = {}
    state["flm"] = {}
    state["flm"][0] = {}
    state["flm"][0][0] = (
        norm
        * density_profile[:, :, None]
        * jnp.exp(-cfg["grid"]["v"][None, None, :] ** 2.0 / 2.0 / temperature_profile[:, :, None])
    )
    state["flm"][0][0] = jnp.array(state["flm"][0][0], dtype=jnp.complex128).view(dtype=jnp.float64)

    for il in range(1, cfg["grid"]["nl"] + 1):
        state["flm"][il] = {}
        for im in range(0, il + 1):
            state["flm"][il][im] = jnp.zeros((nx, ny, nv), dtype=jnp.complex128).view(dtype=jnp.float64)

    state["e"] = jnp.zeros((nx, ny, 3))
    state["b"] = jnp.zeros((nx, ny, 3))

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
    push_vlasov: eqx.Module
    push_driver: eqx.Module
    poisson_solver: eqx.Module

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.push_vlasov = vlasov.Vlasov(cfg)
        self.push_driver = vlasov.Driver(cfg["grid"]["x"], cfg["grid"]["y"])
        # cfg["profiles"]["ion_charge"]
        self.poisson_solver = field.SpectralPoissonSolver(
            jnp.ones((cfg["grid"]["nx"], cfg["grid"]["ny"])),
            cfg["grid"]["one_over_kx"],
            cfg["grid"]["one_over_ky"],
            cfg["grid"]["dv"],
            cfg["grid"]["v"],
        )

    def __call__(self, t: float, y: Dict, args: Dict):
        """
        This function is used by the time integrators specified in diffrax

        :param t:
        :param y:
        :param args:
        :return:
        """
        for il in range(0, self.cfg["grid"]["nl"] + 1):
            for im in range(0, il + 1):
                y["flm"][il][im] = y["flm"][il][im].view(dtype=jnp.complex128)

        y["e"] = self.poisson_solver(y["flm"][0][0])
        # y["b"] = self.b_solver(y)

        ed = 0.0

        for p_ind in self.cfg["drivers"]["ex"].keys():
            ed += self.push_driver(args["driver"]["ex"][p_ind], t)

        total_e = y["e"] + jnp.concatenate(
            [ed[:, :, None], jnp.zeros_like(ed[:, :, None]), jnp.zeros_like(ed[:, :, None])], axis=-1
        )
        total_b = y["b"] + args["b_ext"]

        dydt = {
            "flm": self.push_vlasov(y["flm"], total_e, total_b),
            "e": jnp.zeros_like(total_e),
            "b": jnp.zeros_like(total_e),
        }

        for il in range(0, self.cfg["grid"]["nl"] + 1):
            for im in range(0, il + 1):
                y["flm"][il][im] = y["flm"][il][im].view(dtype=jnp.float64)
                dydt["flm"][il][im] = dydt["flm"][il][im].view(dtype=jnp.float64)

        return dydt
