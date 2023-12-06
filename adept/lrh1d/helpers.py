from collections import defaultdict
from typing import Callable, Dict
from functools import partial

import os

import jax.random
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from jax import tree_util as jtu
from flatdict import FlatDict
import equinox as eqx
from diffrax import ODETerm, Tsit5

from jax import numpy as jnp
from adept.lrh1d import pushers
from equinox import nn


def save_arrays(result, td, cfg, label):
    # if label is None:
    #     label = "x"
    #     flattened_dict = dict(FlatDict(result.ys, delimiter="-"))
    #     save_ax = cfg["grid"]["x"]
    # else:
    #     flattened_dict = dict(FlatDict(result.ys[label], delimiter="-"))
    #     save_ax = cfg["save"][label]["ax"]
    # data_vars = {
    #     k: xr.DataArray(v, coords=(("t", cfg["save"]["t"]["ax"]), (label, save_ax))) for k, v in flattened_dict.items()
    # }
    #
    # saved_arrays_xr = xr.Dataset(data_vars)
    # saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"state_vs_{label}.nc"))
    # return saved_arrays_xr
    pass


# def plot_xrs(which, td, xrs):
#     os.makedirs(os.path.join(td, "plots", which))
#     os.makedirs(os.path.join(td, "plots", which, "ion"))
#     os.makedirs(os.path.join(td, "plots", which, "electron"))
#
#     for k, v in xrs.items():
#         fname = f"{'-'.join(k.split('-')[1:])}.png"
#         fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
#         v.plot(ax=ax, cmap="gist_ncar")
#         ax.grid()
#         fig.savefig(os.path.join(td, "plots", which, k.split("-")[0], fname), bbox_inches="tight")
#         plt.close(fig)
#
#         if which == "kx":
#             os.makedirs(os.path.join(td, "plots", which, "ion", "hue"), exist_ok=True)
#             os.makedirs(os.path.join(td, "plots", which, "electron", "hue"), exist_ok=True)
#             # only plot
#             if v.coords["kx"].size > 8:
#                 hue_skip = v.coords["kx"].size // 8
#             else:
#                 hue_skip = 1
#
#             for log in [True, False]:
#                 fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)
#                 v[:, ::hue_skip].plot(ax=ax, hue="kx")
#                 ax.set_yscale("log" if log else "linear")
#                 ax.grid()
#                 fig.savefig(
#                     os.path.join(
#                         td, "plots", which, k.split("-")[0], f"hue", f"{'-'.join(k.split('-')[1:])}-log-{log}.png"
#                     ),
#                     bbox_inches="tight",
#                 )
#                 plt.close(fig)


def post_process(result, cfg: Dict, td: str) -> Dict:
    # os.makedirs(os.path.join(td, "binary"))
    # os.makedirs(os.path.join(td, "plots"))
    #
    datasets = {}
    # if any(x in ["x", "kx"] for x in cfg["save"]):
    #     if "x" in cfg["save"].keys():
    #         datasets["x"] = save_arrays(result, td, cfg, label="x")
    #         plot_xrs("x", td, datasets["x"])
    #     if "kx" in cfg["save"].keys():
    #         datasets["kx"] = save_arrays(result, td, cfg, label="kx")
    #         plot_xrs("kx", td, datasets["kx"])
    # else:
    #     datasets["full"] = save_arrays(result, td, cfg, label=None)
    #     plot_xrs("x", td, datasets["full"])

    return datasets


def get_derived_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    # cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    # cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    return cfg_grid


def get_solver_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are arrays

    This is run after the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    # cfg_grid = {
    #     **cfg_grid,
    #     **{
    #         "x": jnp.linspace(
    #             cfg_grid["xmin"] + cfg_grid["dx"] / 2, cfg_grid["xmax"] - cfg_grid["dx"] / 2, cfg_grid["nx"]
    #         ),
    #         "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
    #         "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
    #         "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
    #     },
    # }
    #
    # one_over_kx = np.zeros_like(cfg_grid["kx"])
    # one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    # cfg_grid["one_over_kx"] = jnp.array(one_over_kx)
    #
    # one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    # one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    # cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

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
    state = dict(
        u=jnp.zeros(cfg["grid"]["nx"]+1),
        r=jnp.zeros(cfg["grid"]["nx"]+1),
        Ti=jnp.zeros(cfg["grid"]["nx"]),
        Te=jnp.zeros(cfg["grid"]["nx"]),
        Tr=jnp.zeros(cfg["grid"]["nx"]),
        rho=jnp.zeros(cfg["grid"]["nx"]),
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
        self.eta0 = None
        self.sigma_sb = None
        self.cfg = cfg
        self.eos = pushers.IdealGas(cfg)
        self.dt = cfg["grid"]["dt"]
        self.gamma = 3.0
        self.temperature = pushers.Temperature(cfg)
        # self.velocity = pushers.Temperature(cfg)
        # self.density = pushers.density(cfg)

    def __call__(self, t: float, y: Dict, args: Dict):
        """
        This function is used by the time integrators specified in diffrax

        :param t:
        :param y:
        :param args:
        :return:
        """
        u, r, rho = (
            jnp.concatenate([[0.0], y["u"], y["u"][-1:]]),
            jnp.concatenate([[0.0], y["r"], y["r"][-1:]]),
            jnp.concatenate([y["rho"][0:1], y["rho"], y["rho"][-1:]]),
        )
        delta_r, vol = r[1:] - r[:-1], jnp.pi * 4.0 / 3.0 * r**3.0
        Te, Ti, Tr = y["Te"], y["Ti"], y["Tr"]

        ne, ni = self.get_density_from_mass_density(rho)
        pi, pe, pr = self.eos(ne=ne, ni=ni, Te=Te, Ti=Ti, Tr=Tr)

        p_all = pi + pe + pr

        new_u = u - self.dt / rho[1:-1] / delta_r[:-1] * ((p_all + q)[2:] - (p_all + q)[1:-1])
        new_r = r - 0.5 * self.dt * (new_u + u[1:-1])
        padded_new_r = jnp.concatenate([[0.0], new_r, new_r[-1:]])

        new_vol = 4.0 / 3.0 * jnp.pi * new_r**3.0
        new_rho = rho * (vol[1:-1] / new_vol)

        new_Ti, new_Te, new_Tr = self.temperature(new_rho, new_u, new_r, Ti, Te, Tr, pi, pe, pr)

        new_y = {"u": new_u, "r": new_r, "rho": new_rho, "Ti": new_Ti, "Te": new_Te, "Tr": new_Tr}

        return new_y


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
