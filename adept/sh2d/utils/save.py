import os
from functools import partial
from jax import tree_util as jtu

import xarray as xr
from flatdict import FlatDict
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt


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


#
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
#


def save_vector_fields(result, cfg, td):
    """
    saves vector fields to an xarray dataset

    :param result:
    :param cfg:
    :param td:
    :return:
    """
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
        for k in ["e", "b", "de", "db"]
    }

    data_vars["j"] = xr.DataArray(
        calc_j(result.ys["flm"][1], v=cfg["grid"]["v"]),
        coords=(
            ("t", cfg["save"]["t"]["ax"]),
            ("x", cfg["grid"]["x"]),
            ("y", cfg["grid"]["y"]),
            ("component", ["x", "y", "z"]),
        ),
    )

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"vector-fields.nc"))

    num_plots = 8
    t_skip = int(saved_arrays_xr.coords["t"].size // num_plots)
    if t_skip == 1:
        slc = slice(0, num_plots, 1)
    elif t_skip == 0:
        slc = slice(0, -1)
    else:
        slc = slice(t_skip // 2, -1, t_skip)

    for k, v in saved_arrays_xr.items():
        plot_path = os.path.join(td, "plots", k)
        os.makedirs(plot_path, exist_ok=True)
        for i, comp in enumerate(saved_arrays_xr.coords["component"].data):
            v[slc, ..., i].plot(figsize=(12, 6), x="x", y="y", col="t", col_wrap=4)
            plt.savefig(os.path.join(plot_path, f"{comp}.png"), bbox_inches="tight")
            plt.close()

    return saved_arrays_xr


def calc_n(f00, v):
    return np.real(
        4.0 * np.pi * (v[2] - v[1]) * np.sum(f00.view(np.complex128) * v[None, None, None, :] ** 2.0, axis=-1)
    )


def calc_T(f00, v):
    return np.real(
        4.0 * np.pi * (v[2] - v[1]) * np.sum(0.5 * f00.view(np.complex128) * v[None, None, None, :] ** 4.0, axis=-1)
    )


def calc_j(f1, v):
    return jnp.concatenate(
        [
            -np.real(
                4.0 * np.pi * (v[2] - v[1]) * np.sum(f1[0].view(np.complex128) * v[None, None, None, :] ** 3.0, axis=-1)
            )[..., None],
            -8.0
            * np.pi
            * (v[2] - v[1])
            * np.sum(np.real(f1[1].view(np.complex128)) * v[None, None, None, :] ** 3.0, axis=-1)[..., None],
            8.0
            * np.pi
            * (v[2] - v[1])
            * np.sum(np.imag(f1[1].view(np.complex128)) * v[None, None, None, :] ** 3.0, axis=-1)[..., None],
        ],
        axis=-1,
    )


def calc_q():
    pass


def save_scalar_fields(result, cfg, td):
    """
    saves vector fields to an xarray dataset

    :param result:
    :param cfg:
    :param td:
    :return:
    """
    data_vars = {
        k: xr.DataArray(
            func(result.ys["flm"][0][0], cfg["grid"]["v"]),
            coords=(
                ("t", cfg["save"]["t"]["ax"]),
                ("x", cfg["grid"]["x"]),
                ("y", cfg["grid"]["y"]),
            ),
        )
        for k, func in zip(["n", "T"], [calc_n, calc_T])
    }

    saved_arrays_xr = xr.Dataset(data_vars)
    saved_arrays_xr.to_netcdf(os.path.join(td, "binary", f"scalar-fields.nc"))

    num_plots = 8
    t_skip = int(saved_arrays_xr.coords["t"].size // num_plots)
    if t_skip == 1:
        slc = slice(0, num_plots, 1)
    elif t_skip == 0:
        slc = slice(0, -1)
    else:
        slc = slice(t_skip // 2, -1, t_skip)

    for k, v in saved_arrays_xr.items():
        plot_path = os.path.join(td, "plots", f"{k}.png")
        v[slc].plot(figsize=(12, 6), x="x", y="y", col="t", col_wrap=4)
        plt.savefig(os.path.join(plot_path), bbox_inches="tight")
        plt.close()

    return saved_arrays_xr


def save_dists(result, cfg, td):
    """
    Saves flm distributions to an xarray dataset

    :param result:
    :param cfg:
    :param td:
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

    return saved_arrays_xr


def save_arrays(result, td, cfg):
    """

    :param result:
    :param td:
    :param cfg:
    :return:
    """
    dists = save_dists(result, cfg, td)
    vector_fields = save_vector_fields(result, cfg, td)
    scalar_fields = save_scalar_fields(result, cfg, td)

    return dists, vector_fields
