import itertools
from typing import Dict
import os

# import interpax
from functools import partial
from jax import numpy as jnp, vmap
import numpy as np
import xarray as xr


def store_fields(cfg: Dict, binary_dir: str, fields: Dict, this_t: np.ndarray, prefix: str) -> xr.Dataset:
    """
    Stores fields to netcdf

    :param prefix:
    :param cfg:
    :param td:
    :param fields:
    :param this_t:
    :return:
    """

    if any(x in ["x", "kx"] for x in cfg["save"][prefix].keys()):
        crds = set(cfg["save"][prefix].keys()) - {"t", "func"}
        if {"x"} == crds:
            xnm = "x"
        elif {"kx"} == crds:
            xnm = "kx"
        else:
            raise NotImplementedError

        xx = cfg["save"][prefix][xnm]["ax"]

        das = {f"{prefix}-{k}": xr.DataArray(v, coords=(("t", this_t), (xnm, xx))) for k, v in fields.items()}
    else:
        das = {}
        for k, v in fields.items():
            das[f"{prefix}-{k}"] = xr.DataArray(
                v[:, 1:-1] if k in ["a", "prev_a"] else v, coords=(("t", this_t), ("x", cfg["grid"]["x"]))
            )

        if len(cfg["drivers"]["ey"].keys()) > 0:
            ey = -(fields["a"][:, 1:-1] - fields["prev_a"][:, 1:-1]) / cfg["grid"]["dt"]
            bz = jnp.gradient(fields["a"], cfg["grid"]["dx"], axis=1)[:, 1:-1]

            ep = ey + cfg["units"]["derived"]["c_light"].magnitude * bz
            em = ey - cfg["units"]["derived"]["c_light"].magnitude * bz

            das[f"{prefix}-ep"] = xr.DataArray(ep, coords=(("t", this_t), ("x", cfg["grid"]["x"])))
            das[f"{prefix}-em"] = xr.DataArray(em, coords=(("t", this_t), ("x", cfg["grid"]["x"])))

    fields_xr = xr.Dataset(das)
    fields_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-t={round(this_t[-1],4)}.nc"))

    return fields_xr


def store_f(cfg: Dict, this_t: Dict, binary_dir: str, ys: Dict) -> Dict:
    """
    Stores f to netcdf

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    f_store = {}
    for k, save_dict in cfg["save"].items():
        if k.startswith("electron"):
            if {"t", "vx", "vy", "x", "func"} == set(save_dict.keys()):
                f_store[k] = xr.DataArray(
                    ys[k],
                    coords=(
                        ("t", this_t[k]),
                        ("x", save_dict["x"]["ax"]),
                        ("vx", save_dict["vx"]["ax"]),
                        ("vy", save_dict["vy"]["ax"]),
                    ),
                )
                f_store[k].to_netcdf(os.path.join(binary_dir, "f_t-x-vx-vy.nc"))
            elif {"func", "t", "vx", "x"} == set(save_dict.keys()):
                f_store[k] = xr.DataArray(
                    ys[k], coords=(("t", this_t[k]), ("x", save_dict["x"]["ax"]), ("vx", save_dict["vx"]["ax"]))
                )
                f_store[k].to_netcdf(os.path.join(binary_dir, "f_t-x-vx.nc"))

    return f_store


# def clean_td(td):
#     _ = [os.remove(os.path.join(td, "binary", "fields", fl)) for fl in os.listdir(os.path.join(td, "binary", "fields"))]
#     _ = [os.remove(os.path.join(td, "binary", "f", fl)) for fl in os.listdir(os.path.join(td, "binary", "f"))]
#
#
# def first_store(td, cfg):
#     os.makedirs(os.path.join(td, "binary", "f"))
#     os.makedirs(os.path.join(td, "binary", "fields"))
#
#     start_f = jnp.array(cfg["grid"]["f"])
#     store_f(cfg, np.array([0.0]), td, start_f)
#     fields = {
#         "ex": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
#         "ey": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
#         "total_ex": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
#         "total_ey": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
#         "dex": np.zeros((1, cfg["grid"]["nx"], cfg["grid"]["ny"])),
#     }
#     store_fields(cfg, td, fields, np.array([0.0]))
#     mlflow.log_artifacts(td)
#     clean_td(td)
#
#     return start_f
#
#
# def store_everything(td, cfg, this_t, fields, this_driver, i, running_f):
#     fields["dex"] = this_driver[:, 0]
#     store_fields(cfg, td, fields, this_t)
#
#     if i % (cfg["grid"]["num_checkpoints"] // cfg["grid"]["num_fs"]) == 0:
#         store_f(cfg, this_t[-1:], td, running_f)
#         mlflow.log_artifacts(td)
#         clean_td(td)


def get_field_save_func(cfg, k):
    if {"t"} == set(cfg["save"][k].keys()):

        def _calc_moment_(inp):
            return jnp.sum(jnp.sum(inp, axis=2), axis=1) * cfg["grid"]["dv"] * cfg["grid"]["dv"]

        def fields_save_func(t, y, args):
            temp = {
                "n": _calc_moment_(y["electron"]),
                "vx": _calc_moment_(y["electron"] * cfg["grid"]["v"][None, :, None]),
                "vy": _calc_moment_(y["electron"] * cfg["grid"]["v"][None, None, :]),
            }
            vx_m_vxbar = cfg["grid"]["v"][None, :, None] - temp["vx"][:, None, None]
            vy_m_vybar = cfg["grid"]["v"][None, None, :] - temp["vy"][:, None, None]
            temp["px"] = _calc_moment_(y["electron"] * vx_m_vxbar**2.0)
            temp["qx"] = _calc_moment_(y["electron"] * vx_m_vxbar**3.0)
            temp["py"] = _calc_moment_(y["electron"] * vy_m_vybar**2.0)
            temp["qy"] = _calc_moment_(y["electron"] * vy_m_vybar**3.0)
            temp["-flogf"] = _calc_moment_(y["electron"] * jnp.log(jnp.abs(y["electron"])))
            temp["f^2"] = _calc_moment_(y["electron"] * y["electron"])
            temp["e"] = y["e"]
            temp["de"] = y["de"]
            temp["a"] = y["a"]
            temp["prev_a"] = y["prev_a"]
            temp["pond"] = -0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1]

            return temp

    else:
        raise NotImplementedError

    return fields_save_func


def _get_x_save_func_(save_dict, cfg_grid):
    nx = save_dict["nx"]
    if "xmin" in save_dict:
        xmin = save_dict["xmin"]
        xmax = save_dict["xmax"]
    else:
        xmin = cfg_grid["xmin"]
        xmax = cfg_grid["xmax"]

    dx = (xmax - xmin) / nx

    save_x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
    interp_x = vmap(partial(jnp.interp, x=save_x, xp=cfg_grid["x"]), in_axes=1, out_axes=1)

    return save_x, interp_x


def _get_vx_save_func_(save_dict, cfg_grid):
    nvx = save_dict["nvx"]
    if "vxmin" in save_dict:
        vxmin = save_dict["vxmin"]
        vxmax = save_dict["vxmax"]
    else:
        vxmin = -cfg_grid["vmax"]
        vxmax = cfg_grid["vmax"]

    dvx = (vxmax - vxmin) / nvx

    save_v = np.linspace(vxmin + dvx / 2.0, vxmax - dvx / 2.0, nvx)
    interp_v = vmap(partial(jnp.interp, x=save_v, xp=cfg_grid["v"]), in_axes=0, out_axes=0)

    return save_v, interp_v


def get_dist_save_func(cfg, k):
    if {"t"} == set(cfg["save"][k].keys()):
        cfg["save"][k] = cfg["save"][k] | {"x": {}, "vx": {}, "vy": {}}
        cfg["save"][k]["x"]["ax"] = cfg["grid"]["x"]
        cfg["save"][k]["vx"]["ax"] = cfg["grid"]["v"]
        cfg["save"][k]["vy"]["ax"] = cfg["grid"]["v"]

        def dist_save_func(t, y, args):
            return y["electron"]

    elif {"t", "x", "vx"} == set(cfg["save"][k].keys()):

        if "nx" in cfg["save"][k]["x"]:
            cfg["save"][k]["x"]["ax"], interp_x = _get_x_save_func_(cfg["save"][k]["x"], cfg["grid"])
        else:
            cfg["save"][k]["x"]["ax"] = cfg["grid"]["x"]

            def interp_x(fp):
                return fp

        if "nvx" in cfg["save"][k]["vx"]:
            cfg["save"][k]["vx"]["ax"], interp_vx = _get_vx_save_func_(cfg["save"][k]["vx"], cfg["grid"])
        else:
            cfg["save"][k]["vx"]["ax"] = cfg["grid"]["v"]

            def interp_vx(fp):
                return fp

        def dist_save_func(t, y, args):
            fxvx = jnp.sum(y["electron"], axis=2) * cfg["grid"]["dv"]
            f_interp_x = interp_x(fp=fxvx)
            f_interp_xv = interp_vx(fp=f_interp_x)
            return f_interp_xv

    else:
        raise NotImplementedError

    cfg["save"][k]["func"] = dist_save_func

    return cfg["save"][k]


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    for k in cfg["save"].keys():  # this can be fields or electron or scalar?
        # for k2 in cfg["save"][k].keys():  # this can be t, x, y, kx, ky (eventually)
        #     if k2 == "x":
        #         dx = (cfg["save"][k][k2][f"{k2}max"] - cfg["save"][k][k2][f"{k2}min"]) / cfg["save"][k][k2][f"n{k2}"]
        #         cfg["save"][k][k2]["ax"] = np.linspace(
        #             cfg["save"][k][k2][f"{k2}min"] + dx / 2.0,
        #             cfg["save"][k][k2][f"{k2}max"] - dx / 2.0,
        #             cfg["save"][k][k2][f"n{k2}"],
        #         )

        #     else:
        k2 = "t"
        cfg["save"][k][k2]["ax"] = np.linspace(
            cfg["save"][k][k2][f"{k2}min"], cfg["save"][k][k2][f"{k2}max"], cfg["save"][k][k2][f"n{k2}"]
        )

        if k.startswith("fields"):
            cfg["save"][k]["func"] = get_field_save_func(cfg, k)

        elif k.startswith("electron"):
            cfg["save"][k] = get_dist_save_func(cfg, k)

    cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}, "func": get_default_save_func(cfg)}

    return cfg


def get_default_save_func(cfg):
    v = cfg["grid"]["v"][None, :]
    dv = cfg["grid"]["dv"]

    def _calc_mean_moment_(inp):
        return jnp.mean(jnp.sum(jnp.sum(inp, axis=2), axis=1)) * dv * dv

    def save(t, y, args):
        scalars = {
            "mean_P": _calc_mean_moment_(y["electron"] * v**2.0),
            "mean_j": _calc_mean_moment_(y["electron"] * v),
            "mean_n": _calc_mean_moment_(y["electron"]),
            "mean_q": _calc_mean_moment_(y["electron"] * v**3.0),
            "mean_-flogf": _calc_mean_moment_(-jnp.log(jnp.abs(y["electron"])) * jnp.abs(y["electron"])),
            "mean_f2": _calc_mean_moment_(y["electron"] * y["electron"]),
            "mean_de2": jnp.mean(y["de"] ** 2.0),
            "mean_e2": jnp.mean(y["e"] ** 2.0),
            "mean_pond": jnp.mean(-0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1]),
        }

        return scalars

    return save
