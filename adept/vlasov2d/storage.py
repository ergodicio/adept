import itertools
from typing import Dict
import os

# import interpax
from jax import numpy as jnp
import numpy as np
import xarray as xr


def store_fields(cfg: Dict, binary_dir: str, fields: Dict, this_t: np.ndarray, prefix: str) -> xr.Dataset:
    """
    Stores fields to netcdf

    :param prefix:
    :param cfg:
    :param binary_dir:
    :param fields:
    :param this_t:
    :return:
    """

    if any(x in ["x", "kx", "y", "ky"] for x in cfg["save"][prefix].keys()):
        crds = set(cfg["save"][prefix].keys()) - {"t", "func"}
        if {"x", "y"} == crds:
            xnm, ynm = "x", "y"
        elif {"kx", "y"} == crds:
            xnm, ynm = "kx", "y"
        elif {"x", "ky"} == crds:
            xnm, ynm = "x", "ky"
        elif {"kx", "ky"} == crds:
            xnm, ynm = "kx", "ky"
        else:
            raise NotImplementedError

        xx, yx = cfg["save"][prefix][xnm]["ax"], cfg["save"][prefix][ynm]["ax"]
        das = {
            [f"{prefix}-{k}"]: xr.DataArray(v, coords=(("t", this_t), (xnm, xx), (ynm, yx))) for k, v in fields.items()
        }
    else:
        das = {
            f"{prefix}-{k}": xr.DataArray(v, coords=(("t", this_t), ("x", cfg["grid"]["x"]), ("y", cfg["grid"]["y"])))
            for k, v in fields.items()
        }
    fields_xr = xr.Dataset(das)
    fields_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-t={round(this_t[-1],4)}.nc"))

    return fields_xr


def store_f(cfg: Dict, this_t: Dict, td: str, ys: Dict) -> xr.Dataset:
    """
    Stores f to netcdf

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    f_store = xr.Dataset(
        {
            spc: xr.DataArray(
                ys[spc],
                coords=(
                    ("t", this_t[spc]),
                    ("x", cfg["grid"]["x"]),
                    ("y", cfg["grid"]["y"]),
                    ("v_x", cfg["grid"]["vx"]),
                    ("v_y", cfg["grid"]["vy"]),
                ),
            )
            for spc in ["electron"]
        }
    )
    f_store.to_netcdf(os.path.join(td, "binary", "dist.nc"))

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

        def fields_save_func(t, y, args):
            return {this_k: y[this_k] for this_k in ["ex", "ey", "bz", "dex", "dey"]}

    else:
        raise NotImplementedError
        # if {"t", "x", "y"} == set(cfg["save"][k].keys()):
        #     xhat, yhat = cfg["save"][k]["x"]["ax"], cfg["save"][k]["y"]["ax"]
        #     xax, yax = cfg["grid"]["x"], cfg["grid"]["y"]
        #     nx, ny = xax.size, yax.size
        #
        #     def trans_func(arr):
        #         return arr
        #
        # elif {"t", "kx", "y"} == set(cfg["save"][k].keys()):
        #     xhat, yhat = cfg["save"][k]["kx"]["ax"], cfg["save"][k]["y"]["ax"]
        #     xax, yax = cfg["grid"]["kx"], cfg["grid"]["y"]
        #     nx, ny = xax.size, yax.size
        #
        #     def trans_func(arr):
        #         return jnp.fft.rfft(arr, axis=0) * 2 / nx
        #
        # elif {"t", "x", "ky"} == set(cfg["save"][k].keys()):
        #     xhat, yhat = cfg["save"][k]["x"]["ax"], cfg["save"][k]["ky"]["ax"]
        #     xax, yax = cfg["grid"]["x"], cfg["grid"]["ky"]
        #     nx, ny = xax.size, yax.size
        #
        #     def trans_func(arr):
        #         return jnp.fft.rfft(arr, axis=1) * 2 / ny
        #
        # elif {"t", "kx", "ky"} == set(cfg["save"][k].keys()):
        #     xhat, yhat = cfg["save"][k]["kx"]["ax"], cfg["save"][k]["ky"]["ax"]
        #     xax, yax = cfg["grid"]["kx"], cfg["grid"]["ky"]
        #     nx, ny = xax.size, yax.size
        #
        #     def trans_func(arr):
        #         return jnp.fft.rfft(jnp.fft.rfft(arr, axis=0), axis=1) * 4 / (nx * ny)
        #
        # else:
        #     raise NotImplementedError
        #
        # coords = list(itertools.product(xhat, yhat))
        # xq = [tup[0] for tup in coords]
        # yq = [tup[1] for tup in coords]
        #
        # def fields_save_func(t, y, args):
        #     flds = {}
        #     for this_k in ["e", "b", "de"]:
        #         arr = trans_func(y[this_k])
        #         fldx = interpax.interp2d(xq, yq, xax, yax, arr[..., 0], method="linear").reshape(
        #             (xhat.size, yhat.size, 1)
        #         )
        #         fldy = interpax.interp2d(xq, yq, xax, yax, arr[..., 1], method="linear").reshape(
        #             (xhat.size, yhat.size, 1)
        #         )
        #
        #         flds[this_k] = jnp.concatenate([fldx, fldy], axis=-1)
        #     return flds

    return fields_save_func


def get_dist_save_func(cfg, k):
    if {"t"} == set(cfg["save"][k].keys()):

        def dist_save_func(t, y, args):
            return y["electron"]

    elif {"t", "x", "y"} == set(cfg["save"][k].keys()):
        pass
    elif {"t", "kx", "y"} == set(cfg["save"][k].keys()):
        pass

    elif {"t", "x", "ky"} == set(cfg["save"][k].keys()):
        pass

    elif {"t", "kx", "ky"} == set(cfg["save"][k].keys()):
        pass
    else:
        raise NotImplementedError

    return dist_save_func


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    for k in cfg["save"].keys():
        for k2 in cfg["save"][k].keys():
            if (k2 == "x") or (k2 == "y"):
                dx = (cfg["save"][k][k2][f"{k2}max"] - cfg["save"][k][k2][f"{k2}min"]) / cfg["save"][k][k2][f"n{k2}"]
                cfg["save"][k][k2]["ax"] = np.linspace(
                    cfg["save"][k][k2][f"{k2}min"] + dx / 2.0,
                    cfg["save"][k][k2][f"{k2}max"] - dx / 2.0,
                    cfg["save"][k][k2][f"n{k2}"],
                )

            else:
                cfg["save"][k][k2]["ax"] = np.linspace(
                    cfg["save"][k][k2][f"{k2}min"], cfg["save"][k][k2][f"{k2}max"], cfg["save"][k][k2][f"n{k2}"]
                )

        if k.startswith("fields"):
            cfg["save"][k]["func"] = get_field_save_func(cfg, k)

        elif k.startswith("electron"):
            cfg["save"][k]["func"] = get_dist_save_func(cfg, k)

    cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}, "func": get_default_save_func(cfg)}

    return cfg


def get_default_save_func(cfg):
    vx = cfg["grid"]["vx"][None, None, :, None]
    vy = cfg["grid"]["vx"][None, None, None, :]
    dvx = cfg["grid"]["dvx"]
    dvy = cfg["grid"]["dvx"]

    def _calc_mean_moment_(inp):
        return jnp.mean(jnp.sum(jnp.sum(inp, axis=3), axis=2)) * dvy * dvx

    def save(t, y, args):
        scalars = {
            "mean_P": _calc_mean_moment_(y["electron"] * vx**2.0 * vy**2.0),
            "mean_j": _calc_mean_moment_(y["electron"] * vx * vy),
            "mean_n": _calc_mean_moment_(y["electron"]),
            "mean_q": _calc_mean_moment_(y["electron"] * vx**3.0 * vy**3.0),
            "mean_-flogf": _calc_mean_moment_(-jnp.log(jnp.abs(y["electron"])) * jnp.abs(y["electron"])),
            "mean_f2": _calc_mean_moment_(y["electron"] * y["electron"]),
            "mean_de2": jnp.mean(y["dex"] ** 2.0 + y["dey"] ** 2.0),
            "mean_e2": jnp.mean(y["ex"] ** 2.0 + y["ey"] ** 2.0),
        }

        return scalars

    return save
