import itertools
from typing import Dict
import os

# import interpax
from jax import numpy as jnp
import numpy as np
import xarray as xr


def store_fields(cfg: Dict, td: str, fields: Dict, this_t: np.ndarray, prefix: str) -> xr.Dataset:
    """
    Stores fields to netcdf

    :param prefix:
    :param cfg:
    :param td:
    :param fields:
    :param this_t:
    :return:
    """
    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir, exist_ok=True)

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
        das = {
            f"{prefix}-{k}": xr.DataArray(v, coords=(("t", this_t), ("x", cfg["grid"]["x"]))) for k, v in fields.items()
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
            spc: xr.DataArray(ys[spc], coords=(("t", this_t[spc]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"])))
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
            temp = {
                "n": jnp.trapz(y["electron"], dx=cfg["grid"]["dv"], axis=1),
                "v": jnp.trapz(y["electron"] * cfg["grid"]["v"][None, :], dx=cfg["grid"]["dv"], axis=1),
            }
            v_m_vbar = cfg["grid"]["v"][None, :] - temp["v"][:, None]
            temp["p"] = jnp.trapz(y["electron"] * v_m_vbar**2.0, dx=cfg["grid"]["dv"], axis=1)
            temp["q"] = jnp.trapz(y["electron"] * v_m_vbar**3.0, dx=cfg["grid"]["dv"], axis=1)

            return temp

    else:
        raise NotImplementedError

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
            if k2 == "x":
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

    return cfg
