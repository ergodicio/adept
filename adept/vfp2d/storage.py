import itertools
from typing import Dict, Tuple
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
    :param td:
    :param fields:
    :param this_t:
    :return:
    """

    xax = cfg["units"]["derived"]["x0"].to("micron").value * cfg["grid"]["x"]
    tax = this_t * cfg["units"]["derived"]["tp0"].to("ps").value

    if any(x in ["x", "kx"] for x in cfg["save"][prefix].keys()):
        crds = set(cfg["save"][prefix].keys()) - {"t", "func"}
        if {"x"} == crds:
            xnm = "x"
        elif {"kx"} == crds:
            xnm = "kx"
        else:
            raise NotImplementedError

        xx = cfg["save"][prefix][xnm]["ax"]

        das = {f"{prefix}-{k}": xr.DataArray(v, coords=(("t (ps)", tax), (xnm, xx))) for k, v in fields.items()}
    else:
        das = {}
        for k, v in fields.items():
            units, scaling = get_unit(k, cfg)

            das[f"{prefix}-{k} {units}"] = xr.DataArray(v * scaling, coords=(("t (ps)", tax), ("x (um)", xax)))

    fields_xr = xr.Dataset(das)
    fields_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-t={round(tax[-1],4)}.nc"))

    return fields_xr


def get_unit(k, cfg: Dict = None) -> Tuple[str, float]:
    # if k == "e":
    #     return "V/m", 1.0
    if k == "T":
        return "keV", 510.999
    elif k == "n":
        return "n_c", 1.0
    else:
        return "a.u.", 1.0


def store_f(cfg: Dict, this_t: Dict, td: str, ys: Dict) -> xr.Dataset:
    """
    Stores f to netcdf

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    xax = cfg["units"]["derived"]["x0"].to("micron").value * cfg["grid"]["x"]
    tax = this_t["electron"] * cfg["units"]["derived"]["tp0"].to("ps").value

    f_store = xr.Dataset(
        {
            dist: xr.DataArray(
                ys["electron"][dist],
                coords=(("t (ps)", this_t["electron"]), ("x (um)", xax), ("v (c)", cfg["grid"]["v"])),
            )
            for dist in ys["electron"].keys()
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

        def _calc_f0_moment_(f0):
            return 4 * jnp.pi * jnp.trapz(f0 * cfg["grid"]["v"] ** 2.0, dx=cfg["grid"]["dv"], axis=1)

        def _calc_f1_moment_(f1):
            return 4 / 3 * jnp.pi * jnp.trapz(f1 * cfg["grid"]["v"] ** 3.0, dx=cfg["grid"]["dv"], axis=1)

        def fields_save_func(t, y, args):
            temp = {"n": _calc_f0_moment_(y["f0"]), "v": _calc_f1_moment_(y["f10"])}
            temp["U"] = _calc_f0_moment_(0.5 * y["f0"] * cfg["grid"]["v"] ** 2.0)
            temp["P"] = temp["U"] / 1.5
            temp["T"] = temp["P"] / temp["n"]
            temp["q"] = _calc_f1_moment_(0.5 * y["f10"] * cfg["grid"]["v"] ** 2.0)
            temp["e"] = y["e"]
            temp["b"] = y["b"]

            return temp

    else:
        raise NotImplementedError

    return fields_save_func


def get_dist_save_func(cfg, k):
    if {"t"} == set(cfg["save"][k].keys()):

        def dist_save_func(t, y, args):
            return {"f0": y["f0"], "f10": y["f10"]}

    else:
        raise NotImplementedError

    return dist_save_func


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    for k in cfg["save"].keys():  # this can be fields or electron or scalar?
        for k2 in cfg["save"][k].keys():  # this can be t, x, y, kx, ky (eventually)
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

    cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}, "func": get_default_save_func(cfg)}

    return cfg


def get_default_save_func(cfg):
    v = cfg["grid"]["v"][None, :]
    dv = cfg["grid"]["dv"]

    def _calc_f0_moment_(f0):
        return 4 * jnp.pi * jnp.trapz(f0 * cfg["grid"]["v"] ** 2.0, dx=cfg["grid"]["dv"], axis=1)

    def _calc_f1_moment_(f1):
        return 4 / 3 * jnp.pi * jnp.trapz(f1 * cfg["grid"]["v"] ** 2.0, dx=cfg["grid"]["dv"], axis=1)

    def save(t, y, args):
        scalars = {
            "mean_U": jnp.mean(0.5 * _calc_f0_moment_(y["f0"] * v**2.0)),
            "mean_j": jnp.mean(_calc_f1_moment_(y["f10"])),
            "mean_n": jnp.mean(_calc_f0_moment_(y["f0"])),
            "mean_q": jnp.mean(_calc_f1_moment_(0.5 * y["f10"] * v**2.0)),
            # "mean_-flogf": jnp.mean(-jnp.log(jnp.abs(y["f0"])) * jnp.abs(y["electron"])),
            # "mean_f2": jnp.mean(y["f0"] * y["f0"]),
            # "mean_e2": jnp.mean(y["e"] ** 2.0),
        }

        return scalars

    return save
