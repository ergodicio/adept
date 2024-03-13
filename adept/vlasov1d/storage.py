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

        def _calc_moment_(inp):
            return jnp.sum(inp, axis=1) * cfg["grid"]["dv"]

        def fields_save_func(t, y, args):
            temp = {"n": _calc_moment_(y["electron"]), "v": _calc_moment_(y["electron"] * cfg["grid"]["v"][None, :])}
            v_m_vbar = cfg["grid"]["v"][None, :] - temp["v"][:, None]
            temp["T"] = 0.5 * _calc_moment_(y["electron"] * v_m_vbar**2.0)
            temp["q"] = _calc_moment_(y["electron"] * v_m_vbar**3.0)
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

    def _calc_mean_moment_(inp):
        return jnp.mean(jnp.sum(inp, axis=1)) * dv

    def save(t, y, args):
        scalars = {
            "mean_T_eV": 0.5 * _calc_mean_moment_(y["electron"] * v**2.0) * 510999,
            "mean_T": 0.5 * _calc_mean_moment_(y["electron"] * v**2.0),
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
