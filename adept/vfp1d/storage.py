from typing import Dict, Tuple, Callable
import os

from matplotlib import pyplot as plt
from diffrax import Solution
from jax import numpy as jnp
import numpy as np
import xarray as xr
from astropy.units import Quantity as _Q


def calc_EH(this_Z: int, this_wt: float) -> float:
    """
    This function calculates the Epperlein-Haines transport coefficient for a given Z and wt

    :param this_Z: The atomic number
    :param this_wt: The magnetization parameter

    :return: The Epperlein-Haines transport coefficient
    """
    Z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 20, 30, 60, 1e6 + 1])
    g0 = np.array(
        [3.203, 4.931, 6.115, 6.995, 7.680, 8.231, 8.685, 9.067, 9.673, 10.13, 10.5, 11.23, 11.9, 12.67, 13.58]
    )
    g0p = np.array([6.18, 9.3, 10.2, 9.14, 8.6, 8.57, 8.84, 7.93, 7.44, 7.32, 7.08, 6.79, 6.74, 6.36, 6.21])
    g1p = np.array([4.66, 3.96, 3.72, 3.6, 3.53, 3.49, 3.49, 3.43, 3.39, 3.37, 3.35, 3.32, 3.3, 3.27, 3.25])
    c0p = np.array([1.93, 1.89, 1.66, 1.31, 1.12, 1.04, 1.02, 0.875, 0.77, 0.722, 0.674, 0.605, 0.566, 0.502, 0.457])
    c1p = np.array([2.31, 3.78, 4.76, 4.63, 4.62, 4.83, 5.19, 4.74, 4.63, 4.7, 4.64, 4.65, 4.81, 4.71, 4.81])
    c2p = np.array(
        [5.35, 7.78, 8.88, 8.8, 8.8, 8.96, 9.24, 8.84, 8.71, 8.73, 8.65, 8.6, 8.66, 8.52, 8.53],
    )
    g0w = np.array(
        [6.071, 15.75, 25.65, 34.95, 43.45, 51.12, 58.05, 64.29, 75.04, 83.93, 91.38, 107.8, 124.3, 145.2, 172.7]
    )
    g0pp = np.array(
        [4.01, 2.46, 1.13, 0.628, 0.418, 0.319, 0.268, 0.238, 0.225, 0.212, 0.202, 0.2, 0.194, 0.189, 0.186]
    )
    g1pp = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    c0pp = np.array(
        [
            0.661,
            0.156,
            0.0442,
            0.018,
            0.00963,
            0.00625,
            0.00461,
            0.00371,
            0.003,
            0.00252,
            0.00221,
            0.00185,
            0.00156,
            0.0013,
            0.00108,
        ]
    )
    c1pp = np.array(
        [
            0.931,
            0.398,
            0.175,
            0.101,
            0.0702,
            0.0551,
            0.0465,
            0.041,
            0.0354,
            0.0317,
            0.0291,
            0.0256,
            0.0228,
            0.0202,
            0.018,
        ]
    )
    c2pp = np.array([2.5, 1.71, 1.05, 0.775, 0.646, 0.578, 0.539, 0.515, 0.497, 0.482, 0.471, 0.461, 0.45, 0.44, 0.43])

    this_g0p = np.interp(this_Z, Z, g0p)
    this_c0p = np.interp(this_Z, Z, c0p)

    eh = this_g0p / this_c0p

    return eh


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
            units, scalings = get_unit(k, cfg)

            for unit, scaling in zip(units, scalings):
                das[f"{prefix}-{k} {unit}"] = xr.DataArray(v * scaling, coords=(("t (ps)", tax), ("x (um)", xax)))

    das[f"{prefix}-kappa_c"] = calc_kappa(cfg, das[f"{prefix}-T a.u."], das[f"{prefix}-q a.u."], das[f"{prefix}-n n_c"])

    fields_xr = xr.Dataset(das)
    fields_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-t={round(tax[-1],4)}.nc"))

    return fields_xr


def calc_kappa(cfg: Dict, T: xr.DataArray, q: xr.DataArray, n: xr.DataArray) -> xr.DataArray:
    """
    This rescales using the EH tau ei.

    This does NOT work with nuei_shk and nuei_NRL

    It needs a Z dependence which those other two do not have except maybe through the log Lambda.

    1. Epperlein, E. M. & Haines, M. G. Plasma transport coefficients in a magnetic field by direct numerical solution of the Fokker–Planck equation. Physics of Fluids 29, 1029 (1986).

    """

    kappa = (
        -q.data
        / n.data
        / T.data
        / np.gradient(T.data, cfg["grid"]["dx"], axis=1)
        * (cfg["units"]["derived"]["nuei_epphaines"] / cfg["units"]["derived"]["wp0"]).to("").value
    )

    return xr.DataArray(kappa, coords=(("t (ps)", T.coords["t (ps)"].data), ("x (um)", T.coords["x (um)"].data)))


def get_unit(k, cfg: Dict = None) -> Tuple[str, float]:
    """
    Returns the units and scalings for a given key

    :param k: The key
    :param cfg: The config
    :return: The units and scalings

    """
    # if k == "e":
    #     return "V/m", 1.0
    if k == "T":
        return ("keV", "a.u."), (510.999, 1.0)
    elif k == "n":
        return ("cm-3", "n_c"), (9.09e21, 1.0)
    else:
        return ("a.u.",), (1.0,)


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
                coords=(("t (ps)", tax), ("x (um)", xax), ("v (c)", cfg["grid"]["v"])),
            )
            for dist in ys["electron"].keys()
        }
    )
    f_store.to_netcdf(os.path.join(td, "binary", "dist.nc"))

    return f_store


def post_process(soln: Solution, cfg: Dict, td: str, args: Dict = None) -> Dict:
    """
    Post-processing function

    :param result: The result
    :param cfg: The config
    :param td: The directory

    :return: The fields and distributions

    """

    # t0 = time()
    os.makedirs(os.path.join(td, "plots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "lineouts"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "logplots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "scalars"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "dist"), exist_ok=True)

    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir)

    # merge
    for k in soln.ys.keys():
        if k.startswith("field"):
            fields_xr = store_fields(cfg, binary_dir, soln.ys[k], soln.ts[k], k)
            t_skip = int(fields_xr.coords["t (ps)"].data.size // 8)
            t_skip = t_skip if t_skip > 1 else 1
            tslice = slice(0, -1, t_skip)

            for nm, fld in fields_xr.items():
                fld.plot()
                plt.savefig(os.path.join(td, "plots", "fields", f"spacetime-{nm[7:]}.png"), bbox_inches="tight")
                plt.close()

                np.log10(np.abs(fld)).plot()
                plt.savefig(
                    os.path.join(td, "plots", "fields", "logplots", f"spacetime-log-{nm[7:]}.png"), bbox_inches="tight"
                )
                plt.close()

                fld[tslice].T.plot(col="t (ps)", col_wrap=4)
                plt.savefig(os.path.join(td, "plots", "fields", "lineouts", f"{nm[7:]}.png"), bbox_inches="tight")
                plt.close()

        elif k.startswith("default"):

            tax = soln.ts["default"] * cfg["units"]["derived"]["tp0"].to("ps").value
            scalars_xr = xr.Dataset(
                {k: xr.DataArray(v, coords=(("t (ps)", tax),)) for k, v in soln.ys["default"].items()}
            )
            scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(tax[-1], 4)}.nc"))

            for nm, srs in scalars_xr.items():
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                srs.plot(ax=ax[0])
                ax[0].grid()
                np.log10(np.abs(srs)).plot(ax=ax[1])
                ax[1].grid()
                ax[1].set_ylabel("$log_{10}$(|" + nm + "|)")
                fig.savefig(os.path.join(td, "plots", "scalars", f"{nm}.png"), bbox_inches="tight")
                plt.close()

    f_xr = store_f(cfg, soln.ts, td, soln.ys)

    t_skip = int(f_xr.coords["t (ps)"].data.size // 4)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    for k in ["f0", "f10"]:
        f_xr[k][tslice].plot(x="x (um)", y="v (c)", col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(td, "plots", "dist", f"{k}.png"))
        plt.close()

    metrics = {
        "kappa": float(round(np.amax(fields_xr["fields-kappa_c"][-1].data), 4)),
        "kappa_eh": float(round(calc_EH(cfg["units"]["Z"], 0.0), 4)),
    }

    return {"fields": fields_xr, "dists": f_xr, "metrics": metrics}


def get_field_save_func(cfg: Dict, k: str) -> Callable:
    """
    This function returns the field save function

    :param cfg: The config
    :param k: The key
    :return: The save function

    """
    if {"t"} == set(cfg["save"][k].keys()):

        def _calc_f0_moment_(f0):
            return 4 * jnp.pi * jnp.sum(f0 * cfg["grid"]["v"] ** 2.0, axis=1) * cfg["grid"]["dv"]

        def _calc_f1_moment_(f1):
            return 4 / 3 * jnp.pi * jnp.sum(f1 * cfg["grid"]["v"] ** 3.0, axis=1) * cfg["grid"]["dv"]

        def fields_save_func(t, y, args):
            temp = {"n": _calc_f0_moment_(y["f0"]), "v": _calc_f1_moment_(y["f10"])}
            temp["U"] = _calc_f0_moment_(y["f0"] * cfg["grid"]["v"] ** 2.0)
            temp["P"] = temp["U"] / 3.0
            temp["T"] = temp["P"] / temp["n"]
            temp["q"] = _calc_f1_moment_(0.5 * y["f10"] * cfg["grid"]["v"] ** 2.0)
            temp["e"] = y["e"]
            temp["b"] = y["b"]
            temp["ni"] = y["ni"]

            return temp

    else:
        raise NotImplementedError

    return fields_save_func


def get_dist_save_func(cfg: Dict, k: str) -> Callable:
    """
    This function returns the distribution save function

    :param cfg: The config
    :param k: The key

    :return: The save function

    """
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
    :return: The updated config
    """

    for k in cfg["save"].keys():  # this can be fields or electron or scalar?

        tmin = (_Q(cfg["save"][k]["t"]["tmin"]) / cfg["units"]["derived"]["tp0"]).to("").value
        tmax = (_Q(cfg["save"][k]["t"]["tmax"]) / cfg["units"]["derived"]["tp0"]).to("").value
        cfg["save"][k]["t"]["ax"] = jnp.linspace(tmin, tmax, cfg["save"][k]["t"]["nt"])

        if k.startswith("fields"):
            cfg["save"][k]["func"] = get_field_save_func(cfg, k)

        elif k.startswith("electron"):
            cfg["save"][k]["func"] = get_dist_save_func(cfg, k)

    cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}, "func": get_default_save_func(cfg)}

    return cfg


def get_default_save_func(cfg: Dict) -> Callable:
    """
    This function returns the default save function

    :param cfg: The config
    :return: The save function

    """
    v = cfg["grid"]["v"][None, :]
    dv = cfg["grid"]["dv"]

    def _calc_f0_moment_(f0):
        return 4 * jnp.pi * jnp.sum(f0 * cfg["grid"]["v"] ** 2.0, axis=1) * cfg["grid"]["dv"]

    def _calc_f1_moment_(f1):
        return 4 / 3 * jnp.pi * jnp.sum(f1 * cfg["grid"]["v"] ** 3.0, axis=1) * cfg["grid"]["dv"]

    def save(t, y, args):
        scalars = {
            "mean_U": jnp.mean(_calc_f0_moment_(y["f0"] * v**2.0)),
            "mean_j": jnp.mean(_calc_f1_moment_(y["f10"])),
            "mean_n": jnp.mean(_calc_f0_moment_(y["f0"])),
            "mean_q": jnp.mean(_calc_f1_moment_(0.5 * y["f10"] * v**2.0)),
            # "mean_-flogf": jnp.mean(-jnp.log(jnp.abs(y["f0"])) * jnp.abs(y["electron"])),
            # "mean_f2": jnp.mean(y["f0"] * y["f0"]),
            # "mean_e2": jnp.mean(y["e"] ** 2.0),
        }

        return scalars

    return save
