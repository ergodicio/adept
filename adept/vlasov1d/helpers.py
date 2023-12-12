#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from typing import Dict
import os

from time import time

import numpy as np
import xarray, mlflow
from jax import numpy as jnp
from diffrax import ODETerm, SubSaveAt
from matplotlib import pyplot as plt

from adept.vlasov1d.integrator import VlasovMaxwell, Stepper
from adept.vlasov1d.storage import store_f, store_fields, get_save_quantities

gamma_da = xarray.open_dataarray(os.path.join(os.path.dirname(__file__), "gamma_func_for_sg.nc"))
m_ax = gamma_da.coords["m"].data
g_3_m = np.squeeze(gamma_da.loc[{"gamma": "3/m"}].data)
g_5_m = np.squeeze(gamma_da.loc[{"gamma": "5/m"}].data)


def gamma_3_over_m(m):
    return np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m):
    return np.interp(m, m_ax, g_5_m)


def _initialize_distribution_(
    nx: int,
    nv: int,
    v0=0.0,
    m=2.0,
    T0=1.0,
    vmax=6.0,
    n_prof=np.ones(1),
    noise_val=0.0,
    noise_seed=42,
    noise_type="Uniform",
):
    """

    :param nxs:
    :param nvs:
    :param n_prof:
    :param vmax:
    :return:
    """
    """
    Initializes a Maxwell-Boltzmann distribution

    TODO: temperature and density pertubations

    :param nx: size of grid in x (single int)
    :param nv: size of grid in v (single int)
    :param vmax: maximum absolute value of v (single float)
    :return:
    """

    # noise_generator = np.random.default_rng(seed=noise_seed)

    dv = 2.0 * vmax / nv
    vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    alpha = np.sqrt(3.0 * gamma_3_over_m(m) / gamma_5_over_m(m))
    # cst = m / (4 * np.pi * alpha**3.0 * gamma(3.0 / m))

    single_dist = -(np.power(np.abs((vax[None, :] - v0) / alpha / np.sqrt(T0)), m))

    single_dist = np.exp(single_dist)
    # single_dist = np.exp(-(vaxs[0][None, None, :, None]**2.+vaxs[1][None, None, None, :]**2.)/2/T0)

    # for ix in range(nx):
    f = np.repeat(single_dist, nx, axis=0)
    # normalize
    f = f / np.trapz(f, dx=dv, axis=1)[:, None]

    if n_prof.size > 1:
        # scale by density profile
        f = n_prof[:, None] * f

    # if noise_type.casefold() == "uniform":
    #     f = (1.0 + noise_generator.uniform(-noise_val, noise_val, nx)[:, None]) * f
    # elif noise_type.casefold() == "gaussian":
    #     f = (1.0 + noise_generator.normal(-noise_val, noise_val, nx)[:, None]) * f

    return f, vax


def _initialize_total_distribution_(cfg, cfg_grid):
    params = cfg["density"]
    n_prof_total = np.zeros([cfg_grid["nx"]])
    f = np.zeros([cfg_grid["nx"], cfg_grid["nv"]])
    species_found = False
    for name, species_params in cfg["density"].items():
        if name.startswith("species-"):
            v0 = species_params["v0"]
            T0 = species_params["T0"]
            m = species_params["m"]
            if name in params:
                if "v0" in params[name]:
                    v0 = params[name]["v0"]

                if "T0" in params[name]:
                    T0 = params[name]["T0"]

                if "m" in params[name]:
                    m = params[name]["m"]

            if species_params["basis"] == "uniform":
                nprof = np.ones_like(n_prof_total)
            else:
                raise NotImplementedError

            n_prof_total += nprof

            # Distribution function
            temp_f, _ = _initialize_distribution_(
                nx=int(cfg_grid["nx"]),
                nv=int(cfg_grid["nv"]),
                v0=v0,
                m=m,
                T0=T0,
                vmax=cfg_grid["vmax"],
                n_prof=nprof,
                noise_val=species_params["noise_val"],
                noise_seed=int(species_params["noise_seed"]),
                noise_type=species_params["noise_type"],
            )
            f += temp_f
            species_found = True
        else:
            pass

    if not species_found:
        raise ValueError("No species found! Check the config")

    return n_prof_total, f


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dv"] = 2.0 * cfg_grid["vmax"] / cfg_grid["nv"]

    # cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    cfg["grid"] = cfg_grid

    return cfg_grid


def get_solver_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are arrays

    This is run after the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    cfg_grid = {
        **cfg_grid,
        **{
            "x": jnp.linspace(
                cfg_grid["xmin"] + cfg_grid["dx"] / 2, cfg_grid["xmax"] - cfg_grid["dx"] / 2, cfg_grid["nx"]
            ),
            "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "v": jnp.linspace(
                -cfg_grid["vmax"] + cfg_grid["dv"] / 2, cfg_grid["vmax"] - cfg_grid["dv"] / 2, cfg_grid["nv"]
            ),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kv": jnp.fft.fftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
            "kvr": jnp.fft.rfftfreq(cfg_grid["nv"], d=cfg_grid["dv"]) * 2.0 * np.pi,
        },
    }

    # config axes
    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

    one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

    # velocity axes
    one_over_kv = np.zeros_like(cfg_grid["kv"])
    one_over_kv[1:] = 1.0 / cfg_grid["kv"][1:]
    cfg_grid["one_over_kv"] = jnp.array(one_over_kv)

    one_over_kvr = np.zeros_like(cfg_grid["kvr"])
    one_over_kvr[1:] = 1.0 / cfg_grid["kvr"][1:]
    cfg_grid["one_over_kvr"] = jnp.array(one_over_kvr)

    cfg_grid["nuprof"] = 1.0
    # get_profile_with_mask(cfg["nu"]["time-profile"], t, cfg["nu"]["time-profile"]["bump_or_trough"])
    cfg_grid["ktprof"] = 1.0
    # get_profile_with_mask(cfg["krook"]["time-profile"], t, cfg["krook"]["time-profile"]["bump_or_trough"])
    cfg_grid["n_prof_total"], cfg_grid["starting_f"] = _initialize_total_distribution_(cfg, cfg_grid)

    cfg_grid["kprof"] = np.ones_like(cfg_grid["n_prof_total"])
    # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

    cfg_grid["ion_charge"] = np.ones_like(cfg_grid["n_prof_total"])

    return cfg_grid


def init_state(cfg: Dict) -> Dict:
    """
    This function initializes the state

    :param cfg:
    :return:
    """
    n_prof_total, f = _initialize_total_distribution_(cfg, cfg["grid"])

    state = {}
    for species in ["electron"]:
        state[species] = f

    for field in ["e", "de"]:
        state[field] = jnp.zeros(cfg["grid"]["nx"])

    for field in ["a", "da", "prev_a"]:
        state[field] = jnp.zeros(cfg["grid"]["nx"])

    return state


def get_diffeqsolve_quants(cfg):
    return dict(
        terms=ODETerm(VlasovMaxwell(cfg)),
        solver=Stepper(),
        saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in cfg["save"].items()}),
    )


def post_process(result, cfg: Dict, td: str):
    t0 = time()
    os.makedirs(os.path.join(td, "plots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "lineouts"), exist_ok=True)
    # merge
    # flds_paths = [os.path.join(flds_path, tf) for tf in flds_list]
    # arr = xarray.open_mfdataset(flds_paths, combine="by_coords", parallel=True)
    for k in result.ys.keys():
        if k.startswith("field"):
            fields_xr = store_fields(cfg, td, result.ys[k], result.ts[k], k)
            t_skip = int(fields_xr.coords["t"].data.size // 8)
            t_skip = t_skip if t_skip > 1 else 1
            tslice = slice(0, -1, t_skip)

            for nm, fld in fields_xr.items():
                fld.plot()
                plt.savefig(os.path.join(td, "plots", "fields", f"spacetime-{nm[7:]}.png"), bbox_inches="tight")
                plt.close()

                fld[tslice].T.plot(col="t", col_wrap=4)
                plt.savefig(os.path.join(td, "plots", "fields", "lineouts", f"{nm[7:]}.png"), bbox_inches="tight")
                plt.close()

    f_xr = store_f(cfg, result.ts, td, result.ys)

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})

    return {"fields": fields_xr, "dists": f_xr}
