#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from typing import Dict
import os

from time import time


import numpy as np
from astropy import constants as const
import xarray, mlflow, pint, yaml
from jax import numpy as jnp
from diffrax import ODETerm, SubSaveAt
from matplotlib import pyplot as plt
from astropy.units import Quantity as _Q

from adept.vlasov1d.integrator import VlasovMaxwell, Stepper
from adept.vlasov1d.storage import store_f, store_fields, get_save_quantities
from adept.tf1d.pushers import get_envelope
from adept.vfp1d.helpers import write_units

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
    vth=1.0,
    vmax=6.0,
    n_prof=np.ones(1),
    T_prof=np.ones(1),
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

    :param nx: size of grid in x (single int)
    :param nv: size of grid in v (single int)
    :param vmax: maximum absolute value of v (single float)
    :return:
    """

    # noise_generator = np.random.default_rng(seed=noise_seed)

    dv = 2.0 * vmax / nv
    vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    # alpha = np.sqrt(3.0 * gamma_3_over_m(m) / gamma_5_over_m(m))

    f = np.zeros([nx, nv])
    for ix, (tn, tt) in enumerate(zip(n_prof, T_prof)):
        # cst = m / (4 * np.pi * alpha**3.0 * gamma_3_over_m(m)) / (2 * tt * vth**2.0) ** 1.5
        # single_dist = cst * np.exp(-(np.power(np.abs((vax[None, :] - v0) / (alpha * vth * np.sqrt(2 * tt))), m)))
        single_dist = (2 * np.pi * tt * (vth**2.0)) ** -0.5 * np.exp(-(vax**2.0) / (2 * tt * (vth**2.0)))
        f[ix, :] = tn * single_dist

    # f = f / np.sum(f, axis=1)[:, None] / dv

    # if noise_type.casefold() == "uniform":
    #     f = (1.0 + noise_generator.uniform(-noise_val, noise_val, nx)[:, None]) * f
    # elif noise_type.casefold() == "gaussian":
    #     f = (1.0 + noise_generator.normal(-noise_val, noise_val, nx)[:, None]) * f

    return f, vax


def get_1D_n_and_T_profile(profs, species_params, cfg):
    """
    This function initializes the density profile for a given species

    It is capable of initializing a uniform, linear, exponential, tanh, or sine profile

    Args:
        profs: empty dictionary of profiles
        species_params: dictionary of species parameters
        cfg: dictionary of config parameters

    Returns:
        profs: dictionary of profiles
    """
    cfg_grid = cfg["grid"]
    for k in ["n", "T"]:
        if species_params[k]["basis"] == "uniform":
            profs[k] = np.ones(cfg_grid["nx"])

        elif species_params[k]["basis"] == "linear":
            left = species_params[k]["center"] - species_params[k]["width"] * 0.5
            right = species_params[k]["center"] + species_params[k]["width"] * 0.5
            rise = species_params[k]["rise"]
            mask = get_envelope(rise, rise, left, right, cfg_grid["x"])

            L = (
                _Q(species_params[k]["gradient scale length"]).to("nm").value
                / cfg["units"]["derived"]["x0"].to("nm").value
            )
            nprof = species_params[k]["val at center"] + (cfg_grid["x"] - species_params[k]["center"]) / L
            profs[k] = mask * nprof
        elif species_params[k]["basis"] == "exponential":
            left = species_params[k]["center"] - species_params[k]["width"] * 0.5
            right = species_params[k]["center"] + species_params[k]["width"] * 0.5
            rise = species_params[k]["rise"]
            mask = get_envelope(rise, rise, left, right, cfg_grid["x"])

            L = (
                _Q(species_params[k]["gradient scale length"]).to("nm").value
                / cfg["units"]["derived"]["x0"].to("nm").value
            )
            nprof = species_params[k]["val at center"] * np.exp((cfg_grid["x"] - species_params[k]["center"]) / L)
            profs[k] = mask * nprof

        elif species_params[k]["basis"] == "tanh":
            left = species_params[k]["center"] - species_params[k]["width"] * 0.5
            right = species_params[k]["center"] + species_params[k]["width"] * 0.5
            rise = species_params[k]["rise"]
            nprof = get_envelope(rise, rise, left, right, cfg_grid["x"])

            if species_params[k]["bump_or_trough"] == "trough":
                nprof = 1 - nprof
            profs[k] = species_params[k]["baseline"] + species_params[k]["bump_height"] * nprof

        elif species_params[k]["basis"] == "sine":
            baseline = species_params[k]["baseline"]
            amp = species_params[k]["amplitude"]
            kk = species_params[k]["wavenumber"]
            profs[k] = baseline * (1.0 + amp * jnp.sin(kk * cfg["grid"]["x"]))
        else:
            raise NotImplementedError

    return profs


def _initialize_total_distribution_(cfg):
    """
    Initializes the total distribution function for all species

    Args:
        cfg: dictionary of config parameters
        cfg_grid: dictionary of grid parameters

    Returns:
        f: 2D array of distribution function

    """

    cfg_grid = cfg["grid"]

    params = cfg["density"]
    prof_total = {"n": np.zeros([cfg_grid["nx"]]), "T": np.zeros([cfg_grid["nx"]])}
    f = np.zeros([cfg_grid["nx"], cfg_grid["nv"]])
    species_found = False

    for name, species_params in cfg["density"].items():
        if name.startswith("species-"):
            profs = {}
            v0 = species_params["v0"]
            m = species_params["m"]
            if name in params:
                if "v0" in params[name]:
                    v0 = params[name]["v0"]

                if "m" in params[name]:
                    m = params[name]["m"]

            profs = get_1D_n_and_T_profile(profs, species_params, cfg)
            profs["n"] *= (cfg["units"]["derived"]["ne"] / cfg["units"]["derived"]["n0"]).value

            # Distribution function
            temp_f, _ = _initialize_distribution_(
                nx=int(cfg_grid["nx"]),
                nv=int(cfg_grid["nv"]),
                v0=v0,
                m=m,
                vth=cfg_grid["beta"],
                vmax=cfg_grid["vmax"],
                n_prof=profs["n"],
                T_prof=profs["T"],
                noise_val=species_params["noise_val"],
                noise_seed=int(species_params["noise_seed"]),
                noise_type=species_params["noise_type"],
            )

            # total density and distribution function
            f += temp_f
            prof_total["n"] += profs["n"]
            species_found = True
        else:
            pass

    if not species_found:
        raise ValueError("No species found! Check the config")

    return f, prof_total["n"]


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["vmax"] = (
        8 * np.sqrt((_Q(cfg["units"]["reference electron temperature"]) / (const.m_e * const.c**2.0)).to("")).value
    )
    cfg_grid["dv"] = 2.0 * cfg_grid["vmax"] / cfg_grid["nv"]

    if len(cfg["drivers"]["ey"].keys()) > 0:
        print("overriding dt to ensure wave solver stability")
        cfg_grid["dt"] = 0.95 * cfg_grid["dx"] / cfg["units"]["derived"]["c_light"]

    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]
    cfg["grid"] = cfg_grid

    return cfg


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

    cfg_grid["kprof"] = np.ones(cfg_grid["nx"])
    # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

    cfg_grid["x_a"] = np.concatenate(
        [[cfg_grid["x"][0] - cfg_grid["dx"]], cfg_grid["x"], [cfg_grid["x"][-1] + cfg_grid["dx"]]]
    )

    return cfg_grid


def init_state(cfg: Dict, td) -> Dict:
    """
    This function initializes the state

    :param cfg:
    :return:
    """
    f, ne_prof = _initialize_total_distribution_(cfg)

    state = {}
    for species in ["electron"]:
        state[species] = f

    for field in ["e", "de"]:
        state[field] = jnp.zeros(cfg["grid"]["nx"])

    for field in ["a", "da", "prev_a"]:
        state[field] = jnp.zeros(cfg["grid"]["nx"] + 2)  # need boundary cells

    state["Z"] = jnp.ones(cfg["grid"]["nx"]) * cfg["units"]["Z"]
    state["ni"] = ne_prof / cfg["units"]["Z"]

    return state


def get_diffeqsolve_quants(cfg):

    cfg = process_driver_units(cfg)

    return dict(
        terms=ODETerm(VlasovMaxwell(cfg)),
        solver=Stepper(),
        saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in cfg["save"].items()}),
        args={"drivers": cfg["drivers"], "terms": cfg["terms"]},
    )


def process_driver_units(cfg: Dict) -> Dict:
    for k1, efield in cfg["drivers"].items():
        for k2, pulse in efield.items():
            # this is with respect to the "reference electron temperature" in the input deck for all debye lengths
            # it is with respect to the "reference electron density" in the input deck for all timescales
            if "kl_D" in pulse["k0"]:
                cfg["drivers"][k1][k2]["k0"] = (
                    float(pulse["k0"].replace("kl_D", ""))
                    * (cfg["units"]["derived"]["x0"] / cfg["units"]["derived"]["lambda_D"]).to("").value
                )

            if "wp" in pulse["w0"]:
                cfg["drivers"][k1][k2]["w0"] = (
                    float(pulse["w0"].replace("wp", ""))
                    * np.sqrt(cfg["units"]["derived"]["ne"] / cfg["units"]["derived"]["n0"]).to("").value
                )

            if "l_D" in pulse["x_width"]:
                conv_fac = (cfg["units"]["derived"]["x0"] / cfg["units"]["derived"]["lambda_D"]).to("").value
                cfg["drivers"][k1][k2]["x_width"] = float(pulse["x_width"].replace("l_D", "")) / conv_fac
                cfg["drivers"][k1][k2]["x_center"] = float(pulse["x_center"].replace("l_D", "")) / conv_fac
                cfg["drivers"][k1][k2]["x_rise"] = float(pulse["x_rise"].replace("l_D", "")) / conv_fac

            if "tp" in pulse["t_width"]:
                conv_fac = (cfg["units"]["derived"]["tp0"] / cfg["units"]["derived"]["tpe"]).to("").value
                cfg["drivers"][k1][k2]["t_width"] = float(pulse["t_width"].replace("tp", "")) / conv_fac
                cfg["drivers"][k1][k2]["t_center"] = float(pulse["t_center"].replace("tp", "")) / conv_fac
                cfg["drivers"][k1][k2]["t_rise"] = float(pulse["t_rise"].replace("tp", "")) / conv_fac

    return cfg


def post_process(result, cfg: Dict, td: str):
    t0 = time()
    os.makedirs(os.path.join(td, "plots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "lineouts"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "logplots"), exist_ok=True)

    os.makedirs(os.path.join(td, "plots", "scalars"), exist_ok=True)

    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir)
    # merge
    # flds_paths = [os.path.join(flds_path, tf) for tf in flds_list]
    # arr = xarray.open_mfdataset(flds_paths, combine="by_coords", parallel=True)
    for k in result.ys.keys():
        if k.startswith("field"):
            fields_xr = store_fields(cfg, binary_dir, result.ys[k], result.ts[k], k)
            t_skip = int(fields_xr.coords["t"].data.size // 8)
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

                fld[tslice].T.plot(col="t", col_wrap=4)
                plt.savefig(os.path.join(td, "plots", "fields", "lineouts", f"{nm[7:]}.png"), bbox_inches="tight")
                plt.close()

        elif k.startswith("default"):
            scalars_xr = xarray.Dataset(
                {k: xarray.DataArray(v, coords=(("t", result.ts["default"]),)) for k, v in result.ys["default"].items()}
            )
            scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(scalars_xr.coords['t'].data[-1], 4)}.nc"))

            for nm, srs in scalars_xr.items():
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                srs.plot(ax=ax[0])
                ax[0].grid()
                np.log10(np.abs(srs)).plot(ax=ax[1])
                ax[1].grid()
                ax[1].set_ylabel("$log_{10}$(|" + nm + "|)")
                fig.savefig(os.path.join(td, "plots", "scalars", f"{nm}.png"), bbox_inches="tight")
                plt.close()

    f_xr = store_f(cfg, result.ts, td, result.ys)

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})

    return {"fields": fields_xr, "dists": f_xr, "scalars": scalars_xr}
