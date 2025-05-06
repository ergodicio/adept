#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from typing import Dict
import os

from time import time


import numpy as np
import xarray, mlflow, pint
from jax import numpy as jnp
from scipy.special import gamma
from diffrax import Solution
from matplotlib import pyplot as plt

from adept._base_ import get_envelope
from adept._vlasov1d.storage import store_f, store_fields

# gamma_da = xarray.open_dataarray(os.path.join(os.path.dirname(__file__), "gamma_func_for_sg.nc"))
# m_ax = gamma_da.coords["m"].data
# g_3_m = np.squeeze(gamma_da.loc[{"gamma": "3/m"}].data)
# g_5_m = np.squeeze(gamma_da.loc[{"gamma": "5/m"}].data)


def gamma_3_over_m(m):
    return gamma(3.0 / m)  # np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m):
    return gamma(5.0 / m)  # np.interp(m, m_ax, g_5_m)


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

            elif species_params["basis"] == "linear":
                left = species_params["center"] - species_params["width"] * 0.5
                right = species_params["center"] + species_params["width"] * 0.5
                rise = species_params["rise"]
                mask = get_envelope(rise, rise, left, right, cfg_grid["x"])

                ureg = pint.UnitRegistry()
                _Q = ureg.Quantity

                L = (
                    _Q(species_params["gradient scale length"]).to("nm").magnitude
                    / cfg["units"]["derived"]["x0"].to("nm").magnitude
                )
                nprof = species_params["val at center"] + (cfg_grid["x"] - species_params["center"]) / L
                nprof = mask * nprof
            elif species_params["basis"] == "exponential":
                left = species_params["center"] - species_params["width"] * 0.5
                right = species_params["center"] + species_params["width"] * 0.5
                rise = species_params["rise"]
                mask = get_envelope(rise, rise, left, right, cfg_grid["x"])

                ureg = pint.UnitRegistry()
                _Q = ureg.Quantity

                L = (
                    _Q(species_params["gradient scale length"]).to("nm").magnitude
                    / cfg["units"]["derived"]["x0"].to("nm").magnitude
                )
                nprof = species_params["val at center"] * np.exp((cfg_grid["x"] - species_params["center"]) / L)
                nprof = mask * nprof

            elif species_params["basis"] == "tanh":
                left = species_params["center"] - species_params["width"] * 0.5
                right = species_params["center"] + species_params["width"] * 0.5
                rise = species_params["rise"]
                nprof = get_envelope(rise, rise, left, right, cfg_grid["x"])

                if species_params["bump_or_trough"] == "trough":
                    nprof = 1 - nprof
                nprof = species_params["baseline"] + species_params["bump_height"] * nprof

            elif species_params["basis"] == "sine":
                baseline = species_params["baseline"]
                amp = species_params["amplitude"]
                kk = species_params["wavenumber"]
                nprof = baseline * (1.0 + amp * jnp.sin(kk * cfg_grid["x"]))
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


def post_process(result: Solution, cfg: Dict, td: str, args: Dict):

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

    diags_dict = {}
    for k in ["diag-vlasov-dfdt", "diag-fp-dfdt"]:
        if cfg["diagnostics"][k]:
            diags_dict[k] = xarray.DataArray(
                result.ys[k], coords=(("t", result.ts[k]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"]))
            )

    if len(diags_dict.keys()):
        diags_xr = xarray.Dataset(diags_dict)
        diags_xr.to_netcdf(os.path.join(td, "binary", "diags.nc"))

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})

    return {"fields": fields_xr, "dists": f_xr, "scalars": scalars_xr}
