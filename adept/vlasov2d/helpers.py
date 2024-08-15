#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from typing import Dict, List, Tuple
import os

from time import time

import numpy as np
import xarray, mlflow, pint, yaml
from jax import numpy as jnp
from diffrax import ODETerm, SubSaveAt, diffeqsolve, SaveAt
from matplotlib import pyplot as plt
from equinox import filter_jit

from adept.vlasov2d.pushers import time as time_integrator
from adept.vlasov2d.storage import store_f, store_fields, get_save_quantities

gamma_da = xarray.open_dataarray(os.path.join(os.path.dirname(__file__), "gamma_func_for_sg.nc"))
m_ax = gamma_da.coords["m"].data
g_3_m = np.squeeze(gamma_da.loc[{"gamma": "3/m"}].data)
g_5_m = np.squeeze(gamma_da.loc[{"gamma": "5/m"}].data)


def gamma_3_over_m(m):
    return np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m):
    return np.interp(m, m_ax, g_5_m)


def write_units(cfg, td):
    ureg = pint.UnitRegistry()
    _Q = ureg.Quantity

    n0 = _Q(cfg["units"]["normalizing_density"]).to("1/cc")
    T0 = _Q(cfg["units"]["normalizing_temperature"]).to("eV")

    wp0 = np.sqrt(n0 * ureg.e**2.0 / (ureg.m_e * ureg.epsilon_0)).to("rad/s")
    tp0 = (1 / wp0).to("fs")

    v0 = np.sqrt(2.0 * T0 / ureg.m_e).to("m/s")
    x0 = (v0 / wp0).to("nm")
    c_light = _Q(1.0 * ureg.c).to("m/s") / v0
    beta = (v0 / ureg.c).to("dimensionless")

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * x0).to("microns")
    if "ymax" in cfg["grid"].keys():
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * x0).to("microns")
    else:
        box_width = "inf"
    sim_duration = (cfg["grid"]["tmax"] * tp0).to("ps")

    # collisions
    logLambda_ee = 23.5 - np.log(n0.magnitude**0.5 / T0.magnitude**-1.25)
    logLambda_ee -= (1e-5 + (np.log(T0.magnitude) - 2) ** 2.0 / 16) ** 0.5
    nuee = _Q(2.91e-6 * n0.magnitude * logLambda_ee / T0.magnitude**1.5, "Hz")
    nuee_norm = nuee / wp0

    all_quantities = {
        "wp0": wp0,
        "tp0": tp0,
        "n0": n0,
        "v0": v0,
        "T0": T0,
        "c_light": c_light,
        "beta": beta,
        "x0": x0,
        "nuee": nuee,
        "logLambda_ee": logLambda_ee,
        "box_length": box_length,
        "box_width": box_width,
        "sim_duration": sim_duration,
    }

    cfg["units"]["derived"] = all_quantities

    cfg["grid"]["beta"] = beta.magnitude

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump({k: str(v) for k, v in all_quantities.items()}, fi)

    return cfg


def _initialize_distribution_(
    nxs: List,
    nvs: List,
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

    dvs = [2.0 * vmax / nv for nv in nvs]
    vaxs = [np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv) for dv, nv in zip(dvs, nvs)]

    alpha = np.sqrt(3.0 * gamma_3_over_m(m) / gamma_5_over_m(m))
    # cst = m / (4 * np.pi * alpha**3.0 * gamma(3.0 / m))

    single_dist = -(
        np.power(np.abs((vaxs[0][None, None, :, None] - v0) / alpha / np.sqrt(T0)), m)
        + np.power(np.abs((vaxs[1][None, None, None, :] - v0) / alpha / np.sqrt(T0)), m)
    )

    single_dist = np.exp(single_dist)
    # single_dist = np.exp(-(vaxs[0][None, None, :, None]**2.+vaxs[1][None, None, None, :]**2.)/2/T0)

    # for ix in range(nx):
    f = np.repeat(np.repeat(single_dist, nxs[0], axis=0), nxs[1], axis=1)
    # normalize
    f = f / np.trapz(np.trapz(f, dx=dvs[0], axis=2), dx=dvs[1], axis=2)[:, :, None, None]

    if n_prof.size > 1:
        # scale by density profile
        f = n_prof[:, :, None, None] * f

    # if noise_type.casefold() == "uniform":
    #     f = (1.0 + noise_generator.uniform(-noise_val, noise_val, nx)[:, None]) * f
    # elif noise_type.casefold() == "gaussian":
    #     f = (1.0 + noise_generator.normal(-noise_val, noise_val, nx)[:, None]) * f

    return f, vaxs


def _initialize_total_distribution_(cfg, cfg_grid):
    params = cfg["density"]
    xs = (cfg_grid["x"], cfg_grid["y"])
    n_prof_total = np.zeros([x.size for x in xs])
    f = np.zeros([x.size for x in xs] + [cfg_grid["nvx"], cfg_grid["nvy"]])
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
                nxs=[int(cfg_grid["nx"]), int(cfg_grid["ny"])],
                nvs=[int(cfg_grid["nvx"]), int(cfg_grid["nvy"])],
                v0=v0,  # * cfg_grid["beta"],
                m=m,
                T0=T0,  # * cfg_grid["beta"] ** 2.0,
                vmax=cfg_grid["vmax"],  # * cfg_grid["beta"],
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
    cfg_grid["dy"] = cfg_grid["ymax"] / cfg_grid["ny"]

    # cfg_grid["vmax"] *= cfg_grid["beta"]
    cfg_grid["dvx"] = 2.0 * cfg_grid["vmax"] / cfg_grid["nvx"]
    cfg_grid["dvy"] = 2.0 * cfg_grid["vmax"] / cfg_grid["nvy"]

    # cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

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
            "y": jnp.linspace(
                cfg_grid["ymin"] + cfg_grid["dy"] / 2, cfg_grid["ymax"] - cfg_grid["dy"] / 2, cfg_grid["ny"]
            ),
            "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "vx": jnp.linspace(
                -cfg_grid["vmax"] + cfg_grid["dvx"] / 2, cfg_grid["vmax"] - cfg_grid["dvx"] / 2, cfg_grid["nvx"]
            ),
            "vy": jnp.linspace(
                -cfg_grid["vmax"] + cfg_grid["dvy"] / 2, cfg_grid["vmax"] - cfg_grid["dvy"] / 2, cfg_grid["nvy"]
            ),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "ky": jnp.fft.fftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
            "kyr": jnp.fft.rfftfreq(cfg_grid["ny"], d=cfg_grid["dy"]) * 2.0 * np.pi,
            "kvx": jnp.fft.fftfreq(cfg_grid["nvx"], d=cfg_grid["dvx"]) * 2.0 * np.pi,
            "kvxr": jnp.fft.rfftfreq(cfg_grid["nvx"], d=cfg_grid["dvx"]) * 2.0 * np.pi,
            "kvy": jnp.fft.fftfreq(cfg_grid["nvy"], d=cfg_grid["dvy"]) * 2.0 * np.pi,
            "kvyr": jnp.fft.rfftfreq(cfg_grid["nvy"], d=cfg_grid["dvy"]) * 2.0 * np.pi,
        },
    }

    # config axes
    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

    one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

    one_over_ky = np.zeros_like(cfg_grid["ky"])
    one_over_ky[1:] = 1.0 / cfg_grid["ky"][1:]
    cfg_grid["one_over_ky"] = jnp.array(one_over_ky)

    one_over_kyr = np.zeros_like(cfg_grid["kyr"])
    one_over_kyr[1:] = 1.0 / cfg_grid["kyr"][1:]
    cfg_grid["one_over_kyr"] = jnp.array(one_over_kyr)

    # velocity axes
    one_over_kvx = np.zeros_like(cfg_grid["kvx"])
    one_over_kvx[1:] = 1.0 / cfg_grid["kvx"][1:]
    cfg_grid["one_over_kvx"] = jnp.array(one_over_kvx)

    one_over_kvxr = np.zeros_like(cfg_grid["kvxr"])
    one_over_kvxr[1:] = 1.0 / cfg_grid["kvxr"][1:]
    cfg_grid["one_over_kvxr"] = jnp.array(one_over_kvxr)

    one_over_kvy = np.zeros_like(cfg_grid["kvy"])
    one_over_kvy[1:] = 1.0 / cfg_grid["kvy"][1:]
    cfg_grid["one_over_kvy"] = jnp.array(one_over_kvy)

    one_over_kvyr = np.zeros_like(cfg_grid["kvyr"])
    one_over_kvyr[1:] = 1.0 / cfg_grid["kvyr"][1:]
    cfg_grid["one_over_kvyr"] = jnp.array(one_over_kvyr)

    cfg_grid["nuprof"] = 1.0
    # get_profile_with_mask(cfg["nu"]["time-profile"], t, cfg["nu"]["time-profile"]["bump_or_trough"])
    cfg_grid["ktprof"] = 1.0
    # get_profile_with_mask(cfg["krook"]["time-profile"], t, cfg["krook"]["time-profile"]["bump_or_trough"])
    cfg_grid["n_prof_total"], cfg_grid["starting_f"] = _initialize_total_distribution_(cfg, cfg_grid)

    cfg_grid["kprof"] = np.ones_like(cfg_grid["n_prof_total"])
    # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

    cfg_grid["ion_charge"] = np.ones_like(cfg_grid["n_prof_total"])

    return cfg_grid


def get_run_fn(cfg):
    diffeqsolve_quants = get_diffeqsolve_quants(cfg)

    @filter_jit
    def _run_(_models_, _state_, _args_, time_quantities: Dict):

        _state_, _args_ = apply_models(_models_, _state_, _args_, cfg)
        # if "terms" in cfg.keys():
        #     args["terms"] = cfg["terms"]
        solver_result = diffeqsolve(
            terms=diffeqsolve_quants["terms"],
            solver=diffeqsolve_quants["solver"],
            t0=time_quantities["t0"],
            t1=time_quantities["t1"],
            max_steps=cfg["grid"]["max_steps"],
            dt0=cfg["grid"]["dt"],
            y0=_state_,
            args=_args_,
            saveat=SaveAt(**diffeqsolve_quants["saveat"]),
        )

        return solver_result, _state_, _args_

    return _run_


def init_state(cfg: Dict, td) -> Tuple[Dict, Dict]:
    """
    This function initializes the state

    :param cfg:
    :return:
    """
    n_prof_total, f = _initialize_total_distribution_(cfg, cfg["grid"])

    state = {}
    for species in ["electron"]:
        state[species] = f

    for field in ["ex", "ey", "bz", "dex", "dey"]:
        state[field] = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]))

    # transform
    # for nm, quant in state.items():
    #     state[nm] = jnp.fft.fft2(quant, axes=(0, 1)).view(dtype=jnp.float64)

    return state, {"drivers": cfg["drivers"]}


def get_diffeqsolve_quants(cfg):
    # if cfg["solver"]["field"] == "poisson":
    #     VectorField = time_integrator.LeapfrogIntegrator(cfg)
    # elif cfg["solver"]["field"] == "maxwell":
    VectorField = time_integrator.ChargeConservingMaxwell(cfg)
    # else:
    #     raise NotImplementedError

    return dict(
        terms=ODETerm(VectorField),
        solver=time_integrator.Stepper(),
        saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in cfg["save"].items()}),
    )


def post_process(result, cfg: Dict, td: str):
    t0 = time()
    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir, exist_ok=True)

    os.makedirs(os.path.join(td, "plots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "scalars"), exist_ok=True)

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
                fld[tslice].T.plot(col="t", col_wrap=4)  # ax=ax[this_ax_row, this_ax_col])
                plt.savefig(os.path.join(td, "plots", "fields", f"{nm[7:]}.png"), bbox_inches="tight")

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

    return {"fields": fields_xr, "dists": f_xr}


def apply_models(models, state, args, cfg):
    return state, args
