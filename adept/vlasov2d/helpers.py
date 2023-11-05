#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
from typing import Dict, List
import os

import numpy as np
import xarray
from jax import numpy as jnp
from adept.vlasov2d.integrator import LeapfrogIntegrator, Stepper
from diffrax import ODETerm

gamma_da = xarray.open_dataarray(os.path.join(os.path.dirname(__file__), "gamma_func_for_sg.nc"))
m_ax = gamma_da.coords["m"].data
g_3_m = np.squeeze(gamma_da.loc[{"gamma": "3/m"}].data)
g_5_m = np.squeeze(gamma_da.loc[{"gamma": "5/m"}].data)


def gamma_3_over_m(m):
    return np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m):
    return np.interp(m, m_ax, g_5_m)


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


def initialize_total_distribution(cfg):
    params = cfg["density"]
    xs = (cfg["grid"]["x"], cfg["grid"]["y"])
    n_prof_total = np.zeros([x.size for x in xs])
    f = np.zeros([x.size for x in xs] + [cfg["grid"]["nvx"], cfg["grid"]["nvy"]])
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
                nxs=[int(cfg["grid"]["nx"]), int(cfg["grid"]["ny"])],
                nvs=[int(cfg["grid"]["nvx"]), int(cfg["grid"]["nvy"])],
                v0=v0,
                m=m,
                T0=T0,
                vmax=cfg["grid"]["vmax"],
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


def get_derived_quantities(cfg_grid: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dy"] = cfg_grid["ymax"] / cfg_grid["ny"]
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
            "kyvr": jnp.fft.rfftfreq(cfg_grid["nvy"], d=cfg_grid["dvy"]) * 2.0 * np.pi,
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
    cfg_grid["n_prof_total"], cfg_grid["starting_f"] = initialize_total_distribution(cfg)

    cfg_grid["kprof"] = np.ones_like(cfg_grid["n_prof_total"])
    # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

    cfg_grid["ion_charge"] = np.ones_like(cfg_grid["n_prof_total"])

    return cfg


def init_state(cfg: Dict) -> Dict:
    """
    This function initializes the state

    :param cfg:
    :return:
    """
    n_prof_total, f = initialize_total_distribution(cfg)

    state = {}
    for species in ["electron"]:
        state[species] = dict(
            dist=f,
            total_e=jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"])),
            e=jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"])),
            b=jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"])),
        )

    return state


def get_terms_and_solver(cfg):
    return dict(terms=ODETerm(LeapfrogIntegrator(cfg)), solver=Stepper())


# def add_derived_quantities(cfg):
#     """
#
#     In order to keep the main time loop clean, this function handles all the
#     necessary initialization and array creation for the main simulation time loop.
#
#     Initialized here:
#     spatial grid
#     velocity grid
#     distribution function
#     time grid
#     driver array
#
#     :param cfg: Dictionary
#     :return:
#     """
#
#     # Initialize machinery
#     # Spatial Grid
#     # dxs, xs, kxs, one_over_kxs = initialize_spatial_quantities(
#     #     mins=[cfg["grid"]["xmin"], cfg["grid"]["ymin"]],
#     #     maxs=[cfg["grid"]["xmax"], cfg["grid"]["ymax"]],
#     #     ns=[cfg["grid"]["nx"], cfg["grid"]["ny"]],
#     # )
#
#     # Distribution function
#     n_prof_total, f = initialize_total_distribution(cfg["density"], cfg, xs)
#
#     kprof = np.ones_like(
#         n_prof_total
#     )  # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])
#
#     if cfg["density"]["quasineutrality"]:
#         ion_charge = np.copy(n_prof_total)
#     else:
#         ion_charge = np.ones_like(n_prof_total)
#
#     # Velocity grid
#     # dvs, vs, kvs = initialize_velocity_quantities(
#     #     vmax=cfg["grid"]["vmax"], nvs=[int(cfg["grid"]["nvx"]), int(cfg["grid"]["nvy"])]
#     # )
#
#     # dt = cfg["grid"]["tmax"] / cfg["grid"]["nt"]
#     # t = dt * np.arange(0, cfg["grid"]["nt"] + 1)
#
#     cfg["grid"]["nt"] = int(cfg["grid"]["tmax"] / cfg["grid"]["dt"] + 1)
#     cfg["grid"]["tmax"] = cfg["grid"]["dt"] * cfg["grid"]["nt"]
#
#     if cfg["grid"]["nt"] > 1e6:
#         cfg["grid"]["max_steps"] = int(1e6)
#         print(r"Only running $10^6$ steps")
#     else:
#         cfg["grid"]["max_steps"] = cfg["grid"]["nt"] + 4
#
#     nuprof = 1.0  # get_profile_with_mask(cfg["nu"]["time-profile"], t, cfg["nu"]["time-profile"]["bump_or_trough"])
#     ktprof = (
#         1.0  # get_profile_with_mask(cfg["krook"]["time-profile"], t, cfg["krook"]["time-profile"]["bump_or_trough"])
#     )
#
#     driver_function = get_driver_function(xs=xs)
#
#     cfg["derived"] = {
#         "e": np.zeros([x.size for x in xs]),
#         "f": f,
#         "nx": int(cfg["grid"]["nx"]),
#         "ny": int(cfg["grid"]["ny"]),
#         "kx": kxs[0],
#         "ky": kxs[1],
#         "x": xs[0],
#         "y": xs[1],
#         "dx": dxs[0],
#         "dy": dxs[1],
#         "c_light": cfg["grid"]["c_light"],
#         # "a": (jnp.zeros(x.size + 2), jnp.zeros(x.size + 2)),
#         "nprof": n_prof_total,
#         "kr_prof": kprof,
#         "iprof": ion_charge,
#         "one_over_kx": one_over_kxs[0],
#         "one_over_ky": one_over_kxs[1],
#         "vx": vs[0],
#         "kvx": kvs[0],
#         "nvx": int(cfg["grid"]["nvx"]),
#         "dvx": dvs[0],
#         "vy": vs[1],
#         "kvy": kvs[1],
#         "nvy": int(cfg["grid"]["nvy"]),
#         "dvy": dvs[1],
#         "driver_function": driver_function,
#         # "dt": dt,
#         # "nu_prof": nuprof,
#         # "kt_prof": ktprof,
#         "t": np.linspace(0, cfg["grid"]["tmax"], cfg["grid"]["nt"]),
#     }
#     # cfg["derived"]["dt"] = float(cfg["derived"]["dt"][1] - cfg["derived"]["dt"][0])
#
#     return cfg


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    for k in cfg["save"].keys():
        cfg["save"][k]["t"]["ax"] = np.linspace(
            cfg["save"][k]["t"]["tmin"], cfg["save"][k]["t"]["tmax"], cfg["save"][k]["t"]["nt"]
        )

    if "fields" in cfg["save"].keys():

        def fields_save_func(t, y, args):
            return {"fields": y["fields"]}

        cfg["save"]["fields"]["func"] = fields_save_func

    if "dist" in cfg["save"].keys():

        def dist_save_func(t, y, args):
            return {"dist": y["dist"]}

        cfg["save"]["dist"]["func"] = dist_save_func

    return cfg


def get_envelope(p_wL, p_wR, p_L, p_R, ax, a0, bump_or_trough="trough", this_np=np):
    if bump_or_trough.casefold() == "trough":
        return a0 * 0.5 * (this_np.tanh((ax - p_L) / p_wL) - this_np.tanh((ax - p_R) / p_wR))
    elif bump_or_trough.casefold() == "bump":
        return a0 * 0.5 * (2.0 - this_np.tanh((ax - p_L) / p_wL) + this_np.tanh((ax - p_R) / p_wR))


def get_profile_with_mask(prof_dict, axs, b_or_t, this_np=np):
    profile = 1.0 - (
        get_envelope(
            p_wL=prof_dict["rise"][0],
            p_L=prof_dict["center"][0] - 0.5 * prof_dict["width"][0],
            p_wR=prof_dict["rise"][0],
            p_R=prof_dict["center"][0] + 0.5 * prof_dict["width"][0],
            ax=axs[0],
            a0=1.0,
            bump_or_trough=b_or_t,
            this_np=this_np,
        )[:, None]
        * get_envelope(
            p_wL=prof_dict["rise"][1],
            p_L=prof_dict["center"][1] - 0.5 * prof_dict["width"][1],
            p_wR=prof_dict["rise"][1],
            p_R=prof_dict["center"][1] + 0.5 * prof_dict["width"][1],
            ax=axs[1],
            a0=1.0,
            bump_or_trough=b_or_t,
            this_np=this_np,
        )[None, :]
    )
    mask = (
        get_envelope(
            p_wL=prof_dict["rise"][0],
            p_L=prof_dict["center"][0] - 0.5 * prof_dict["width"][0],
            p_wR=prof_dict["rise"][0],
            p_R=prof_dict["center"][0] + 0.5 * prof_dict["width"][0],
            ax=axs[0],
            a0=1.0,
            bump_or_trough="trough",
            this_np=this_np,
        )[:, None]
        * get_envelope(
            p_wL=prof_dict["rise"][1],
            p_L=prof_dict["center"][1] - 0.5 * prof_dict["width"][1],
            p_wR=prof_dict["rise"][1],
            p_R=prof_dict["center"][1] + 0.5 * prof_dict["width"][1],
            ax=axs[1],
            a0=1.0,
            bump_or_trough="trough",
            this_np=this_np,
        )[None, :]
    )
    profile *= prof_dict["wall_height"]
    profile += prof_dict["baseline"]
    profile *= prof_dict["slope"] * mask * ax + 1.0

    return profile
