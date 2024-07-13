import os
from typing import Dict, Tuple
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import yaml, mlflow
from jax import Array, numpy as jnp
import numpy as np
import equinox as eqx
import xarray as xr
import interpax

from astropy.units import Quantity as _Q

from adept.lpse2d import nn

from adept import get_envelope


def write_units(cfg: Dict) -> Dict:
    """
    Write the units to a file

    :param cfg:
    :param td:
    :return: cfg
    """
    timeScale = 1e-12  # cgs (ps)
    spatialScale = 1e-4  # cgs (um)
    velocityScale = spatialScale / timeScale
    massScale = 1
    chargeScale = spatialScale ** (3 / 2) * massScale ** (1 / 2) / timeScale
    fieldScale = massScale ** (1 / 2) / spatialScale ** (1 / 2) / timeScale
    # forceScale = massScale * spatialScale/timeScale^2

    Te = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
    Ti = _Q(cfg["units"]["reference ion temperature"]).to("keV").value
    Z = cfg["units"]["ionization state"]
    A = cfg["units"]["atomic number"]
    lam0 = _Q(cfg["units"]["laser wavelength"]).to("um").value
    I0 = _Q(cfg["units"]["laser intensity"]).to("W/cm^2").value
    envelopeDensity = cfg["units"]["envelope density"]

    # Scaled constants
    c_cgs = 2.99792458e10
    me_cgs = 9.10938291e-28
    mp_cgs = 1.6726219e-24
    e_cgs = 4.8032068e-10
    c = c_cgs / velocityScale
    me = me_cgs / massScale
    mi = mp_cgs * A / massScale
    e = e_cgs / chargeScale
    w0 = 2 * np.pi * c / lam0  # 1/ps
    wp0 = w0 * np.sqrt(envelopeDensity)
    w1 = w0 - wp0
    # nc = (w0*1e12)^2 * me / (4*pi*e^2) * (1e-4)^3
    vte = c * np.sqrt(Te / 511)
    vte_sq = vte**2
    cs = c * np.sqrt((Z * Te + 3 * Ti) / (A * 511 * 1836))

    # nu_sideloss = 1e-1

    # nu_ei = calc_nuei(ne, Te, Z, ni, Ti)
    # nu_ee = calc_nuee(ne, Te)

    nc = w0**2 * me / (4 * np.pi * e**2)

    E0_source = np.sqrt(8 * np.pi * I0 * 1e7 / c_cgs) / fieldScale

    ne_cc = nc * envelopeDensity * 1e4**3
    Te_eV = Te * 1000

    coulomb_log = (
        23.0 - np.log(np.sqrt(ne_cc) * Z / Te_eV**1.5) if Te_eV < 10 * Z**2 else 24.0 - np.log(np.sqrt(ne_cc) / Te_eV)
    )
    fract = 1
    Zbar = Z * fract
    ni = fract * ne_cc / Zbar

    # logLambda_ei = np.zeros(len(Z))
    # for iZ in range(len(Z)):
    if cfg["terms"]["epw"]["damping"]["collisions"]:
        if Te_eV < 0.01 * Z**2:
            logLambda_ei = 22.8487 - np.log(np.sqrt(ne_cc) * Z / (Te * 1000) ** (3 / 2))
        elif Te_eV > 0.01 * Z**2:
            logLambda_ei = 24 - np.log(np.sqrt(ne_cc) / (Te * 1000))

        e_sq = 510.9896 * 2.8179e-13
        this_me = 510.9896 / 2.99792458e10**2
        nu_coll = (
            float(
                (4 * np.sqrt(2 * np.pi) / 3 * e_sq**2 / np.sqrt(this_me) * Z**2 * ni * logLambda_ei / Te**1.5)
                / 2
                * timeScale
            )
            * cfg["terms"]["epw"]["damping"]["collisions"]
        )
    else:
        nu_coll = 1e-4  # nu_ee + nu_ei + nu_sideloss

    gradient_scale_length = _Q(cfg["density"]["gradient scale length"]).to("um").value
    I_thresh = calc_threshold_intensity(Te, Ln=gradient_scale_length, w0=w0)
    # for k in ["delta_omega", "initial_phase", "amplitudes"]:

    # Derived units
    cfg["units"]["derived"] = {
        "c": c,
        "me": me,
        "mi": mi,
        "e": e,
        "w0": w0,
        "wp0": wp0,
        "w1": w1,
        "vte": vte,
        "vte_sq": vte_sq,
        "cs": cs,
        "nc": nc,
        "nu_coll": nu_coll,
        "I_thresh": I_thresh,
        "E0_source": E0_source,
        "timeScale": timeScale,
        "spatialScale": spatialScale,
        "velocityScale": velocityScale,
        "massScale": massScale,
        "chargeScale": chargeScale,
        "fieldScale": fieldScale,
    }

    return {k: str(v) for k, v in cfg["units"]["derived"].items()}


def calc_threshold_intensity(Te: float, Ln: float, w0: float) -> float:
    """
    Calculate the TPD threshold intensity

    :param Te:
    :return: intensity
    """

    c = 2.99792458e10
    me_keV = 510.998946  # keV/c^2
    me_cgs = 9.10938291e-28
    e = 4.8032068e-10

    vte = np.sqrt(Te / me_keV) * c
    I_threshold = 4 * 4.134 * 1 / (8 * np.pi) * (me_cgs * c / e) ** 2 * w0 * vte**2 / (Ln / 100) * 1e-7

    return I_threshold


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    # cfg_grid["xmax"] = _Q(cfg_grid["xmax"]).to("um").value
    # cfg_grid["xmin"] = _Q(cfg_grid["xmin"]).to("um").value
    L = _Q(cfg["density"]["gradient scale length"]).to("um").value
    nmax = cfg["density"]["max"]
    nmin = cfg["density"]["min"]
    Lgrid = L / 0.25 * (nmax - nmin)

    print("Ignoring xmax and xmin and using the density gradient scale length to set the grid size")
    print("Grid size = L / 0.25 * (nmax - nmin) = ", Lgrid, "um")
    cfg_grid["xmax"] = Lgrid
    cfg_grid["xmin"] = 0.0

    if "x" in cfg["save"]:
        cfg["save"]["x"]["xmax"] = cfg_grid["xmax"]

    cfg_grid["ymax"] = _Q(cfg_grid["ymax"]).to("um").value
    cfg_grid["ymin"] = _Q(cfg_grid["ymin"]).to("um").value
    cfg_grid["dx"] = _Q(cfg_grid["dx"]).to("um").value

    cfg_grid["nx"] = int(cfg_grid["xmax"] / cfg_grid["dx"]) + 1
    # cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dy"] = cfg_grid["dx"]  # cfg_grid["ymax"] / cfg_grid["ny"]
    cfg_grid["ny"] = int(cfg_grid["ymax"] / cfg_grid["dy"]) + 1

    # midpt = (cfg_grid["xmax"] + cfg_grid["xmin"]) / 2

    # max_density = cfg["density"]["val at center"] + (cfg["grid"]["xmax"] - midpt) / L

    norm_n_max = np.abs(nmax / cfg["units"]["envelope density"] - 1)
    # cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
    cfg_grid["dt"] = _Q(cfg_grid["dt"]).to("ps").value
    cfg_grid["tmax"] = _Q(cfg_grid["tmax"]).to("ps").value
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    # change driver parameters to the right units
    for k in cfg["drivers"].keys():
        cfg["drivers"][k]["envelope"]["tw"] = _Q(cfg["drivers"][k]["envelope"]["tw"]).to("ps").value
        cfg["drivers"][k]["envelope"]["tc"] = _Q(cfg["drivers"][k]["envelope"]["tc"]).to("ps").value
        cfg["drivers"][k]["envelope"]["tr"] = _Q(cfg["drivers"][k]["envelope"]["tr"]).to("ps").value
        cfg["drivers"][k]["envelope"]["xr"] = _Q(cfg["drivers"][k]["envelope"]["xr"]).to("um").value
        cfg["drivers"][k]["envelope"]["xc"] = _Q(cfg["drivers"][k]["envelope"]["xc"]).to("um").value
        cfg["drivers"][k]["envelope"]["xw"] = _Q(cfg["drivers"][k]["envelope"]["xw"]).to("um").value
        cfg["drivers"][k]["envelope"]["yw"] = _Q(cfg["drivers"][k]["envelope"]["yw"]).to("um").value
        cfg["drivers"][k]["envelope"]["yr"] = _Q(cfg["drivers"][k]["envelope"]["yr"]).to("um").value
        cfg["drivers"][k]["envelope"]["yc"] = _Q(cfg["drivers"][k]["envelope"]["yc"]).to("um").value

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

    Lx = cfg_grid["xmax"] - cfg_grid["xmin"]
    Ly = cfg_grid["ymax"] - cfg_grid["ymin"]

    cfg_grid = {
        **cfg_grid,
        **{
            "x": np.linspace(
                cfg_grid["xmin"] + cfg_grid["dx"] / 2,
                cfg_grid["xmax"] - cfg_grid["dx"] / 2,
                cfg_grid["nx"],
            ),
            "y": np.linspace(
                cfg_grid["ymin"] + cfg_grid["dy"] / 2,
                cfg_grid["ymax"] - cfg_grid["dy"] / 2,
                cfg_grid["ny"],
            ),
            "t": np.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "kx": np.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"] / 2.0 / np.pi),
            "kxr": np.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"] / 2.0 / np.pi),
            "ky": np.fft.fftfreq(cfg_grid["ny"], d=cfg_grid["dy"] / 2.0 / np.pi),
            "kyr": np.fft.rfftfreq(cfg_grid["ny"], d=cfg_grid["dy"] / 2.0 / np.pi),
        },
    }

    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = np.array(one_over_kx)

    one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    cfg_grid["one_over_kxr"] = np.array(one_over_kxr)

    one_over_ky = np.zeros_like(cfg_grid["ky"])
    one_over_ky[1:] = 1.0 / cfg_grid["ky"][1:]
    cfg_grid["one_over_ky"] = np.array(one_over_ky)

    one_over_kyr = np.zeros_like(cfg_grid["kyr"])
    one_over_kyr[1:] = 1.0 / cfg_grid["kyr"][1:]
    cfg_grid["one_over_kyr"] = np.array(one_over_kyr)

    one_over_ksq = np.array(1.0 / (cfg_grid["kx"][:, None] ** 2.0 + cfg_grid["ky"][None, :] ** 2.0))
    one_over_ksq[0, 0] = 0.0
    cfg_grid["one_over_ksq"] = np.array(one_over_ksq)

    boundary_width = _Q(cfg_grid["boundary_width"]).to("um").value
    rise = boundary_width / 5

    if cfg["terms"]["epw"]["boundary"]["x"] == "absorbing":
        left = cfg["grid"]["xmin"] + boundary_width
        right = cfg["grid"]["xmax"] - boundary_width

        envelope_x = get_envelope(rise, rise, left, right, cfg_grid["x"])[:, None]
    else:
        envelope_x = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

    if cfg["terms"]["epw"]["boundary"]["y"] == "absorbing":
        left = cfg["grid"]["ymin"] + boundary_width
        right = cfg["grid"]["ymax"] - boundary_width
        envelope_y = get_envelope(rise, rise, left, right, cfg_grid["y"])[None, :]
    else:
        envelope_y = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

    cfg_grid["absorbing_boundaries"] = np.exp(
        -float(cfg_grid["boundary_abs_coeff"]) * cfg_grid["dt"] * (1.0 - envelope_x * envelope_y)
    )

    cfg_grid["zero_mask"] = (
        np.where(cfg_grid["kx"][:, None] * cfg_grid["ky"][None, :] == 0, 0, 1) if cfg["terms"]["zero_mask"] else 1
    )
    # sqrt(kx**2 + ky**2) < 2/3 kmax
    cfg_grid["low_pass_filter"] = np.where(
        np.sqrt(cfg_grid["kx"][:, None] ** 2 + cfg_grid["ky"][None, :] ** 2)
        < cfg_grid["low_pass_filter"] * cfg_grid["kx"].max(),
        1,
        0,
    )

    return cfg_grid


def init_state(cfg: Dict, td=None) -> Tuple[Dict, Dict]:
    """
    This function initializes the state for the PDE solve

    The state is initialized using
    random seeds
    drivers


    :param cfg:
    :return: state: Dict
    """

    if cfg["density"]["noise"]["type"] == "uniform":
        random_amps = np.random.uniform(
            cfg["density"]["noise"]["min"], cfg["density"]["noise"]["max"], (cfg["grid"]["nx"], cfg["grid"]["ny"])
        )

    elif cfg["density"]["noise"]["type"] == "normal":
        loc = 0.5 * (cfg["density"]["noise"]["min"] + cfg["density"]["noise"]["max"])
        scale = 1.0
        random_amps = np.random.normal(loc, scale, (cfg["grid"]["nx"], cfg["grid"]["ny"]))

    else:
        raise NotImplementedError

    random_phases = np.random.uniform(0, 2 * np.pi, (cfg["grid"]["nx"], cfg["grid"]["ny"]))
    phi_noise = 1 * np.exp(1j * random_phases)
    epw = 0 * phi_noise

    background_density = get_density_profile(cfg)
    vte_sq = np.ones((cfg["grid"]["nx"], cfg["grid"]["ny"])) * cfg["units"]["derived"]["vte"] ** 2
    E0 = np.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"], 2), dtype=np.complex128)
    state = {"background_density": background_density, "epw": epw, "E0": E0, "vte_sq": vte_sq}

    drivers = assemble_bandwidth(cfg)
    return {k: v.view(dtype=np.float64) for k, v in state.items()}, {"drivers": drivers}


def assemble_bandwidth(cfg: Dict) -> Dict:
    """
    Assemble the amplitudes, initial phases, and frequency shifts associated with each color

    :param cfg: Dict
    :return: drivers: Dict

    """
    drivers = {"E0": {}}
    num_colors = cfg["drivers"]["E0"]["num_colors"]

    if num_colors == 1:
        drivers["E0"]["delta_omega"] = np.zeros(1)
        drivers["E0"]["initial_phase"] = np.zeros(1)
        drivers["E0"]["amplitudes"] = np.ones(1)
    else:
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        delta_omega = np.linspace(-delta_omega_max, delta_omega_max, num_colors)

        drivers["E0"]["delta_omega"] = delta_omega
        drivers["E0"]["initial_phase"] = np.random.uniform(0, 2 * np.pi, num_colors)

        if cfg["drivers"]["E0"]["amplitude_shape"] == "uniform":
            drivers["E0"]["amplitudes"] = np.ones(num_colors)
        elif cfg["drivers"]["E0"]["amplitude_shape"] == "gaussian":
            drivers["E0"]["amplitudes"] = (
                2
                * np.log(2)
                / delta_omega_max
                / np.sqrt(np.pi)
                * np.exp(-4 * np.log(2) * (delta_omega / delta_omega_max) ** 2.0)
            )
            drivers["E0"]["amplitudes"] = np.sqrt(drivers["E0"]["amplitudes"])  # for amplitude from intensity

        elif cfg["drivers"]["E0"]["amplitude_shape"] == "lorentzian":
            drivers["E0"]["amplitudes"] = (
                1 / np.pi * (delta_omega_max / 2) / (delta_omega**2.0 + (delta_omega_max / 2) ** 2.0)
            )
            drivers["E0"]["amplitudes"] = np.sqrt(drivers["E0"]["amplitudes"])  # for amplitude from intensity
        elif cfg["drivers"]["E0"]["amplitude_shape"] == "ML" or cfg["drivers"]["E0"]["amplitude_shape"] == "opt":
            drivers["E0"]["amplitudes"] = np.ones(num_colors)  # will be modified elsewhere
        elif cfg["drivers"]["E0"]["amplitude_shape"] == "file":
            import tempfile

            with tempfile.TemporaryDirectory() as td:

                import pickle

                if cfg["drivers"]["E0"]["file"].startswith("s3"):
                    import boto3

                    fname = cfg["drivers"]["E0"]["file"]

                    bucket = fname.split("/")[2]
                    key = "/".join(fname.split("/")[3:])
                    s3 = boto3.client("s3")
                    s3.download_file(bucket, key, local_fname := os.path.join(td, "drivers.pkl"))
                else:
                    local_fname = cfg["drivers"]["E0"]["file"]

                with open(local_fname, "rb") as fi:
                    drivers = pickle.load(fi)
        else:
            raise NotImplemented

        drivers["E0"]["amplitudes"] /= np.sum(drivers["E0"]["amplitudes"])

    return drivers


def get_density_profile(cfg: Dict) -> Array:
    """
    Helper function for initializing the density profile

    It can be uniform, linear, exponential, tanh, or sine

    :param cfg: Dict
    """
    if cfg["density"]["basis"] == "uniform":
        nprof = np.ones((cfg["grid"]["nx"], cfg["grid"]["ny"]))

    elif cfg["density"]["basis"] == "linear":
        left = cfg["grid"]["xmin"] + _Q("5.0um").to("um").value
        right = cfg["grid"]["xmax"] - _Q("5.0um").to("um").value
        rise = _Q("0.5um").to("um").value
        # mask = np.repeat(get_envelope(rise, rise, left, right, cfg["grid"]["x"])[:, None], cfg["grid"]["ny"], axis=-1)
        # midpt = (cfg["grid"]["xmax"] + cfg["grid"]["xmin"]) / 2

        nprof = (
            cfg["density"]["min"]
            + (cfg["density"]["max"] - cfg["density"]["min"]) * cfg["grid"]["x"] / cfg["grid"]["xmax"]
        )
        # nprof = mask * nprof[:, None]
        nprof = np.repeat(nprof[:, None], cfg["grid"]["ny"], axis=-1)

    elif cfg["density"]["basis"] == "exponential":
        left = cfg["density"]["center"] - cfg["density"]["width"] * 0.5
        right = cfg["density"]["center"] + cfg["density"]["width"] * 0.5
        rise = cfg["density"]["rise"]
        mask = get_envelope(rise, rise, left, right, cfg["grid"]["x"])

        L = _Q(cfg["density"]["gradient scale length"]).to("nm").value / cfg["units"]["derived"]["x0"].to("nm").value
        nprof = cfg["density"]["val at center"] * np.exp((cfg["grid"]["x"] - cfg["density"]["center"]) / L)
        nprof = mask * nprof

    elif cfg["density"]["basis"] == "tanh":
        left = cfg["density"]["center"] - cfg["density"]["width"] * 0.5
        right = cfg["density"]["center"] + cfg["density"]["width"] * 0.5
        rise = cfg["density"]["rise"]
        nprof = get_envelope(rise, rise, left, right, cfg["grid"]["x"])

        if cfg["density"]["bump_or_trough"] == "trough":
            nprof = 1 - nprof
        nprof = cfg["density"]["baseline"] + cfg["density"]["bump_height"] * nprof

    elif cfg["density"]["basis"] == "sine":
        baseline = cfg["density"]["baseline"]
        amp = cfg["density"]["amplitude"]
        kk = cfg["density"]["wavenumber"]
        nprof = baseline * (1.0 + amp * np.sin(kk * cfg["grid"]["x"]))
    else:
        raise NotImplementedError

    return nprof


def plot_fields(fields, td):
    t_skip = int(fields.coords["t (ps)"].data.size // 8)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    dx = fields.coords["x (um)"].data[1] - fields.coords["x (um)"].data[0]
    dy = fields.coords["y (um)"].data[1] - fields.coords["y (um)"].data[0]

    for k, v in fields.items():
        fld_dir = os.path.join(td, "plots", k)
        os.makedirs(fld_dir)

        np.abs(v[tslice]).T.plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_x.png"), bbox_inches="tight")
        plt.close()

        np.real(v[tslice]).T.plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_x_r.png"), bbox_inches="tight")
        plt.close()

        # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        # np.abs(v[:, 1, 0]).plot(ax=ax)
        # fig.savefig(os.path.join(td, "plots", f"{k}_k1.png"))
        # plt.close()
        ymidpt = int(fields.coords["y (um)"].data.size // 2)
        slice_dir = os.path.join(fld_dir, "slice-along-x")
        os.makedirs(slice_dir)
        np.log10(np.abs(v[tslice, :, ymidpt])).plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(slice_dir, f"log-{k}.png"))
        plt.close()

        np.abs(v[tslice, :, ymidpt]).plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(slice_dir, f"{k}.png"))
        plt.close()

        np.real(v[tslice, :, ymidpt]).plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(slice_dir, f"real-{k}.png"))
        plt.close()

        np.log10(np.abs(v[:, :, ymidpt])).plot()
        plt.savefig(os.path.join(slice_dir, f"spacetime-log-{k}.png"))
        plt.close()

        np.abs(v[:, :, ymidpt]).plot()
        plt.savefig(os.path.join(slice_dir, f"spacetime-{k}.png"))
        plt.close()

        np.real(v[:, :, ymidpt]).plot()
        plt.savefig(os.path.join(slice_dir, f"spacetime-real-{k}.png"))
        plt.close()

    # plot total electric field energy in box vs time
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    total_e_sq = np.abs(fields["ex"].data ** 2 + fields["ey"].data ** 2).sum(axis=(1, 2)) * dx * dy
    ax[0].plot(fields.coords["t (ps)"].data, total_e_sq)
    ax[0].set_xlabel("t (ps)")
    ax[0].set_ylabel("Total E^2")

    ax[1].semilogy(fields.coords["t (ps)"].data, total_e_sq)
    ax[1].set_xlabel("t (ps)")
    ax[1].set_ylabel("Total E^2")

    ax[0].grid()
    ax[1].grid()

    fig.savefig(os.path.join(td, "plots", "total_e_sq.png"))
    plt.close()


def plot_kt(kfields, td):
    t_skip = int(kfields.coords["t (ps)"].data.size // 8)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    k_min = -2.5
    k_max = 2.5

    ikx_min = np.argmin(np.abs(kfields.coords["kx"].data - k_min))
    ikx_max = np.argmin(np.abs(kfields.coords["kx"].data - k_max))
    iky_min = np.argmin(np.abs(kfields.coords["ky"].data - k_min))
    iky_max = np.argmin(np.abs(kfields.coords["ky"].data - k_max))

    kx_slice = slice(ikx_min, ikx_max)
    ky_slice = slice(iky_min, iky_max)

    for k, v in kfields.items():
        fld_dir = os.path.join(td, "plots", k)
        os.makedirs(fld_dir, exist_ok=True)

        np.log10(np.abs(v[tslice, kx_slice, 0])).T.plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"log_{k}_kx.png"), bbox_inches="tight")
        plt.close()

        np.abs(v[tslice, kx_slice, ky_slice]).T.plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_kx_ky.png"), bbox_inches="tight")
        plt.close()

        np.log10(np.abs(v[tslice, kx_slice, ky_slice])).T.plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"log_{k}_kx_ky.png"), bbox_inches="tight")
        plt.close()
        #
        # kx = kfields.coords["kx"].data


def post_process(result, cfg: Dict, td: str, args) -> Tuple[xr.Dataset, xr.Dataset]:

    # if isinstance(sim_out, tuple):
    #     val, actual_sim_out = sim_out[0]
    #     grad = sim_out[1]
    #     result = actual_sim_out["solver_result"]
    #     used_driver = actual_sim_out["args"]["drivers"]
    # # else:
    # result = sim_out["solver_result"]
    used_driver = args["drivers"]
    import pickle

    with open(os.path.join(td, "used_driver.pkl"), "wb") as fi:
        pickle.dump(used_driver, fi)

    dw_over_w = used_driver["E0"]["delta_omega"]  # / cfg["units"]["derived"]["w0"] - 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 5), tight_layout=True)
    ax[0].plot(dw_over_w, used_driver["E0"]["amplitudes"], "o")
    ax[0].grid()
    ax[0].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[0].set_ylabel("$|E|$", fontsize=14)
    ax[1].semilogy(dw_over_w, used_driver["E0"]["amplitudes"], "o")
    ax[1].grid()
    ax[1].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[1].set_ylabel("$|E|$", fontsize=14)
    ax[2].plot(dw_over_w, used_driver["E0"]["initial_phase"], "o")
    ax[2].grid()
    ax[2].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[2].set_ylabel(r"$\angle E$", fontsize=14)
    plt.savefig(os.path.join(td, "learned_bandwidth.png"), bbox_inches="tight")
    plt.close()

    os.makedirs(os.path.join(td, "binary"))
    kfields, fields = make_xarrays(cfg, result.ts, result.ys, td)

    plot_fields(fields, td)
    plot_kt(kfields, td)

    dx = fields.coords["x (um)"].data[1] - fields.coords["x (um)"].data[0]
    dy = fields.coords["y (um)"].data[1] - fields.coords["y (um)"].data[0]
    dt = fields.coords["t (ps)"].data[1] - fields.coords["t (ps)"].data[0]

    metrics = {}
    metrics["total_e_sq"] = float(
        np.sum(np.abs(fields["ex"][-20:].data) ** 2 + np.abs(fields["ey"][-20:].data ** 2) * dx * dy * dt)
    )
    metrics["log10_total_e_sq"] = float(np.log10(metrics["total_e_sq"]))

    # if isinstance(sim_out, tuple):
    #     if "loss_dict" in sim_out[0][1]:
    #         for k, v in sim_out[0][1]["loss_dict"].items():
    #             metrics[k] = float(v)

    # mlflow.log_metrics(metrics)

    return {"k": kfields, "x": fields, "metrics": metrics}


def make_xarrays(cfg, this_t, state, td):
    if "x" in cfg["save"]:
        kx = cfg["save"]["kx"]
        ky = cfg["save"]["ky"]
        xax = cfg["save"]["x"]["ax"]
        yax = cfg["save"]["y"]["ax"]
        nx = cfg["save"]["x"]["ax"].size
        ny = cfg["save"]["y"]["ax"].size

    else:
        kx = cfg["grid"]["kx"]
        ky = cfg["grid"]["ky"]
        xax = cfg["grid"]["x"]
        yax = cfg["grid"]["y"]
        nx = cfg["grid"]["nx"]
        ny = cfg["grid"]["ny"]

    shift_kx = np.fft.fftshift(kx) * cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]
    shift_ky = np.fft.fftshift(ky) * cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]

    tax_tuple = ("t (ps)", this_t)
    xax_tuple = ("x (um)", xax)
    yax_tuple = ("y (um)", yax)

    phi_vs_t = state["epw"].view(np.complex128)
    phi_k_np = np.fft.fft2(phi_vs_t, axes=(1, 2))
    ex_k_np = -1j * kx[None, :, None] * phi_k_np
    ey_k_np = -1j * ky[None, None, :] * phi_k_np

    phi_k = xr.DataArray(np.fft.fftshift(phi_k_np, axes=(1, 2)), coords=(tax_tuple, ("kx", shift_kx), ("ky", shift_ky)))
    ex_k = xr.DataArray(np.fft.fftshift(ex_k_np, axes=(1, 2)), coords=(tax_tuple, ("kx", shift_kx), ("ky", shift_ky)))
    ey_k = xr.DataArray(np.fft.fftshift(ey_k_np, axes=(1, 2)), coords=(tax_tuple, ("kx", shift_kx), ("ky", shift_ky)))
    phi_x = xr.DataArray(phi_vs_t, coords=(tax_tuple, xax_tuple, yax_tuple))
    ex = xr.DataArray(np.fft.ifft2(ex_k_np, axes=(1, 2)) / nx / ny * 4, coords=(tax_tuple, xax_tuple, yax_tuple))
    ey = xr.DataArray(np.fft.ifft2(ey_k_np, axes=(1, 2)) / nx / ny * 4, coords=(tax_tuple, xax_tuple, yax_tuple))
    e0x = xr.DataArray(state["E0"].view(np.complex128)[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e0y = xr.DataArray(state["E0"].view(np.complex128)[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))

    background_density = xr.DataArray(state["background_density"], coords=(tax_tuple, xax_tuple, yax_tuple))

    # delta = xr.DataArray(state["delta"], coords=(tax_tuple, xax_tuple, yax_tuple))

    kfields = xr.Dataset({"phi": phi_k, "ex": ex_k, "ey": ey_k})
    fields = xr.Dataset(
        {"phi": phi_x, "ex": ex, "ey": ey, "e0_x": e0x, "e0_y": e0y, "background_density": background_density}
    )
    kfields.to_netcdf(os.path.join(td, "binary", "k-fields.xr"), engine="h5netcdf", invalid_netcdf=True)
    fields.to_netcdf(os.path.join(td, "binary", "fields.xr"), engine="h5netcdf", invalid_netcdf=True)

    return kfields, fields


def get_models(all_models_config: Dict) -> defaultdict[eqx.Module]:
    models = {}
    for nm, this_models_config in all_models_config.items():
        if "file" in this_models_config:
            file_path = this_models_config["file"]
            if file_path.endswith(".pkl"):
                import pickle

                with open(file_path, "rb") as fi:
                    models[nm] = pickle.load(fi)
                    print(models)
                print(f"Loading {nm} weights from file {file_path} and ignoring any other specifications.")
            elif file_path.endswith(".eqx"):
                models[nm], _ = nn.load(file_path)

                print(f"Loading {nm} model from file {file_path} and ignoring any other specifications.")
        else:
            if this_models_config["type"] == "MLP":
                models[nm] = nn.DriverModel(**this_models_config["config"])
            elif this_models_config["type"] == "VAE":
                models[nm] = nn.VAE(**this_models_config["config"])
            else:
                raise NotImplementedError

    return models


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """

    # cfg["save"]["func"] = {**cfg["save"]["func"], **{"callable": get_save_func(cfg)}}
    tmin = _Q(cfg["save"]["t"]["tmin"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    tmax = _Q(cfg["save"]["t"]["tmax"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    dt = _Q(cfg["save"]["t"]["dt"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    nt = int((tmax - tmin) / dt) + 1

    cfg["save"]["t"]["dt"] = dt
    cfg["save"]["t"]["ax"] = jnp.linspace(tmin, tmax, nt)

    if "x" in cfg["save"]:
        xmin = cfg["grid"]["xmin"]
        xmax = cfg["grid"]["xmax"]
        dx = _Q(cfg["save"]["x"]["dx"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
        nx = int((xmax - xmin) / dx)
        cfg["save"]["x"]["dx"] = dx
        cfg["save"]["x"]["ax"] = jnp.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
        cfg["save"]["kx"] = np.fft.fftfreq(nx, d=dx / 2.0 / np.pi)

        if "y" in cfg["save"]:
            ymin = cfg["grid"]["ymin"]
            ymax = cfg["grid"]["ymax"]
            dy = _Q(cfg["save"]["y"]["dy"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
            ny = int((ymax - ymin) / dy)
            cfg["save"]["y"]["dy"] = dy
            cfg["save"]["y"]["ax"] = jnp.linspace(ymin + dy / 2.0, ymax - dy / 2.0, ny)
            cfg["save"]["ky"] = np.fft.fftfreq(ny, d=dy / 2.0 / np.pi)
        else:
            raise NotImplementedError("Must specify y in save")

        xq, yq = jnp.meshgrid(cfg["save"]["x"]["ax"], cfg["save"]["y"]["ax"], indexing="ij")

        interpolator = partial(
            interpax.interp2d,
            xq=jnp.reshape(xq, (nx * ny), order="F"),
            yq=jnp.reshape(yq, (nx * ny), order="F"),
            x=cfg["grid"]["x"],
            y=cfg["grid"]["y"],
            method="linear",
        )

        def save_func(t, y, args):
            save_y = {}
            for k, v in y.items():
                if k == "E0":
                    cmplx_fld = v.view(jnp.complex128)
                    save_y[k] = jnp.concatenate(
                        [
                            jnp.reshape(interpolator(f=cmplx_fld[..., ivec]), (nx, ny), order="F")[..., None]
                            for ivec in range(2)
                        ],
                        axis=-1,
                    ).view(jnp.float64)
                elif k == "epw":
                    cmplx_fld = v.view(jnp.complex128)
                    save_y[k] = jnp.reshape(interpolator(f=cmplx_fld), (nx, ny), order="F").view(jnp.float64)
                else:
                    save_y[k] = jnp.reshape(interpolator(f=v), (nx, ny), order="F")

            return save_y

    else:
        save_func = lambda t, y, args: y

    cfg["save"]["func"] = save_func

    return cfg
