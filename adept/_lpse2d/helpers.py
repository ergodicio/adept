import os
from functools import partial

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid", "no-latex"])

import interpax
import numpy as np
import xarray as xr
from astropy.units import Quantity as _Q
from jax import Array
from jax import numpy as jnp

from adept._base_ import get_envelope


def write_units(cfg: dict) -> dict:
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
    lam0 = _Q(cfg["units"]["laser_wavelength"]).to("um").value
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
    ld = vte / w0
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
        if isinstance(cfg["terms"]["epw"]["damping"]["collisions"], bool):
            if Te_eV < 0.01 * Z**2:
                logLambda_ei = 22.8487 - np.log(np.sqrt(ne_cc) * Z / (Te * 1000) ** (3 / 2))
            elif Te_eV > 0.01 * Z**2:
                logLambda_ei = 24 - np.log(np.sqrt(ne_cc) / (Te * 1000))

            e_sq = 510.9896 * 2.8179e-13
            this_me = 510.9896 / 2.99792458e10**2
            nu_coll = float(
                (4 * np.sqrt(2 * np.pi) / 3 * e_sq**2 / np.sqrt(this_me) * Z**2 * ni * logLambda_ei / Te**1.5)
                / 2
                * timeScale
            )
        elif isinstance(cfg["terms"]["epw"]["damping"]["collisions"], float):
            nu_coll = cfg["terms"]["epw"]["damping"]["collisions"]
    else:
        nu_coll = 0.0  # nu_ee + nu_ei + nu_sideloss

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
        "lambda_D": ld,
        "nu_coll": nu_coll,
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


def get_derived_quantities(cfg: dict) -> dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    # cfg_grid["xmax"] = _Q(cfg_grid["xmax"]).to("um").value
    # cfg_grid["xmin"] = _Q(cfg_grid["xmin"]).to("um").value

    if "linear" in cfg["density"]["basis"]:
        L = _Q(cfg["density"]["gradient scale length"]).to("um").value
        nmax = cfg["density"]["max"]
        nmin = cfg["density"]["min"]
        Lgrid = L / 0.25 * (nmax - nmin)

        print("Ignoring xmax and xmin and using the density gradient scale length to set the grid size")
        print("Grid size = L / 0.25 * (nmax - nmin) = ", Lgrid, "um")
    else:
        Lgrid = _Q(cfg_grid["xmax"]).to("um").value

    cfg_grid["xmax"] = Lgrid
    cfg_grid["xmin"] = 0.0

    if "x" in cfg["save"]:
        cfg["save"]["x"]["xmax"] = cfg_grid["xmax"]

    cfg_grid["ymax"] = _Q(cfg_grid["ymax"]).to("um").value
    cfg_grid["ymin"] = _Q(cfg_grid["ymin"]).to("um").value
    cfg_grid["dx"] = _Q(cfg_grid["dx"]).to("um").value

    cfg_grid["nx"] = int((cfg_grid["xmax"] - cfg_grid["xmin"]) / cfg_grid["dx"]) + 1

    cfg_grid["dy"] = cfg_grid["dx"]
    cfg_grid["ny"] = int((cfg_grid["ymax"] - cfg_grid["ymin"]) / cfg_grid["dy"]) + 1

    cfg_grid["dt"] = _Q(cfg_grid["dt"]).to("ps").value
    cfg_grid["tmax"] = _Q(cfg_grid["tmax"]).to("ps").value
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    cfg_grid["max_steps"] = cfg_grid["nt"] + 2048

    # change driver parameters to the right units
    for k in cfg["drivers"].keys():
        cfg["drivers"][k]["derived"] = {}
        cfg["drivers"][k]["derived"]["tw"] = _Q(cfg["drivers"][k]["envelope"]["tw"]).to("ps").value
        cfg["drivers"][k]["derived"]["tc"] = _Q(cfg["drivers"][k]["envelope"]["tc"]).to("ps").value
        cfg["drivers"][k]["derived"]["tr"] = _Q(cfg["drivers"][k]["envelope"]["tr"]).to("ps").value
        cfg["drivers"][k]["derived"]["xr"] = _Q(cfg["drivers"][k]["envelope"]["xr"]).to("um").value
        cfg["drivers"][k]["derived"]["xc"] = _Q(cfg["drivers"][k]["envelope"]["xc"]).to("um").value
        cfg["drivers"][k]["derived"]["xw"] = _Q(cfg["drivers"][k]["envelope"]["xw"]).to("um").value
        cfg["drivers"][k]["derived"]["yw"] = _Q(cfg["drivers"][k]["envelope"]["yw"]).to("um").value
        cfg["drivers"][k]["derived"]["yr"] = _Q(cfg["drivers"][k]["envelope"]["yr"]).to("um").value
        cfg["drivers"][k]["derived"]["yc"] = _Q(cfg["drivers"][k]["envelope"]["yc"]).to("um").value
        if "k0" in cfg["drivers"][k]:
            cfg["drivers"][k]["derived"]["k0"] = cfg["drivers"][k]["k0"]
            cfg["drivers"][k]["derived"]["w0"] = cfg["drivers"][k]["w0"]
            cfg["drivers"][k]["derived"]["a0"] = cfg["drivers"][k]["a0"]

    cfg["grid"] = cfg_grid

    return cfg


def get_solver_quantities(cfg: dict) -> dict:
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
        np.where(np.sqrt(cfg_grid["kx"][:, None] ** 2 + cfg_grid["ky"][None, :] ** 2) == 0, 0, 1)
        if cfg["terms"]["zero_mask"]
        else 1
    )
    # sqrt(kx**2 + ky**2) < low_pass_filter * kmax
    cfg_grid["low_pass_filter"] = np.where(
        np.sqrt(cfg_grid["kx"][:, None] ** 2 + cfg_grid["ky"][None, :] ** 2)
        < cfg_grid["low_pass_filter"] * cfg_grid["kx"].max(),
        1,
        0,
    )

    return cfg_grid


def get_density_profile(cfg: dict) -> Array:
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

        np.log10(np.abs(v[:, :, ymidpt])).plot(size=10, aspect=1)
        plt.savefig(os.path.join(slice_dir, f"spacetime-log-{k}.png"))
        plt.close()

        np.abs(v[:, :, ymidpt]).plot(size=10, aspect=1)
        plt.savefig(os.path.join(slice_dir, f"spacetime-{k}.png"))
        plt.close()

        np.real(v[:, :, ymidpt]).plot(size=10, aspect=1)
        plt.savefig(os.path.join(slice_dir, f"spacetime-real-{k}.png"))
        plt.close()


def plot_kt(kfields, td):
    t_skip = int(kfields.coords["t (ps)"].data.size // 8)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    k_min = -2.5
    k_max = 2.5

    ikx_min = np.argmin(np.abs(kfields.coords[r"kx ($kc\omega_0^{-1}$)"].data - k_min))
    ikx_max = np.argmin(np.abs(kfields.coords[r"kx ($kc\omega_0^{-1}$)"].data - k_max))
    iky_min = np.argmin(np.abs(kfields.coords[r"ky ($kc\omega_0^{-1}$)"].data - k_min))
    iky_max = np.argmin(np.abs(kfields.coords[r"ky ($kc\omega_0^{-1}$)"].data - k_max))

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


def post_process(result, cfg: dict, td: str) -> tuple[xr.Dataset, xr.Dataset]:
    os.makedirs(os.path.join(td, "binary"))

    kfields, fields = make_field_xarrays(cfg, result.ts["fields"], result.ys["fields"], td)
    series = make_series_xarrays(cfg, result.ts["default"], result.ys["default"], td)

    os.makedirs(os.path.join(td, "plots"))
    plot_series(series, td)
    plot_fields(fields, td)
    plot_kt(kfields, td)

    return {"k": kfields, "x": fields, "series": series, "metrics": {}}


def plot_series(series, td):
    for k in series.keys():
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        series[k].plot(ax=ax[0])
        series[k].plot(ax=ax[1])
        ax[1].set_yscale("log")
        fig.savefig(os.path.join(td, "plots", f"{k}_vs_t.png"), bbox_inches="tight")
        fig.savefig(os.path.join(td, "plots", f"{k}_vs_t.pdf"), bbox_inches="tight")
        plt.close()


def make_series_xarrays(cfg, this_t, state, td):
    esq = state["e_sq"]
    max_phi = state["max_phi"]
    series_xr = xr.Dataset(
        {
            "e_sq": xr.DataArray(esq, coords=(("t (ps)", this_t),)),
            "max_phi": xr.DataArray(max_phi, coords=(("t (ps)", this_t),)),
        }
    )
    series_xr.to_netcdf(os.path.join(td, "binary", "series.xr"), engine="h5netcdf", invalid_netcdf=True)
    return series_xr


def make_field_xarrays(cfg, this_t, state, td):
    fld_save = cfg["save"]["fields"]
    if "x" in fld_save:
        kx = fld_save["kx"]
        ky = fld_save["ky"]
        xax = fld_save["x"]["ax"]
        yax = fld_save["y"]["ax"]
        nx = fld_save["x"]["ax"].size
        ny = fld_save["y"]["ax"].size

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

    phi_k_np = state["epw"].view(np.complex128)
    phi_vs_t = np.fft.ifft2(state["epw"].view(np.complex128), axes=(1, 2))
    ex_k_np = -1j * kx[None, :, None] * phi_k_np
    ey_k_np = -1j * ky[None, None, :] * phi_k_np

    phi_k = xr.DataArray(
        np.fft.fftshift(phi_k_np, axes=(1, 2)),
        coords=(tax_tuple, (r"kx ($kc\omega_0^{-1}$)", shift_kx), (r"ky ($kc\omega_0^{-1}$)", shift_ky)),
    )
    ex_k = xr.DataArray(
        np.fft.fftshift(ex_k_np, axes=(1, 2)),
        coords=(tax_tuple, (r"kx ($kc\omega_0^{-1}$)", shift_kx), (r"ky ($kc\omega_0^{-1}$)", shift_ky)),
    )
    ey_k = xr.DataArray(
        np.fft.fftshift(ey_k_np, axes=(1, 2)),
        coords=(tax_tuple, (r"kx ($kc\omega_0^{-1}$)", shift_kx), (r"ky ($kc\omega_0^{-1}$)", shift_ky)),
    )
    phi_x = xr.DataArray(phi_vs_t, coords=(tax_tuple, xax_tuple, yax_tuple))
    ex = xr.DataArray(np.fft.ifft2(ex_k_np, axes=(1, 2)) / nx / ny * 4, coords=(tax_tuple, xax_tuple, yax_tuple))
    ey = xr.DataArray(np.fft.ifft2(ey_k_np, axes=(1, 2)) / nx / ny * 4, coords=(tax_tuple, xax_tuple, yax_tuple))
    e0x = xr.DataArray(state["E0"].view(np.complex128)[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e0y = xr.DataArray(state["E0"].view(np.complex128)[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))
    e1x = xr.DataArray(state["E1"].view(np.complex128)[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e1y = xr.DataArray(state["E1"].view(np.complex128)[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))

    background_density = xr.DataArray(state["background_density"], coords=(tax_tuple, xax_tuple, yax_tuple))

    # delta = xr.DataArray(state["delta"], coords=(tax_tuple, xax_tuple, yax_tuple))

    kfields = xr.Dataset({"phi": phi_k, "ex": ex_k, "ey": ey_k})
    fields = xr.Dataset(
        {
            "phi": phi_x,
            "ex": ex,
            "ey": ey,
            "e0_x": e0x,
            "e0_y": e0y,
            "e1_x": e1x,
            "e1_y": e1y,
            "background_density": background_density,
        }
    )
    kfields.to_netcdf(os.path.join(td, "binary", "k-fields.xr"), engine="h5netcdf", invalid_netcdf=True)
    fields.to_netcdf(os.path.join(td, "binary", "fields.xr"), engine="h5netcdf", invalid_netcdf=True)

    return kfields, fields


def get_save_quantities(cfg: dict) -> dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """

    # cfg["save"]["func"] = {**cfg["save"]["func"], **{"callable": get_save_func(cfg)}}
    tmin = _Q(cfg["save"]["fields"]["t"]["tmin"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    tmax = _Q(cfg["save"]["fields"]["t"]["tmax"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    dt = _Q(cfg["save"]["fields"]["t"]["dt"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    nt = int((tmax - tmin) / dt) + 1

    cfg["save"]["fields"]["t"]["dt"] = dt
    cfg["save"]["fields"]["t"]["ax"] = jnp.linspace(tmin, tmax, nt)

    if "x" in cfg["save"]["fields"]:
        xmin = cfg["grid"]["xmin"]
        xmax = cfg["grid"]["xmax"]
        dx = _Q(cfg["save"]["fields"]["x"]["dx"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
        nx = int((xmax - xmin) / dx)
        cfg["save"]["fields"]["x"]["dx"] = dx
        cfg["save"]["fields"]["x"]["ax"] = jnp.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
        cfg["save"]["fields"]["kx"] = np.fft.fftfreq(nx, d=dx / 2.0 / np.pi)

        if "y" in cfg["save"]["fields"]:
            ymin = cfg["grid"]["ymin"]
            ymax = cfg["grid"]["ymax"]
            dy = _Q(cfg["save"]["fields"]["y"]["dy"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
            ny = int((ymax - ymin) / dy)
            cfg["save"]["fields"]["y"]["dy"] = dy
            cfg["save"]["fields"]["y"]["ax"] = jnp.linspace(ymin + dy / 2.0, ymax - dy / 2.0, ny)
            cfg["save"]["fields"]["ky"] = np.fft.fftfreq(ny, d=dy / 2.0 / np.pi)
        else:
            raise NotImplementedError("Must specify y in save")

        xq, yq = jnp.meshgrid(cfg["save"]["fields"]["x"]["ax"], cfg["save"]["fields"]["y"]["ax"], indexing="ij")

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
                if k in ["E0", "E1"]:
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

    cfg["save"]["fields"]["func"] = save_func

    cfg["save"]["default"] = get_default_save_func(cfg)

    return cfg


def get_default_save_func(cfg):
    def save_func(t, y, args):
        phi_k = y["epw"].view(jnp.complex128)
        ex = -1j * cfg["grid"]["kx"][:, None] * phi_k
        ey = -1j * cfg["grid"]["ky"][None, :] * phi_k
        ex = jnp.fft.ifft2(ex)
        ey = jnp.fft.ifft2(ey)
        e_sq = jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2

        return {"e_sq": jnp.sum(e_sq * cfg["grid"]["dx"] * cfg["grid"]["dy"]), "max_phi": jnp.max(jnp.abs(phi_k))}

    return {"t": {"ax": cfg["grid"]["t"]}, "func": save_func}
