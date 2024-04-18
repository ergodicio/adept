import os
from typing import Dict, Callable, Tuple, List
from collections import defaultdict


import matplotlib.pyplot as plt
import jax, yaml
from jax import numpy as jnp
import numpy as np
import equinox as eqx
from diffrax import ODETerm
import xarray as xr
from astropy.units import Quantity as _Q

from adept.lpse2d.core import integrator
from adept.vlasov1d.integrator import Stepper

# from adept.vfp1d.helpers import write_units
from adept.tf1d.pushers import get_envelope


def write_units(cfg, td):
    timeScale = 1e-12
    spatialScale = 1e-4
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
    w0 = 2 * np.pi * c / (lam0 * 1e-6 / spatialScale)  # 1/ps
    wp0 = w0 * np.sqrt(envelopeDensity)
    w1 = w0 - wp0
    # nc = (w0*1e12)^2 * me / (4*pi*e^2) * (1e-4)^3
    vte = c * np.sqrt(Te / 511)
    vte_sq = vte**2
    cs = c * np.sqrt((Z * Te + 3 * Ti) / (A * 511 * 1836))

    nc = w0**2 * me / (4 * np.pi * e**2)

    E0_source = np.sqrt(8 * np.pi * np.pi * I0 * 1e7 / c_cgs) / fieldScale

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
        "E0_source": E0_source,
        "timeScale": timeScale,
        "spatialScale": spatialScale,
        "velocityScale": velocityScale,
        "massScale": massScale,
        "chargeScale": chargeScale,
        "fieldScale": fieldScale,
    }

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump({k: str(v) for k, v in cfg["units"]["derived"].items()}, fi)

    return cfg


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    cfg_grid["xmax_norm"] = _Q(cfg_grid["xmax"]).to("m").value / cfg["units"]["derived"]["spatialScale"]
    cfg_grid["xmin_norm"] = _Q(cfg_grid["xmin"]).to("m").value / cfg["units"]["derived"]["spatialScale"]
    cfg_grid["ymax_norm"] = _Q(cfg_grid["ymax"]).to("m").value / cfg["units"]["derived"]["spatialScale"]
    cfg_grid["ymin_norm"] = _Q(cfg_grid["ymin"]).to("m").value / cfg["units"]["derived"]["spatialScale"]
    cfg_grid["dx_norm"] = _Q(cfg_grid["dx"]).to("m").value / cfg["units"]["derived"]["spatialScale"]

    cfg_grid["nx"] = int(cfg_grid["xmax_norm"] / cfg_grid["dx_norm"]) + 1
    # cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["dy_norm"] = cfg_grid["dx_norm"]  # cfg_grid["ymax"] / cfg_grid["ny"]
    cfg_grid["ny"] = int(cfg_grid["ymax_norm"] / cfg_grid["dy_norm"]) + 1

    midpt = (cfg_grid["xmax_norm"] + cfg_grid["xmin_norm"]) / 2

    L = _Q(cfg["density"]["gradient scale length"]).to("m").value / cfg["units"]["derived"]["spatialScale"]
    max_density = cfg["density"]["val at center"] + (cfg["grid"]["xmax_norm"] - midpt) / L

    n_max = np.abs(max_density / cfg["units"]["envelope density"] - 1)
    # cfg_grid["dt"] = 0.05 * cfg_grid["dx"]
    cfg_grid["dt"] = (
        0.5
        * 1
        / (
            2
            * 2
            / _Q(cfg_grid["dx"]).to("micron").value ** 2
            * 3
            * cfg["units"]["derived"]["vte"] ** 2
            / (2 * cfg["units"]["derived"]["wp0"])
            + cfg["units"]["derived"]["wp0"] * n_max / 4
        )
    )
    cfg_grid["tmax"] = _Q(cfg_grid["tmax"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    if cfg_grid["nt"] > 1e6:
        cfg_grid["max_steps"] = int(1e6)
        print(r"Only running $10^6$ steps")
    else:
        cfg_grid["max_steps"] = cfg_grid["nt"] + 4

    # cfg = get_more_units(cfg)

    cfg["grid"] = cfg_grid

    return cfg


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

    cfg["save"]["t"]["ax"] = jnp.linspace(tmin, tmax, nt)

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
                cfg_grid["xmin_norm"] + cfg_grid["dx_norm"] / 2,
                cfg_grid["xmax_norm"] - cfg_grid["dx_norm"] / 2,
                cfg_grid["nx"],
            ),
            "y": jnp.linspace(
                cfg_grid["ymin_norm"] + cfg_grid["dy_norm"] / 2,
                cfg_grid["ymax_norm"] - cfg_grid["dy_norm"] / 2,
                cfg_grid["ny"],
            ),
            "t": jnp.linspace(0, cfg_grid["tmax"], cfg_grid["nt"]),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx_norm"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx_norm"]) * 2.0 * np.pi,
            "ky": jnp.fft.fftfreq(cfg_grid["ny"], d=cfg_grid["dy_norm"]) * 2.0 * np.pi,
            "kyr": jnp.fft.rfftfreq(cfg_grid["ny"], d=cfg_grid["dy_norm"]) * 2.0 * np.pi,
        },
    }

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

    one_over_ksq = np.array(1.0 / (cfg_grid["kx"][:, None] ** 2.0 + cfg_grid["ky"][None, :] ** 2.0))
    one_over_ksq[0, 0] = 0.0
    cfg_grid["one_over_ksq"] = jnp.array(one_over_ksq)

    rise = _Q("0.5um").to("m").value / cfg["units"]["derived"]["spatialScale"]
    boundary_width = _Q("3um").to("m").value / cfg["units"]["derived"]["spatialScale"]

    if cfg["terms"]["epw"]["boundary"]["x"] == "absorbing":
        left = cfg["grid"]["xmin_norm"] + boundary_width
        right = cfg["grid"]["xmax_norm"] - boundary_width

        envelope_x = get_envelope(rise, rise, left, right, cfg_grid["x"])[:, None]
    else:
        envelope_x = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

    if cfg["terms"]["epw"]["boundary"]["y"] == "absorbing":
        left = cfg["grid"]["ymin_norm"] + boundary_width
        right = cfg["grid"]["ymax_norm"] - boundary_width
        envelope_y = get_envelope(rise, rise, left, right, cfg_grid["y"])[None, :]
    else:
        envelope_y = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

    cfg_grid["absorbing_boundaries"] = np.exp(-1e4 * cfg_grid["dt"] * (1.0 - envelope_x * envelope_y))

    return cfg_grid


def init_state(cfg: Dict, td=None) -> Dict:
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
    phi_noise = random_amps * jnp.exp(1j * random_phases)
    # ex_noise = -1j * np.fft.ifft(cfg["grid"]["kx"][:, None] * phi_noise)
    # ey_noise = -1j * np.fft.ifft(cfg["grid"]["ky"][None, :] * phi_noise)

    # div_E = (
    #     np.gradient(ex_noise, axis=0) / cfg["grid"]["dx_norm"] + np.gradient(ey_noise, axis=1) / cfg["grid"]["dy_norm"]
    # )
    epw = phi_noise

    background_density = get_density_profile(cfg)
    # permittivity_0 = 1 - background_density

    vte_sq = np.ones((cfg["grid"]["nx"], cfg["grid"]["ny"])) * cfg["units"]["derived"]["vte"] ** 2
    # k0 = cfg["units"]["derived"]["w0"] / cfg["units"]["derived"]["c"] * jnp.sqrt(permittivity_0[x.size // 2])
    # noise_amps = random_amps
    # initial_amp = jnp.ones((cfg["grid"]["nx"], cfg["grid"]["ny"]))
    # div_E = initial_amp * jnp.exp(1j * k0 * x[:, None])
    # div_E *= jnp.exp(-((x[:, None] - np.mean(x)) ** 2.0) / (cfg["grid"]["dx"] * 0.1) ** 2.0)

    E0 = np.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"], 2), dtype=np.complex128)

    state = {"background_density": background_density, "epw": epw, "E0": E0, "vte_sq": vte_sq}

    return {k: v.view(dtype=np.float64) for k, v in state.items()}


def _startup_plots_(state, cfg, td):
    plot_dir = os.path.join(td, "plots", "init_state")
    os.makedirs(plot_dir)

    for comp, label in zip(np.arange(2), ["x", "y"]):
        for func, nm in zip([np.real, np.abs], ["real", "abs"]):
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
            cb = ax.contourf(cfg["grid"]["x"], cfg["grid"]["y"], func(state["e0"][..., comp].T), 32)
            ax.grid()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{nm}(e0_{label}(x,y))")
            fig.colorbar(cb)
            fig.savefig(os.path.join(plot_dir, f"{nm}-e0-{label}.png"), bbox_inches="tight")
            plt.close()

    for k in ["nb", "dn", "temperature", "delta"]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
        cb = ax.contourf(cfg["grid"]["x"], cfg["grid"]["y"], state[k].T, 32)
        ax.grid()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(k)
        fig.colorbar(cb)
        fig.savefig(os.path.join(plot_dir, f"{k}.png"), bbox_inches="tight")
        plt.close()

    for func, nm in zip([np.real, np.abs], ["real", "abs"]):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
        cb = ax.contourf(cfg["grid"]["x"], cfg["grid"]["y"], func(state["phi"].T), 32)
        ax.grid()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{nm}(phi(x,y))")
        fig.colorbar(cb)
        fig.savefig(os.path.join(plot_dir, f"{nm}-phi.png"), bbox_inches="tight")
        plt.close()


def get_density_profile(cfg: Dict):
    if cfg["density"]["basis"] == "uniform":
        nprof = np.ones((cfg["grid"]["nx"], cfg["grid"]["ny"]))

    elif cfg["density"]["basis"] == "linear":
        left = cfg["grid"]["xmin_norm"] + _Q("5.0um").to("m").value / cfg["units"]["derived"]["spatialScale"]
        right = cfg["grid"]["xmax_norm"] - _Q("5.0um").to("m").value / cfg["units"]["derived"]["spatialScale"]
        rise = _Q("0.5um").to("m").value / cfg["units"]["derived"]["spatialScale"]
        # mask = np.repeat(get_envelope(rise, rise, left, right, cfg["grid"]["x"])[:, None], cfg["grid"]["ny"], axis=-1)
        midpt = (cfg["grid"]["xmax_norm"] + cfg["grid"]["xmin_norm"]) / 2

        L = _Q(cfg["density"]["gradient scale length"]).to("m").value / cfg["units"]["derived"]["spatialScale"]
        nprof = cfg["density"]["val at center"] + (cfg["grid"]["x"] - midpt) / L
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
        nprof = baseline * (1.0 + amp * jnp.sin(kk * cfg["grid"]["x"]))
    else:
        raise NotImplementedError

    return nprof


def plot_fields(fields, td):
    t_skip = int(fields.coords["t (ps)"].data.size // 8)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

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


def plot_kt(kfields, td):
    t_skip = int(kfields.coords["t"].data.size // 8)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    for k, v in kfields.items():
        fld_dir = os.path.join(td, "plots", k)
        os.makedirs(fld_dir, exist_ok=True)

        np.log10(np.abs(v[tslice, :, 0])).T.plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_kx.png"), bbox_inches="tight")
        plt.close()

        np.abs(v[tslice, :, :]).T.plot(col="t (ps)", col_wrap=4)
        plt.savefig(os.path.join(fld_dir, f"{k}_kx_ky.png"), bbox_inches="tight")
        plt.close()

        # np.log10(np.abs(v[tslice, :, :])).T.plot(col="t (ps)", col_wrap=4)
        # plt.savefig(os.path.join(fld_dir, f"{k}_kx_ky.png"), bbox_inches="tight")
        # plt.close()
        #
        # kx = kfields.coords["kx"].data


def post_process(result, cfg: Dict, td: str) -> Tuple[xr.Dataset, xr.Dataset]:
    os.makedirs(os.path.join(td, "binary"))
    kfields, fields = make_xarrays(cfg, result.ts, result.ys, td)

    plot_fields(fields, td)
    # plot_kt(kfields, td)

    return kfields, fields


def make_xarrays(cfg, this_t, state, td):
    shift_kx = np.fft.fftshift(cfg["grid"]["kx"])
    shift_ky = np.fft.fftshift(cfg["grid"]["ky"])

    tax_tuple = ("t (ps)", this_t * cfg["units"]["derived"]["timeScale"] * 1e12)
    xax_tuple = ("x (um)", cfg["grid"]["x"] * cfg["units"]["derived"]["spatialScale"] * 1e6)
    yax_tuple = ("y (um)", cfg["grid"]["y"] * cfg["units"]["derived"]["spatialScale"] * 1e6)

    phi_vs_t = state["epw"].view(np.complex128)
    phi_k_np = np.fft.fft2(phi_vs_t, axes=(1, 2))
    ex_k_np = -1j * cfg["grid"]["kx"][None, :, None] * phi_k_np
    ey_k_np = -1j * cfg["grid"]["ky"][None, None, :] * phi_k_np

    phi_k = xr.DataArray(np.fft.fftshift(phi_k_np, axes=(1, 2)), coords=(tax_tuple, ("kx", shift_kx), ("ky", shift_ky)))
    ex_k = xr.DataArray(np.fft.fftshift(ex_k_np, axes=(1, 2)), coords=(tax_tuple, ("kx", shift_kx), ("ky", shift_ky)))
    ey_k = xr.DataArray(np.fft.fftshift(ey_k_np, axes=(1, 2)), coords=(tax_tuple, ("kx", shift_kx), ("ky", shift_ky)))
    phi_x = xr.DataArray(phi_vs_t, coords=(tax_tuple, xax_tuple, yax_tuple))
    ex = xr.DataArray(
        np.fft.ifft2(ex_k_np, axes=(1, 2)) / cfg["grid"]["nx"] / cfg["grid"]["ny"] * 4,
        coords=(tax_tuple, xax_tuple, yax_tuple),
    )
    ey = xr.DataArray(
        np.fft.ifft2(ey_k_np, axes=(1, 2)) / cfg["grid"]["nx"] / cfg["grid"]["ny"] * 4,
        coords=(tax_tuple, xax_tuple, yax_tuple),
    )
    e0x = xr.DataArray(state["E0"].view(np.complex128)[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e0y = xr.DataArray(state["E0"].view(np.complex128)[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))

    background_density = xr.DataArray(state["background_density"], coords=(tax_tuple, xax_tuple, yax_tuple))

    # delta = xr.DataArray(state["delta"], coords=(tax_tuple, xax_tuple, yax_tuple))

    kfields = xr.Dataset({"phi": phi_k, "ex": ex_k, "ey": ey_k})
    fields = xr.Dataset(
        {"phi": phi_x, "ex": ex, "ey": ey, "e0_x": e0x, "e0_y": e0y, "background_density": background_density}
    )
    # kfields.to_netcdf(os.path.join(td, "binary", "k-fields.xr"), engine="h5netcdf", invalid_netcdf=True)
    fields.to_netcdf(os.path.join(td, "binary", "fields.xr"), engine="h5netcdf", invalid_netcdf=True)

    return kfields, fields


def get_models(model_config: Dict) -> defaultdict[eqx.Module]:
    if model_config:
        model_keys = jax.random.split(jax.random.PRNGKey(420), len(model_config.keys()))
        model_dict = defaultdict(eqx.Module)
        for (term, config), this_key in zip(model_config.items(), model_keys):
            if term == "file":
                pass
            else:
                for act in ["activation", "final_activation"]:
                    if config[act] == "tanh":
                        config[act] = jnp.tanh

                model_dict[term] = eqx.nn.MLP(**{**config, "key": this_key})
        if model_config["file"]:
            model_dict = eqx.tree_deserialise_leaves(model_config["file"], model_dict)

        return model_dict
    else:
        return False


def mva(actual_ek1, mod_defaults, results, td, coords):
    loss_t = np.linspace(200, 400, 64)
    ek1 = -1j * mod_defaults["grid"]["kx"][None, :, None] * results.ys["phi"].view(np.complex128)
    ek1 = jnp.mean(jnp.abs(ek1[:, -1, :]), axis=-1)
    rescaled_ek1 = ek1 / jnp.amax(ek1) * np.amax(actual_ek1.data)
    # interp_ek1 = jnp.interp(loss_t, mod_defaults["save"]["t"]["ax"], rescaled_ek1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    ax[0].plot(coords["t"].data, actual_ek1, label="Vlasov")
    ax[0].plot(mod_defaults["save"]["t"]["ax"], rescaled_ek1, label="NN + Fluid")
    ax[0].axvspan(loss_t[0], loss_t[-1], alpha=0.1)
    ax[1].semilogy(coords["t"].data, actual_ek1, label="Vlasov")
    ax[1].semilogy(mod_defaults["save"]["t"]["ax"], rescaled_ek1, label="NN + Fluid")
    ax[1].axvspan(loss_t[0], loss_t[-1], alpha=0.1)
    ax[0].set_xlabel(r"t ($\omega_p^{-1}$)", fontsize=12)
    ax[1].set_xlabel(r"t ($\omega_p^{-1}$)", fontsize=12)
    ax[0].set_ylabel(r"$|\hat{n}|^{1}$", fontsize=12)
    ax[0].grid()
    ax[1].grid()
    ax[0].legend(fontsize=14)
    fig.savefig(os.path.join(td, "plots", "vlasov_v_fluid.png"), bbox_inches="tight")
    plt.close(fig)


def get_diffeqsolve_quants(cfg):
    return dict(
        terms=ODETerm(integrator.SplitStep(cfg)),
        solver=Stepper(),
        saveat=dict(ts=cfg["save"]["t"]["ax"]),  # , fn=cfg["save"]["func"]["callable"]),
    )
