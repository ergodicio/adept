#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from typing import Dict
import os

from time import time


import numpy as np
import xarray, mlflow, yaml, plasmapy
from astropy import units as u, constants as csts
from jax import numpy as jnp
from diffrax import ODETerm, SubSaveAt
from matplotlib import pyplot as plt


from adept.vfp2d.storage import store_fields, store_f, get_save_quantities
from adept.vlasov1d.integrator import Stepper
from adept.tf1d.pushers import get_envelope
from adept.vfp2d.integrator import IMPACT, OSHUN1D

gamma_da = xarray.open_dataarray(os.path.join(os.path.dirname(__file__), "..", "vlasov1d", "gamma_func_for_sg.nc"))
m_ax = gamma_da.coords["m"].data
g_3_m = np.squeeze(gamma_da.loc[{"gamma": "3/m"}].data)
g_5_m = np.squeeze(gamma_da.loc[{"gamma": "5/m"}].data)


def gamma_3_over_m(m):
    return np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m):
    return np.interp(m, m_ax, g_5_m)


def write_units(cfg, td):

    ne = u.Quantity(cfg["units"]["reference electron density"]).to("1/cm^3")
    ni = ne / cfg["units"]["Z"]
    Te = u.Quantity(cfg["units"]["reference electron temperature"]).to("eV")
    Ti = u.Quantity(cfg["units"]["reference ion temperature"]).to("eV")
    Z = cfg["units"]["Z"]
    n0 = u.Quantity("9.09e21/cm^3")
    ion_species = cfg["units"]["Ion"]

    wp0 = np.sqrt(n0 * csts.e.to("C") ** 2.0 / (csts.m_e * csts.eps0)).to("Hz")
    tp0 = (1 / wp0).to("fs")

    vthe = np.sqrt(2.0 * Te / csts.m_e).to("m/s")
    c_light = u.Quantity(1.0 * csts.c).to("m/s")

    x0 = (c_light / wp0).to("nm")

    beta_e = vthe / c_light

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * x0).to("micron")
    if "ymax" in cfg["grid"].keys():
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * x0).to("micron")
    else:
        box_width = "inf"
    sim_duration = (cfg["grid"]["tmax"] * tp0).to("ps")

    logLambda_ei = plasmapy.formulary.Coulomb_logarithm(n_e=ne, T=Te, z_mean=Z, species=("e", ion_species))
    logLambda_ee = plasmapy.formulary.Coulomb_logarithm(n_e=ne, T=Te, z_mean=1.0, species=("e", "e"))
    tauei = (
        0.75
        * np.sqrt(9.11e-31)
        * Te.to("J").value ** 1.5
        / (np.sqrt(2 * np.pi) * ni.to("1/m^3").value * Z**2.0 * csts.e.to("C").value ** 4.0 * logLambda_ei)
    )  # from Epperlein and Haines (1986)

    nuei = u.Quantity(f"{1 / tauei} Hz")
    tauei = tauei * u.s

    all_quantities = {
        "wp0": wp0,
        "n0": n0,
        "tp0": tp0,
        "n0": n0,
        "vthe": vthe,
        "Te": Te,
        "Ti": Ti,
        "logLambda_ei": logLambda_ei,
        "logLambda_ee": logLambda_ee,
        "c_light": c_light,
        "beta": beta_e,
        "x0": x0,
        "nuei": nuei,
        "tauei": tauei,
        "box_length": box_length,
        "box_width": box_width,
        "sim_duration": sim_duration,
    }

    cfg["units"]["derived"] = all_quantities
    cfg["grid"]["beta"] = beta_e.value

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump({k: str(v) for k, v in all_quantities.items()}, fi)

    return cfg


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

    TODO: temperature and density pertubations

    :param nx: size of grid in x (single int)
    :param nv: size of grid in v (single int)
    :param vmax: maximum absolute value of v (single float)
    :return:
    """

    # noise_generator = np.random.default_rng(seed=noise_seed)

    dv = vmax / nv
    vax = np.linspace(dv / 2.0, vmax - dv / 2.0, nv)

    # alpha = np.sqrt(3.0 * gamma_3_over_m(m) / gamma_5_over_m(m))
    # cst = m / (4 * np.pi * alpha**3.0 * gamma(3.0 / m))

    f = np.zeros([nx, nv])
    for ix, (tn, tt) in enumerate(zip(n_prof, T_prof)):
        single_dist = -vax[None, :] ** 2.0 / (tt * vth**2.0)
        single_dist = np.exp(single_dist)
        single_dist = single_dist / np.sum(4 * np.pi * single_dist * vax**2.0 * dv, axis=1)
        f[ix, :] = tn * single_dist

    # if noise_type.casefold() == "uniform":
    #     f = (1.0 + noise_generator.uniform(-noise_val, noise_val, nx)[:, None]) * f
    # elif noise_type.casefold() == "gaussian":
    #     f = (1.0 + noise_generator.normal(-noise_val, noise_val, nx)[:, None]) * f

    return f, vax


def _initialize_total_distribution_(cfg, cfg_grid):
    params = cfg["density"]
    prof_total = {"n": np.zeros([cfg_grid["nx"]]), "T": np.zeros([cfg_grid["nx"]])}

    f = np.zeros([cfg_grid["nx"], cfg_grid["nv"]])
    species_found = False
    for name, species_params in cfg["density"].items():
        if name.startswith("species-"):
            profs = {}
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

            for k in ["n", "T"]:
                if species_params[k]["basis"] == "uniform":
                    profs[k] = np.ones_like(prof_total[k])

                elif species_params[k]["basis"] == "tanh":
                    left = species_params[k]["center"] - species_params[k]["width"] * 0.5
                    right = species_params[k]["center"] + species_params[k]["width"] * 0.5
                    rise = species_params[k]["rise"]
                    prof = get_envelope(rise, rise, left, right, cfg_grid["x"])

                    if species_params[k]["bump_or_trough"] == "trough":
                        prof = 1 - prof
                    profs[k] = species_params[k]["baseline"] + species_params[k]["bump_height"] * prof

                elif species_params[k]["basis"] == "sine":
                    baseline = species_params[k]["baseline"]
                    amp = species_params[k]["amplitude"]
                    kk = species_params[k]["wavenumber"]

                    profs[k] = baseline * (1.0 + amp * jnp.sin(kk * cfg_grid["x"]))

                else:
                    raise NotImplementedError

            prof_total["n"] += profs["n"]

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
            f += temp_f
            species_found = True
        else:
            pass

    if not species_found:
        raise ValueError("No species found! Check the config")

    return f


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]

    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]
    cfg_grid["vmax"] = cfg_grid["vmax"] * cfg["grid"]["beta"]
    cfg_grid["dv"] = cfg_grid["vmax"] / cfg_grid["nv"]

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
            "v": jnp.linspace(cfg_grid["dv"] / 2, cfg_grid["vmax"] - cfg_grid["dv"] / 2, cfg_grid["nv"]),
            "kx": jnp.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
            "kxr": jnp.fft.rfftfreq(cfg_grid["nx"], d=cfg_grid["dx"]) * 2.0 * np.pi,
        },
    }

    # config axes
    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = jnp.array(one_over_kx)

    one_over_kxr = np.zeros_like(cfg_grid["kxr"])
    one_over_kxr[1:] = 1.0 / cfg_grid["kxr"][1:]
    cfg_grid["one_over_kxr"] = jnp.array(one_over_kxr)

    cfg_grid["nuprof"] = 1.0
    # get_profile_with_mask(cfg["nu"]["time-profile"], t, cfg["nu"]["time-profile"]["bump_or_trough"])
    cfg_grid["ktprof"] = 1.0
    # get_profile_with_mask(cfg["krook"]["time-profile"], t, cfg["krook"]["time-profile"]["bump_or_trough"])
    cfg_grid["starting_f"] = _initialize_total_distribution_(cfg, cfg_grid)

    cfg_grid["kprof"] = np.ones_like(cfg_grid["x"])
    # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

    cfg_grid["ion_charge"] = np.zeros_like(cfg_grid["x"]) + cfg_grid["x"]

    cfg_grid["x_a"] = np.concatenate(
        [
            [cfg_grid["x"][0] - cfg_grid["dx"]],
            cfg_grid["x"],
            [cfg_grid["x"][-1] + cfg_grid["dx"]],
        ]
    )

    return cfg_grid


def init_state(cfg: Dict) -> Dict:
    """
    This function initializes the state

    :param cfg:
    :return:
    """
    f = _initialize_total_distribution_(cfg, cfg["grid"])

    state = {"f0": f, "f10": jnp.zeros_like(f)}

    for field in ["e", "b"]:
        state[field] = jnp.zeros(cfg["grid"]["nx"])

    return state


def get_diffeqsolve_quants(cfg):
    return dict(
        terms=ODETerm(OSHUN1D(cfg)),
        solver=Stepper(),
        saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in cfg["save"].items()}),
    )


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

            tax = result.ts["default"] * cfg["units"]["derived"]["tp0"].to("ps").value
            scalars_xr = xarray.Dataset(
                {k: xarray.DataArray(v, coords=(("t (ps)", tax),)) for k, v in result.ys["default"].items()}
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

    f_xr = store_f(cfg, result.ts, td, result.ys)

    mlflow.log_metrics({"qx": np.amax(fields_xr["fields-q a.u."][-1].data)})
    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})

    return {"fields": fields_xr, "dists": f_xr}  # , "scalars": scalars_xr}
