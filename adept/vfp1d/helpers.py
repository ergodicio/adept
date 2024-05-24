#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from typing import Dict
import os

from time import time


import numpy as np
import xarray, yaml, plasmapy
from astropy import units as u, constants as csts
from jax import numpy as jnp
from diffrax import ODETerm, SubSaveAt
from matplotlib import pyplot as plt


from adept.vfp1d.storage import post_process, get_save_quantities
from adept.vlasov1d.integrator import Stepper
from adept.tf1d.pushers import get_envelope
from adept.vfp1d.integrator import IMPACT, OSHUN1D

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
    n0 = u.Quantity("9.0663e21/cm^3")
    ion_species = cfg["units"]["Ion"]

    wp0 = np.sqrt(n0 * csts.e.to("C") ** 2.0 / (csts.m_e * csts.eps0)).to("Hz")
    tp0 = (1 / wp0).to("fs")

    vth = np.sqrt(2 * Te / csts.m_e).to("m/s")  # mean square velocity eq 4-51a in Shkarofsky

    x0 = (csts.c / wp0).to("nm")

    beta = vth / csts.c

    box_length = ((cfg["grid"]["xmax"] - cfg["grid"]["xmin"]) * x0).to("micron")
    if "ymax" in cfg["grid"].keys():
        box_width = ((cfg["grid"]["ymax"] - cfg["grid"]["ymin"]) * x0).to("micron")
    else:
        box_width = "inf"
    sim_duration = (cfg["grid"]["tmax"] * tp0).to("ps")

    logLambda_ei, logLambda_ee = calc_logLambda(cfg, ne, Te, Z, ion_species)
    logLambda_ee = logLambda_ei

    nD_NRL = 1.72e9 * Te.value**1.5 / np.sqrt(ne.value)
    nD_Shkarofsky = np.exp(logLambda_ei) * Z / 9

    nuei_shk = np.sqrt(2.0 / np.pi) * wp0 * logLambda_ei / np.exp(logLambda_ei)
    nuei_nrl = np.sqrt(2.0 / np.pi) * wp0 * logLambda_ei / nD_NRL

    lambda_mfp_shk = (vth / nuei_shk).to("micron")
    lambda_mfp_nrl = (vth / nuei_nrl).to("micron")

    nuei_epphaines = (
        1 / (0.75 * np.sqrt(csts.m_e) * Te**1.5 / (np.sqrt(2 * np.pi) * ni * Z**2.0 * csts.e.gauss**4.0 * logLambda_ei))
    ).to("Hz")

    all_quantities = {
        "wp0": wp0,
        "n0": n0,
        "tp0": tp0,
        "ne": ne,
        "vth": vth,
        "Te": Te,
        "Ti": Ti,
        "logLambda_ei": logLambda_ei,
        "logLambda_ee": logLambda_ee,
        "beta": beta,
        "x0": x0,
        "nuei_shk": nuei_shk,
        "nuei_nrl": nuei_nrl,
        "nuei_epphaines": nuei_epphaines,
        "nuei_shk_norm": nuei_shk / wp0,
        "nuei_nrl_norm": nuei_nrl / wp0,
        "nuei_epphaines_norm": nuei_epphaines / wp0,
        "lambda_mfp_shk": lambda_mfp_shk,
        "lambda_mfp_nrl": lambda_mfp_nrl,
        "lambda_mfp_epphaines": (vth / nuei_epphaines).to("micron"),
        "nD_NRL": nD_NRL,
        "nD_Shkarofsky": nD_Shkarofsky,
        "box_length": box_length,
        "box_width": box_width,
        "sim_duration": sim_duration,
    }

    cfg["units"]["derived"] = all_quantities
    cfg["grid"]["beta"] = beta.value

    with open(os.path.join(td, "units.yaml"), "w") as fi:
        yaml.dump({k: str(v) for k, v in all_quantities.items()}, fi)

    return cfg


def calc_logLambda(cfg, ne, Te, Z, ion_species):
    """
    Calculate the Coulomb logarithm


    """
    if isinstance(cfg["units"]["logLambda"], str):
        if cfg["units"]["logLambda"].casefold() == "plasmapy":
            logLambda_ei = plasmapy.formulary.Coulomb_logarithm(n_e=ne, T=Te, z_mean=Z, species=("e", ion_species))
            logLambda_ee = plasmapy.formulary.Coulomb_logarithm(n_e=ne, T=Te, z_mean=1.0, species=("e", "e"))

        elif cfg["units"]["logLambda"].casefold() == "nrl":
            log_ne = np.log(ne.to("1/cm^3").value)
            log_Te = np.log(Te.to("eV").value)
            log_Z = np.log(Z)

            logLambda_ee = max(
                2.0, 23.5 - 0.5 * log_ne + 1.25 * log_Te - np.sqrt(1e-5 + 0.0625 * (log_Te - 2.0) ** 2.0)
            )

            if Te.to("eV").value > 10 * Z**2.0:
                logLambda_ei = max(2.0, 24.0 - 0.5 * log_ne + log_Te)
            else:
                logLambda_ei = max(2.0, 23.0 - 0.5 * log_ne + 1.5 * log_Te - log_Z)

        else:
            raise NotImplementedError("This logLambda method is not implemented")
    elif isinstance(cfg["units"]["logLambda"], (int, float)):
        logLambda_ei = cfg["units"]["logLambda"]
        logLambda_ee = cfg["units"]["logLambda"]
    return logLambda_ei, logLambda_ee


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

    # single_dist = np.exp(-3 * (vax**2.0) / (2 * vth**2.0)) / (2 * np.pi * vth**2.0) ** 1.5
    # norm = 1.0 / np.sum(4 * np.pi * single_dist * vax**2.0 * dv, axis=0)

    f = np.zeros([nx, nv])
    for ix, (tn, tt) in enumerate(zip(n_prof, T_prof)):
        # eq 4-51b in Shkarofsky
        single_dist = (2 * np.pi * tt * (vth**2.0 / 2)) ** -1.5 * np.exp(-(vax**2.0) / (2 * tt * (vth**2.0 / 2)))
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
                    ll = species_params[k]["wavelength"]

                    profs[k] = baseline * (1.0 + amp * jnp.sin(2 * jnp.pi / ll * cfg_grid["x"]))

                else:
                    raise NotImplementedError

            profs["n"] *= (cfg["units"]["derived"]["ne"] / cfg["units"]["derived"]["n0"]).value

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

    return f, prof_total["n"]


def get_derived_quantities(cfg: Dict) -> Dict:
    """
    This function just updates the config with the derived quantities that are only integers or strings.

    This is run prior to the log params step

    :param cfg_grid:
    :return:
    """
    cfg_grid = cfg["grid"]
    # cfg_grid["xmax"] = u.Quantity(cfg_grid["xmax"])

    cfg_grid["dx"] = cfg_grid["xmax"] / cfg_grid["nx"]

    # sqrt(2 * k * T / m)
    cfg_grid["vmax"] = (
        8
        * np.sqrt((u.Quantity(cfg["units"]["reference electron temperature"]) / (csts.m_e * csts.c**2.0)).to("")).value
    )

    cfg_grid["dv"] = cfg_grid["vmax"] / cfg_grid["nv"]

    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"]) + 1

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
    cfg_grid["kprof"] = np.ones_like(cfg_grid["x"])
    # get_profile_with_mask(cfg["krook"]["space-profile"], xs, cfg["krook"]["space-profile"]["bump_or_trough"])

    cfg_grid["ion_charge"] = np.zeros_like(cfg_grid["x"]) + cfg_grid["x"]

    cfg_grid["x_a"] = np.concatenate(
        [[cfg_grid["x"][0] - cfg_grid["dx"]], cfg_grid["x"], [cfg_grid["x"][-1] + cfg_grid["dx"]]]
    )

    return cfg_grid


def init_state(cfg: Dict, td=None) -> tuple[Dict, Dict]:
    """
    This function initializes the state

    :param cfg:
    :return:
    """
    f, ne_prof = _initialize_total_distribution_(cfg, cfg["grid"])

    state = {"f0": f}
    for il in range(1, cfg["grid"]["nl"] + 1):
        for im in range(0, il + 1):
            state[f"f{il}{im}"] = jnp.zeros_like(f)

    for field in ["e", "b"]:
        state[field] = jnp.zeros(cfg["grid"]["nx"])

    state["Z"] = jnp.ones(cfg["grid"]["nx"])
    state["ni"] = ne_prof / cfg["units"]["Z"]

    return state, {"drivers": cfg["drivers"]}


def get_diffeqsolve_quants(cfg):
    return dict(
        terms=ODETerm(OSHUN1D(cfg)),
        solver=Stepper(),
        saveat=dict(subs={k: SubSaveAt(ts=v["t"]["ax"], fn=v["func"]) for k, v in cfg["save"].items()}),
    )


def apply_models(models, state, args, cfg):
    return state, args
