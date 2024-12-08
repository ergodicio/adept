#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from typing import Dict, Tuple

import numpy as np
from jax import Array
from scipy.special import gamma
from astropy.units import Quantity as _Q
from jax import numpy as jnp
from adept._base_ import get_envelope

# ideally this should be passed as as an argument and not re-initialised
from adept.vfp1d.vector_field import OSHUN1D


def gamma_3_over_m(m: float) -> Array:
    """
    Interpolates gamma(3/m) function from a previous calculation. This is used in the super gaussian initialization scheme

    :param m: float between 2 and 5
    :return: Array

    """
    return gamma(3.0 / m)  # np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m: float) -> Array:
    """
    Interpolates gamma(5/m) function from a previous calculation. This is used in the super gaussian initialization scheme

    :param m: float between 2 and 5
    :return: Array
    """
    return gamma(5.0 / m)  # np.interp(m, m_ax, g_5_m)


def calc_logLambda(cfg: Dict, ne: float, Te: float, Z: int, ion_species: str) -> Tuple[float, float]:
    """
    Calculate the Coulomb logarithm

    :param cfg: Dict
    :param ne: float
    :param Te: float
    :param Z: int
    :param ion_species: str

    :return: Tuple[float, float]

    """
    if isinstance(cfg["units"]["logLambda"], str):
        if cfg["units"]["logLambda"].casefold() == "nrl":
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

    TODO: temperature and density pertubations (JPB - this is done right so can delete this line?)

    :param nx: size of grid in x (single int)
    :param nv: size of grid in v (single int)
    :param vmax: maximum absolute value of v (single float)
    :return: f, vax (nx, nv), (nv,)
    """

    # noise_generator = np.random.default_rng(seed=noise_seed)

    dv = vmax / nv
    vax = np.linspace(dv / 2.0, vmax - dv / 2.0, nv)

    f = np.zeros([nx, nv])
    # obviously not a bottleneck as initialisation, but this should be trivial to vectorize
    for ix, (tn, tt) in enumerate(zip(n_prof, T_prof)):
        # eq 4-51b in Shkarofsky
        # redundant
        # single_dist = (2 * np.pi * tt * (vth**2.0 / 2)) ** -1.5 * np.exp(-(vax**2.0) / (2 * tt * (vth**2.0 / 2)))

        # from Ridgers2008, allows initialisation as a super-Gaussian with temperature tt
        vth_x = np.sqrt(tt) * vth
        alpha = np.sqrt(3.0 * gamma_3_over_m(m) / 2.0 / gamma_5_over_m(m))
        cst = m / (4 * np.pi * alpha**3.0 * gamma_3_over_m(m))
        single_dist = cst / vth_x**3.0 * np.exp(-((vax / alpha / vth_x) ** m))

        f[ix, :] = tn * single_dist

    # if noise_type.casefold() == "uniform":
    #     f = (1.0 + noise_generator.uniform(-noise_val, noise_val, nx)[:, None]) * f
    # elif noise_type.casefold() == "gaussian":
    #     f = (1.0 + noise_generator.normal(-noise_val, noise_val, nx)[:, None]) * f

    return f, vax


def _initialize_total_distribution_(cfg, cfg_grid):
    """
    This function initializes the distribution function as a sum of the individual species

    :param cfg: Dict
    :param cfg_grid: Dict
    :return: distribution function, density profile (nx, nv), (nx,)

    """
    params = cfg["density"]
    prof_total = {"n": np.zeros([cfg_grid["nx"]]), "T": np.zeros([cfg_grid["nx"]])}

    f0 = np.zeros([cfg_grid["nx"], cfg_grid["nv"]])
    f10 = np.zeros([cfg_grid["nx"], cfg_grid["nv"]])
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
                    profs[k] = species_params[k]["baseline"] * np.ones_like(prof_total[k])

                elif species_params[k]["basis"] == "tanh":
                    center = (_Q(species_params[k]["center"]) / cfg["units"]["derived"]["x0"]).to("").value
                    width = (_Q(species_params[k]["width"]) / cfg["units"]["derived"]["x0"]).to("").value
                    rise = (_Q(species_params[k]["rise"]) / cfg["units"]["derived"]["x0"]).to("").value

                    left = center - width * 0.5
                    right = center + width * 0.5
                    # rise = species_params[k]["rise"]
                    prof = get_envelope(rise, rise, left, right, cfg_grid["x"])

                    if species_params[k]["bump_or_trough"] == "trough":
                        prof = 1 - prof
                    profs[k] = species_params[k]["baseline"] + species_params[k]["bump_height"] * prof

                elif species_params[k]["basis"] == "sine":
                    baseline = species_params[k]["baseline"]
                    amp = species_params[k]["amplitude"]
                    ll = (_Q(species_params[k]["wavelength"]) / cfg["units"]["derived"]["x0"]).to("").value

                    profs[k] = baseline * (1.0 + amp * jnp.sin(2 * jnp.pi / ll * cfg_grid["x"]))

                else:
                    raise NotImplementedError

            profs["n"] *= (cfg["units"]["derived"]["ne"] / cfg["units"]["derived"]["n0"]).value

            prof_total["n"] += profs["n"]

            # Distribution function
            temp_f0, _ = _initialize_distribution_(
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
            f0 += temp_f0

            # initialize f1 by taking a big time step while keeping f0 fix (essentailly sets electron inertia to 0)
            # I don't like having to reinitialise oshun to get helper functions, either we pass as an argument or refactor
            oshun = OSHUN1D(cfg)
            big_dt = 1e12
            ni = prof_total["n"] / cfg["units"]["Z"]
            f10_star = -big_dt * oshun.v[None, :] * oshun.ddx(f0)
            f10_from_adv = oshun.ei(Z=jnp.ones(cfg["grid"]["nx"]), ni=ni, f0=f0, f10=f10_star, dt=big_dt)
            jx_from_adv = oshun.calc_j(f10_from_adv)

            df0dv = oshun.ddv(f0)
            f10_from_df0dv = oshun.ei(Z=jnp.ones(cfg["grid"]["nx"]), ni=ni, f0=f0, f10=df0dv, dt=big_dt)
            jx_from_df0dv = oshun.calc_j(f10_from_df0dv)

            # directly solve for ex field
            e_tmp = -jx_from_adv / jx_from_df0dv

            f10 += f10_from_adv + e_tmp[:, None] * f10_from_df0dv

            species_found = True
        else:
            pass

    if not species_found:
        raise ValueError("No species found! Check the config")

    return f0, f10, prof_total["n"]
