#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io


import numpy as np
from astropy.units import Quantity as _Q
from jax import Array
from jax import numpy as jnp
from scipy.special import gamma

from adept._base_ import get_envelope

# ideally this should be passed as as an argument and not re-initialised
from adept.vfp1d.vector_field import OSHUN1D


def gamma_3_over_m(m: float) -> Array:
    """
    Interpolates gamma(3/m) function from a previous calculation.

    This is used in the super gaussian initialization scheme

    :param m: float between 2 and 5
    :return: Array

    """
    return gamma(3.0 / m)  # np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m: float) -> Array:
    """
    Interpolates gamma(5/m) function from a previous calculation.

    This is used in the super gaussian initialization scheme

    :param m: float between 2 and 5
    :return: Array
    """
    return gamma(5.0 / m)  # np.interp(m, m_ax, g_5_m)


def calc_logLambda(
    cfg: dict, ne: float, Te: float, Z: int, ion_species: str, force_ee_equal_ei: bool = False
) -> tuple[float, float]:
    """
    Calculate the Coulomb logarithm

    :param cfg: Dict
    :param ne: float
    :param Te: float
    :param Z: int
    :param ion_species: str
    :param force_ee_equal_ei: if True, set logLambda_ee = logLambda_ei

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
    elif isinstance(cfg["units"]["logLambda"], int | float):
        logLambda_ei = cfg["units"]["logLambda"]
        logLambda_ee = cfg["units"]["logLambda"]

    if force_ee_equal_ei:
        logLambda_ee = logLambda_ei

    return logLambda_ei, logLambda_ee


def _initialize_distribution_(
    nv: int,
    m: float = 2.0,
    vth: float = 1.0,
    vmax: float = 6.0,
    n_prof: Array = np.ones(1),
    T_prof: Array = np.ones(1),
) -> tuple[Array, Array]:
    """
    Initializes a super-Gaussian distribution function on a positive-only velocity grid.

    Uses the Ridgers2008 parameterization which reduces to Maxwell-Boltzmann when m=2.

    :param nv: size of grid in v (single int)
    :param m: super-Gaussian exponent (2.0 = Maxwellian)
    :param vth: thermal velocity
    :param vmax: maximum absolute value of v (single float)
    :param n_prof: density profile (nx,)
    :param T_prof: temperature profile (nx,)
    :return: f, vax (nx, nv), (nv,)
    """

    dv = vmax / nv
    vax = jnp.linspace(dv / 2.0, vmax - dv / 2.0, nv)

    # from Ridgers2008, allows initialisation as a super-Gaussian with temperature tt
    alpha = jnp.sqrt(3.0 * gamma_3_over_m(m) / 2.0 / gamma_5_over_m(m))
    cst = m / (4 * jnp.pi * alpha**3.0 * gamma_3_over_m(m))
    vth_x = jnp.sqrt(T_prof) * vth  # (nx,)
    single_dist = cst / vth_x[:, None] ** 3.0 * jnp.exp(-((vax[None, :] / alpha / vth_x[:, None]) ** m))
    f = n_prof[:, None] * single_dist

    return f, vax


def _initialize_total_distribution_(cfg: dict, grid, plasma_norm) -> tuple[Array, Array, Array]:
    """
    This function initializes the distribution function as a sum of the individual species

    :param cfg: Dict
    :param grid: Grid object
    :param plasma_norm: PlasmaNorm object (provides beta for thermal velocity)
    :return: distribution function, density profile (nx, nv), (nx,)

    """
    params = cfg["density"]
    prof_total = {"n": np.zeros([grid.nx]), "T": np.zeros([grid.nx])}

    f0 = np.zeros([grid.nx, grid.nv])
    f10 = np.zeros([grid.nx + 1, grid.nv])
    species_found = False
    for name, species_params in cfg["density"].items():
        if name.startswith("species-"):
            profs = {}
            m = species_params["m"]
            if name in params:
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
                    prof = get_envelope(rise, rise, left, right, grid.x)

                    if species_params[k]["bump_or_trough"] == "trough":
                        prof = 1 - prof
                    profs[k] = species_params[k]["baseline"] + species_params[k]["bump_height"] * prof

                elif species_params[k]["basis"] == "sine":
                    baseline = species_params[k]["baseline"]
                    amp = species_params[k]["amplitude"]
                    ll = (_Q(species_params[k]["wavelength"]) / cfg["units"]["derived"]["x0"]).to("").value

                    profs[k] = baseline * (1.0 + amp * jnp.sin(2 * jnp.pi / ll * grid.x))

                elif species_params[k]["basis"] == "cosine":
                    baseline = species_params[k]["baseline"]
                    amp = species_params[k]["amplitude"]
                    ll = (_Q(species_params[k]["wavelength"]) / cfg["units"]["derived"]["x0"]).to("").value

                    profs[k] = baseline * (1.0 + amp * jnp.cos(2 * jnp.pi / ll * grid.x))

                else:
                    raise NotImplementedError

            profs["n"] *= (cfg["units"]["derived"]["ne"] / cfg["units"]["derived"]["n0"]).value

            prof_total["n"] += profs["n"]

            # Distribution function
            temp_f0, _ = _initialize_distribution_(
                nv=int(grid.nv),
                m=m,
                vth=plasma_norm.beta,
                vmax=grid.vmax,
                n_prof=profs["n"],
                T_prof=profs["T"],
            )
            f0 += temp_f0

            # initialize f1 by taking a big time step while keeping f0 fixed (essentially sets electron inertia to 0)
            # TODO: add switch to opt in/out
            oshun = OSHUN1D(cfg, grid=grid)

            big_dt = 1e12
            ni = prof_total["n"] / cfg["units"]["Z"]

            Z_edge = oshun.interp_c2e(jnp.ones(grid.nx))
            ni_edge = oshun.interp_c2e(jnp.array(ni))
            f0_at_edges = oshun.interp_c2e(jnp.array(f0))

            f10_star = -big_dt * grid.v[None, :] * oshun.ddx_c2e(jnp.array(f0))
            f10_from_adv = oshun.solve_aniso(
                Z=Z_edge,
                ni=ni_edge,
                f0=f0_at_edges,
                f10=f10_star,
                dt=big_dt,
                include_ee_offdiag_explicitly=False,
            ).block_until_ready()
            jx_from_adv = oshun.calc_j(f10_from_adv)

            df0dv_at_edges = oshun.ddv(f0_at_edges)
            f10_from_df0dv = oshun.solve_aniso(
                Z=Z_edge,
                ni=ni_edge,
                f0=f0_at_edges,
                f10=df0dv_at_edges,
                dt=big_dt,
                include_ee_offdiag_explicitly=False,
            )
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
