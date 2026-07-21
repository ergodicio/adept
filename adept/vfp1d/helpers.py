#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io


import numpy as np
from jax import Array
from jax import numpy as jnp
from scipy.special import gamma

from adept._base_ import get_envelope
from adept.normalization import UREG, PlasmaNormalization, normalize

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
        if cfg["units"]["logLambda"].casefold() in ("lee-more", "lee_more", "leemore"):
            # Lee & More 1984 (Phys. Fluids 27, 1273):
            #   lnLambda = max(2, 0.5 * ln(1 + (bmax/bmin)^2))
            # bmax: Debye-Hückel length with electron + ion screening, floored at
            #       the ion-sphere radius R0
            # bmin: max(classical closest approach, electron thermal de Broglie)
            # With the Zeff convention (Z*ni = ne, Z^2*ni = sum_i n_i Z_i^2) the ion
            # screening term Z^2*ni/Ti is exact for mixtures.
            ne_cc = ne.to("1/cc").magnitude
            Te_eV = Te.to("eV").magnitude
            Ti_eV = UREG.Quantity(cfg["units"]["reference ion temperature"]).to("eV").magnitude
            ni_cc = ne_cc / Z

            e2 = 1.44e-7  # e^2 in eV cm (Gaussian)
            hbar_c = 1.9732697e-5  # eV cm
            me_c2 = 510998.95  # eV

            inv_lD2 = 4 * np.pi * e2 * (ne_cc / Te_eV + Z**2 * ni_cc / Ti_eV)
            R0 = (3.0 / (4 * np.pi * ni_cc)) ** (1.0 / 3.0)
            b_max = max(inv_lD2**-0.5, R0)

            b_classical = Z * e2 / (3.0 * Te_eV)
            b_quantum = hbar_c / (2.0 * np.sqrt(3.0 * Te_eV * me_c2))
            b_min = max(b_classical, b_quantum)

            logLambda_ei = max(2.0, 0.5 * np.log(1.0 + (b_max / b_min) ** 2))
            logLambda_ee = logLambda_ei

        elif cfg["units"]["logLambda"].casefold() == "nrl":
            log_ne = np.log(ne.to("1/cc").magnitude)
            log_Te = np.log(Te.to("eV").magnitude)
            log_Z = np.log(Z)

            logLambda_ee = max(
                2.0, 23.5 - 0.5 * log_ne + 1.25 * log_Te - np.sqrt(1e-5 + 0.0625 * (log_Te - 2.0) ** 2.0)
            )

            if Te.to("eV").magnitude > 10 * Z**2.0:
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


def _read_two_column_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a two-column (coordinate, value) text file.

    Tries comma-delimited first, then whitespace-delimited. Header rows and any
    rows that fail to parse as numbers are dropped. Rows are sorted by coordinate.

    :param path: path to the file
    :return: (coordinate, value) arrays
    """
    data = None
    for delimiter in (",", None):
        try:
            candidate = np.genfromtxt(path, delimiter=delimiter, comments="#")
        except ValueError:
            continue
        if candidate.ndim == 2 and candidate.shape[1] >= 2:
            data = candidate
            break

    if data is None:
        raise ValueError(f"Could not parse two numeric columns from {path}")

    data = data[:, :2]
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < 2:
        raise ValueError(f"Fewer than 2 valid (coordinate, value) rows in {path}")

    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def load_profile_on_grid(prof_cfg: dict, x_grid: Array, norm: PlasmaNormalization):
    """
    Loads a (coordinate, value) profile from a file and interpolates it onto the
    simulation grid. Values outside the file's coordinate range are clamped to
    the endpoint values.

    Expected config::

        basis: file
        path: /path/to/profile.csv
        units: 1/cm^3          # units of the value column
        coordinate_units: um   # units of the coordinate column (default um)

    :param prof_cfg: the profile config dict (must contain "path" and "units")
    :param x_grid: normalized spatial grid to interpolate onto (nx,)
    :param norm: plasma normalization (provides L0 for coordinate conversion)
    :return: pint Quantity of interpolated values on the grid (nx,)
    """
    x_file, val_file = _read_two_column_file(prof_cfg["path"])

    coord_scale = float((UREG.Quantity(1.0, prof_cfg.get("coordinate_units", "um")) / norm.L0).to("").magnitude)
    x_file_norm = x_file * coord_scale

    on_grid = np.interp(np.asarray(x_grid), x_file_norm, val_file)
    return UREG.Quantity(on_grid, prof_cfg["units"])


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


def _initialize_total_distribution_(
    cfg: dict, grid, vth_norm: float, norm: PlasmaNormalization
) -> tuple[Array, Array, Array]:
    """
    This function initializes the distribution function as a sum of the individual species

    :param cfg: Dict
    :param grid: Grid object
    :param vth_norm: Thermal velocity in normalized units (vth/v0)
    :param norm: Plasma normalization (used for unit conversion of spatial profiles)
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
                    center = normalize(species_params[k]["center"], norm, dim="x")
                    width = normalize(species_params[k]["width"], norm, dim="x")
                    rise = normalize(species_params[k]["rise"], norm, dim="x")

                    left = center - width * 0.5
                    right = center + width * 0.5
                    prof = get_envelope(rise, rise, left, right, grid.x)

                    if species_params[k]["bump_or_trough"] == "trough":
                        prof = 1 - prof
                    profs[k] = species_params[k]["baseline"] + species_params[k]["bump_height"] * prof

                elif species_params[k]["basis"] == "sine":
                    baseline = species_params[k]["baseline"]
                    amp = species_params[k]["amplitude"]
                    ll = normalize(species_params[k]["wavelength"], norm, dim="x")

                    profs[k] = baseline * (1.0 + amp * jnp.sin(2 * jnp.pi / ll * grid.x))

                elif species_params[k]["basis"] == "cosine":
                    baseline = species_params[k]["baseline"]
                    amp = species_params[k]["amplitude"]
                    ll = normalize(species_params[k]["wavelength"], norm, dim="x")

                    profs[k] = baseline * (1.0 + amp * jnp.cos(2 * jnp.pi / ll * grid.x))

                elif species_params[k]["basis"] == "file":
                    prof_q = load_profile_on_grid(species_params[k], grid.x, norm)
                    if k == "n":
                        # relative to the reference electron density -- the physical
                        # scaling to n/n0 happens in init_state_and_args
                        ref = UREG.Quantity(cfg["units"]["reference electron density"])
                        profs[k] = np.asarray((prof_q / ref).to("").magnitude)
                    else:  # k == "T", relative to the reference electron temperature
                        profs[k] = np.asarray((prof_q / norm.T0).to("").magnitude)

                else:
                    raise NotImplementedError

            prof_total["n"] += profs["n"]

            # Distribution function
            temp_f0, _ = _initialize_distribution_(
                nv=grid.nv,
                m=m,
                vth=vth_norm,
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
