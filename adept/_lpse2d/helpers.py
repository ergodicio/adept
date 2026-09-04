import os
from functools import partial

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid", "no-latex"])

import time

import interpax
import jax
import numpy as np
import xarray as xr
from astropy.units import Quantity as _Q
from jax import Array
from jax import numpy as jnp

from adept._base_ import get_envelope


def next_smooth_fft_size(n, max_prime=7):
    """
    Find the smallest integer >= n that has only small prime factors.

    Parameters:
    -----------
    n : int
        Minimum size needed
    max_prime : int
        Largest prime factor allowed (default: 7)
        Use 5 for best performance, 7 for more flexibility

    Returns:
    --------
    int
        Optimal FFT size >= n
    """
    if n <= 1:
        return 1

    # Generate smooth numbers up to a reasonable limit
    # We'll generate more than we need and find the first one >= n
    limit = n * 2  # generous upper bound

    # Allowed prime factors
    primes = [2, 3, 5]
    if max_prime >= 7:
        primes.append(7)

    # Generate all smooth numbers using dynamic programming
    smooth = [1]
    indices = [0] * len(primes)

    while smooth[-1] < limit:
        # Next candidates: multiply smallest smooth number by each prime
        candidates = [smooth[indices[i]] * primes[i] for i in range(len(primes))]
        next_smooth = min(candidates)
        smooth.append(next_smooth)

        # Increment indices for primes that produced this value
        for i in range(len(primes)):
            if candidates[i] == next_smooth:
                indices[i] += 1

    # Binary search for first value >= n
    idx = np.searchsorted(smooth, n)
    return smooth[idx]


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
        # Conversions to OSIRIS code units, for one-to-one metric comparison with PIC runs:
        # multiply an E-field (code units) by e_norm to get it in me*c*w0/e units, and a
        # length (um) by x_norm to get it in c/w0. I0_code is the nominal incident pump
        # flux in this code's units (c * |E_envelope|^2); the WKB-swelled pump satisfies
        # sqrt(eps) * |E0_local|^2 = E0_source^2 so this holds at any density below nc.
        "e_norm": e / (me * c * w0),
        "x_norm": w0 / c,
        "I0_code": c * E0_source**2,
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

    # Explicit nulls mean "absent". The datamodel advertises `X | None = None` for these
    # fields, so normalize them here -- before anything reads them -- to keep every
    # spelling of "not set" (missing key, or `key: null` in the YAML) equivalent.
    for term_key in ("hpe", "iaw", "light"):
        if term_key in cfg["terms"] and cfg["terms"][term_key] is None:
            cfg["terms"][term_key] = {}
    if "light_substeps" in cfg_grid and cfg_grid["light_substeps"] is None:
        del cfg_grid["light_substeps"]
    for driver in cfg["drivers"].values():
        if isinstance(driver, dict):
            for opt_key in [k for k, v in driver.items() if v is None]:
                del driver[opt_key]

    # A missing (or mistyped) drivers.E0 is legitimate only for the seed-only / direct-EPW-driver
    # test paths; in every other case it silently zeroes the pump and the run completes with
    # nothing but noise. Warn loudly rather than guess.
    if "E0" not in cfg["drivers"]:
        print(
            "WARNING: no drivers.E0 -- the pump laser is identically zero. This is only sensible "
            "for seed-only or direct-EPW-driver (drivers.E2) runs; if you expected a pump, check "
            "the spelling of drivers.E0."
        )

    # Default save.*.t.tmin/tmax to grid values (preserves unit strings)
    for save_type in cfg.get("save", {}).keys():
        if "t" in cfg["save"][save_type]:
            t_cfg = cfg["save"][save_type]["t"]
            t_cfg.setdefault("tmin", cfg_grid.get("tmin", "0ps"))
            t_cfg.setdefault("tmax", cfg_grid["tmax"])

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

    xmax = cfg_grid["xmax"] = Lgrid
    xmin = cfg_grid["xmin"] = 0.0

    if "x" in cfg["save"]:
        cfg["save"]["x"]["xmax"] = cfg_grid["xmax"]

    ymax = cfg_grid["ymax"] = _Q(cfg_grid["ymax"]).to("um").value
    ymin = cfg_grid["ymin"] = _Q(cfg_grid["ymin"]).to("um").value
    dx = cfg_grid["dx"] = _Q(cfg_grid["dx"]).to("um").value

    # round to the nearest even number
    cfg_grid["nx"] = int((xmax - xmin) / dx)
    cfg_grid["nx"] = next_smooth_fft_size(cfg_grid["nx"], max_prime=5)
    cfg_grid["dx"] = dx = (xmax - xmin) / cfg_grid["nx"]  # recalculate dx based on optimal nx

    cfg_grid["dy"] = dx  # we want square cells
    cfg_grid["ny"] = int((ymax - ymin) / dx)  # recalculate ny based on dx
    cfg_grid["ny"] = next_smooth_fft_size(cfg_grid["ny"], max_prime=5)
    # ymax and ymin have to be symmetric about 0 and have to be recalculated
    cfg_grid["ymax"] = ymax = dx * cfg_grid["ny"] / 2
    cfg_grid["ymin"] = ymin = -ymax
    cfg_grid["dt"] = _Q(cfg_grid["dt"]).to("ps").value
    cfg_grid["tmax"] = _Q(cfg_grid["tmax"]).to("ps").value
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    cfg_grid["max_steps"] = cfg_grid["nt"] + 2048

    # SRS: the Raman light is advanced with an explicit conditionally-stable scheme, so it is
    # sub-cycled within each EPW step. The stability bound follows MATLAB line 500
    # (dt_max_seed), generalized to 2D
    # EPW noise source: resolve the amplitude/seed here (prior to log_params) so the
    # run is reproducible and the actual seed lands in MLflow. noise_seed: null (or
    # absent) draws a random seed once, then pins it in the cfg.
    epw_source = cfg["terms"]["epw"]["source"]
    if epw_source.get("noise", False):
        epw_source.setdefault("noise_amplitude", 1e-10)
        if epw_source.get("noise_seed") is None:
            epw_source["noise_seed"] = int(np.random.randint(2**20))

    # Ion-acoustic waves: resolve defaults before parameter logging. The MATLAB
    # split step is a symplectic-Euler update of (div u_i, delta n_i/n0), whose
    # acoustic branch is stable for omega_max * dt < 2.
    iaw = cfg["terms"].get("iaw", {})
    if iaw.get("active", False):
        from adept._lpse2d.datamodel import IAWModel

        iaw = {**iaw, **IAWModel(**iaw).model_dump()}
        if iaw["boundary"] is None:
            iaw["boundary"] = dict(cfg["terms"]["epw"]["boundary"])
        if iaw["damping"]["collisions"] < 0.0:
            raise ValueError("terms.iaw.damping.collisions must be non-negative")
        if iaw["damping"]["landau"] < 0.0:
            raise ValueError("terms.iaw.damping.landau must be non-negative")
        max_dn = iaw["max_density_perturbation"]
        if max_dn is not None and max_dn <= 0.0:
            raise ValueError("terms.iaw.max_density_perturbation must be positive or null")

        inv_dy_sq = 0.0 if cfg_grid["ny"] == 1 else 1.0 / cfg_grid["dy"] ** 2
        omega_max = 2.0 * cfg["units"]["derived"]["cs"] * np.sqrt(1.0 / cfg_grid["dx"] ** 2 + inv_dy_sq)
        if omega_max * cfg_grid["dt"] >= 2.0:
            raise ValueError(
                "The ion-acoustic update is unstable: omega_iaw,max * grid.dt must be < 2 "
                f"(got {omega_max * cfg_grid['dt']:.3g}). Reduce grid.dt or increase grid.dx."
            )
        cfg["terms"]["iaw"] = iaw
        print(
            "IAWs are on -- evolving density and velocity divergence with "
            f"omega_iaw,max * dt = {omega_max * cfg_grid['dt']:.3g}"
        )

    # HPE (Follett-style test-particle Landau damping): resolve defaults, convert
    # units, and derive the substep count here so everything lands in MLflow params
    hpe = cfg["terms"].get("hpe", {})
    if hpe.get("active", False):
        if cfg["terms"]["epw"]["source"].get("tpd", False):
            raise NotImplementedError(
                "terms.hpe cannot be combined with TPD yet: TPD requires transverse modes, "
                "while the current particle tracker evolves only (x, p_x) in a ny == 1 box"
            )
        if cfg_grid["ny"] != 1:
            raise NotImplementedError("terms.hpe requires a quasi-1D box (ny == 1); shrink the y extent")
        if not cfg["terms"]["epw"]["damping"].get("landau", True):
            raise ValueError("terms.hpe requires terms.epw.damping.landau: true (it replaces the static rate)")
        # defaults and type coercion come from the datamodel's HPEModel so they are
        # defined in exactly one place; unknown keys are passed through untouched
        from adept._lpse2d.datamodel import HPEModel

        hpe = {**hpe, **HPEModel(**hpe).model_dump()}
        if hpe["omega_res"] not in ("bohm_gross", "wp0"):
            raise ValueError("terms.hpe.omega_res must be 'bohm_gross' or 'wp0'")
        hpe["tau_damping_ps"] = _Q(hpe["tau_damping"]).to("ps").value
        hpe["t_start_ps"] = _Q(hpe["t_start"]).to("ps").value
        wp0 = cfg["units"]["derived"]["wp0"]
        hpe["substeps"] = int(np.ceil(wp0 * cfg_grid["dt"] / float(hpe["substep_courant"])))
        cfg["terms"]["hpe"] = hpe
        print(
            f"HPE is on -- {hpe['n_particles']} tail particles (|v| > {hpe['v_min']} vte), "
            f"{hpe['substeps']} particle substeps per EPW step"
        )

    pump_depletion = cfg["terms"].get("light", {}).get("pump_depletion", False)
    source_terms = cfg["terms"]["epw"]["source"]
    srs_on = bool(source_terms.get("srs", False))
    tpd_on = bool(source_terms.get("tpd", False))
    if pump_depletion:
        if not (srs_on or tpd_on):
            raise ValueError("terms.light.pump_depletion requires at least one of terms.epw.source.srs/tpd")
        if "E0" not in cfg["drivers"]:
            raise ValueError(
                "terms.light.pump_depletion requires drivers.E0 (the evolved pump is launched "
                "by a boundary injector built from the E0 driver parameters)"
            )
        if cfg["drivers"].get("E0", {}).get("speckle", {}).get("enabled", False):
            raise ValueError("terms.light.pump_depletion does not support drivers.E0.speckle yet")
        if cfg["terms"]["epw"]["boundary"]["x"] != "absorbing":
            raise ValueError(
                "terms.light.pump_depletion requires terms.epw.boundary.x: absorbing "
                "(the pump is launched by a boundary injector and must exit the box)"
            )

    if srs_on:
        derived = cfg["units"]["derived"]

        # The SRS source filter only passes wavenumbers up to the local Raman light
        # wavenumber k1(n_min). If the box's minimum density reaches the w1 critical
        # density, that band is empty and E1_filter (epw.py) silently zeroes every
        # mode of the source -- the run completes with reflectivity ~0 and reads as
        # "below threshold". The seeded path already dies with a clear error at the
        # injector; fail the noise-seeded path just as loudly here.
        if cfg["density"]["basis"] == "uniform":
            n_box_min = float(cfg["density"].get("val", 1.0))
        elif "min" in cfg["density"]:
            n_box_min = float(cfg["density"]["min"])
        else:
            n_box_min = None
        n_crit_w1 = (derived["w1"] / derived["w0"]) ** 2
        if n_box_min is not None and n_box_min >= n_crit_w1:
            raise ValueError(
                f"terms.epw.source.srs is on but the minimum box density {n_box_min:.3f} nc is at or "
                f"above the Raman critical density {n_crit_w1:.3f} nc, so the scattered light is "
                "evanescent everywhere and the SRS source is filtered to zero. Lower the density "
                "(or the envelope density) if you want SRS."
            )

    if srs_on or pump_depletion:
        derived = cfg["units"]["derived"]

        # The detuning term's operator norm is set by the density endpoint farthest
        # from each evolved carrier's critical density, not the largest density.
        if cfg["density"]["basis"] == "uniform":
            n_endpoints = [float(cfg["density"].get("val", 1.0))]
        else:
            n_endpoints = [float(cfg["density"][k]) for k in ("min", "max") if k in cfg["density"]] or [1.0]

        def _worst_detuning_sq(w_carrier: float) -> float:
            return max(abs(w_carrier**2 - derived["w0"] ** 2 * n) for n in n_endpoints)

        dt_limits = []
        evolved_carriers = []
        if srs_on:
            dt_limits.append(
                1.0
                / (
                    2.0 * derived["c"] ** 2 / (cfg_grid["dx"] ** 2 * derived["w1"])
                    + _worst_detuning_sq(derived["w1"]) / (4.0 * derived["w1"])
                )
            )
            evolved_carriers.append("Raman")
        if pump_depletion:
            dt_limits.append(
                1.0
                / (
                    2.0 * derived["c"] ** 2 / (cfg_grid["dx"] ** 2 * derived["w0"])
                    + _worst_detuning_sq(derived["w0"]) / (4.0 * derived["w0"])
                )
            )
            evolved_carriers.append("pump")
        dt_max = min(dt_limits)
        if "light_substeps" in cfg_grid:
            n_sub = int(cfg_grid["light_substeps"])
            if cfg_grid["dt"] / n_sub > dt_max:
                raise ValueError(
                    f"grid.light_substeps = {n_sub} gives a light step of {cfg_grid['dt'] / n_sub:.2e} ps "
                    f"which exceeds the dynamic-light stability limit of {dt_max:.2e} ps"
                )
        else:
            n_sub = int(np.ceil(cfg_grid["dt"] / (0.9 * dt_max)))
        cfg_grid["light_substeps"] = n_sub
        carriers = " + ".join(evolved_carriers)
        print(f"{carriers} light is sub-cycled {n_sub}x per EPW step (dt_light limit {dt_max:.2e} ps)")

    # change driver parameters to the right units
    for k in cfg["drivers"].keys():
        cfg["drivers"][k]["derived"] = {}
        if k == "E1":
            # Raman seed injector -- different parameter set than the envelope drivers
            c_cgs = 2.99792458e10
            seed_intensity = _Q(cfg["drivers"][k]["intensity"]).to("W/cm^2").value
            # the injector must sit clear of the absorbing boundary, whose tanh skirt
            # (rise = boundary_width / 5) extends past xmax - boundary_width into the box
            boundary_width = _Q(cfg["grid"]["boundary_width"]).to("um").value
            min_offset = 1.6 * boundary_width
            if "offset" in cfg["drivers"][k]:
                offset = _Q(cfg["drivers"][k]["offset"]).to("um").value
                if offset < min_offset:
                    print(
                        f"WARNING: drivers.E1.offset = {offset}um is inside the absorbing-boundary skirt "
                        f"(< 1.6 * boundary_width = {min_offset}um); the seed will be damped at the source"
                    )
            else:
                offset = min_offset
            cfg["drivers"][k]["derived"] = {
                "amplitude": np.sqrt(8 * np.pi * seed_intensity * 1e7 / c_cgs) / cfg["units"]["derived"]["fieldScale"],
                "delta_omega": float(cfg["drivers"][k].get("delta_omega", 0.0)),
                "turn_on_time": _Q(cfg["drivers"][k].get("turn_on_time", "10fs")).to("ps").value,
                "offset": offset,
                "yw": _Q(cfg["drivers"][k]["yw"]).to("um").value if "yw" in cfg["drivers"][k] else 0.0,
            }
            continue
        if k == "E0" and pump_depletion:
            # boundary-injector parameters for the evolved pump: the injector sits at
            # xmin + offset (default 2*boundary_width, clear of the absorber skirt)
            boundary_width = _Q(cfg["grid"]["boundary_width"]).to("um").value
            if "offset" in cfg["drivers"][k]:
                cfg["drivers"][k]["derived"]["offset"] = _Q(cfg["drivers"][k]["offset"]).to("um").value
            else:
                cfg["drivers"][k]["derived"]["offset"] = 2.0 * boundary_width
            cfg["drivers"][k]["derived"]["turn_on_time"] = (
                _Q(cfg["drivers"][k].get("turn_on_time", "10fs")).to("ps").value
            )
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


def _pump_k_support(cfg: dict) -> tuple[float, float]:
    """
    Half-widths of the pump's support in k-space, returned as ``(kx, ky)`` in 1/um.

    The TPD and SRS source terms are products of the pump with a plasma-wave field, so in k-space
    the plasma-wave spectrum is *translated* by the pump wavenumber rather than convolved against a
    broad kernel. These half-widths are how far that translation can reach, which is what sets the
    part of the band that has to be left empty for the product not to wrap around Nyquist.

    ``laser.Light.laser_update`` builds each color as a plane wave at ``k0(delta_omega)`` along x and
    applies the speckle envelope as a function of y only, so the kx support is a single wavenumber
    and the ky support is zero unless a speckle profile is attached -- in which case the beamlets
    span the aperture and the numerical aperture bounds ky.

    :param cfg: the full config, after ``write_units`` and ``get_derived_quantities`` have run
    :return: ``(kx, ky)`` half-widths in 1/um; ``(0, 0)`` when there is no E0 driver
    """
    if "E0" not in cfg["drivers"]:
        return 0.0, 0.0

    derived = cfg["units"]["derived"]
    w0, c, wp0 = derived["w0"], derived["c"], derived["wp0"]

    # the largest k0 over the colors -- mirrors the per-color k0 in laser.py
    delta_omega_max = cfg["drivers"]["E0"].get("delta_omega_max", 0.0)
    k0 = w0 / c * np.sqrt((1.0 + delta_omega_max) ** 2 - (wp0 / w0) ** 2)

    speckle_cfg = cfg["drivers"]["E0"].get("speckle", {})
    if speckle_cfg.get("enabled", False):
        focal_length_m = _Q(speckle_cfg["focal_length"]).to("m").value
        beam_aperture_m = [_Q(a).to("m").value for a in speckle_cfg["beam_aperture"]]
        numerical_aperture = max(beam_aperture_m) / (2.0 * focal_length_m)
        ky = k0 * numerical_aperture
    else:
        ky = 0.0

    return float(k0), float(ky)


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

    def absorbing_boundary(boundary):
        if boundary["x"] == "absorbing":
            left = cfg_grid["xmin"] + boundary_width
            right = cfg_grid["xmax"] - boundary_width
            envelope_x = get_envelope(rise, rise, left, right, cfg_grid["x"])[:, None]
        else:
            envelope_x = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

        if boundary["y"] == "absorbing":
            left = cfg_grid["ymin"] + boundary_width
            right = cfg_grid["ymax"] - boundary_width
            envelope_y = get_envelope(rise, rise, left, right, cfg_grid["y"])[None, :]
        else:
            envelope_y = np.ones((cfg_grid["nx"], cfg_grid["ny"]))

        return np.exp(-float(cfg_grid["boundary_abs_coeff"]) * cfg_grid["dt"] * (1.0 - envelope_x * envelope_y))

    cfg_grid["absorbing_boundaries"] = absorbing_boundary(cfg["terms"]["epw"]["boundary"])
    iaw = cfg["terms"].get("iaw", {})
    if iaw.get("active", False):
        cfg_grid["iaw_absorbing_boundaries"] = absorbing_boundary(iaw["boundary"])

    cfg_grid["zero_mask"] = (
        np.where(np.sqrt(cfg_grid["kx"][:, None] ** 2 + cfg_grid["ky"][None, :] ** 2) == 0, 0, 1)
        if cfg["terms"]["zero_mask"]
        else 1
    )

    k_mag = np.sqrt(cfg_grid["kx"][:, None] ** 2 + cfg_grid["ky"][None, :] ** 2)
    kmax = cfg_grid["kx"].max()
    cutoff = cfg_grid["low_pass_filter"] * kmax
    taper_fraction = cfg_grid.get("low_pass_taper_fraction", 0.0)

    if cutoff <= 0:
        cfg_grid["low_pass_filter_grid"] = np.ones_like(k_mag)
    elif taper_fraction <= 0.0:
        cfg_grid["low_pass_filter_grid"] = np.where(k_mag < cutoff, 1.0, 0.0)
    else:
        taper_start = cutoff * (1.0 - taper_fraction)
        taper_start = max(taper_start, 0.0)
        filter_grid = np.ones_like(k_mag)
        outside_cutoff = k_mag >= cutoff
        filter_grid[outside_cutoff] = 0.0
        taper_region = (k_mag >= taper_start) & (k_mag < cutoff)
        if cutoff > taper_start:
            xi = (k_mag[taper_region] - taper_start) / (cutoff - taper_start)
            filter_grid[taper_region] = 0.5 * (1.0 + np.cos(np.pi * xi))
        cfg_grid["low_pass_filter_grid"] = filter_grid

    # The isotropic cutoff above is a physics knob -- it is what keeps the retained band inside the
    # range where the asymptotic Landau damping rate in epw.py is still valid. Dealiasing is a
    # separate constraint, and an isotropic circle is the wrong shape for it: the pump translates
    # the spectrum along x only, so the band that has to stay empty is a rectangle, not a disc.
    dealias = cfg_grid.get("dealias", "isotropic")
    if dealias == "shifted-band":
        kx_pump, ky_pump = _pump_k_support(cfg)
        kx_nyquist = float(np.abs(cfg_grid["kx"]).max())
        ky_nyquist = float(np.abs(cfg_grid["ky"]).max())
        kx_limit = kx_nyquist - kx_pump
        ky_limit = ky_nyquist - ky_pump

        if kx_limit <= 0.0 or ky_limit <= 0.0:
            raise ValueError(
                f"The pump support (kx={kx_pump:.2f}, ky={ky_pump:.2f} 1/um) does not leave any room "
                f"inside the grid's Nyquist wavenumber (kx={kx_nyquist:.2f}, ky={ky_nyquist:.2f} 1/um), "
                "so no choice of band is alias free. Decrease grid.dx."
            )

        alias_free_band = (np.abs(cfg_grid["kx"])[:, None] <= kx_limit) & (np.abs(cfg_grid["ky"])[None, :] <= ky_limit)
        cfg_grid["low_pass_filter_grid"] = cfg_grid["low_pass_filter_grid"] * alias_free_band

        print(
            f"dealias='shifted-band': alias-free band is |kx| <= {kx_limit:.2f}, |ky| <= {ky_limit:.2f} 1/um "
            f"(Nyquist {kx_nyquist:.2f}, {ky_nyquist:.2f}); low_pass_filter caps |k| <= {cutoff:.2f}"
        )
    elif dealias != "isotropic":
        raise ValueError(f"Unknown grid.dealias '{dealias}'. Choose 'isotropic' or 'shifted-band'.")

    retained = float(np.mean(cfg_grid["low_pass_filter_grid"] > 0))
    debye_length = np.sqrt(cfg["units"]["derived"]["vte_sq"]) / cfg["units"]["derived"]["wp0"]
    k_edge = float(np.max(k_mag * (cfg_grid["low_pass_filter_grid"] > 0)))
    print(
        f"dealias='{dealias}' retains {100 * retained:.1f}% of the {cfg_grid['nx']}x{cfg_grid['ny']} k-grid; "
        f"band edge reaches k*lambda_D = {k_edge * debye_length:.2f}"
    )

    # Initialize LASY speckle profile if configured
    if cfg["drivers"].get("E0", {}).get("speckle", {}).get("enabled", False):
        import jax

        from adept._lpse2d.core.speckle import SpeckleProfile

        speckle_cfg = cfg["drivers"]["E0"]["speckle"]

        # Get wavelength in meters from config (laser_wavelength is stored as string like "351nm")
        wavelength_m = _Q(cfg["units"]["laser_wavelength"]).to("m").value

        # Get smoothing type
        smoothing_type = speckle_cfg.get("smoothing_type", "CPP").upper()

        # Get t_max for time-varying methods (GP methods need this)
        # cfg_grid["tmax"] is already in ps at this point
        t_max_seconds = cfg_grid["tmax"] * 1e-12  # ps -> s

        # Get bandwidth (required for SSD/ISI, default small value for RPP/CPP)
        relative_laser_bandwidth = speckle_cfg.get("relative_laser_bandwidth", 1e-10)

        # Parse focal_length and beam_aperture with units
        focal_length_m = _Q(speckle_cfg["focal_length"]).to("m").value
        beam_aperture_m = [_Q(a).to("m").value for a in speckle_cfg["beam_aperture"]]

        cfg["drivers"]["E0"]["speckle_profile"] = SpeckleProfile(
            wavelength=wavelength_m,
            pol=(1, 0),
            focal_length=focal_length_m,
            beam_aperture=beam_aperture_m,
            n_beamlets=speckle_cfg["n_beamlets"],
            temporal_smoothing_type=smoothing_type,
            key=jax.random.PRNGKey(speckle_cfg.get("seed", 42)),
            t_max=t_max_seconds,
            relative_laser_bandwidth=relative_laser_bandwidth,
            ssd_phase_modulation_amplitude=speckle_cfg.get("ssd_phase_modulation_amplitude"),
            ssd_number_color_cycles=speckle_cfg.get("ssd_number_color_cycles"),
            ssd_transverse_bandwidth_distribution=speckle_cfg.get("ssd_transverse_bandwidth_distribution"),
        )

    return cfg_grid


def get_density_profile(cfg: dict) -> Array:
    """
    Helper function for initializing the density profile

    It can be uniform, linear, exponential, tanh, or sine

    :param cfg: Dict
    """
    if cfg["density"]["basis"] == "uniform":
        nprof = cfg["density"].get("val", 1.0) * np.ones((cfg["grid"]["nx"], cfg["grid"]["ny"]))

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

    ny = fields.coords["y (um)"].data.size

    for k, v in fields.items():
        fld_dir = os.path.join(td, "plots", k)
        os.makedirs(fld_dir)

        if ny > 1:
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
    t_skip = int(kfields.coords["t (ps)"].data.size // 6)
    t_skip = t_skip if t_skip > 1 else 1
    tslice = slice(0, -1, t_skip)

    for abs_kmax in [2.5, 1.25]:
        # k_min = -2.5
        # k_max = 2.5
        k_min = -abs_kmax
        k_max = abs_kmax

        ikx_min = np.argmin(np.abs(kfields.coords[r"kx ($kc\omega_0^{-1}$)"].data - k_min))
        ikx_max = np.argmin(np.abs(kfields.coords[r"kx ($kc\omega_0^{-1}$)"].data - k_max))
        iky_min = np.argmin(np.abs(kfields.coords[r"ky ($kc\omega_0^{-1}$)"].data - k_min))
        iky_max = np.argmin(np.abs(kfields.coords[r"ky ($kc\omega_0^{-1}$)"].data - k_max))

        kx_slice = slice(ikx_min, ikx_max)
        ky_slice = slice(iky_min, iky_max)
        n_ky = kfields.coords[r"ky ($kc\omega_0^{-1}$)"].data.size

        for k, v in kfields.items():
            fld_dir = os.path.join(td, "plots", k)
            os.makedirs(fld_dir, exist_ok=True)

            if n_ky == 1:
                np.log10(np.abs(v[tslice, kx_slice, 0])).plot(col="t (ps)", col_wrap=4)
                plt.savefig(os.path.join(fld_dir, f"log_{k}_kx_absmax{abs_kmax}.png"), bbox_inches="tight")
                plt.close()
                continue

            np.abs(v[tslice, kx_slice, ky_slice]).T.plot(col="t (ps)", col_wrap=4)
            plt.savefig(os.path.join(fld_dir, f"{k}_kx_ky_absmax{abs_kmax}.png"), bbox_inches="tight")
            plt.close()

            np.log10(np.abs(v[tslice, kx_slice, ky_slice])).T.plot(col="t (ps)", col_wrap=4)
            plt.savefig(os.path.join(fld_dir, f"log_{k}_kx_ky_absmax{abs_kmax}.png"), bbox_inches="tight")
            plt.close()
            #
            # kx = kfields.coords["kx"].data


def post_process(result, cfg: dict, td: str) -> tuple[xr.Dataset, xr.Dataset]:
    from adept._lpse2d.diagnostics import series_metrics

    os.makedirs(os.path.join(td, "binary"))
    metrics = {}
    t0 = time.time()
    kfields, fields = make_field_xarrays(cfg, result.ts["fields"], result.ys["fields"], td)
    series = make_series_xarrays(cfg, result.ts["default"], result.ys["default"], td)
    metrics["write_time"] = time.time() - t0
    os.makedirs(os.path.join(td, "plots"))

    # OSIRIS-comparable scalars (laser budget, EPW growth fit, electron energy)
    metrics.update(series_metrics(series, cfg))

    t0 = time.time()
    plot_series(series, td)
    plot_srs_diagnostics(series, metrics, cfg, td)
    plot_fields(fields, td)
    plot_kt(kfields, td)
    metrics["plot_time"] = time.time() - t0

    return {"k": kfields, "x": fields, "series": series, "metrics": metrics}


def plot_srs_diagnostics(series, metrics, cfg, td):
    """Composite diagnostic plots mirroring the OSIRIS scan2 figures: the laser
    budget channels vs time, the EPW energy with the growth-fit window shaded, and
    the cumulative electron energy (integrated EPW dissipation)."""
    t = np.asarray(series["t (ps)"].values, dtype=float)

    if "incident_flux" in series:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        for a in ax:
            for key, label in [
                ("incident_flux", "incident"),
                ("reflected_flux", "reflected"),
                ("transmitted_flux", "transmitted"),
                ("backrefl_flux", "back-reflected"),
            ]:
                a.plot(t, np.asarray(series[key].values, dtype=float), label=label)
            a.set_xlabel("t (ps)")
            a.set_ylabel("flux / I0")
        ax[1].set_yscale("log")
        ax[0].legend(fontsize=8)
        fig.savefig(os.path.join(td, "plots", "laser_budget_vs_t.png"), bbox_inches="tight")
        plt.close(fig)

    if "epw_energy" in series:
        w0 = cfg["units"]["derived"]["w0"]
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        ax.semilogy(t, np.asarray(series["epw_energy"].values, dtype=float))
        if metrics.get("epw_growth_measurable"):
            # fit window is stored in 1/w0 (OSIRIS code time); convert back to ps
            ax.axvspan(metrics["epw_growth_fit_tstart"] / w0, metrics["epw_growth_fit_tend"] / w0, alpha=0.2)
            ax.set_title(f"gamma/w0 = {metrics['epw_growth_rate']:.2e}, r2 = {metrics['epw_growth_rate_r2']:.3f}")
        ax.set_xlabel("t (ps)")
        ax.set_ylabel("W_epw (OSIRIS units)")
        fig.savefig(os.path.join(td, "plots", "epw_energy_fit.png"), bbox_inches="tight")
        plt.close(fig)

    if "epw_dissipation" in series:
        dissip = np.asarray(series["epw_dissipation"].values, dtype=float)
        electron_energy = np.concatenate([[0.0], np.cumsum(0.5 * (dissip[1:] + dissip[:-1]) * np.diff(t))])
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        ax.plot(t, electron_energy)
        ax.set_xlabel("t (ps)")
        ax.set_ylabel("cumulative electron energy (OSIRIS units)")
        fig.savefig(os.path.join(td, "plots", "electron_energy_vs_t.png"), bbox_inches="tight")
        plt.close(fig)

    if "hpe_hist" in series:
        # tail distribution vs time: flattening at the resonant v_phi and a growing
        # high-energy shoulder are the HPE signatures
        hist = np.asarray(series["hpe_hist"].values, dtype=float)
        v = np.asarray(series["v (c)"].values, dtype=float)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        with np.errstate(divide="ignore"):
            log_h = np.log10(np.where(hist > 0, hist, np.nan))
        pcm = ax[0].pcolormesh(t, v, log_h.T, shading="auto")
        fig.colorbar(pcm, ax=ax[0], label="log10 f(v)")
        ax[0].set_xlabel("t (ps)")
        ax[0].set_ylabel("v (c)")
        for it in np.linspace(0, len(t) - 1, 5).astype(int):
            ax[1].semilogy(v, np.where(hist[it] > 0, hist[it], np.nan), label=f"t = {t[it]:.1f} ps")
        ax[1].set_xlabel("v (c)")
        ax[1].set_ylabel("f(v)")
        ax[1].legend(fontsize=7)
        fig.savefig(os.path.join(td, "plots", "hpe_distribution.png"), bbox_inches="tight")
        plt.close(fig)

    if "hpe_gamma_ratio_min" in series:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        ax.plot(t, np.asarray(series["hpe_gamma_ratio_min"].values, dtype=float))
        ax.set_xlabel("t (ps)")
        ax.set_ylabel("min gamma_HPE / gamma_analytic")
        ax.set_ylim(bottom=0)
        fig.savefig(os.path.join(td, "plots", "hpe_damping_reduction_vs_t.png"), bbox_inches="tight")
        plt.close(fig)


def plot_series(series, td):
    for k in series.keys():
        if series[k].ndim != 1:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        series[k].plot(ax=ax[0])
        series[k].plot(ax=ax[1])
        ax[1].set_yscale("log")
        fig.savefig(os.path.join(td, "plots", f"{k}_vs_t.png"), bbox_inches="tight")
        fig.savefig(os.path.join(td, "plots", f"{k}_vs_t.pdf"), bbox_inches="tight")
        plt.close()


def make_series_xarrays(cfg, this_t, state, td):
    data = {}
    for k, v in state.items():
        v = np.asarray(v)
        if k == "hpe_hist":
            # (nt, nv) velocity histogram from the HPE module; the axis is v/c
            # (a slash in a coord name is illegal in netCDF)
            hpe = cfg["terms"]["hpe"]
            edges = np.linspace(-hpe["v_max"], hpe["v_max"], hpe["nv"] + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            data[k] = xr.DataArray(v, coords=(("t (ps)", this_t), ("v (c)", centers)))
        else:
            data[k] = xr.DataArray(v, coords=(("t (ps)", this_t),))
    series_xr = xr.Dataset(data)
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

    # the state is stored as a float view of a complex array; pick the matching complex dtype
    if state["epw"].dtype in (np.float64, np.complex128):
        _complex = np.complex128
    else:
        _complex = np.complex64

    phi_k_np = np.array(state["epw"]).view(_complex)
    phi_vs_t = np.fft.ifft2(np.array(state["epw"]).view(_complex), axes=(1, 2))
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
    e0x = xr.DataArray(np.array(state["E0"]).view(_complex)[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e0y = xr.DataArray(np.array(state["E0"]).view(_complex)[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))
    e1x = xr.DataArray(np.array(state["E1"]).view(_complex)[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e1y = xr.DataArray(np.array(state["E1"]).view(_complex)[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))

    from scipy import interpolate

    if ny == 1:
        # Quasi-1D: RegularGridInterpolator cannot take a single-node y axis (and the
        # save row lies off it -> fill_value=0). Interpolate the density in x only.
        density_1d = np.asarray(cfg["grid"]["background_density"])[:, 0]
        density_interpolator = interpolate.interp1d(
            np.asarray(cfg["grid"]["x"]), density_1d, bounds_error=False, fill_value="extrapolate"
        )
        density_on_save_grid = density_interpolator(np.asarray(xax)).reshape((nx, ny))
    else:
        density_interpolator = interpolate.RegularGridInterpolator(
            (cfg["grid"]["x"], cfg["grid"]["y"]),
            cfg["grid"]["background_density"],
            bounds_error=False,
            fill_value=0.0,
        )
        grid_x, grid_y = np.meshgrid(xax, yax, indexing="ij")
        points = np.array([grid_x.flatten(), grid_y.flatten()]).T
        density_on_save_grid = density_interpolator(points).reshape((nx, ny))

    background_density = xr.DataArray(
        np.repeat(density_on_save_grid[None, ...], repeats=len(this_t), axis=0),
        coords=(tax_tuple, xax_tuple, yax_tuple),
    )

    kfield_data = {"phi": phi_k, "ex": ex_k, "ey": ey_k}
    field_data = {
        "phi": phi_x,
        "ex": ex,
        "ey": ey,
        "e0_x": e0x,
        "e0_y": e0y,
        "e1_x": e1x,
        "e1_y": e1y,
        "background_density": background_density,
    }
    if "iaw_density" in state:
        iaw_density_np = np.asarray(state["iaw_density"])
        iaw_velocity_np = np.asarray(state["iaw_velocity_divergence"])
        field_data["iaw_density"] = xr.DataArray(iaw_density_np, coords=(tax_tuple, xax_tuple, yax_tuple))
        field_data["iaw_velocity_divergence"] = xr.DataArray(iaw_velocity_np, coords=(tax_tuple, xax_tuple, yax_tuple))
        k_coords = (
            tax_tuple,
            (r"kx ($kc\omega_0^{-1}$)", shift_kx),
            (r"ky ($kc\omega_0^{-1}$)", shift_ky),
        )
        kfield_data["iaw_density"] = xr.DataArray(
            np.fft.fftshift(np.fft.fft2(iaw_density_np, axes=(1, 2)), axes=(1, 2)), coords=k_coords
        )
        kfield_data["iaw_velocity_divergence"] = xr.DataArray(
            np.fft.fftshift(np.fft.fft2(iaw_velocity_np, axes=(1, 2)), axes=(1, 2)), coords=k_coords
        )

    kfields = xr.Dataset(kfield_data)
    fields = xr.Dataset(field_data)
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
        nx = cfg["grid"]["nx"]
        # nx = int((xmax - xmin) / dx)
        cfg["save"]["fields"]["x"]["dx"] = dx
        cfg["save"]["fields"]["x"]["ax"] = jnp.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
        cfg["save"]["fields"]["kx"] = np.fft.fftfreq(nx, d=dx / 2.0 / np.pi)

        if "y" in cfg["save"]["fields"]:
            ymin = cfg["grid"]["ymin"]
            ymax = cfg["grid"]["ymax"]
            dy = _Q(cfg["save"]["fields"]["y"]["dy"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
            ny = cfg["grid"]["ny"]
            cfg["save"]["fields"]["y"]["dy"] = dy
            cfg["save"]["fields"]["y"]["ax"] = jnp.linspace(ymin + dy / 2.0, ymax - dy / 2.0, ny)
            cfg["save"]["fields"]["ky"] = np.fft.fftfreq(ny, d=dy / 2.0 / np.pi)
        else:
            raise NotImplementedError("Must specify y in save")

        xq, yq = jnp.meshgrid(cfg["save"]["fields"]["x"]["ax"], cfg["save"]["fields"]["y"]["ax"], indexing="ij")

        if ny == 1:
            # Quasi-1D (single transverse cell): interpax.interp2d needs >=2 nodes per
            # axis and returns NaN off the lone y-node, so interpolate in x only and
            # keep the single y-row. save_func reshapes the flat (nx,) result to (nx, 1).
            x_save = cfg["save"]["fields"]["x"]["ax"]
            x_src = cfg["grid"]["x"]

            def interpolator(f):
                return interpax.interp1d(x_save, x_src, jnp.reshape(f, (-1,)), method="linear")
        else:
            interpolator = partial(
                interpax.interp2d,
                xq=jnp.reshape(xq, (nx * ny), order="F"),
                yq=jnp.reshape(yq, (nx * ny), order="F"),
                x=cfg["grid"]["x"],
                y=cfg["grid"]["y"],
                method="linear",
            )

        def save_func(t, y, args):
            from adept._lpse2d.core.hpe import PARTICLE_KEYS

            save_y = {}
            for k, v in y.items():
                if k in PARTICLE_KEYS:
                    # particle arrays are (Np,) and gamma_L/epw_hist live in k/v space --
                    # none of them fit the spatial interpolator; the histogram and the
                    # damping-reduction scalars are saved through the default series
                    continue
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

        def save_func(t, y, args):
            from adept._lpse2d.core.hpe import PARTICLE_KEYS

            return {k: v for k, v in y.items() if k not in PARTICLE_KEYS}

    cfg["save"]["fields"]["func"] = save_func

    cfg["save"]["default"] = get_default_save_func(cfg)

    return cfg


def get_default_save_func(cfg):
    from adept._lpse2d.core.epw import landau_damping_rate

    srs_on = cfg["terms"]["epw"]["source"].get("srs", False)
    pump_evolved = cfg["terms"].get("light", {}).get("pump_depletion", False)
    iaw_on = cfg["terms"].get("iaw", {}).get("active", False)
    derived = cfg["units"]["derived"]
    dt = cfg["grid"]["dt"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx, ky = cfg["grid"]["kx"], cfg["grid"]["ky"]

    # OSIRIS-normalized EPW energy: W = 1/2 * dx * sum_x <e^2>_cycle with fields in
    # me*c*w0/e and lengths in c/w0 (osiris_lpi/epw_growth.py convention). The complex
    # envelope carries <e^2>_cycle = |E|^2/2, hence the 0.25. The transverse mean makes
    # the 2D case a per-unit-y version of the same quantity; for ny=1 it is identical
    # to the OSIRIS 1D reduction.
    epw_energy_prefactor = 0.25 * cfg["grid"]["dx"] * derived["x_norm"] * derived["e_norm"] ** 2

    # Per-step dissipated EPW energy, evaluated with the *same* rates the solver
    # applies (phi_k *= exp(-(gamma_landau + nu_coll)*dt) => energy factor exp(-2(..)dt)).
    # Parseval: sum_x <.>_y |E|^2 = (1/(nx*ny^2)) * sum_k k^2 |phi_k|^2.
    k_sq = np.array(kx[:, None] ** 2 + ky[None, :] ** 2)
    zero_mask = np.where(k_sq > 0, 1.0, 0.0)
    if cfg["terms"]["epw"]["damping"].get("landau", True):
        gamma_total = np.array(
            landau_damping_rate(jnp.array(k_sq), derived["wp0"], derived["vte_sq"], jnp.array(zero_mask))
        )
    else:
        gamma_total = np.zeros_like(k_sq)
    gamma_total = gamma_total + derived.get("nu_coll", 0.0) * zero_mask
    energy_loss_factor = (1.0 - np.exp(-2.0 * gamma_total * dt)) / dt  # 1/ps, per k mode
    boundary_sq_loss = (1.0 - np.array(cfg["grid"]["absorbing_boundaries"]) ** 2) / dt  # 1/ps, per cell

    # Total (electric + kinetic + thermal) EPW energy per unit of the electric energy
    # that epw_energy counts. For the warm-fluid EPW, W_total/W_E = d(w*eps)/dw =
    # 1 + wp^2/w^2 + 9 k^2 vte^2 wp^2 / w^4 evaluated at the envelope carrier w = wp0;
    # the solver's detuning relation 3 k^2 vte^2 = wp0^2 - wp^2 with wp^2 = wp0^2 * n/n_env
    # collapses it to the density-only factor 2*(2 - n/n_env). At n = n_env this is the
    # familiar 2 (electric = kinetic); on the shipped ramps it reaches 2.56 at the
    # low-density end, which the previously hard-coded 2 understated by ~28%.
    energy_total_factor = 2.0 * (2.0 - np.array(cfg["grid"]["background_density"]) / cfg["units"]["envelope density"])

    # HPE: the damping is dynamic (y["gamma_L"]), so the dissipation diagnostic must
    # use the state's rate; also emit the hot-electron scalars and the histogram
    hpe_on = cfg["terms"].get("hpe", {}).get("active", False)
    if hpe_on:
        from adept._lpse2d.core.hpe import resonance_arrays

        hpe_arrays = resonance_arrays(cfg)
        n_p = int(cfg["terms"]["hpe"]["n_particles"])
        w_tail = hpe_arrays["f_tail_frac"] / n_p  # fraction of all electrons per particle
        c_light = derived["c"]
        nu_coll_arr = derived.get("nu_coll", 0.0) * zero_mask
        gamma_an_1d = np.array(hpe_arrays["gamma_analytic"][:, 0])
        # damping-reduction band: resonant modes where the analytic rate is non-negligible
        ratio_band = np.array(hpe_arrays["mask_res"]) & (gamma_an_1d > 1.0e-4 * derived["wp0"])
        have_ratio_band = bool(np.any(ratio_band))
        gamma_an_safe = np.where(ratio_band, gamma_an_1d, 1.0)

    if srs_on or pump_evolved:
        # Flux probes for the dynamic-light laser budget. They are also retained for
        # prescribed-pump SRS so the legacy one-way Raman reflectivity remains available.
        boundary_width = _Q(cfg["grid"]["boundary_width"]).to("um").value
        x_probe = cfg["grid"]["xmin"] + 1.6 * boundary_width
        ix_probe = int(np.argmin(np.abs(np.array(cfg["grid"]["x"]) - x_probe)))
        w0 = derived["w0"]
        w1 = derived["w1"]
        n_probe = float(np.mean(np.array(cfg["grid"]["background_density"])[ix_probe, :]))
        sqrt_eps1 = np.sqrt(max(1.0 - n_probe * w0**2 / w1**2, 0.0)) if srs_on else 0.0
        E0_source_sq = derived["E0_source"] ** 2

        # Flux probes for the OSIRIS-style laser budget. F_j = c^2/(w dx) * Im(E_j* E_j+1)
        # is the exactly-conserved flux of the FD Schroedinger operator (equals
        # c*sqrt(eps)|E|^2*sinc(k dx) for a plane wave, i.e. the grid's own dispersion is
        # accounted for). Probes sit at probe_offset (default 2*boundary_width, clear of
        # the absorber skirt whose transmission at 1.6*bw is only ~0.91).
        if "probe_offset" in cfg["grid"]:
            probe_offset = _Q(cfg["grid"]["probe_offset"]).to("um").value
        else:
            probe_offset = 2.0 * boundary_width
        x_grid = np.array(cfg["grid"]["x"])
        ix_left = int(np.argmin(np.abs(x_grid - (cfg["grid"]["xmin"] + probe_offset))))
        ix_right = int(np.argmin(np.abs(x_grid - (cfg["grid"]["xmax"] - probe_offset))))
        # with an evolved pump, the incident probe must sit downstream (+x) of the pump
        # injector rows or it reads the near-field of the two-point source
        ix_left_e0 = ix_left
        if pump_evolved:
            pump_offset = cfg["drivers"]["E0"]["derived"]["offset"]
            ix_inject = int(np.argmin(np.abs(x_grid - (cfg["grid"]["xmin"] + pump_offset))))
            ix_left_e0 = max(ix_left, ix_inject + 4)
        flux_coeff_w0 = derived["c"] ** 2 / (derived["w0"] * cfg["grid"]["dx"])
        flux_coeff_w1 = derived["c"] ** 2 / (derived["w1"] * cfg["grid"]["dx"])
        I0_code = derived["I0_code"]

        def flux_correction(w, ix):
            # The discrete two-point flux of the FD mode at local wavenumber k is
            # |E|^2 * v_g,discrete with v_g,disc = (c^2/w) sin(k_grid dx)/dx, where
            # k_grid satisfies the FD dispersion (2/dx^2)(1 - cos k_grid dx) = k^2.
            # Dividing by sin(k_grid dx)/(k dx) converts it to the physical flux
            # |E|^2 * c * sqrt(eps). Evanescent probes get 1 (their flux is ~0 anyway).
            n_loc = float(np.mean(np.array(cfg["grid"]["background_density"])[ix, :]))
            eps = 1.0 - n_loc * w0**2 / w**2
            if eps <= 0:
                return 1.0
            k_dx = w / derived["c"] * np.sqrt(eps) * cfg["grid"]["dx"]
            cos_kg = 1.0 - k_dx**2 / 2.0
            if cos_kg <= -1.0:
                return 1.0
            sin_kg = float(np.sqrt(1.0 - cos_kg**2))
            return float(sin_kg / k_dx)

        corr_e0_left = flux_correction(w0, ix_left_e0)
        corr_e0_right = flux_correction(w0, ix_right)
        corr_e1_left = flux_correction(w1, ix_left)
        corr_e1_right = flux_correction(w1, ix_right)

        def discrete_flux(E, ix, coeff):
            # sum over polarization components, mean over y
            cross = jnp.sum(jnp.conj(E[ix, :, :]) * E[ix + 1, :, :], axis=-1)
            return coeff * jnp.mean(jnp.imag(cross))

    def save_func(t, y, args):
        phi_k = y["epw"].view(jnp.complex128)
        ex = -1j * kx[:, None] * phi_k
        ey = -1j * ky[None, :] * phi_k
        ex = jnp.fft.ifft2(ex)
        ey = jnp.fft.ifft2(ey)
        e_sq = jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2

        out = {"e_sq": jnp.sum(e_sq * cfg["grid"]["dx"] * cfg["grid"]["dy"]), "max_phi": jnp.max(jnp.abs(phi_k))}

        out["epw_energy"] = epw_energy_prefactor * jnp.sum(jnp.mean(e_sq, axis=1))
        # dissipation/boundary channels are TOTAL EPW-energy rates: epw_energy counts
        # only the electric part (the OSIRIS field-only convention), so the energy
        # actually handed to electrons -- and the budget sink -- carries the local
        # total-to-electric factor energy_total_factor = 2*(2 - n/n_env)
        if hpe_on:
            # the applied rate is dynamic: read it from the state
            gamma_dyn = y["gamma_L"] + nu_coll_arr
            loss_factor = (1.0 - jnp.exp(-2.0 * gamma_dyn * dt)) / dt
        else:
            loss_factor = energy_loss_factor
        # the per-k loss rate does not commute with the x-dependent energy factor, so
        # build a local electric-energy loss density from the sqrt-weighted fields --
        # its box integral equals the k-space total exactly (Parseval) and reduces to
        # the previous 2x k-space sum on a uniform n = n_env box
        sqrt_loss = jnp.sqrt(loss_factor)
        ex_loss = jnp.fft.ifft2(-1j * kx[:, None] * phi_k * sqrt_loss)
        ey_loss = jnp.fft.ifft2(-1j * ky[None, :] * phi_k * sqrt_loss)
        loss_density = jnp.abs(ex_loss) ** 2 + jnp.abs(ey_loss) ** 2
        out["epw_dissipation"] = epw_energy_prefactor * jnp.sum(jnp.mean(energy_total_factor * loss_density, axis=1))
        out["epw_boundary_loss"] = epw_energy_prefactor * jnp.sum(
            jnp.mean(energy_total_factor * e_sq * boundary_sq_loss, axis=1)
        )

        if iaw_on:
            iaw_density = y["iaw_density"]
            out["iaw_density_sq"] = jnp.mean(iaw_density**2)
            out["iaw_density_abs_max"] = jnp.max(jnp.abs(iaw_density))

        if hpe_on:
            u = y["u_e"]
            gamma_rel = jnp.sqrt(1.0 + (u / c_light) ** 2)
            ke_kev = 510.999 * (gamma_rel - 1.0)
            out["fhot_50keV"] = w_tail * jnp.sum(ke_kev > 50.0)
            out["fhot_100keV"] = w_tail * jnp.sum(ke_kev > 100.0)
            out["hpe_mean_energy_keV"] = jnp.mean(ke_kev)
            out["hpe_hist"] = y["epw_hist"]
            if have_ratio_band:
                ratio = y["gamma_L"][:, 0] / gamma_an_safe
                # inflation-o-meters: worst-case reduction across the resonant band
                # (noisy at low n_particles -- one clamped mode reads 0), and the
                # reduction at the band mode carrying the most EPW energy (robust,
                # and the physically relevant one)
                out["hpe_gamma_ratio_min"] = jnp.min(jnp.where(ratio_band, ratio, jnp.inf))
                phi_amp = jnp.where(ratio_band, jnp.abs(phi_k[:, 0]), 0.0)
                # before the EPW has any energy in the band (e.g. the zero-initialized
                # first steps) argmax lands on index 0 where the ratio is meaningless;
                # emit NaN so the reduction metrics (nanmin / windowed means) skip it
                out["hpe_gamma_ratio_kpeak"] = jnp.where(jnp.any(phi_amp > 0.0), ratio[jnp.argmax(phi_amp)], jnp.nan)

        if srs_on or pump_evolved:
            e0 = y["E0"].view(jnp.complex128)
            out["incident_flux"] = discrete_flux(e0, ix_left_e0, flux_coeff_w0) / corr_e0_left / I0_code
            out["transmitted_flux"] = discrete_flux(e0, ix_right, flux_coeff_w0) / corr_e0_right / I0_code
            if srs_on:
                e1 = y["E1"].view(jnp.complex128)
                out["e1_sq"] = jnp.mean(jnp.sum(jnp.abs(e1) ** 2, axis=-1))
                out["reflectivity"] = sqrt_eps1 * jnp.mean(jnp.abs(e1[ix_probe, :, 1]) ** 2) / E0_source_sq
                out["reflected_flux"] = -discrete_flux(e1, ix_left, flux_coeff_w1) / corr_e1_left / I0_code
                out["backrefl_flux"] = discrete_flux(e1, ix_right, flux_coeff_w1) / corr_e1_right / I0_code
            else:
                # TPD-only: any backward pump content is already included in the signed
                # net E0 flux. Keep the four-channel budget schema with zero Raman flux.
                out["reflected_flux"] = jnp.asarray(0.0)
                out["backrefl_flux"] = jnp.asarray(0.0)

        return out

    return {"t": {"ax": cfg["grid"]["t"]}, "func": save_func}
