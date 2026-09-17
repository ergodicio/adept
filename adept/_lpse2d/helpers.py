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


def range_restriction(x: np.ndarray, y: np.ndarray, window: dict) -> np.ndarray:
    """LPSE ``ZakharovSolver::restrictRange``: per axis a flat top of ``width`` about
    ``center`` with linear ramps of ``edge_width`` to zero outside, multiplied over the axes
    (an axis with width 0 / omitted is unrestricted). Coordinates are measured from the box
    centre as in LPSE. Returns the (nx, ny) multiplier."""
    xc = x - 0.5 * (x[0] + x[-1])
    yc = y - 0.5 * (y[0] + y[-1])
    widths = list(window.get("width") or [])
    centers = list(window.get("center") or [])
    edge = _Q(window.get("edge_width", "0um")).to("um").value
    multiplier = np.ones((x.size, y.size))
    for axis, coord in enumerate((xc[:, None], yc[None, :])):
        width = _Q(widths[axis]).to("um").value if axis < len(widths) and widths[axis] is not None else 0.0
        if width <= 0.0:
            continue
        center = _Q(centers[axis]).to("um").value if axis < len(centers) and centers[axis] is not None else 0.0
        lo, hi = center - 0.5 * width, center + 0.5 * width
        if edge > 0.0:
            ramp = np.clip(np.minimum((coord - lo) / edge + 1.0, (hi - coord) / edge + 1.0), 0.0, 1.0)
        else:
            ramp = np.where((coord > lo) & (coord <= hi), 1.0, 0.0)
        multiplier = multiplier * np.broadcast_to(ramp, multiplier.shape)
    return multiplier


def source_mask(cfg: dict, cfg_grid: dict, which: str) -> np.ndarray:
    """The x-space source multiplier of ``which`` (``epw`` or ``iaw``): the range restriction
    of ``terms.<which>.source_window`` times, with ``terms.light.suppress_sources_in_absorbers``,
    zero inside the absorbing layers (``boundary_width`` from the walls, LPSE
    ``suppressSourcesInAbsorbingRegions``) and, with ``terms.light.suppress_sources_at_injectors``
    (LPSE ``suppressSourcesAtInjectors``), zero across the pump and seed injector rows."""
    x = np.asarray(cfg_grid["x"], dtype=np.float64)
    y = np.asarray(cfg_grid["y"], dtype=np.float64)
    window = cfg["terms"].get(which, {}).get("source_window")
    mask = range_restriction(x, y, window) if window else np.ones((x.size, y.size))
    light = cfg["terms"].get("light", {})
    if light.get("suppress_sources_in_absorbers", False):
        boundary_width = _Q(cfg_grid["boundary_width"]).to("um").value
        inside_x = (x < cfg_grid["xmin"] + boundary_width) | (x > cfg_grid["xmax"] - boundary_width)
        mask = mask * np.where(inside_x, 0.0, 1.0)[:, None]
        boundary = cfg["terms"][which]["boundary"] if which in cfg["terms"] else cfg["terms"]["epw"]["boundary"]
        if y.size > 1 and str(boundary.get("y", "periodic")) != "periodic":
            inside_y = (y < cfg_grid["ymin"] + boundary_width) | (y > cfg_grid["ymax"] - boundary_width)
            mask = mask * np.where(inside_y, 0.0, 1.0)[None, :]
    if light.get("suppress_sources_at_injectors", False):
        dx = float(cfg_grid["dx"])
        rows = np.ones(x.size)
        pump = cfg["drivers"].get("E0", {}).get("derived", {})
        if light.get("pump_depletion", False) and "offset" in pump:
            leftward = np.asarray(pump.get("beam_leftward", [False]), dtype=bool)
            if not np.all(leftward):
                rows = rows * _injector_rows(
                    x, cfg_grid["xmin"] + pump["offset"], pump.get("injector_width"), dx, light
                )
            if np.any(leftward):
                rows = rows * _injector_rows(
                    x, cfg_grid["xmax"] - pump["offset"], pump.get("injector_width"), dx, light
                )
        seed = cfg["drivers"].get("E1", {}).get("derived", {})
        if "offset" in seed:
            rows = rows * _injector_rows(x, cfg_grid["xmax"] - seed["offset"], seed.get("injector_width"), dx, light)
        mask = mask * rows[:, None]
    return mask


def _injector_rows(x: np.ndarray, x_inject: float, injector_width, dx: float, light: dict) -> np.ndarray:
    """Zero over the injector's cells: the rows of the FD plane-wave injector (two at second
    order, ``fd_order`` rows centred on the plane in general -- LPSE ``solverOrder / 2 + 1``
    on either side), or the Gaussian source's width in cells for the spectral solver."""
    i0 = int(np.argmin(np.abs(x - x_inject)))
    if str(light.get("solver", "fd")) == "spectral":
        width_cells = int(np.ceil(float(injector_width) / dx)) if injector_width else 2
        lo, hi = i0 - width_cells, i0 + width_cells
    else:
        half = int(light.get("fd_order", 2)) // 2
        lo, hi = i0 - half + 1, i0 + half
    rows = np.ones(x.size)
    rows[max(lo, 0) : min(hi, x.size - 1) + 1] = 0.0
    return rows


def initial_perturbation_field(cfg: dict) -> np.ndarray:
    """The x-space plane wave of ``initial_perturbation`` on the grid, (nx, ny) complex, in this
    code's units of the target field (LPSE ``InitialPerturbation::create`` with its
    ``scaleFactor``): the potential ``e phi / (m_e c^2)`` converts with ``1 / (e_norm x_norm)``,
    a light field ``e E / (m_e w0 c)`` with ``1 / e_norm``. Coordinates are measured from the
    box centre as in LPSE."""
    ip = cfg["initial_perturbation"]
    grid, derived = cfg["grid"], cfg["units"]["derived"]
    x = np.asarray(grid["x"], dtype=np.float64)
    y = np.asarray(grid["y"], dtype=np.float64)
    xc = x - 0.5 * (x[0] + x[-1])
    yc = y - 0.5 * (y[0] + y[-1])
    direction = np.asarray(list(ip.get("direction", [1.0, 0.0]))[:2] + [0.0, 0.0], dtype=np.float64)[:2]
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("initial_perturbation.direction must be non-zero")
    k_vec = 2.0 * np.pi / _Q(ip["wavelength"]).to("um").value * direction / norm
    envelope = np.ones((x.size, y.size))
    sizes = ip.get("envelope_size") or []
    offsets = ip.get("envelope_offset") or []
    order = float(ip.get("envelope_sg_order", 4.0))
    for axis, coord in enumerate((xc[:, None], yc[None, :])):
        size = _Q(sizes[axis]).to("um").value if axis < len(sizes) and sizes[axis] is not None else 0.0
        if size > 0.0:
            offset = _Q(offsets[axis]).to("um").value if axis < len(offsets) and offsets[axis] is not None else 0.0
            envelope = envelope * np.exp(-(np.abs((coord - offset) / (0.5 * size)) ** order))
    phase = np.exp(1j * (k_vec[0] * xc[:, None] + k_vec[1] * yc[None, :]))
    amplitude = float(ip.get("amplitude", 1.0))
    if ip.get("field", "epw") == "epw":
        scale = 1.0 / (derived["e_norm"] * derived["x_norm"])  # e phi / (m_e c^2) -> potential
    else:
        scale = 1.0 / derived["e_norm"]  # e E / (m_e w0 c) -> field
    return amplitude * scale * envelope * phase


def _polarization_rad(value) -> float:
    """``drivers.*.polarization``: degrees about the beam axis, or ``"p"`` (0, in-plane) / ``"s"``
    (90, along z); returned in radians."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "p":
            return 0.0
        if key == "s":
            return float(np.pi / 2.0)
        value = float(value)
    return float(np.deg2rad(float(value)))


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
        if isinstance(cfg["save"][save_type], dict) and "t" in cfg["save"][save_type]:
            t_cfg = cfg["save"][save_type]["t"]
            t_cfg.setdefault("tmin", cfg_grid.get("tmin", "0ps"))
            t_cfg.setdefault("tmax", cfg_grid["tmax"])

    # cfg_grid["xmax"] = _Q(cfg_grid["xmax"]).to("um").value
    # cfg_grid["xmin"] = _Q(cfg_grid["xmin"]).to("um").value

    if cfg["density"]["basis"] == "linear":
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
        if iaw["stride"] < 1:
            raise ValueError("terms.iaw.stride must be a positive integer")
        if iaw["solver"] == "explicit":
            if iaw["stride"] != 1:
                raise ValueError("terms.iaw.stride > 1 requires terms.iaw.solver: spectral")
            if iaw["flow"] is not None:
                raise ValueError("terms.iaw.flow requires terms.iaw.solver: spectral")
            if omega_max * cfg_grid["dt"] >= 2.0:
                raise ValueError(
                    "The ion-acoustic update is unstable: omega_iaw,max * grid.dt must be < 2 "
                    f"(got {omega_max * cfg_grid['dt']:.3g}). Reduce grid.dt or increase grid.dx, "
                    "or use terms.iaw.solver: spectral (unconditionally stable)."
                )
        cfg["terms"]["iaw"] = iaw
        print(
            f"IAWs are on ({iaw['solver']} solver, every {iaw['stride']} EPW step(s)) -- evolving density and "
            f"velocity divergence with omega_iaw,max * dt = {omega_max * cfg_grid['dt'] * iaw['stride']:.3g}"
        )

    # HPE (Follett-style test-particle Landau damping): resolve defaults, convert
    # units, and derive the substep count here so everything lands in MLflow params
    hpe = cfg["terms"].get("hpe", {})
    if hpe.get("active", False):
        if not cfg["terms"]["epw"]["damping"].get("landau", True):
            raise ValueError("terms.hpe requires terms.epw.damping.landau: true (it replaces the static rate)")
        # defaults and type coercion come from the datamodel's HPEModel so they are
        # defined in exactly one place; unknown keys are passed through untouched
        from adept._lpse2d.datamodel import HPEModel

        hpe = {**hpe, **HPEModel(**hpe).model_dump()}
        if hpe["omega_res"] not in ("bohm_gross", "wp0"):
            raise ValueError("terms.hpe.omega_res must be 'bohm_gross' or 'wp0'")
        if hpe["n_angles"] < 4:
            raise ValueError("terms.hpe.n_angles must be at least 4")
        if hpe["v_min"] < 0.0:
            raise ValueError("terms.hpe.v_min must be non-negative")
        vte = np.sqrt(cfg["units"]["derived"]["vte_sq"])
        if hpe["v_min"] * vte >= 0.99 * cfg["units"]["derived"]["c"]:
            raise ValueError("terms.hpe.v_min * vte must be below the 0.99c particle-speed cap")
        hpe["tau_damping_ps"] = _Q(hpe["tau_damping"]).to("ps").value
        hpe["t_start_ps"] = _Q(hpe["t_start"]).to("ps").value
        wp0 = cfg["units"]["derived"]["wp0"]
        hpe["substeps"] = int(np.ceil(wp0 * cfg_grid["dt"] / float(hpe["substep_courant"])))
        cfg["terms"]["hpe"] = hpe
        dimensionality = f"2D2V with {hpe['n_angles']} projections" if cfg_grid["ny"] > 1 else "1D1V"
        print(
            f"HPE is on -- {hpe['n_particles']} box-averaged tail particles ({dimensionality}, "
            f"|v| > {hpe['v_min']} vte), "
            f"{hpe['substeps']} particle substeps per EPW step"
        )

    # light-wave options: defaults and validation (coupling scheme, filter) come from
    # the datamodel's LightModel; unknown keys are passed through untouched
    from adept._lpse2d.datamodel import LightModel

    light = cfg["terms"].get("light", {})
    if light:
        light = {**light, **LightModel(**light).model_dump()}
        cfg["terms"]["light"] = light
    pump_depletion = light.get("pump_depletion", False)
    if light.get("resonance_absorption"):
        if not pump_depletion or light.get("solver", "fd") != "fd":
            raise ValueError(
                "terms.light.resonance_absorption needs terms.light.pump_depletion with the fd light solver "
                "(LPSE refuses it with the spectral solver too)"
            )
    light_solver = light.get("solver", "fd")
    combined = cfg["terms"]["epw"].get("solver", "separate") == "combined"
    if combined and not light:
        cfg["terms"]["light"] = light = {**LightModel().model_dump()}
    if (
        not pump_depletion
        and not combined
        and (light.get("coupling", "explicit") != "explicit" or light.get("filter") is not None)
    ):
        raise ValueError(
            "terms.light.coupling and terms.light.filter act on the coupled (pump-depletion) light solver "
            "and require terms.light.pump_depletion: true"
        )
    source_terms = cfg["terms"]["epw"]["source"]
    srs_on = bool(source_terms.get("srs", False))
    tpd_on = bool(source_terms.get("tpd", False))
    epw_solver = cfg["terms"]["epw"].get("solver", "separate")
    if epw_solver == "combined":
        if srs_on != tpd_on:
            raise ValueError(
                "terms.epw.solver: combined needs terms.epw.source.tpd and srs both on or both off "
                "(LPSE: 'When lw.solver=combined, both SRS and TPD must be enabled or neither')"
            )
        if light_solver != "spectral":
            raise ValueError("terms.epw.solver: combined requires terms.light.solver: spectral")
        if "E2" in cfg["drivers"]:
            raise ValueError("terms.epw.solver: combined does not support the direct EPW driver drivers.E2")
        if cfg["terms"]["epw"].get("energy_ledger", False):
            raise ValueError("terms.epw.energy_ledger is only available with terms.epw.solver: separate")
        if light.get("coupling", "explicit") != "explicit" or light.get("filter") is not None:
            raise ValueError("terms.light.coupling / filter are options of the separate solver's FD light path")
    elif srs_on and tpd_on:
        print(
            "WARNING: terms.epw.source.tpd and srs are both on with terms.epw.solver: separate. The original "
            "LPSE refuses this ('Must use lw.solver=combined when both SRS and TPD are enabled'): the four "
            "separate envelope equations are not valid for simultaneous TPD and SRS. Use terms.epw.solver: "
            "combined for the LPSE formulation."
        )
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

    if srs_on and cfg["terms"]["epw"]["source"].get("srs_k_filter", True):
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
        # the stencil's largest eigenvalue per dimension relative to the compact stencil's
        # 4/dx^2 (1, 4/3, 68/45 at orders 2, 4, 6) scales the propagation term of the bound
        from adept._lpse2d.core.stencils import max_eigenvalue_factor

        stencil_factor = max_eigenvalue_factor(cfg["terms"].get("light", {}).get("fd_order", 2))
        if srs_on:
            dt_limits.append(
                1.0
                / (
                    stencil_factor * 2.0 * derived["c"] ** 2 / (cfg_grid["dx"] ** 2 * derived["w1"])
                    + _worst_detuning_sq(derived["w1"]) / (4.0 * derived["w1"])
                )
            )
            evolved_carriers.append("Raman")
        if pump_depletion:
            dt_limits.append(
                1.0
                / (
                    stencil_factor * 2.0 * derived["c"] ** 2 / (cfg_grid["dx"] ** 2 * derived["w0"])
                    + _worst_detuning_sq(derived["w0"]) / (4.0 * derived["w0"])
                )
            )
            evolved_carriers.append("pump")
        dt_max = min(dt_limits)
        if light_solver == "spectral":
            # the exact k-space propagator has no CFL limit, but the smooth injectors add their
            # source with an Euler step: the light must not cross more than one cell per sub-step
            # (LPSE's spectral decks use raman.dt ~ dx/c), or the injected amplitude is wrong
            c_light = cfg["units"]["derived"]["c"]
            n_needed = int(np.ceil(c_light * cfg_grid["dt"] / cfg_grid["dx"]))
            if "light_substeps" in cfg_grid:
                n_sub = int(cfg_grid["light_substeps"])
                if n_sub < 1:
                    raise ValueError("grid.light_substeps must be a positive integer")
                if n_sub < n_needed:
                    cells = c_light * cfg_grid["dt"] / n_sub / cfg_grid["dx"]
                    print(
                        f"WARNING: grid.light_substeps = {n_sub} lets light cross {cells:.1f} cells per "
                        f"sub-step; the spectral injectors want <= 1 ({n_needed} sub-steps)"
                    )
            else:
                n_sub = max(1, n_needed)
        elif "light_substeps" in cfg_grid:
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
        print(
            f"{carriers} light ({light_solver} solver) is sub-cycled {n_sub}x per EPW step "
            f"(FD dt_light limit {dt_max:.2e} ps)"
        )

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
                "polarization": _polarization_rad(cfg["drivers"][k].get("polarization", "p")),
            }
            if cfg["drivers"][k].get("injector_width") is not None:
                cfg["drivers"][k]["derived"]["injector_width"] = _Q(cfg["drivers"][k]["injector_width"]).to("um").value
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
            if cfg["drivers"][k].get("injector_width") is not None:
                cfg["drivers"][k]["derived"]["injector_width"] = _Q(cfg["drivers"][k]["injector_width"]).to("um").value
        if k == "E0":
            angle_deg = float(cfg["drivers"][k].get("angle", 0.0))
            cfg["drivers"][k]["derived"]["angle"] = float(np.deg2rad(angle_deg))
            e0_cfg = cfg["drivers"][k]
            polarization = _polarization_rad(e0_cfg.get("polarization", "p"))
            cfg["drivers"][k]["derived"]["polarization"] = polarization
            beams = e0_cfg.get("beams") or [{"intensity": 1.0, "angle": angle_deg, "phase": 0.0, "delta_omega": 0.0}]
            # LPSE rotateBeam: the field starts along y, is rotated about the beam axis by the
            # polarization angle, then carried onto the beam direction -- cos(psi) in-plane + sin(psi) z
            cfg["drivers"][k]["derived"]["beam_polarization"] = np.array(
                [_polarization_rad(b.get("polarization", e0_cfg.get("polarization", "p"))) for b in beams],
                dtype=np.float64,
            )
            fraction = np.array([float(b.get("intensity", 1.0)) for b in beams], dtype=np.float64)
            if np.any(fraction < 0) or fraction.sum() <= 0:
                raise ValueError("drivers.E0.beams intensities must be non-negative with a positive sum")
            cfg["drivers"][k]["derived"]["beam_fraction"] = fraction / fraction.sum()
            cfg["drivers"][k]["derived"]["beam_angle"] = np.deg2rad(
                np.array([float(b.get("angle", angle_deg)) for b in beams], dtype=np.float64)
            )
            cfg["drivers"][k]["derived"]["beam_phase"] = np.array(
                [float(b.get("phase", 0.0)) for b in beams], dtype=np.float64
            )
            cfg["drivers"][k]["derived"]["beam_delta_omega"] = np.array(
                [float(b.get("delta_omega", 0.0)) for b in beams], dtype=np.float64
            )
            cfg["drivers"][k]["derived"]["beam_width"] = (
                _Q(e0_cfg["beam_width"]).to("um").value if e0_cfg.get("beam_width") else 0.0
            )
            cfg["drivers"][k]["derived"]["beam_sg_order"] = float(e0_cfg.get("beam_sg_order", 2.0))
            cfg["drivers"][k]["derived"]["beam_offset"] = (
                _Q(e0_cfg["beam_offset"]).to("um").value if e0_cfg.get("beam_offset") else 0.0
            )
            cfg["drivers"][k]["derived"]["kap_bandwidth"] = float(e0_cfg.get("kap_bandwidth", 0.0))
            cfg["drivers"][k]["derived"]["kap_seed"] = int(e0_cfg.get("kap_seed", 0))
            if e0_cfg.get("pulse_file"):
                table = np.loadtxt(e0_cfg["pulse_file"], dtype=np.float64)
                if table.ndim != 2 or table.shape[1] < 2:
                    raise ValueError("drivers.E0.pulse_file must be a two-column (t_ps, amplitude) table")
                cfg["drivers"][k]["derived"]["pulse_t"] = table[:, 0]
                cfg["drivers"][k]["derived"]["pulse_amp"] = table[:, 1]
            beam_angles = [float(b.get("angle", angle_deg)) for b in beams]
            if any(abs(abs(a) - 90.0) < 1e-9 for a in beam_angles):
                raise ValueError("drivers.E0 beams at +-90 deg (injection from a y face) are not supported")
            # a beam with |angle| > 90 propagates leftward and is launched from the x-max face
            cfg["drivers"][k]["derived"]["beam_leftward"] = np.array([abs(a) > 90.0 for a in beam_angles], dtype=bool)
            multi = len(beams) > 1 or any(a != 0.0 for a in beam_angles)
            if multi and cfg["drivers"][k].get("speckle", {}).get("enabled", False):
                raise ValueError("drivers.E0.angle / beams are not supported together with drivers.E0.speckle")
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
    # an oblique pump (drivers.E0.angle) carries k0 sin(angle) in y
    angle = np.deg2rad(float(cfg["drivers"]["E0"].get("angle", 0.0)))
    kx_support = k0 * abs(np.cos(angle))
    ky = max(ky, k0 * abs(np.sin(angle)))

    return float(kx_support), float(ky)


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
    boundary_profile = str(cfg_grid.get("boundary_profile", "tanh"))
    if boundary_profile not in ("tanh", "exp"):
        raise ValueError(f"grid.boundary_profile must be 'tanh' or 'exp', got {boundary_profile!r}")

    def absorbing_rate(boundary, max_rate=None):
        """Amplitude damping rate (1/ps) of the absorbing layers, shape (nx, ny).

        ``tanh``: MATLAB's envelope, ``boundary_abs_coeff * (1 - tanh-envelope)``.
        ``exp``: LPSE absorbingBoundaries.cpp, ``rate = max_rate (e^{lambda s/L} - 1)/(e^lambda - 1)``
        with s the distance into the layer of width L, per axis, the two axes combined by max.
        """
        if boundary_profile == "tanh":
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

            return float(cfg_grid["boundary_abs_coeff"]) * (1.0 - envelope_x * envelope_y)

        lam = float(cfg_grid.get("boundary_lambda", 7.0))
        gamma_max = float(cfg_grid.get("boundary_max_rate", 200.0)) if max_rate is None else float(max_rate)
        coeff = gamma_max / np.expm1(lam)

        def axis_rate(ax, lo, hi):
            # LPSE measures the distance into the layer in whole cells from the edge cell
            # (absorbingBoundaries.cpp), so the first/last cell carries the full rate and
            # the layer spans boundary_width / dx cells
            half = 0.5 * (ax[1] - ax[0]) if ax.size > 1 else 0.0
            s = np.maximum(lo + boundary_width + half - ax, ax - (hi - boundary_width - half))
            s = np.clip(s, 0.0, boundary_width)
            return coeff * np.expm1(lam * s / boundary_width)

        rate_x = np.zeros((cfg_grid["nx"], cfg_grid["ny"]))
        rate_y = np.zeros((cfg_grid["nx"], cfg_grid["ny"]))
        if boundary["x"] == "absorbing":
            rate_x = axis_rate(cfg_grid["x"], cfg_grid["xmin"], cfg_grid["xmax"])[:, None] * np.ones(
                (1, cfg_grid["ny"])
            )
        if boundary["y"] == "absorbing":
            rate_y = axis_rate(cfg_grid["y"], cfg_grid["ymin"], cfg_grid["ymax"])[None, :] * np.ones(
                (cfg_grid["nx"], 1)
            )
        return np.maximum(rate_x, rate_y)

    def absorbing_boundary(boundary, max_rate=None):
        return np.exp(-absorbing_rate(boundary, max_rate) * cfg_grid["dt"])

    cfg_grid["absorbing_rate"] = absorbing_rate(cfg["terms"]["epw"]["boundary"])
    cfg_grid["absorbing_boundaries"] = np.exp(-cfg_grid["absorbing_rate"] * cfg_grid["dt"])
    # the light fields get their own absorber strength (LPSE {laser|raman}.evolution.abc.maxDampingRate,
    # default 5e3/ps against 200/ps for the EPW): light crosses a 3 um layer in 0.01 ps, so at the
    # EPW rate the exp profile removes only ~25 % per crossing and the pump builds a coherent
    # standing wave between the walls (1.45x the launched amplitude on the LPSE test_010 deck).
    # The tanh profile keeps boundary_abs_coeff for both (no change for existing configs).
    light_max_rate = cfg["terms"].get("light", {}).get("boundary_max_rate")
    if light_max_rate is None and boundary_profile == "exp":
        light_max_rate = 5.0e3
    cfg_grid["light_absorbing_boundaries"] = absorbing_boundary(cfg["terms"]["epw"]["boundary"], light_max_rate)
    iaw = cfg["terms"].get("iaw", {})
    if iaw.get("active", False):
        # LPSE's IAW absorber default is half the EPW one (IawSolver.cpp abc.maxDampingRate = 100)
        iaw_max_rate = iaw.get("boundary_max_rate")
        if iaw_max_rate is None and boundary_profile == "exp":
            iaw_max_rate = 0.5 * float(cfg_grid.get("boundary_max_rate", 200.0))
        cfg_grid["iaw_absorbing_boundaries"] = absorbing_boundary(iaw["boundary"], iaw_max_rate)

    cfg_grid["zero_mask"] = (
        np.where(np.sqrt(cfg_grid["kx"][:, None] ** 2 + cfg_grid["ky"][None, :] ** 2) == 0, 0, 1)
        if cfg["terms"]["zero_mask"]
        else 1
    )

    # source windows (plan 2 I.3): x-space multipliers on the EPW sources (TPD, SRS, the
    # combined solver's unified source) and on the IAW ponderomotive drive
    cfg_grid["epw_source_mask"] = source_mask(cfg, cfg_grid, "epw")
    if iaw.get("active", False):
        cfg_grid["iaw_source_mask"] = source_mask(cfg, cfg_grid, "iaw")

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
    elif dealias == "rectangular":
        # the original LPSE mask (grid.antiAliasing.range): the outer fraction of *each* k axis
        # is zeroed, i.e. |kx| < low_pass_filter * kmax_x and |ky| < low_pass_filter * kmax_y
        kx_nyquist = float(np.abs(cfg_grid["kx"]).max())
        ky_nyquist = float(np.abs(cfg_grid["ky"]).max())
        frac = float(cfg_grid["low_pass_filter"])
        band = (np.abs(cfg_grid["kx"])[:, None] < frac * kx_nyquist) & (
            (np.abs(cfg_grid["ky"])[None, :] < frac * ky_nyquist) | (cfg_grid["ny"] == 1)
        )
        cfg_grid["low_pass_filter_grid"] = np.where(band, 1.0, 0.0)
    elif dealias != "isotropic":
        raise ValueError(f"Unknown grid.dealias '{dealias}'. Choose 'isotropic', 'shifted-band' or 'rectangular'.")

    # LPSE lw.maxWavenumber: an additional hard cap on the retained EPW band, in units
    # of the vacuum laser wavenumber k0 = w0/c
    max_wavenumber = cfg["terms"]["epw"].get("max_wavenumber")
    if max_wavenumber is not None:
        k0_vac = cfg["units"]["derived"]["w0"] / cfg["units"]["derived"]["c"]
        cfg_grid["low_pass_filter_grid"] = cfg_grid["low_pass_filter_grid"] * np.where(
            k_mag < float(max_wavenumber) * k0_vac, 1.0, 0.0
        )

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

    elif str(cfg["density"]["basis"]).startswith("lpse-"):
        nprof = _lpse_density_profile(cfg)
    else:
        raise NotImplementedError

    return nprof


LPSE_DENSITY_SHAPES = ("linear", "exp", "gaussian", "inverse-power", "quadratic", "qd", "gd", "file")


def _lpse_density_profile(cfg: dict) -> np.ndarray:
    """The original LPSE density profiles (``ZakharovSolver::backgroundDensityShape``),
    ``density.basis: lpse-<shape>`` with shape in ``LPSE_DENSITY_SHAPES``.

    With ``r`` the distance from the N_max location along the unit vector towards the N_min
    location (``geometry: cartesian``) or the radial distance from it (``spherical``), ``dr``
    the min-max separation and ``p = sg_order`` (default 2):

    - ``linear``: ``N_max + (N_min - N_max) r/dr``, clipped to [N_min, N_max]
    - ``exp``: ``N_max exp(-r/L)``, ``L = dr/ln(N_max/N_min)``, clipped to [N_min, N_max]
    - ``gaussian``: ``N_max exp(-|r/s|^p)``, ``s = dr / ln(N_max/N_min)^(1/p)`` (not clipped below)
    - ``inverse-power``: ``N_min (dr/|r|)^p`` for ``|r| > r_c = dr (N_min/N_max)^(1/p)``, else ``N_max``
    - ``quadratic``: ``A0 + A1 x + A2 x^2`` through ``(0, central_density)``, ``(x_min, N_min)``,
      ``(x_max, N_max)`` with x measured from ``origin`` (LPSE's box centre)
    - ``qd`` / ``gd``: linear plus a parabolic / super-Gaussian dip of depth ``dip_depth``, full
      width ``dip_width`` at ``dip_offset`` from ``origin``
    - ``file``: a (nx, ny) or (nx,) array from ``file`` (``.npy``, a text table, or an LPSE
      grid file read with ``lpse_deck.read_frames``)

    Every profile is clipped at ``max_density`` (LPSE ``maxBackgroundDensity``, 1.25 n_c here)."""
    d = cfg["density"]
    shape = str(d["basis"])[5:]
    if shape not in LPSE_DENSITY_SHAPES:
        raise ValueError(f"density.basis lpse-{shape}: shape must be one of {LPSE_DENSITY_SHAPES}")
    x = np.asarray(cfg["grid"]["x"], dtype=np.float64)
    y = np.asarray(cfg["grid"]["y"], dtype=np.float64)
    nx, ny = len(x), cfg["grid"]["ny"]
    max_density = float(d.get("max_density", 1.25))

    if shape == "file":
        path = str(d["file"])
        if path.endswith(".npy"):
            arr = np.load(path)
        elif path.endswith((".txt", ".csv", ".dat")):
            arr = np.loadtxt(path)
        else:
            from adept._lpse2d.lpse_deck import read_frames

            arr = np.real(read_frames(path)[-1][1])  # already (nx, ny)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = np.repeat(arr[:, None], ny, axis=-1)
        if arr.shape != (nx, ny):
            raise ValueError(f"density.file {path} has shape {arr.shape}, the grid is {(nx, ny)}")
        return np.minimum(arr, max_density)

    n_min, n_max = float(d["min"]), float(d["max"])
    p = float(d.get("sg_order", 2.0))
    loc_min = np.array([_Q(d["min_location"]).to("um").value, _Q(d.get("min_location_y", "0um")).to("um").value])
    loc_max = np.array([_Q(d["max_location"]).to("um").value, _Q(d.get("max_location_y", "0um")).to("um").value])
    X, Y = np.meshgrid(x, y, indexing="ij")
    dr = float(np.linalg.norm(loc_min - loc_max))
    if dr == 0.0 and n_min != n_max and shape not in ("quadratic",):
        raise ValueError("density: min_location and max_location coincide but min != max")
    if str(d.get("geometry", "cartesian")) == "spherical":
        r = np.hypot(X - loc_max[0], Y - loc_max[1])
    else:
        direction = (loc_min - loc_max) / dr if dr > 0 else np.array([1.0, 0.0])
        r = (X - loc_max[0]) * direction[0] + (Y - loc_max[1]) * direction[1]
    lo, hi = min(n_min, n_max), max(n_min, n_max)

    def _linear():
        if dr == 0.0 or n_min == n_max:
            return np.full_like(X, n_max)
        return np.clip(n_max + (n_min - n_max) * r / dr, lo, hi)

    # the quadratic and the dips use LPSE's box-centred x; ``origin`` is that point here
    origin = _Q(d["origin"]).to("um").value if d.get("origin") is not None else 0.5 * (x[0] + x[-1])
    xc = X - origin

    if shape == "linear":
        nprof = _linear()
    elif shape == "exp":
        if n_min == n_max:
            nprof = np.full_like(X, n_max)
        else:
            nprof = np.clip(n_max * np.exp(-r / (dr / np.log(n_max / n_min))), lo, hi)
    elif shape == "gaussian":
        if n_max <= n_min:
            raise ValueError("density lpse-gaussian needs max > min")
        sd = dr / np.log(n_max / n_min) ** (1.0 / p)
        nprof = n_max * np.exp(-(np.abs(r / sd) ** p))
    elif shape == "inverse-power":
        if n_max <= n_min:
            raise ValueError("density lpse-inverse-power needs max > min")
        rc = dr * (n_min / n_max) ** (1.0 / p)
        safe = np.where(np.abs(r) > 0, np.abs(r), 1.0)
        nprof = np.where(np.abs(r) > rc, n_min * (dr / safe) ** p, n_max)
    elif shape == "quadratic":
        a0 = float(d["central_density"])
        x1, x2 = loc_min[0] - origin, loc_max[0] - origin
        if x1 == 0.0 or x2 == 0.0 or x1 == x2:
            raise ValueError("density lpse-quadratic: min/max locations must differ from each other and from origin")
        a2 = (n_max - n_min * x2 / x1 + a0 * (x2 / x1 - 1.0)) / (x2**2 - x1 * x2)
        a1 = (n_min - a0 - a2 * x1**2) / x1
        nprof = a0 + a1 * xc + a2 * xc**2
    elif shape in ("qd", "gd"):
        dn = float(d["dip_depth"])
        wd = 0.5 * _Q(d["dip_width"]).to("um").value
        xd = _Q(d.get("dip_offset", "0um")).to("um").value
        nprof = _linear()
        if shape == "qd":
            inside = np.abs(xc - xd) <= wd
            nprof = nprof + np.where(inside, dn * (((xc - xd) / wd) ** 2 - 1.0), 0.0)
        else:
            nprof = nprof - dn * np.exp(-(np.abs((xc - xd) / wd) ** p))
    if np.any(nprof < 0):
        raise ValueError("density: the LPSE profile is negative somewhere")
    return np.minimum(nprof, max_density)


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
    if "checkpoint" in result.ys:
        target = cfg["save"]["checkpoint"]
        path = target if isinstance(target, str) else os.path.join(td, "binary", "checkpoint.npz")
        np.savez(
            path,
            t=np.asarray(result.ts["checkpoint"][-1]),
            **{k: np.asarray(v[-1]) for k, v in result.ys["checkpoint"].items()},
        )
        cfg["save"]["checkpoint_file"] = path
        print(f"checkpoint written to {path}")
    metrics["write_time"] = time.time() - t0
    os.makedirs(os.path.join(td, "plots"))

    # OSIRIS-comparable scalars (laser budget, EPW growth fit, electron energy)
    metrics.update(series_metrics(series, cfg))

    t0 = time.time()
    plot_series(series, td)
    plot_srs_diagnostics(series, metrics, cfg, td)
    plot_fields(fields, td)
    plot_kt(kfields, td)
    light_spectra = make_light_spectrum_xarrays(cfg, result, td)
    metrics["plot_time"] = time.time() - t0

    return {"k": kfields, "x": fields, "series": series, "metrics": metrics, "light_spectrum": light_spectra}


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
        # high-energy shoulder are the HPE signatures. The 2-D tracker stores one
        # projected histogram per angle; show their box-wide angular mean here.
        hist = np.asarray(series["hpe_hist"].values, dtype=float)
        if hist.ndim == 3:
            hist = np.mean(hist, axis=1)
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
            # Quasi-1D stores (nt, nv). The 2-D tracker stores the same global
            # projected distribution at n_angles oriented axes: (nt, angle, nv).
            hpe = cfg["terms"]["hpe"]
            edges = np.linspace(-hpe["v_max"], hpe["v_max"], hpe["nv"] + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            if v.ndim == 3:
                angles = 2.0 * np.pi * np.arange(hpe["n_angles"]) / hpe["n_angles"]
                data[k] = xr.DataArray(
                    v,
                    coords=(("t (ps)", this_t), ("projection angle (rad)", angles), ("v (c)", centers)),
                )
            else:
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
    e0_all = np.array(state["E0"]).view(_complex)
    e1_all = np.array(state["E1"]).view(_complex)
    e0x = xr.DataArray(e0_all[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e0y = xr.DataArray(e0_all[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))
    e1x = xr.DataArray(e1_all[..., 0], coords=(tax_tuple, xax_tuple, yax_tuple))
    e1y = xr.DataArray(e1_all[..., 1], coords=(tax_tuple, xax_tuple, yax_tuple))
    # the out-of-plane components (plan 2 F.1); saved only when the state carries them
    z_fields = {}
    if e0_all.shape[-1] == 3:
        z_fields["e0z"] = xr.DataArray(e0_all[..., 2], coords=(tax_tuple, xax_tuple, yax_tuple))
        z_fields["e1z"] = xr.DataArray(e1_all[..., 2], coords=(tax_tuple, xax_tuple, yax_tuple))

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
    poynting = {}
    if cfg["save"]["fields"].get("poynting", False):
        # LPSE {laser|raman}.save.S0: the envelope energy-flux density S_j = (c^2/w) Im(E* . d_j E)
        # (v_g |E|^2 for a plane wave), in field^2 * um/ps, on the save grid
        derived = cfg["units"]["derived"]
        for name, arr, w in (
            ("s0", np.array(state["E0"]).view(_complex), derived["w0"]),
            ("s1", np.array(state["E1"]).view(_complex), derived["w1"]),
        ):
            dxs = float(xax[1] - xax[0]) if len(xax) > 1 else 1.0
            dys = float(yax[1] - yax[0]) if len(yax) > 1 else 1.0
            sx = np.zeros(arr.shape[:-1])
            sy = np.zeros(arr.shape[:-1])
            for comp in range(2):
                e = arr[..., comp]
                sx += np.imag(np.conj(e) * np.gradient(e, dxs, axis=1))
                if arr.shape[2] > 1:
                    sy += np.imag(np.conj(e) * np.gradient(e, dys, axis=2))
            factor = derived["c"] ** 2 / w
            poynting[f"{name}_x"] = xr.DataArray(factor * sx, coords=(tax_tuple, xax_tuple, yax_tuple))
            poynting[f"{name}_y"] = xr.DataArray(factor * sy, coords=(tax_tuple, xax_tuple, yax_tuple))
    field_data = {
        "phi": phi_x,
        "ex": ex,
        "ey": ey,
        "e0_x": e0x,
        "e0_y": e0y,
        "e1_x": e1x,
        "e1_y": e1y,
        **({"e0_z": z_fields["e0z"], "e1_z": z_fields["e1z"]} if z_fields else {}),
        "background_density": background_density,
        **poynting,
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
                if k in PARTICLE_KEYS or k == "epw_ledger":
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

            return {k: v for k, v in y.items() if k not in PARTICLE_KEYS and k != "epw_ledger"}

    cfg["save"]["fields"]["func"] = save_func

    cfg["save"]["default"] = get_default_save_func(cfg)

    for i_probe, probe in enumerate(cfg["save"].get("light_spectrum") or []):
        cfg["save"][f"light_spectrum_{i_probe}"] = light_spectrum_save(cfg, probe)

    return cfg


def light_spectrum_save(cfg: dict, probe: dict) -> dict:
    """The save group of one light spectrum probe (LPSE ``LightSpectrum``): the field ``E0`` or
    ``E1`` on the sub-box ``x: [xmin, xmax]``, ``y: [ymin, ymax]`` (coordinates from the box
    centre, as LPSE's ``location.min / .max``; nearest grid nodes, a whole axis when omitted),
    sampled every ``interval`` from ``tmin`` (default the grid start) to ``tmax`` (default the
    grid end). ``poynting: true`` keeps one guard cell on each side so the Poynting components
    can be differenced in post-processing. The time series and its omega spectrum are
    written by ``make_light_spectrum_xarrays``."""
    field = str(probe.get("field", "E0"))
    if field not in ("E0", "E1"):
        raise ValueError(f"save.light_spectrum field must be E0 or E1, got {field!r}")
    srs_on = bool(cfg["terms"]["epw"]["source"].get("srs", False))
    combined = cfg["terms"]["epw"].get("solver", "separate") == "combined"
    if field == "E1" and not (srs_on or combined):
        raise ValueError("save.light_spectrum on E1 needs terms.epw.source.srs (the Raman field)")
    interval = _Q(probe["interval"]).to("ps").value
    if interval <= 0.0:
        raise ValueError("save.light_spectrum interval must be positive")

    def ps(value):
        return float(value) if isinstance(value, (int, float)) else _Q(value).to("ps").value

    grid_tmin, grid_tmax = ps(cfg["grid"]["tmin"]), ps(cfg["grid"]["tmax"])
    tmin = ps(probe["tmin"]) if probe.get("tmin") is not None else grid_tmin
    tmax = ps(probe["tmax"]) if probe.get("tmax") is not None else grid_tmax
    tmin = max(tmin, grid_tmin)
    tmax = min(tmax, grid_tmax)
    if tmax < tmin:
        raise ValueError("save.light_spectrum has tmax < tmin")
    n_t = int(np.floor((tmax - tmin) / interval + 1.0e-9)) + 1
    t_ax = tmin + interval * np.arange(n_t)
    guard = 1 if bool(probe.get("poynting", False)) else 0

    def index_range(axis, key, n):
        coords = np.asarray(cfg["grid"][axis], dtype=np.float64)
        centre = 0.5 * (coords[0] + coords[-1])
        bounds = probe.get(key)
        if bounds is None or n == 1:
            return 0, n - 1
        lo = _Q(bounds[0]).to("um").value + centre
        hi = _Q(bounds[1]).to("um").value + centre
        if hi < lo:
            raise ValueError(f"save.light_spectrum {key} range is reversed")
        i_lo = int(np.argmin(np.abs(coords - lo)))
        i_hi = int(np.argmin(np.abs(coords - hi)))
        return max(i_lo - guard, 0), min(i_hi + guard, n - 1)

    ix = index_range("x", "x", int(cfg["grid"]["nx"]))
    iy = index_range("y", "y", int(cfg["grid"]["ny"]))
    x_slice = slice(ix[0], ix[1] + 1)
    y_slice = slice(iy[0], iy[1] + 1)

    def save_func(t, y, args):
        return {"E": y[field][x_slice, y_slice, :]}

    return {
        "t": {"ax": jnp.asarray(t_ax), "dt": interval},
        "func": save_func,
        "field": field,
        "ix": ix,
        "iy": iy,
        "guard": guard,
    }


def make_light_spectrum_xarrays(cfg: dict, result, td: str) -> list[xr.Dataset]:
    """One Dataset per ``save.light_spectrum`` probe: the field components ``e_x, e_y, e_z`` on
    the sub-box against ``t (ps)``, their spectra ``spectrum_x, ...`` against the envelope
    frequency offset ``omega / w`` (``w`` the field's own carrier: ``w0`` for E0, ``w1`` for
    E1; positive = above the carrier), the box-summed spectral power ``power`` and, with
    ``poynting``, the Poynting components ``s_x, s_y`` (``(c^2/w) Im(E* . d_j E)``) on the box
    without its guard cells. Written to ``binary/light_spectrum_<i>.xr``."""
    derived = cfg["units"]["derived"]
    out = []
    for i_probe, probe in enumerate(cfg["save"].get("light_spectrum") or []):
        key = f"light_spectrum_{i_probe}"
        if key not in result.ys:
            continue
        sub = cfg["save"][key]
        t_ax = np.asarray(result.ts[key], dtype=np.float64)
        raw = np.asarray(result.ys[key]["E"])
        arr = raw.view(np.complex64 if raw.dtype == np.float32 else np.complex128)  # (nt, nx_sub, ny_sub, ncomp)
        w = derived["w0"] if sub["field"] == "E0" else derived["w1"]
        x_full = np.asarray(cfg["grid"]["x"], dtype=np.float64)
        y_full = np.asarray(cfg["grid"]["y"], dtype=np.float64)
        x_ax = x_full[sub["ix"][0] : sub["ix"][1] + 1]
        y_ax = y_full[sub["iy"][0] : sub["iy"][1] + 1]
        g = sub["guard"]
        data = {}
        if g:
            # the Poynting components from the guarded box, then everything cropped to the probe
            dxs = float(x_full[1] - x_full[0]) if x_full.size > 1 else 1.0
            dys = float(y_full[1] - y_full[0]) if y_full.size > 1 else 1.0
            sx = np.zeros(arr.shape[:-1])
            sy = np.zeros(arr.shape[:-1])
            for comp in range(arr.shape[-1]):
                e = arr[..., comp]
                if arr.shape[1] > 1:
                    sx += np.imag(np.conj(e) * np.gradient(e, dxs, axis=1))
                if arr.shape[2] > 1:
                    sy += np.imag(np.conj(e) * np.gradient(e, dys, axis=2))
            factor = derived["c"] ** 2 / w
            crop_x = slice(g if arr.shape[1] > 2 * g else 0, arr.shape[1] - g if arr.shape[1] > 2 * g else arr.shape[1])
            crop_y = slice(g if arr.shape[2] > 2 * g else 0, arr.shape[2] - g if arr.shape[2] > 2 * g else arr.shape[2])
            sx, sy = factor * sx[:, crop_x, crop_y], factor * sy[:, crop_x, crop_y]
            arr = arr[:, crop_x, crop_y, :]
            x_ax, y_ax = x_ax[crop_x], y_ax[crop_y]
        t_tuple = ("t (ps)", t_ax)
        x_tuple = ("x (um)", x_ax)
        y_tuple = ("y (um)", y_ax)
        names = ("x", "y", "z")[: arr.shape[-1]]
        for comp, name in enumerate(names):
            data[f"e_{name}"] = xr.DataArray(arr[..., comp], coords=(t_tuple, x_tuple, y_tuple))
        if g:
            data["s_x"] = xr.DataArray(sx, coords=(t_tuple, x_tuple, y_tuple))
            data["s_y"] = xr.DataArray(sy, coords=(t_tuple, x_tuple, y_tuple))
        # envelope spectrum: E(t) ~ e^{-i dw t} for a component dw above the carrier, so the
        # frequency axis is -(FFT frequency); in units of the field's carrier
        n_t = t_ax.size
        if n_t > 1:
            dt_probe = float(t_ax[1] - t_ax[0])
            omega = -2.0 * np.pi * np.fft.fftfreq(n_t, d=dt_probe) / w
            order = np.argsort(omega)
            omega_tuple = ("delta omega (w_carrier)", omega[order])
            spectra = np.fft.fft(arr, axis=0)[order]
            power = np.zeros(n_t)
            for comp, name in enumerate(names):
                data[f"spectrum_{name}"] = xr.DataArray(spectra[..., comp], coords=(omega_tuple, x_tuple, y_tuple))
                power += np.sum(np.abs(spectra[..., comp]) ** 2, axis=(1, 2))
            data["power"] = xr.DataArray(power, coords=(omega_tuple,))
        ds = xr.Dataset(data, attrs={"field": sub["field"], "carrier (rad/ps)": float(w)})
        ds.to_netcdf(os.path.join(td, "binary", f"{key}.xr"), engine="h5netcdf", invalid_netcdf=True)
        if "power" in ds:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
            ax.semilogy(ds["delta omega (w_carrier)"].values, np.maximum(ds["power"].values, 1e-300))
            ax.set_xlabel(f"(omega - w) / w  [{sub['field']}]")
            ax.set_ylabel("|E(omega)|^2 summed over the probe")
            ax.set_title(f"light spectrum probe {i_probe}")
            fig.savefig(os.path.join(td, "plots", f"{key}.png"), bbox_inches="tight")
            plt.close(fig)
        out.append(ds)
    return out


def get_default_save_func(cfg):
    from adept._lpse2d.core.epw import analytic_landau_rate

    srs_on = cfg["terms"]["epw"]["source"].get("srs", False)
    pump_evolved = cfg["terms"].get("light", {}).get("pump_depletion", False)
    iaw_on = cfg["terms"].get("iaw", {}).get("active", False)
    derived = cfg["units"]["derived"]
    dt = cfg["grid"]["dt"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx, ky = cfg["grid"]["kx"], cfg["grid"]["ky"]
    thomson_probes = list(cfg["save"].get("thomson") or [])
    thomson_masks = []
    k0_vac = derived["w0"] / derived["c"]
    for probe in thomson_probes:
        if probe.get("field", "epw") == "iaw" and not iaw_on:
            raise ValueError("save.thomson probe on the IAW density needs terms.iaw.active")
        kxp, kyp = (float(v) * k0_vac for v in probe["k"])
        band = float(probe.get("bandwidth", 0.1)) * k0_vac
        dist = np.sqrt((np.asarray(kx)[:, None] - kxp) ** 2 + (np.asarray(ky)[None, :] - kyp) ** 2)
        thomson_masks.append(jnp.asarray(dist < band))

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
    # the same static rate (form, threshold, multiplier) the solver applies
    gamma_total = np.array(analytic_landau_rate(cfg)) + derived.get("nu_coll", 0.0) * zero_mask
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
        gamma_an = np.array(hpe_arrays["gamma_analytic"])
        mask_res = np.array(hpe_arrays["mask_res"])
        if mask_res.ndim == 1:
            mask_res = mask_res[:, None]
        # damping-reduction band: resonant modes where the analytic rate is non-negligible
        ratio_band = mask_res & (gamma_an > 1.0e-4 * derived["wp0"])
        have_ratio_band = bool(np.any(ratio_band))
        gamma_an_safe = np.where(ratio_band, gamma_an, 1.0)

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
        # injector rows or it reads the near-field of the two-point source; a leftward beam
        # (x-max face) likewise keeps the right probe upstream of its rows
        ix_left_e0 = ix_left
        ix_right_e0 = ix_right
        if pump_evolved:
            pump = cfg["drivers"]["E0"]["derived"]
            pump_offset = pump["offset"]
            # clearance from the injector plane: 4 cells past the FD rows; the spectral solver's
            # Gaussian source (injector_width, default half a local wavelength) needs three
            # widths, or the probe reads the source region (incident 0.71 instead of 0.98 on the
            # 20 um test box with the default probe_offset = offset)
            clearance = 4
            if cfg["terms"].get("light", {}).get("solver", "fd") == "spectral":
                k0_inj = w0 / derived["c"] * np.sqrt(max(1.0 - float(np.min(cfg["grid"]["background_density"])), 0.0))
                width = pump.get("injector_width", np.pi / k0_inj if k0_inj > 0 else 0.0)
                clearance = max(clearance, int(np.ceil(3.0 * width / cfg["grid"]["dx"])) + 1)
            ix_inject = int(np.argmin(np.abs(x_grid - (cfg["grid"]["xmin"] + pump_offset))))
            ix_left_e0 = max(ix_left, ix_inject + clearance)
            if np.any(np.asarray(pump.get("beam_leftward", [False]))):
                ix_inject_max = int(np.argmin(np.abs(x_grid - (cfg["grid"]["xmax"] - pump_offset))))
                ix_right_e0 = min(ix_right, ix_inject_max - clearance)
        flux_coeff_w0 = derived["c"] ** 2 / (derived["w0"] * cfg["grid"]["dx"])
        flux_coeff_w1 = derived["c"] ** 2 / (derived["w1"] * cfg["grid"]["dx"])
        I0_code = derived["I0_code"]

        spectral_light = cfg["terms"].get("light", {}).get("solver", "fd") == "spectral"
        fd_order = int(cfg["terms"].get("light", {}).get("fd_order", 2))

        def flux_correction(w, ix):
            # The discrete two-point flux of the FD mode at local wavenumber k is
            # |E|^2 * v_g,discrete with v_g,disc = (c^2/w) sin(k_grid dx)/dx, where
            # k_grid satisfies the stencil's dispersion sigma(k_grid dx) = -(k dx)^2
            # ((2/dx^2)(1 - cos k_grid dx) = k^2 at second order; stencils.grid_wavenumber).
            # Dividing by sin(k_grid dx)/(k dx) converts it to the physical flux
            # |E|^2 * c * sqrt(eps). Evanescent probes get 1 (their flux is ~0 anyway).
            # The spectral solver has no grid dispersion: k_grid = k.
            from adept._lpse2d.core.stencils import grid_wavenumber

            n_loc = float(np.mean(np.array(cfg["grid"]["background_density"])[ix, :]))
            eps = 1.0 - n_loc * w0**2 / w**2
            if eps <= 0:
                return 1.0
            k_dx = w / derived["c"] * np.sqrt(eps) * cfg["grid"]["dx"]
            if spectral_light:
                return float(np.sin(k_dx) / k_dx)
            kg_dx = grid_wavenumber(k_dx, fd_order)
            if kg_dx >= np.pi:
                return 1.0
            return float(np.sin(kg_dx) / k_dx)

        corr_e0_left = flux_correction(w0, ix_left_e0)
        corr_e0_right = flux_correction(w0, ix_right_e0)
        corr_e1_left = flux_correction(w1, ix_left)
        corr_e1_right = flux_correction(w1, ix_right)

        def discrete_flux(E, ix, coeff):
            # sum over polarization components, mean over y
            cross = jnp.sum(jnp.conj(E[ix, :, :]) * E[ix + 1, :, :], axis=-1)
            return coeff * jnp.mean(jnp.imag(cross))

    ledger_on = bool(cfg["terms"]["epw"].get("energy_ledger", False))
    if ledger_on:
        from adept._lpse2d.core.epw import LEDGER_CHANNELS, LEDGER_KEY

        # sum_k k^2 |phi_k|^2 -> the epw_energy normalization (Parseval, see above)
        ledger_prefactor = epw_energy_prefactor / (nx * ny**2)

    combined_solver = cfg["terms"]["epw"].get("solver", "separate") == "combined"
    if combined_solver:
        from adept._lpse2d.core.combined import transverse_part

        one_over_k_sq_c = jnp.asarray(np.where(k_sq > 0, 1.0 / np.where(k_sq > 0, k_sq, 1.0), 0.0))
        kx_c, ky_c = jnp.asarray(kx), jnp.asarray(ky)

    def save_func(t, y, args):
        phi_k = y["epw"].view(jnp.complex128)
        ex = -1j * kx[:, None] * phi_k
        ey = -1j * ky[None, :] * phi_k
        ex = jnp.fft.ifft2(ex)
        ey = jnp.fft.ifft2(ey)
        e_sq = jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2
        if combined_solver:
            # the Raman-light diagnostics see the transverse part of the combined field only
            y = {
                **y,
                "E1": transverse_part(y["E1"].view(jnp.complex128), kx_c, ky_c, one_over_k_sq_c).view(jnp.float64),
            }

        out = {"e_sq": jnp.sum(e_sq * cfg["grid"]["dx"] * cfg["grid"]["dy"]), "max_phi": jnp.max(jnp.abs(phi_k))}

        out["epw_energy"] = epw_energy_prefactor * jnp.sum(jnp.mean(e_sq, axis=1))
        if ledger_on:
            # cumulative energy attributed to each split-step operation, in epw_energy units;
            # epw_energy(t) - epw_energy(0) - sum of the channels closes to round-off
            ledger = y[LEDGER_KEY] * ledger_prefactor
            for i, name in enumerate(LEDGER_CHANNELS):
                out[f"epw_ledger_{name}"] = ledger[i]
            out["epw_ledger_closure"] = out["epw_energy"] - jnp.sum(ledger)
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

        # Thomson-scattering probes (LPSE thomsonScattering.N: wavevector.lw / .iaw, bandwidth):
        # the complex amplitude and the power of the EPW potential (or the IAW density) in the
        # k-window |k - k_probe| < bandwidth k0, as a time series for later spectral analysis
        for i_probe, mask in enumerate(thomson_masks):
            field = "iaw" if thomson_probes[i_probe].get("field", "epw") == "iaw" else "epw"
            spectrum = jnp.fft.fft2(y["iaw_density"]) if field == "iaw" else phi_k
            amplitude = jnp.sum(jnp.where(mask, spectrum, 0.0))
            out[f"thomson_{i_probe}_re"] = jnp.real(amplitude)
            out[f"thomson_{i_probe}_im"] = jnp.imag(amplitude)
            out[f"thomson_{i_probe}_power"] = jnp.sum(jnp.where(mask, jnp.abs(spectrum) ** 2, 0.0))

        if hpe_on:
            u = y["u_e"]
            if u.ndim == 2:
                gamma_rel = jnp.sqrt(1.0 + jnp.sum((u / c_light) ** 2, axis=-1))
            else:
                gamma_rel = jnp.sqrt(1.0 + (u / c_light) ** 2)
            ke_kev = 510.999 * (gamma_rel - 1.0)
            out["fhot_50keV"] = w_tail * jnp.sum(ke_kev > 50.0)
            out["fhot_100keV"] = w_tail * jnp.sum(ke_kev > 100.0)
            out["hpe_mean_energy_keV"] = jnp.mean(ke_kev)
            out["hpe_hist"] = y["epw_hist"]
            # LPSE-style instruments: cumulative energy (keV, per real electron via w_tail) out of
            # each wall per energy bin, inside the acceptance cone, and the LD multiplier
            from adept._lpse2d.core.hpe import WALLS

            wall_flux = y["hpe_wall_flux"]
            for i_wall, wall in enumerate(WALLS):
                for i_bin in range(wall_flux.shape[1]):
                    out[f"hpe_wall_energy_{wall}_bin{i_bin}"] = w_tail * wall_flux[i_wall, i_bin]
            out["hpe_cone_energy"] = w_tail * y["hpe_cone_energy"][0]
            out["hpe_ld_multiplier"] = y["hpe_ld_multiplier"][0]
            # running-average kinetic-energy gain of the particles per step, keV per real electron
            out["hpe_particle_power"] = w_tail * 510.999 * y["hpe_particle_power"][0]
            if have_ratio_band:
                ratio = y["gamma_L"] / gamma_an_safe
                # inflation-o-meters: worst-case reduction across the resonant band
                # (noisy at low n_particles -- one clamped mode reads 0), and the
                # reduction at the band mode carrying the most EPW energy (robust,
                # and the physically relevant one)
                out["hpe_gamma_ratio_min"] = jnp.min(jnp.where(ratio_band, ratio, jnp.inf))
                phi_amp = jnp.where(ratio_band, jnp.abs(phi_k), 0.0)
                # before the EPW has any energy in the band (e.g. the zero-initialized
                # first steps) argmax lands on index 0 where the ratio is meaningless;
                # emit NaN so the reduction metrics (nanmin / windowed means) skip it
                peak_index = jnp.argmax(jnp.ravel(phi_amp))
                out["hpe_gamma_ratio_kpeak"] = jnp.where(jnp.any(phi_amp > 0.0), jnp.ravel(ratio)[peak_index], jnp.nan)

        if srs_on or pump_evolved:
            e0 = y["E0"].view(jnp.complex128)
            out["incident_flux"] = discrete_flux(e0, ix_left_e0, flux_coeff_w0) / corr_e0_left / I0_code
            out["transmitted_flux"] = discrete_flux(e0, ix_right_e0, flux_coeff_w0) / corr_e0_right / I0_code
            if srs_on:
                e1 = y["E1"].view(jnp.complex128)
                out["e1_sq"] = jnp.mean(jnp.sum(jnp.abs(e1) ** 2, axis=-1))
                # the transverse components at the probe (y, and z for an s-polarised seed)
                e1_probe_sq = jnp.sum(jnp.abs(e1[ix_probe, :, 1:]) ** 2, axis=-1)
                out["reflectivity"] = sqrt_eps1 * jnp.mean(e1_probe_sq) / E0_source_sq
                out["reflected_flux"] = -discrete_flux(e1, ix_left, flux_coeff_w1) / corr_e1_left / I0_code
                out["backrefl_flux"] = discrete_flux(e1, ix_right, flux_coeff_w1) / corr_e1_right / I0_code
            else:
                # TPD-only: any backward pump content is already included in the signed
                # net E0 flux. Keep the four-channel budget schema with zero Raman flux.
                out["reflected_flux"] = jnp.asarray(0.0)
                out["backrefl_flux"] = jnp.asarray(0.0)

        return out

    return {"t": {"ax": cfg["grid"]["t"]}, "func": save_func}
