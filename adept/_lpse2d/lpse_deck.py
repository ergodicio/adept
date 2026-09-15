"""Translate original-LPSE decks (``lpse.parms``) to envelope-2d configs and read LPSE outputs.

LPSE (LLE, C++) runs in "ZAK" units chosen so that the EPW dispersion coefficient
``3 v_te^2/(2 wp)`` is exactly 1 (``ParameterManager::createPhysicalParameters``); adept
runs in cgs scaled to um / ps / g. ``ZakUnits`` ports the conversion factors so deck
values (and output fields) can be compared quantitatively. The translation covers the
2-D physics of the parity plan (spectral EPW / combined solver, static or spectral light,
spectral IAWs, HPE on/off); keys outside it are reported, not silently dropped.

Conventions that differ between the two codes and are handled here:

- LPSE grid: ``grid.nodes`` nodes spanning ``grid.sizes`` microns with both end nodes
  included, so ``dx = L/(N - 1)`` and the FFT period is ``N dx``; the box is centred on 0.
  adept: ``nx`` cells of ``dx`` on ``[0, nx dx)``. Locations are shifted by ``+L/2``.
- LPSE anti-aliasing (``grid.antiAliasing.range``) zeroes the outer fraction of *each*
  k axis (``dealias: rectangular``); the EPW time step is
  ``min(lw.spectral.dt, lw.maxLightStepsPerStep * light dt)``.
- LPSE noise (``lw.noise.isCalculated = false``) is a flat-in-potential-kick source
  without the Debye factor (``noise_model: thermal, noise_debye_factor: false``); the
  amplitude converts with ``A_adept = A_lpse * zakUnitsPerMicron * potential_zak_to_adept``
  (both codes use an unnormalized forward FFT).
- ``relativity`` is enabled in LPSE by default, so the Landau rate is its relativistic
  2-D form.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# constants as in LPSE Constants.h / adept helpers
C_CGS = 2.99792458e10
ME_CGS = 9.10938291e-28
MP_CGS = 1.6726219e-24
QE_CGS = 4.8032068e-10
QE_SI = 1.602176634e-19
KB_CGS = 1.380649e-16
KELVIN_PER_EV = 11604.5
FIELD_SCALE_ADEPT = 1.0e14  # cgs statvolt/cm per adept field unit (massScale^1/2 spatialScale^-1/2 / timeScale)
POTENTIAL_SCALE_ADEPT = 1.0e10  # cgs statvolt per adept potential unit (field unit * um)


class ZakUnits:
    """LPSE's ZAK unit system for a plasma (Te, Ti in keV, Z, Mi/Me), envelope density
    ``n_env`` (n0/nc) and laser wavelength (um). Port of ``ParameterManager.cpp:1136-1225``."""

    def __init__(self, te_kev: float, ti_kev: float, z: float, mi_over_me: float, n_env: float, wavelength_um: float):
        self.eta = 1.0 + 3.0 * ti_kev / (z * te_kev)
        self.big_m = mi_over_me / (self.eta * z)  # v_e^2 / c_s^2
        lambda_cm = wavelength_um * 1.0e-4
        te_kelvin = 1000.0 * te_kev * KELVIN_PER_EV
        self.laser_frequency = 2.0 * np.pi * C_CGS / lambda_cm  # rad/s
        self.nc = self.laser_frequency**2 * ME_CGS / (4.0 * np.pi * QE_CGS**2)  # 1/cm^3
        self.n0 = n_env * self.nc
        self.plasma_frequency = np.sqrt(4.0 * np.pi * self.n0 * QE_CGS**2 / ME_CGS)  # rad/s
        self.debye_length_cm = np.sqrt(KB_CGS * te_kelvin / (4.0 * np.pi * self.n0 * QE_CGS**2))
        self.n0_zak = 0.75 * self.big_m
        self.nc_zak = self.n0_zak / n_env
        self.kde = 1.5 * np.sqrt(self.big_m)
        self.wpe = 1.5 * self.big_m
        self.zak_per_ps = 1.0e-12 * self.plasma_frequency / self.wpe
        self.zak_per_um = 1.0e-4 / (self.kde * self.debye_length_cm)
        self.um_per_zak = 1.0 / self.zak_per_um
        self.ps_per_zak = 1.0 / self.zak_per_ps
        self.ve = np.sqrt(self.big_m)  # electron thermal velocity, ZAK
        self.speed_of_light = C_CGS * 1.0e4 * self.zak_per_um / (1.0e12 * self.zak_per_ps)
        self.io_scale_factor = (
            (8.0 / 3.0) * (self.eta / self.big_m) * self.n0 * (QE_SI * 1000.0 * te_kev) * C_CGS
        )  # W/cm^2
        self.field_zak_to_cgs = np.sqrt(1.0e7 * self.io_scale_factor * 8.0 * np.pi / C_CGS)
        self.potential_zak_to_cgs = self.field_zak_to_cgs * self.um_per_zak * 1.0e-4
        self.field_zak_to_adept = self.field_zak_to_cgs / FIELD_SCALE_ADEPT
        self.potential_zak_to_adept = self.potential_zak_to_cgs / POTENTIAL_SCALE_ADEPT

    def summary(self) -> dict:
        return {
            "critical density (1/cm^3)": self.nc,
            "envelope density (1/cm^3)": self.n0,
            "plasma frequency (rad/s)": self.plasma_frequency,
            "Debye length (cm)": self.debye_length_cm,
            "zak per ps": self.zak_per_ps,
            "zak per um": self.zak_per_um,
            "Io scale factor (W/cm^2)": self.io_scale_factor,
        }


# ------------------------------------------------------------------ decks --


def parse_parms(path: str | Path) -> dict[str, str]:
    """Parse an LPSE ``lpse.parms`` deck (``key = value;`` lines, ``#include "file"``,
    ``#`` comments) into a flat dict of raw string values, includes resolved relative
    to the deck's directory."""
    path = Path(path)
    parms: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        inc = re.match(r'#include\s+"([^"]+)"', line)
        if inc:
            parms.update(parse_parms(path.parent / inc.group(1)))
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        line = line.split("//")[0].strip()
        for statement in line.split(";"):
            statement = statement.strip()
            if not statement or "=" not in statement:
                continue
            key, value = statement.split("=", 1)
            parms[key.strip()] = value.strip().strip('"')
    return parms


def _floats(value: str) -> list[float]:
    return [float(v) for v in value.replace(",", " ").split()]


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("true", "t", "1", "yes")


def translate_parms(
    parms: dict[str, str], *, experiment: str = "lpse-parity", run: str = "translated-deck"
) -> tuple[dict, dict]:
    """Build an envelope-2d config from parsed LPSE deck values.

    Returns ``(cfg, report)``; ``report["unsupported"]`` lists deck settings that have no
    adept counterpart (they are also printed), ``report["units"]`` the ZakUnits summary.
    """
    g = parms.get
    report: dict = {"unsupported": [], "notes": []}

    te = float(g("physical.Te", "2"))
    ti = float(g("physical.Ti", "1"))
    z = float(g("physical.Z", "1"))
    mi_over_me = float(g("physical.MiOverMe", "1836"))
    n_env = float(g("lw.envelopeDensity", "0.25"))
    wavelength_um = float(g("laser.wavelength", "0.351"))
    zak = ZakUnits(te, ti, z, mi_over_me, n_env, wavelength_um)
    report["units"] = zak.summary()

    # ---- grid: LPSE nodes span the size with both end nodes, dx = L / (N - 1)
    sizes = _floats(g("grid.sizes", "0 0"))
    nodes = [int(v) for v in _floats(g("grid.nodes", "0 0"))]
    lx = sizes[0]
    nx = nodes[0]
    dx = lx / (nx - 1)
    ny = nodes[1] if len(nodes) > 1 else 1
    xmax = nx * dx
    if ny > 1:
        half_y = 0.5 * ny * dx
    else:
        half_y = 0.4 * dx  # collapses to ny = 1 in adept
    if len(sizes) > 2 and sizes[2] > 0 and len(nodes) > 2 and nodes[2] > 1:
        report["unsupported"].append("3-D grid (grid.sizes/nodes z): adept envelope-2d is 2-D")

    # ---- time stepping
    lw_dt = float(g("lw.spectral.dt", g("simulation.dt", "0.002")))
    light_dt = None
    for key in ("raman.dt", "laser.dt"):
        if key in parms:
            light_dt = float(parms[key])
            break
    max_light_steps = int(float(g("lw.maxLightStepsPerStep", "10")))
    laser_evolves = g("laser.solver", "static").lower() != "static" and _bool(g("laser.enable"), True)
    raman_on = _bool(g("raman.enable"))
    if light_dt is not None and (laser_evolves or raman_on):
        dt = min(lw_dt, max_light_steps * light_dt)
        light_substeps = max(1, round(dt / light_dt))
    else:
        dt = lw_dt
        light_substeps = None
    tmax = float(g("simulation.time.end", "1"))
    sample_period = float(g("simulation.samplePeriod", str(tmax)))

    # ---- anti-aliasing: outer fraction of each axis zeroed
    aa_range = float(g("grid.antiAliasing.range", "0.334"))

    # ---- density profile
    shape = g("densityProfile.shape", "linear").lower()
    n_min = float(g("densityProfile.NminOverNc", str(n_env)))
    n_max = float(g("densityProfile.NmaxOverNc", str(n_env)))
    x_min_loc = _floats(g("densityProfile.NminLocation", f"{-lx / 2}"))[0] + lx / 2.0
    x_max_loc = _floats(g("densityProfile.NmaxLocation", f"{lx / 2}"))[0] + lx / 2.0
    shape_map = {
        "linear": "linear",
        "exp": "exp",
        "exponential": "exp",
        "gaussian": "gaussian",
        "inversesquare": "inverse-power",
        "inversepower": "inverse-power",
        "quadratic": "quadratic",
        "qd": "qd",
        "gd": "gd",
        "file": "file",
    }
    if shape not in shape_map:
        report["unsupported"].append(f"densityProfile.shape = {shape} (translated as linear)")
    adept_shape = shape_map.get(shape, "linear")
    y_min_loc = _floats(g("densityProfile.NminLocation", "0 0"))
    y_max_loc = _floats(g("densityProfile.NmaxLocation", "0 0"))
    density = {
        "basis": f"lpse-{adept_shape}",
        "min": n_min,
        "max": n_max,
        "min_location": f"{x_min_loc}um",
        "max_location": f"{x_max_loc}um",
        "min_location_y": f"{y_min_loc[1] if len(y_min_loc) > 1 else 0.0}um",
        "max_location_y": f"{y_max_loc[1] if len(y_max_loc) > 1 else 0.0}um",
        "geometry": "spherical" if g("densityProfile.geometry", "cartesian").lower() == "spherical" else "cartesian",
        "origin": f"{lx / 2.0}um",
        "max_density": min(float(g("densityProfile.maxBackgroundDensity", "1.25")), 1.25),
    }
    sg = g("densityProfile.sgOrder", g("densityProfile.sgPower", g("densityProfile.power")))
    if shape == "inversesquare":
        density["sg_order"] = 2.0
    elif sg is not None:
        density["sg_order"] = float(sg)
    if adept_shape == "quadratic":
        density["central_density"] = float(g("densityProfile.quadratic.centralDensity", "0"))
    if adept_shape in ("qd", "gd"):
        density["dip_depth"] = float(g("densityProfile.dip.depth", "0"))
        density["dip_width"] = f"{float(g('densityProfile.dip.width', '0'))}um"
        density["dip_offset"] = f"{float(g('densityProfile.dip.offset', '0'))}um"
    if adept_shape == "file":
        density["file"] = g("densityProfile.loadFilename", "")
        report["notes"].append("densityProfile.shape = file: density.file must point at the LPSE grid file")

    # ---- EPW terms
    labc_x = float(g("lw.Labc.min.x", g("lw.Labc", "0")))
    labc_y = float(g("lw.Labc.min.y", g("lw.Labc", "0")))
    boundary = {"x": "absorbing" if labc_x > 0 else "periodic", "y": "absorbing" if labc_y > 0 else "periodic"}
    boundary_width = max(labc_x, labc_y, dx)
    if labc_x > 0 and labc_y > 0 and labc_x != labc_y:
        report["notes"].append("different x/y absorber widths: adept uses one boundary_width (the larger)")
    landau_on = _bool(g("lw.landauDamping.enable"), True)
    relativity = _bool(g("physical.isRelativistic", g("isRelativistic")), True)
    if _bool(g("lw.collisionalDampingRate.isCalculated")):
        collisions: bool | float = True
    else:
        collisions = float(g("lw.collisionalDampingRate", "0"))
    tpd_on = _bool(g("lw.TPD.enable"))
    srs_on = _bool(g("lw.SRS.enable"))
    noise_on = _bool(g("lw.noise.enable"))
    noise_amp_lpse = float(g("lw.noise.amplitude", "1" if _bool(g("lw.noise.isCalculated")) else "0"))
    noise_calculated = _bool(g("lw.noise.isCalculated"))
    if noise_calculated:
        report["notes"].append(
            "lw.noise.isCalculated: LPSE's absolute constant is not reproduced; adept's equipartition "
            "calibration (noise_calibrate) is used with the deck amplitude as multiplier"
        )
    source = {
        "noise": noise_on,
        "noise_model": "thermal",
        "noise_debye_factor": noise_calculated,
        "noise_calibrate": noise_calculated,
        "noise_amplitude": float(
            noise_amp_lpse if noise_calculated else noise_amp_lpse * zak.zak_per_um * zak.potential_zak_to_adept
        ),
        "noise_seed": int(float(g("seed", "1"))),
        "tpd": tpd_on,
        "tpd_form": "lpse",
        "srs": srs_on,
        # LPSE lw.kFilter is off unless enabled; adept's default (on) would zero the SRS
        # source at densities detuned from the envelope density
        "srs_k_filter": _bool(g("lw.kFilter.enable")),
        "srs_k_filter_scale": float(g("lw.kFilter.scale", "1.2")),
    }
    epw_solver = "combined" if g("lw.solver", "spectral").lower() == "combined" else "separate"
    if g("lw.solver", "spectral").lower() == "fd":
        report["unsupported"].append("lw.solver = fd (LPSE itself disables it); translated as spectral")
    epw = {
        "boundary": boundary,
        "damping": {
            "collisions": collisions,
            "landau": landau_on,
            "landau_form": "relativistic" if relativity else "lpse",
            "landau_lower_threshold": float(g("lw.landauDamping.lowerThreshold", "0")),
        },
        "density_gradient": True,
        "linear": True,
        "source": source,
        "solver": epw_solver,
    }
    if "lw.maxWavenumber" in parms:
        epw["max_wavenumber"] = float(parms["lw.maxWavenumber"])

    # ---- light
    raman_solver = g("raman.solver", "spectral").lower()
    laser_solver = g("laser.solver", "static").lower()
    light: dict = {"pump_depletion": laser_evolves}
    if epw_solver == "combined" or raman_solver == "spectral" or laser_solver == "spectral":
        light["solver"] = "spectral"
    else:
        light["solver"] = "fd"
    if laser_evolves and laser_solver != raman_solver and raman_on:
        report["notes"].append(
            f"laser.solver = {laser_solver} but raman.solver = {raman_solver}: "
            f"one adept light solver ({light['solver']})"
        )
    absorption = float(g("raman.evolution.absorption", g("laser.evolution.absorption", "0")))
    if absorption > 0:
        light["absorption"] = absorption
    elif absorption < 0:
        light["absorption"] = True
    if "raman.spectral.maxWavenumber" in parms:
        light["max_wavenumber"] = float(parms["raman.spectral.maxWavenumber"])
    light["boundary_max_rate"] = float(
        g("raman.evolution.abc.maxDampingRate", g("laser.evolution.abc.maxDampingRate", "5000"))
    )
    # LPSE default is false; the reference decks set it true
    light["transverse_source"] = _bool(
        g("raman.takeTransversePartOfSourceTerms", g("laser.takeTransversePartOfSourceTerms", "false"))
    )
    if _bool(g("laser.pumpDepletion.TPD.enable")) or _bool(g("laser.pumpDepletion.SRS.enable")):
        light["pump_depletion"] = True
    if _bool(g("lw.kFilter.enable")):
        light["tpd_k_filter"] = True

    # ---- laser beams (static: intensities summed; direction along +x expected)
    n_beams = int(float(g("laser.nBeams", "1")))
    intensity = 0.0
    angle_deg = 0.0
    for b in range(1, n_beams + 1):
        intensity += float(g(f"laser.{b}.intensity", "0"))
        direction = _floats(g(f"laser.{b}.direction", "1 0 0"))
        dz = direction[2] if len(direction) > 2 else 0.0
        if abs(dz) > 1e-6 * np.linalg.norm(direction) or direction[0] <= 0:
            report["unsupported"].append(f"laser.{b}.direction {direction}: adept pump is in-plane and rightward")
        beam_angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        if b == 1:
            angle_deg = beam_angle
        elif abs(beam_angle - angle_deg) > 1e-6:
            report["unsupported"].append(
                f"laser.{b}.direction differs from beam 1: one pump angle ({angle_deg:.2f} deg)"
            )
        if float(g(f"laser.{b}.polarization", "0")) != 0.0:
            report["unsupported"].append(f"laser.{b}.polarization != 0: adept pump is in-plane (p) polarized")
        if float(g(f"laser.{b}.frequencyShift", "0")) != 0.0:
            report["notes"].append(f"laser.{b}.frequencyShift ignored (single color)")
    if n_beams > 1:
        report["notes"].append(f"{n_beams} beamlets summed into one plane wave (no speckle)")
    drivers = {
        "E0": {
            "shape": "uniform",
            "num_colors": 1,
            "delta_omega_max": 0.0,
            "params": {"phases": {"seed": 42}},
            "angle": angle_deg,
            "envelope": {
                "tw": f"{10 * tmax}ps",
                "tr": "0.01ps",
                "tc": f"{5 * tmax}ps",
                "xr": "0.2um",
                "xw": "1000um",
                "xc": "50um",
                "yr": "0.2um",
                "yw": "1000um",
                "yc": "0um",
            },
        }
    }
    if laser_evolves:
        labc_light = float(g("laser.evolution.Labc.min.x", g("laser.evolution.Labc", str(labc_x))))
        drivers["E0"]["offset"] = f"{2.0 * max(labc_light, dx)}um"
    if raman_on and int(float(g("raman.nBeams", "0"))) > 0:
        drivers["E1"] = {"intensity": f"{float(g('raman.1.intensity', '0'))}W/cm^2"}

    # ---- IAW
    iaw = None
    if _bool(g("iaw.enable")):
        iaw = {
            "active": True,
            "solver": "spectral" if g("iaw.solver", "spectral").lower() == "spectral" else "explicit",
            "damping": {
                "landau": float(g("iaw.landauDampingRate", g("iaw.dampingRate", "0"))),
                "collisions": float(g("iaw.collisionalDampingRate", "0")),
                "landau_form": "simplified" if _bool(g("iaw.fd.damping.isSimplified"), True) else "full",
            },
            "max_density_perturbation": float(g("iaw.amplitudeClamp", "0.9")),
            "noise": _bool(g("iaw.noise.enable")),
        }
        if g("iaw.solver", "spectral").lower() != "spectral":
            report["notes"].append("iaw.solver = fd translated as spectral (adept has no PPM IAW solver)")
        if "iaw.spectral.dt" in parms:
            stride = max(1, round(float(parms["iaw.spectral.dt"]) / dt))
            if stride > 1:
                iaw["stride"] = stride
        if "fluid.velocity" in parms:
            vel = _floats(parms["fluid.velocity"])
            iaw["flow"] = [vel[0], vel[1] if len(vel) > 1 else 0.0]

    # ---- HPE
    hpe = None
    if _bool(g("hpe.enable")):
        hpe = {"active": True}
        for lpse_key, adept_key in (
            ("hpe.numParticles", "n_particles"),
            ("hpe.nParticles", "n_particles"),
            ("hpe.nElectrons", "n_particles"),
        ):
            if lpse_key in parms:
                hpe[adept_key] = int(float(parms[lpse_key]))
        if "hpe.velocityGrid" in parms:
            hpe["nv"] = int(float(parms["hpe.velocityGrid"]))
        if "hpe.startEvolutionAt" in parms:
            hpe["t_start"] = f"{float(parms['hpe.startEvolutionAt'])}ps"
        if "hpe.thermalizationProbability" in parms:
            tp = _floats(parms["hpe.thermalizationProbability"])
            hpe["thermalization_probability"] = [tp[0], tp[1] if len(tp) > 1 else tp[0]]
        if "hpe.magneticField" in parms:
            b = _floats(parms["hpe.magneticField"])
            hpe["magnetic_field"] = b[2] if len(b) > 2 else 0.0
            if any(abs(v) > 0 for v in b[:2]):
                report["unsupported"].append("hpe.magneticField in-plane components (only B_z acts on the 2-D push)")
        if "hpe.gammaLimit.damping" in parms:
            hpe["gamma_limit_damping"] = float(parms["hpe.gammaLimit.damping"])
        if "hpe.gammaLimit.growth" in parms:
            hpe["gamma_limit_growth"] = float(parms["hpe.gammaLimit.growth"])
        if _bool(g("hpe.allowGrowth")):
            hpe["allow_growth"] = True
        if _bool(g("hpe.enforceEnergyConservation")):
            hpe["energy_conservation"] = True
            hpe["energy_conservation_steps"] = float(g("hpe.numStepsToAverageEnergyChange", "1"))
        n_flux = int(float(g("hpe.metrics.nFluxMetrics", "0")))
        if n_flux > 0:
            edges = sorted(
                {float(g(f"hpe.metrics.flux.{i}.energy.min", "0")) for i in range(1, n_flux + 1)}
                | {float(g(f"hpe.metrics.flux.{i}.energy.max", "1e9")) for i in range(1, n_flux + 1)}
            )
            hpe["flux_bins"] = edges
        if int(float(g("hpe.metrics.nPowerMetrics", "0"))) > 0:
            hpe["cone_angle"] = float(g("hpe.metrics.power.1.angle", "30"))
            direction = _floats(g("hpe.metrics.power.1.direction", "1 0 0"))
            hpe["cone_direction"] = [direction[0], direction[1] if len(direction) > 1 else 0.0]
        report["notes"].append(
            "hpe: dt/dtFields/stepsPerLandauUpdate/vdf.*/blend.gamma are LPSE tracker internals without an adept "
            "equivalent (adept sub-cycles at substep_courant and blends with v_blend_buffer)"
        )

    for key in parms:
        if key.startswith("thermalFil") and _bool(parms[key]):
            report["unsupported"].append(f"{key} (thermal filamentation)")
        if key.startswith("qle") and _bool(parms[key]):
            report["unsupported"].append(f"{key} (quasilinear module)")

    cfg = {
        "solver": "envelope-2d",
        "units": {
            "atomic number": mi_over_me / 1836.15,
            "envelope density": n_env,
            "ionization state": round(z),
            "laser intensity": f"{intensity:.6g}W/cm^2",
            "laser_wavelength": f"{wavelength_um}um",
            "reference electron temperature": f"{te}keV",
            "reference ion temperature": f"{ti}keV",
        },
        "density": density,
        "grid": {
            "boundary_abs_coeff": 200.0,
            "boundary_width": f"{boundary_width}um",
            "boundary_profile": "exp",
            "boundary_max_rate": 200.0,
            "low_pass_filter": 1.0 - aa_range,
            "dealias": "rectangular",
            "dt": f"{dt}ps",
            "dx": f"{dx}um",
            "xmax": f"{xmax}um",
            "tmax": f"{tmax}ps",
            "tmin": "0ps",
            "ymax": f"{half_y}um",
            "ymin": f"{-half_y}um",
        },
        "save": {"fields": {"t": {"dt": f"{sample_period}ps", "tmax": f"{tmax}ps", "tmin": "0ps"}}},
        "mlflow": {"experiment": experiment, "run": run},
        "drivers": drivers,
        "terms": {"epw": epw, "light": light, "zero_mask": True},
    }
    if light_substeps is not None:
        cfg["grid"]["light_substeps"] = light_substeps
    if iaw is not None:
        cfg["terms"]["iaw"] = iaw
    if hpe is not None:
        cfg["terms"]["hpe"] = hpe
    for item in report["unsupported"]:
        print(f"lpse_deck: not translated -- {item}")
    return cfg, report


# ---------------------------------------------------------------- outputs --


def read_metrics(path: str | Path) -> dict[str, np.ndarray]:
    """Read an LPSE ``lpse.metrics`` table into ``{column name: array}``."""
    names = []
    rows = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"#\s+\((\d+)\)\s+(\S+)", line)
        if m:
            names.append(m.group(2))
            continue
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(v) for v in line.split()])
    data = np.asarray(rows)
    return {name: data[:, i] for i, name in enumerate(names)}


def read_frames(path: str | Path) -> list[tuple[dict, np.ndarray]]:
    """Read an LPSE binary field file: a list of ``(header, array)`` per frame, the array
    complex ``(Nx, Ny)`` (single precision in the file; ``FileType=real`` gives a real
    array). Header values are parsed to float where possible."""
    blob = Path(path).read_bytes()
    frames = []
    begin = b"# BeginHeaderSegment;"
    data_marker = b"# BeginDataSegment;\n"
    pos = 0
    while True:
        i = blob.find(begin, pos)
        if i < 0:
            break
        j = blob.find(data_marker, i)
        header_text = blob[i + len(begin) : j].decode(errors="replace")
        header: dict = {}
        for pair in header_text.replace("# EndHeaderSegment;", "").replace(";", "").split(","):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                header[k] = float(v) if len(v.split()) == 1 else [float(x) for x in v.split()]
            except ValueError:
                header[k] = v
        nx, ny, nz = int(header["Nx"]), int(header.get("Ny", 1)), int(header.get("Nz", 1))
        is_complex = str(header.get("FileType", "complex")).lower().startswith("c")
        count = nx * ny * nz * (2 if is_complex else 1)
        start = j + len(data_marker)
        raw = np.frombuffer(blob[start : start + 4 * count], dtype="<f4")
        if is_complex:
            arr = raw[0::2] + 1j * raw[1::2]
        else:
            arr = raw.astype(np.float64)
        # LPSE stores x fastest (MATLAB reshape([Nx Ny]) column-major)
        arr = arr.reshape((nz, ny, nx)).transpose(2, 1, 0)[:, :, 0]
        frames.append((header, arr))
        pos = start + 4 * count
    return frames


def main(argv=None):
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="Translate an LPSE lpse.parms deck to an envelope-2d YAML config")
    ap.add_argument("parms")
    ap.add_argument("out")
    ap.add_argument("--experiment", default="lpse-parity")
    ap.add_argument("--run", default=None)
    args = ap.parse_args(argv)
    parms = parse_parms(args.parms)
    run = args.run or Path(args.parms).resolve().parent.name
    cfg, report = translate_parms(parms, experiment=args.experiment, run=run)
    with open(args.out, "w") as fo:
        yaml.safe_dump(cfg, fo, sort_keys=False)
    print(f"wrote {args.out}")
    for k, v in report["units"].items():
        print(f"  {k}: {v:.5g}")
    for note in report["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
