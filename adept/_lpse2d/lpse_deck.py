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
        # LPSE writes its potential frames (lw.save.pots) scaled by this factor -- the file holds
        # e phi / (m_e c^2), not the ZAK potential (saveInNormalizedUnits is forced true,
        # ParameterManager.cpp:442, 1247; ZakharovSolver.cpp:504)
        self.potential_normalization_factor = QE_CGS / (ME_CGS * C_CGS**2) * self.potential_zak_to_cgs
        # the electron density frames (lw.save.rho) carry potentialNormalizationFactor / (k0 in ZAK)^2
        self.rho_normalization_factor = (
            self.potential_normalization_factor / (self.um_per_zak * 1.0e-4 * self.laser_frequency / C_CGS) ** 2
        )

    @classmethod
    def from_cfg(cls, cfg: dict) -> ZakUnits:
        """The ZAK system of an adept config (``units`` block; ``atomic number`` is
        ``MiOverMe / 1836.15`` as the translator writes it)."""
        from astropy.units import Quantity as _Q

        units = cfg["units"]
        return cls(
            _Q(units["reference electron temperature"]).to("keV").value,
            _Q(units["reference ion temperature"]).to("keV").value,
            float(units["ionization state"]),
            float(units["atomic number"]) * 1836.15,
            float(units["envelope density"]),
            _Q(units["laser_wavelength"]).to("um").value,
        )

    def noise_amp_k0(self, h_um: float, nx: int, ny: int = 1, nz: int = 1) -> float:
        """LPSE's ``lw.noise.calcNoiseAmp_K0`` (``ParameterManager.cpp:1440-1444``), the
        k-space potential-kick constant of the ``lw.noise.isCalculated`` source for a grid of
        square cells ``h_um`` (um) and ``nx x ny x nz`` nodes:

            (2 pi / h)^nDim / (3 (2 pi)^{3/2} sqrt(eta) M^{1/4} sqrt(N_zak) L_zak^{3/2})
                / sqrt(No_zak Lambda_d^3) / sqrt(deltaK3)

        with ``N_zak = No / No_zak`` (No in 1/cm^3), ``L_zak = kde * debyeLength`` (cm),
        ``Lambda_d = 1 / kde``, ``deltaK_i = 2 pi / (N_i h)`` (ZAK) and ``deltaK3`` the product
        of ``nDim`` of them, the transverse one squared in 2-D. Its own source comments say
        the derivation "doesn't all match"; this is the expression the code uses. test_010
        (Te 2, Ti 1, Z 1, Mi/Me 1836, n_env 0.25, 20 x 10 um on 360 x 180 nodes) prints
        1.7885e+00; this returns 1.78853."""
        n_dim = 1 + int(ny > 1) + int(nz > 1)
        h = h_um * self.zak_per_um
        two_pi = 2.0 * np.pi
        dk = [two_pi / (nx * h), two_pi / (ny * h), two_pi / (nz * h)]
        if n_dim == 1:
            delta_k3 = dk[0] ** 3
        elif n_dim == 2:
            delta_k3 = dk[0] * dk[1] ** 2
        else:
            delta_k3 = dk[0] * dk[1] * dk[2]
        n_zak = self.n0 / self.n0_zak
        l_zak = self.kde * self.debye_length_cm
        lambda_d = 1.0 / self.kde
        return float(
            (two_pi / h) ** n_dim
            / (3.0 * two_pi**1.5 * np.sqrt(self.eta) * self.big_m**0.25 * np.sqrt(n_zak) * l_zak**1.5)
            / np.sqrt(self.n0_zak * lambda_d**3)
            / np.sqrt(delta_k3)
        )

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
    # the deck's directory, for file references relative to it (private key)
    parms["_deck_dir"] = str(Path(path).resolve().parent)
    return parms


def _floats(value: str) -> list[float]:
    """``"1 0 0"``, ``"1,0,0"`` or ``"[1,0,0]"`` -> floats (LPSE's get_floatVector accepts all three)."""
    return [float(v) for v in value.replace(",", " ").replace("[", " ").replace("]", " ").split()]


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("true", "t", "1", "yes")


C_UM_PER_PS = 299.792458


def lpse_density_max(parms: dict[str, str], lx: float) -> float:
    """The largest background density on LPSE's grid (n / n_c; getMinAndMaxBackgroundDensity reads the
    grid), which its FD light step uses: a linear profile is clipped between N_min and N_max
    (backgroundDensityShape), so a box that ends before the N_max location peaks at its edge; all are
    clipped at maxBackgroundDensity (default 1.25)."""
    g = parms.get
    n_env = float(g("lw.envelopeDensity", "0.25"))
    n_min = float(g("densityProfile.NminOverNc", str(n_env)))
    n_max = float(g("densityProfile.NmaxOverNc", str(n_env)))
    clip = float(g("densityProfile.maxBackgroundDensity", "1.25"))
    peak = max(n_min, n_max)
    if g("densityProfile.shape", "linear").lower() == "linear":
        x0 = _floats(g("densityProfile.NminLocation", f"{-lx / 2}"))[0]
        x1 = _floats(g("densityProfile.NmaxLocation", f"{lx / 2}"))[0]
        if x1 != x0:
            slope = (n_max - n_min) / (x1 - x0)
            lo, hi = min(n_min, n_max), max(n_min, n_max)
            peak = max(min(max(n_min + slope * (x - x0), lo), hi) for x in (-lx / 2.0, lx / 2.0))
    return min(peak, clip)


def lpse_time_steps(parms: dict[str, str], *, dx: float, n_dim: int, n_env: float, n_max: float, aa: float) -> dict:
    """LPSE's solver time steps (ps), ported from ``Lpse::computeMicroTimestep`` / ``setupSolverTimeSteps``
    and the solvers' ``Tstep`` (``LightSolver::computeTimeStep``, ``lw.spectral.dt``, ``IawSolver``).

    Returns the global micro step ``dt`` and each solver's ``n`` micro steps per update (laser,
    raman, lw, iaw), with every step ``n * dt``: what LPSE prints under "Time step sizes"."""
    g = parms.get
    wavelength = float(g("laser.wavelength", "0.351"))
    w0 = 2.0 * np.pi * C_UM_PER_PS / wavelength
    wpe = w0 * np.sqrt(n_env)
    combined = g("lw.solver", "spectral").lower() == "combined"
    use_laser = _bool(g("laser.enable"), False)
    use_raman = _bool(g("raman.enable"))
    use_lw = _bool(g("lw.enable"))
    use_iaw = _bool(g("iaw.enable"))
    solvers = {"laser": g("laser.solver", "static").lower(), "raman": g("raman.solver", "static").lower()}
    evolves = {"laser": use_laser and solvers["laser"] != "static", "raman": use_raman and solvers["raman"] != "static"}
    carrier = {"laser": w0, "raman": wpe if combined else w0 - wpe}
    perturbs = {
        "laser": _bool(g("laser.ionAcousticPerturbations.enable")),
        "raman": _bool(g("raman.ionAcousticPerturbations.enable")),
    }

    def p_polarized(cls):
        # LightSolver::getPolarizationState: 2-D and every beam at polarization 0
        if n_dim != 2:
            return False
        n = int(float(g(f"{cls}.nBeams", "0")))
        pols = [float(g(f"{cls}.{b}.polarization", "0")) for b in range(1, n + 1)]
        field = str(g("initialPerturbation.field", "")).lower()
        if _bool(g("initialPerturbation.enable")) and field == ("e0_z" if cls == "laser" else "e1_z"):
            return False
        # a z-component injector file (LightSolver::getPolarizationState)
        if any(re.fullmatch(rf"{cls}\.E_z\.loadInjector\..*\.filename", k) and v for k, v in parms.items()):
            return False
        return all(p == 0.0 for p in pols)

    def light_tstep(cls):
        if not evolves[cls]:
            # a static field with frequency-shifted beams: the injector's sampling of the shortest
            # beat period (LightSolver::computeTimeStep, waveInjector.sampleRate default 20)
            if cls == "laser" and use_laser:
                n_beams = int(float(g("laser.nBeams", "0")))
                shift = max(
                    (abs(float(g(f"laser.{b}.frequencyShift", "0"))) for b in range(1, n_beams + 1)), default=0.0
                )
                if shift > 0.0:
                    rate = int(float(g("laser.waveInjector.sampleRate", "20")))
                    return 2.0 * np.pi / (carrier[cls] * shift) / rate
            return 0.0
        dt_input = float(g(f"{cls}.dt", "0"))
        if dt_input > 0.0:
            return dt_input
        wo = carrier[cls]
        if solvers[cls] == "spectral":
            t_crit = 4.0 * wo * dx**2 / (np.pi * C_UM_PER_PS**2 * n_dim * (1.0 - aa) ** 2)
        else:
            order = int(float(g(f"{cls}.evolution.solverOrder", "2")))
            if order not in (2, 4, 6):
                # LPSE leaves solverOrderFactor unset; the translator reports the order as unsupported
                # and adept runs its default second-order stencil, whose step this is
                order = 2
            p_pol = p_polarized(cls)
            dim_f = 1.0 if n_dim == 1 or p_pol else 2.0
            order_f = {2: 1.0, 4: 4.0 / 3.0, 6: 68.0 / 45.0}[order]
            extra = 1.0
            if n_dim == 2 and p_pol:
                extra = {4: 1.0 / 0.94, 6: 1.0 / 0.89}.get(order, 1.0)
            wpe_max = w0 * np.sqrt((2.0 if perturbs[cls] else 1.0) * n_max)
            t_crit = 1.0 / (dim_f * order_f * extra * (C_UM_PER_PS / dx) ** 2 / wo - (wo**2 - wpe_max**2) / (4.0 * wo))
        return float(g(f"{cls}.evolution.dtFraction", "0.95")) * t_crit

    t = {cls: light_tstep(cls) for cls in ("laser", "raman")}
    t_lw = float(g("lw.spectral.dt", "1e10"))
    if g("iaw.solver", "spectral").lower() == "fd":
        t_iaw = float(g("iaw.fd.dt", "0"))
        if t_iaw <= 0.0:
            te, ti = float(g("physical.Te", "2")), float(g("physical.Ti", "1"))
            z, mi = float(g("physical.Z", "1")), float(g("physical.MiOverMe", "1836"))
            cs = C_UM_PER_PS * np.sqrt((z * te + 3.0 * ti) / (mi * 510.999))
            u_max = 0.0
            if _bool(g("iaw.velocityProfile.enable")):
                u_max = (
                    max(
                        abs(float(g("iaw.velocityProfile.from.speed", "0"))),
                        abs(float(g("iaw.velocityProfile.to.speed", "0"))),
                    )
                    * cs
                )
            h = dx / int(float(g("iaw.fd.superSamples", "2")))
            t_iaw = float(g("iaw.fd.dtFraction", "0.95")) * h / (np.sqrt(n_dim) * cs + u_max)
    else:
        t_iaw = float(g("iaw.spectral.dt", "1e10"))

    # Lpse::computeMicroTimestep
    dt = np.inf
    for cls in ("raman", "laser"):
        if evolves[cls] and t[cls] > 0.0:
            dt = min(dt, t[cls])
    if use_lw and not combined:
        dt = min(dt, t_lw)
    if use_iaw:
        dt = min(dt, t_iaw)
    if not np.isfinite(dt) and use_laser:
        zak = ZakUnits(
            float(g("physical.Te", "2")),
            float(g("physical.Ti", "1")),
            float(g("physical.Z", "1")),
            float(g("physical.MiOverMe", "1836")),
            n_env,
            wavelength,
        )
        dt = t["laser"] if t["laser"] > 0.0 else 0.01 * zak.ps_per_zak
    if not np.isfinite(dt) or dt >= 1e9:
        # every enabled solver kept LPSE's 1e10 placeholder (lw.spectral.dt / iaw.spectral.dt unset)
        raise ValueError("lpse_time_steps: the deck sets no finite solver time step (lw.spectral.dt, iaw.spectral.dt)")
    max_raman_per_laser = int(float(g("laser.maxRamanStepsPerStep", "2")))
    max_laser_per_raman = int(float(g("raman.maxLaserStepsPerStep", "10")))
    lw_max_light = int(float(g("lw.maxLightStepsPerStep", "10")))
    iaw_max_light = int(float(g("iaw.maxLightStepsPerStep", "10")))
    iaw_max_lw = int(float(g("iaw.maxLwStepsPerStep", "2")))
    if (
        evolves["laser"]
        and evolves["raman"]
        and not combined
        and solvers["laser"] != "spectral"
        and solvers["raman"] != "spectral"
    ):
        one_to_one = (max_raman_per_laser == 1) or (use_lw and lw_max_light == 1) or (use_iaw and iaw_max_light == 1)
        if not one_to_one and 2.0 / dt > 3.0 / t["laser"]:
            dt = t["laser"] / np.ceil(t["laser"] / dt)

    # Lpse::setupSolverTimeSteps
    n = {"laser": 1, "raman": 1}
    if t["laser"] > 0.0:
        n["laser"] = int(np.floor(t["laser"] / dt))
    if use_laser and use_raman and t["raman"] > 0.0 and t["laser"] > t["raman"]:
        n["laser"] = min(n["laser"], max_raman_per_laser)
    n["laser"] = max(n["laser"], 1)
    if t["raman"] > 0.0:
        n["raman"] = int(np.floor(t["raman"] * 1.0001 / dt))
    if use_laser and use_raman and t["raman"] > 0.0 and t["laser"] > 0.0 and t["raman"] > t["laser"]:
        n["raman"] = min(n["raman"], max_laser_per_raman)
    n["raman"] = max(n["raman"], 1)
    if combined:
        n_lw = lw_max_light * n["raman"]
    elif use_lw and (use_laser or use_raman):
        n_lw = int(np.floor(min(t_lw * 1.0001 / dt, float(lw_max_light))))
    else:
        n_lw = 1
    n_lw = max(n_lw, 1)
    n_iaw = 1
    if use_iaw:
        if use_laser or use_raman:
            n_iaw = int(np.floor(min(t_iaw * 1.0001 / dt, float(iaw_max_light))))
            if use_lw:
                iaw_step, lw_step = dt * n_iaw, dt * n_lw
                if iaw_step / lw_step > iaw_max_lw:
                    iaw_step *= iaw_max_lw / (iaw_step / lw_step)
                    n_iaw = int(np.floor(iaw_step * 1.0001 / dt))
        elif use_lw:
            n_iaw = int(np.floor(min(t_iaw * 1.0001 / dt, float(iaw_max_lw))))
    n_iaw = max(n_iaw, 1)
    if use_iaw:
        n["laser"], n["raman"], n_lw = min(n["laser"], n_iaw), min(n["raman"], n_iaw), min(n_lw, n_iaw)
    if use_lw:
        n["laser"], n["raman"] = min(n["laser"], n_lw), min(n["raman"], n_lw)
        if use_laser:
            n_lw -= n_lw % n["laser"]
        if use_raman:
            n_lw -= n_lw % n["raman"]
        n_iaw -= n_iaw % n_lw
    if use_laser:
        n_iaw -= n_iaw % n["laser"]
    if use_raman:
        n_iaw -= n_iaw % n["raman"]
    return {
        "dt": float(dt),
        "n": {"laser": n["laser"], "raman": n["raman"], "lw": max(n_lw, 1), "iaw": max(n_iaw, 1)},
        "evolves": evolves,
        "use": {"laser": use_laser, "raman": use_raman, "lw": use_lw, "iaw": use_iaw},
    }


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
    nx = nodes[0]
    ny = nodes[1] if len(nodes) > 1 else 1
    # LPSE's cell is the smallest of the axes' spacings and the box is rescaled to it (h_xyz =
    # min(d_x, d_y), adjustGridCoordinates, ParameterManager.cpp:1260-1272): Lx = h (Nx - 1)
    dx = sizes[0] / (nx - 1)
    if ny > 1 and len(sizes) > 1 and sizes[1] > 0:
        dx = min(dx, sizes[1] / (ny - 1))
    lx = dx * (nx - 1)
    xmax = nx * dx
    if ny > 1:
        half_y = 0.5 * ny * dx
    else:
        half_y = 0.4 * dx  # collapses to ny = 1 in adept
    if len(sizes) > 2 and sizes[2] > 0 and len(nodes) > 2 and nodes[2] > 1:
        report["unsupported"].append("3-D grid (grid.sizes/nodes z): adept envelope-2d is 2-D")

    # ---- time stepping (the steps themselves: lpse_time_steps, below)
    # LPSE defaults: laser.enable false, laser.solver static (ParameterManager.cpp:268, 343-347)
    laser_evolves = g("laser.solver", "static").lower() != "static" and _bool(g("laser.enable"), False)
    raman_on = _bool(g("raman.enable"))
    tmax = float(g("simulation.time.end", "1"))
    sample_period = float(g("simulation.samplePeriod", str(tmax)))

    # ---- anti-aliasing: outer fraction of each axis zeroed
    # LPSE ParameterManager.cpp:237-244: grid.antiAliasing.range is a vector of up to six
    # fractions (+x, -x, +y, -y, +z, -z faces of k-space, Lpse::setupAntiAliasingRange), each
    # missing entry taking the previous one's value; with the key omitted the range is 0.3334
    aa_values = _floats(g("grid.antiAliasing.range", "0.3334"))
    aa_range = aa_values[0] if aa_values else 0.3334
    if len(aa_values) > 1 and any(v != aa_range for v in aa_values[:4]):
        report["unsupported"].append(
            f"grid.antiAliasing.range = {aa_values}: per-face ranges; adept applies {aa_range} on every face"
        )
    for solver_key in ("lw.antiAliasing.range", "iaw.antiAliasing.range"):
        # LwSolver.cpp:159-169 / IawSolver.cpp:302-313: a solver's own range overrides the grid's
        if solver_key in parms and _floats(parms[solver_key])[:1] != [aa_range]:
            report["unsupported"].append(f"{solver_key}: adept has one anti-aliasing mask for every field")

    # ---- density profile
    shape = g("densityProfile.shape", "linear").lower()
    n_min = float(g("densityProfile.NminOverNc", str(n_env)))
    n_max = float(g("densityProfile.NmaxOverNc", str(n_env)))
    # LPSE's node i sits at i h - Lx/2 (ZakharovSolver::backgroundDensityShape, Lx = (N - 1) h), adept's
    # cell i at (i + 1/2) h from xmin = 0: LPSE's box centre is adept's xmax / 2, not Lx / 2 (half a cell)
    centre = xmax / 2.0
    x_min_loc = _floats(g("densityProfile.NminLocation", f"{-lx / 2}"))[0] + centre
    x_max_loc = _floats(g("densityProfile.NmaxLocation", f"{lx / 2}"))[0] + centre
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
        "origin": f"{centre}um",
        # LPSE densityProfile.maxBackgroundDensity: default 1.25 n_c, up to 1000 (ParameterManager.cpp:610, 619)
        "max_density": float(g("densityProfile.maxBackgroundDensity", "1.25")),
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
    # ---- absorbing layers, one per field (LPSE lw.Labc, {laser|raman}.evolution.Labc, iaw.Labc --
    # ParameterManager.cpp:799-813 / 922-936, LightSolver.cpp:1296-1330; each defaults to 0, no
    # layer): adept keeps one width per field (the x-min one) and reports per-side / x-y differences
    def labc(prefix, axis):
        base = g(f"{prefix}Labc", "0")
        lo, hi = float(g(f"{prefix}Labc.min.{axis}", base)), float(g(f"{prefix}Labc.max.{axis}", base))
        if lo != hi:
            report["unsupported"].append(
                f"{prefix}Labc.min.{axis} = {lo} != max.{axis} = {hi}: adept uses one width per field (min)"
            )
        return lo

    layers = {
        "lw": ("lw.", True),
        "laser": ("laser.evolution.", laser_evolves),
        "raman": ("raman.evolution.", raman_on),
        "iaw": ("iaw.", _bool(g("iaw.enable"))),
    }
    widths = {k: (labc(prefix, "x"), labc(prefix, "y") if ny > 1 else 0.0) for k, (prefix, _) in layers.items()}
    active_layers = [k for k, (_, on) in layers.items() if on]
    labc_x = max((widths[k][0] for k in active_layers), default=0.0)
    labc_y = max((widths[k][1] for k in active_layers), default=0.0)
    deepest_layer = labc_x
    boundary = {"x": "absorbing" if labc_x > 0 else "periodic", "y": "absorbing" if labc_y > 0 else "periodic"}
    if ny == 1:
        # a 1-D deck: LPSE's scalar Labc names no y layer; adept's one-cell y axis would sit
        # entirely inside one (test_030: the whole box damped at 0.78 per light sub-step)
        boundary["y"] = "periodic"
    for k in active_layers:
        wx, wy = widths[k]
        if boundary["y"] == "absorbing" and wy != wx:
            report["unsupported"].append(f"{layers[k][0]}Labc x {wx} / y {wy} um: adept uses one width per field (x)")
    boundary_width = widths["lw"][0]

    # ---- time steps: LPSE's own (Lpse::computeMicroTimestep / setupSolverTimeSteps), mapped onto
    # adept's EPW step (grid.dt), light sub-steps per EPW step and IAW stride in EPW steps -- LPSE
    # makes each a multiple of the finer one, so the ratios are exact
    steps = lpse_time_steps(
        parms, dx=dx, n_dim=2 if ny > 1 else 1, n_env=n_env, n_max=lpse_density_max(parms, lx), aa=aa_range
    )
    light_ns = [steps["n"][c] for c in ("laser", "raman") if steps["evolves"][c]]
    if steps["use"]["lw"]:
        outer = steps["n"]["lw"]
    elif steps["use"]["iaw"]:
        outer = steps["n"]["iaw"]
    else:
        outer = max(light_ns, default=1)
    dt = outer * steps["dt"]
    light_substeps = outer // min(light_ns) if light_ns else None
    if len(set(light_ns)) > 1:
        report["notes"].append(
            f"LPSE steps the pump and the Raman light differently ({light_ns} micro steps): adept uses the finer"
        )
    iaw_stride = max(1, steps["n"]["iaw"] // outer)
    # what LPSE prints under "Time step sizes" (fs), for the enabled solvers
    report["lpse_time_steps_fs"] = {
        cls: steps["n"][cls] * steps["dt"] * 1e3 for cls in ("laser", "raman", "lw", "iaw") if steps["use"][cls]
    }
    lam = float(g("lw.abc.lambda", "7"))
    for prefix in ("laser.evolution.", "raman.evolution.", "iaw."):
        if float(g(f"{prefix}abc.lambda", "7")) != lam:
            report["unsupported"].append(f"{prefix}abc.lambda != lw.abc.lambda: adept uses one layer steepness")
    landau_on = _bool(g("lw.landauDamping.enable"), False)  # LPSE default false (ParameterManager.cpp:782-787)
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
    for key in ("lw.noise.isSymmetric", "lw.noise.isAnalytic"):
        if _bool(g(key)):
            report["unsupported"].append(f"{key}: adept draws LPSE's default (random, non-symmetric) noise")
    if float(g("densityProfile.temporalSlope", "0")) != 0.0:
        report["unsupported"].append("densityProfile.temporalSlope: adept's background density is static")
    source = {
        "noise": noise_on,
        "noise_model": "thermal",
        "noise_debye_factor": noise_calculated,
        # isCalculated: LPSE's own calcNoiseAmp_K0 constant (ZakUnits.noise_amp_k0) with the
        # deck amplitude as multiplier; otherwise the plain amplitude converted to adept units
        "noise_calibrate": "lpse" if noise_calculated else False,
        # LPSE lw.noise.maxWavenumber (units of k0, default unbounded; ParameterManager.cpp:767, 1498)
        **({"noise_max_wavenumber": float(g("lw.noise.maxWavenumber"))} if "lw.noise.maxWavenumber" in parms else {}),
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
    if _bool(g("lw.restrictSourceRange.enable")):
        epw_source_window = _source_window(g, "lw.restrictSourceRange")
    else:
        epw_source_window = None
    epw_solver = "combined" if g("lw.solver", "spectral").lower() == "combined" else "separate"
    combined_fd = epw_solver == "combined" and g("raman.solver", "static").lower() == "fd"
    if combined_fd and g("laser.solver", "static").lower() != "fd":
        # LPSE's fd combined path (ZakharovSolver.cpp:232-238, LwSolver::advanceLW_combinedTPDandSRS_FD) is
        # ported with an FD pump (core/fd_combined.py, test_022, test_083); a mixed pair is not
        report["unsupported"].append(
            "lw.solver = combined with raman.solver = fd and a non-fd laser: adept runs the spectral combined solver"
        )
        combined_fd = False
    if g("lw.solver", "spectral").lower() == "fd":
        report["unsupported"].append("lw.solver = fd (LPSE itself disables it); translated as spectral")
    epw = {
        "boundary": boundary,
        **({"source_window": epw_source_window} if epw_source_window else {}),
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
    # LPSE {lw|laser|raman}.interpolateSourcesInTime, default true (ParameterManager.cpp:330, 367, 756)
    epw["interpolate_sources"] = _bool(g("lw.interpolateSourcesInTime"), True)

    # ---- light
    # LPSE raman.solver defaults to static (ParameterManager.cpp:346); a static Raman light with its
    # EPW source (raman.sourceTerm.lw.enable, default true) is refused (LightSolver.cpp:195)
    raman_solver = g("raman.solver", "static").lower()
    laser_solver = g("laser.solver", "static").lower()
    if raman_on and raman_solver == "static" and _bool(g("raman.sourceTerm.lw.enable"), True):
        raise ValueError("raman.enable with raman.solver = static and its lw source term (LPSE refuses)")
    light: dict = {"pump_depletion": laser_evolves}
    # only the evolved fields vote, so a laser-only fd deck (test_016) stays fd
    voters = [s for s, on in ((raman_solver, raman_on), (laser_solver, laser_evolves)) if on]
    if combined_fd:
        light["solver"] = "fd"  # LPSE's FD combined solver (core/fd_combined.py)
    elif epw_solver == "combined" or "spectral" in voters:
        light["solver"] = "spectral"
    else:
        light["solver"] = "fd"
    if laser_evolves and laser_solver != raman_solver and raman_on:
        report["notes"].append(
            f"laser.solver = {laser_solver} but raman.solver = {raman_solver}: "
            f"one adept light solver ({light['solver']})"
        )
    if laser_evolves and _bool(g("laser.evolution.resonanceAbsorption.enable")):
        if light["solver"] != "fd":
            report["unsupported"].append("laser.evolution.resonanceAbsorption with a spectral light solver")
        else:
            ra = {
                "t_start": f"{float(g('laser.evolution.resonanceAbsorption.time.start', '0'))}ps",
                "filter": _bool(g("laser.evolution.resonanceAbsorption.filter.enable"), True),
                "filter_width": float(g("laser.evolution.resonanceAbsorption.filter.width", "1")),
                "landau_update": int(float(g("laser.evolution.resonanceAbsorption.LdUpdate", "1"))),
            }
            if "laser.evolution.resonanceAbsorption.time.stop" in parms:
                ra["t_stop"] = f"{float(parms['laser.evolution.resonanceAbsorption.time.stop'])}ps"
            light["resonance_absorption"] = ra
    if raman_on and _bool(g("raman.evolution.resonanceAbsorption.enable")):
        report["unsupported"].append("raman.evolution.resonanceAbsorption (adept applies RA to the pump only)")
    if light["solver"] == "fd":
        # evolution.solverOrder (default 2 in LPSE for both fields): one stencil order here
        orders = {}
        for field, active in (("laser", laser_evolves), ("raman", raman_on)):
            key = f"{field}.evolution.solverOrder"
            if active and key in parms:
                orders[field] = int(float(parms[key]))
        if orders:
            order = max(orders.values())
            if order not in (2, 4, 6):
                report["unsupported"].append(f"evolution.solverOrder = {order} (2, 4 or 6)")
            else:
                light["fd_order"] = order
                if len(set(orders.values())) > 1:
                    report["notes"].append(
                        f"laser/raman evolution.solverOrder differ ({orders}): one adept stencil order ({order})"
                    )
    # LPSE {laser|raman}.evolution.absorption (LightSolver.cpp:1349-1350, default 0; negative =
    # calculated): each class's own rate at its own critical density
    for key, target in (
        ("laser.evolution.absorption", "absorption"),
        ("raman.evolution.absorption", "raman_absorption"),
    ):
        rate = float(g(key, "0"))
        light[target] = True if rate < 0 else (rate if rate > 0 else False)
    light["boundary_width"] = f"{widths['laser'][0]}um"
    light["raman_boundary_width"] = f"{widths['raman'][0]}um"
    light["boundary_max_rate"] = float(g("laser.evolution.abc.maxDampingRate", "5000"))
    light["raman_boundary_max_rate"] = float(g("raman.evolution.abc.maxDampingRate", "5000"))
    if "raman.spectral.maxWavenumber" in parms:
        light["max_wavenumber"] = float(parms["raman.spectral.maxWavenumber"])
    abc_type = str(g("laser.evolution.abc.type", g("raman.evolution.abc.type", "exp"))).lower()
    if abc_type == "pml":
        if light["solver"] == "fd":
            light["absorber"] = "pml"
            # LPSE reads the PML denominator as laser.SabcDenom for both light classes
            # (LightSolver.cpp:938-945, default 5, range 2-10)
            if "laser.SabcDenom" in parms:
                light["pml_denominator"] = float(parms["laser.SabcDenom"])
            if "laser.evolution.abc.SabcDenom" in parms:
                report["unsupported"].append("laser.evolution.abc.SabcDenom: not an LPSE key (laser.SabcDenom)")
        else:
            report["notes"].append("abc.type = pml with a spectral light solver: adept keeps the exp layer")
    elif abc_type not in ("exp",):
        report["unsupported"].append(f"abc.type = {abc_type} (exp or pml)")
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
    # LPSE zeroes every source across the injector rows unless told otherwise; in the
    # absorbing layers only on request (plan 2 I.3)
    light["suppress_sources_at_injectors"] = _bool(g("suppressSourcesAtInjectors", "true"))
    light["suppress_sources_in_absorbers"] = _bool(g("suppressSourcesInAbsorbingRegions"))
    # one flag for both light fields; only the evolved ones count (LPSE turns it off for a static field)
    interp = {
        cls: _bool(g(f"{cls}.interpolateSourcesInTime"), True)
        for cls, on in (("laser", laser_evolves), ("raman", raman_on and raman_solver != "static"))
        if on
    }
    if interp:
        light["interpolate_sources"] = interp.get("raman", interp.get("laser"))
        if len(set(interp.values())) > 1:
            report["notes"].append(
                f"interpolateSourcesInTime differs between the light classes {interp}: adept uses the Raman light's"
            )

    # ---- laser beams (static: intensities summed; direction along +x expected)
    n_beams = int(float(g("laser.nBeams", "1")))
    intensity = 0.0
    angle_deg = 0.0
    beams = []
    for b in range(1, n_beams + 1):
        i_b = float(g(f"laser.{b}.intensity", "0"))
        intensity += i_b
        direction = _floats(g(f"laser.{b}.direction", "1 0 0"))
        dz = direction[2] if len(direction) > 2 else 0.0
        dy = direction[1] if len(direction) > 1 else 0.0
        # in-plane beams from the x-min (dx > 0) or x-max (dx < 0) face; a y-face beam
        # (evolution.source = min.y / max.y, or dx = 0) and any out-of-plane direction are
        # not supported (plan 2 L.4b)
        source_face = str(g(f"laser.{b}.evolution.source", "min.x")).lower()
        if abs(dz) > 1e-6 * np.linalg.norm(direction) or direction[0] == 0 or source_face.endswith(".y"):
            report["unsupported"].append(
                f"laser.{b}.direction {direction} / evolution.source = {source_face}: "
                "adept pump beams enter from the x faces"
            )
        beam_angle = float(np.degrees(np.arctan2(dy, direction[0])))
        if b == 1:
            angle_deg = beam_angle
        beams.append(
            {
                "intensity": i_b,
                "angle": beam_angle,
                # LPSE laser.N.phase is in degrees (LightSolver.cpp:1716-1717)
                "phase": float(np.deg2rad(float(g(f"laser.{b}.phase", "0")))),
                "delta_omega": float(g(f"laser.{b}.frequencyShift", "0")),
                # LPSE rotateBeam: degrees about the beam axis, 0 in-plane (p), 90 along z (s)
                "polarization": float(g(f"laser.{b}.polarization", "0")),
            }
        )
    polarization_deg = beams[0]["polarization"] if beams else 0.0
    beam_extras = {}

    def beam_profile(b):
        # LPSE laser.N.evolution.{width, sgOrder | sgPower, offset} (LightSolver.cpp:1754-1762;
        # SchrodingerSolver3::superGaussian: exp(-(r / width)^sgOrder)); width defaults to 0 (flat),
        # sgOrder to 4 (LightSolver.cpp:1700-1704), and sgPower, read second, overrides it. LPSE reads
        # no bare laser.N.width / sgOrder / offset, so neither does the translator
        for bare in ("width", "sgOrder", "sgPower", "offset"):
            if f"laser.{b}.{bare}" in parms:
                report["unsupported"].append(
                    f"laser.{b}.{bare}: not an LPSE key (LPSE reads laser.{b}.evolution.{bare})"
                )
        width = float(g(f"laser.{b}.evolution.width", "0"))
        order = float(g(f"laser.{b}.evolution.sgPower", g(f"laser.{b}.evolution.sgOrder", "4")))
        off = _floats(g(f"laser.{b}.evolution.offset", "0 0"))
        return width, order, (off[1] if len(off) > 1 else 0.0)

    width1, order1, offset1 = beam_profile(1)
    # a zero width or order is LPSE's flat (periodic-injection) beam: no transverse profile
    if width1 > 0 and order1 != 0:
        # adept's beam_width is the Gaussian standard deviation, exp(-(y^2 / (2 s^2))^(n/2)) =
        # exp(-(|y| / (sqrt(2) s))^n): s = width / sqrt(2) reproduces LPSE's profile for any n
        beam_extras["beam_width"] = f"{width1 / np.sqrt(2.0)}um"
        beam_extras["beam_sg_order"] = order1
        beam_extras["beam_offset"] = f"{offset1}um"
        for b in range(2, n_beams + 1):
            if beam_profile(b) != (width1, order1, offset1):
                report["notes"].append(
                    f"laser.{b}.evolution.width/sgOrder/offset differ from beam 1: one adept profile"
                )
                break
    # LPSE KAP bandwidth is per beam, laser.N.bandwidth.KAP.frequency (dW/W0, LightSolver.cpp:1743);
    # each beam its own group unless laser.N.group joins them (LightSolver.cpp:1628)
    if "laser.bandwidth.KAP.frequency" in parms:
        report["unsupported"].append("laser.bandwidth.KAP.frequency: not an LPSE key (laser.N.bandwidth.KAP.frequency)")
    kaps = [float(g(f"laser.{b}.bandwidth.KAP.frequency", "0")) for b in range(1, n_beams + 1)]
    if kaps and max(kaps) > 0:
        beam_extras["kap_bandwidth"] = kaps[0]
        if len(set(kaps)) > 1:
            report["unsupported"].append("per-beam KAP bandwidths differ: adept uses beam 1's for every beam")
        groups = [g(f"laser.{b}.group") for b in range(1, n_beams + 1) if g(f"laser.{b}.group") is not None]
        if len(groups) != len(set(groups)):
            report["unsupported"].append(
                "laser.N.group: beams sharing a KAP group jump together in LPSE; adept's are independent"
            )
    # LPSE laser.pulseShape.{enable, shape, file, period, dutyCycle} (LightSolver.cpp:1175-1214):
    # a power factor, shape file (default; two columns t_ps scale, LightSolver::loadPulseShapingData,
    # resolved relative to the deck's directory), square or sin; period in ps
    if _bool(g("laser.pulseShape.enable")):
        pulse_shape = g("laser.pulseShape.shape", "file").strip().lower()
        if pulse_shape == "file":
            if not g("laser.pulseShape.file"):
                raise ValueError("laser.pulseShape.shape = file needs laser.pulseShape.file (LPSE refuses the deck)")
            pulse_path = Path(g("laser.pulseShape.file"))
            if not pulse_path.is_absolute() and parms.get("_deck_dir"):
                pulse_path = Path(parms["_deck_dir"]) / pulse_path
            beam_extras["pulse_file"] = str(pulse_path)
        elif pulse_shape in ("square", "sin"):
            beam_extras["pulse_shape"] = pulse_shape
            beam_extras["pulse_period"] = f"{float(g('laser.pulseShape.period', '0.1'))}ps"
            beam_extras["pulse_duty_cycle"] = float(g("laser.pulseShape.dutyCycle", "0.5"))
        else:
            raise ValueError(f"laser.pulseShape.shape = {pulse_shape}: LPSE takes file, square or sin")
    if _bool(g("raman.pulseShape.enable")):
        report["unsupported"].append("raman.pulseShape: adept's Raman seed has no pulse shape")
    drivers = {
        "E0": {
            "shape": "uniform",
            "num_colors": 1,
            "delta_omega_max": 0.0,
            "params": {"phases": {"seed": 42}},
            "angle": angle_deg,
            "polarization": polarization_deg,
            # the static field's swelling: LPSE's constant (1 - n_env)^(-1/4) unless
            # laser.static.useSpatiallyVaryingFieldSwelling (LightSolver.cpp:1046-1053)
            "swelling": "local" if _bool(g("laser.static.useSpatiallyVaryingFieldSwelling")) else "constant",
            **({"beams": beams} if n_beams > 1 else {}),
            **beam_extras,
            # LPSE has no pump envelope (an evolved pump ramps with laser.evolution.riseTime only, a
            # static one is on from t = 0): the window opens 0.1 ps before t = 0, where its 10 fs
            # tanh rise is complete (at t = 0 it was the half-way point, a ~1 % deficit in the
            # first 0.1 ps of test_019)
            "envelope": {
                "tw": f"{10 * tmax + 0.2}ps",
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

    def injector_cells(prefix, side):
        """LPSE's injector plane, ``int(Labc / h) + int(Loff / h)`` nodes in from the face
        (SchrodingerSolver3::setAbcParameters / completeBoundaryConditions)."""
        width = float(g(f"{prefix}Labc.{side}.x", g(f"{prefix}Labc", "0")))
        loff = float(g(f"{prefix}Loff.{side}.x", g(f"{prefix}Loff", "0")))
        for name, v in (("Labc", width), ("Loff", loff)):
            if v > 0 and abs(v / dx - round(v / dx)) < 1e-4:
                report["notes"].append(
                    f"{prefix}{name} / h = {v / dx:.6f}: LPSE's single-precision index may differ by one"
                )
        return int(width / dx) + int(loff / dx)

    # adept's drivers.E*.offset names the plane's last scattered-field row: the row before LPSE's
    # first injected node from x-min, the first injected node itself from x-max (cell-centred x)
    leftward_only = bool(beams) and all(abs(float(b["angle"])) > 90.0 for b in beams)
    n_pump = injector_cells("laser.evolution.", "max" if leftward_only else "min") if laser_evolves else 0
    if laser_evolves:
        drivers["E0"]["offset"] = f"{(n_pump + 0.5 if leftward_only else n_pump - 0.5) * dx}um"
        if not leftward_only and any(abs(float(b["angle"])) > 90.0 for b in beams):
            report["notes"].append("beams from both x faces: adept places the x-max plane one cell off LPSE's")
        # LPSE laser.evolution.riseTime (fs, default 30): the 1 - exp(-(t/rise)^2) ramp on every
        # pump source (SchrodingerSolver3::addInjectorSources)
        drivers["E0"]["turn_on_time"] = f"{float(g('laser.evolution.riseTime', '30'))}fs"

    # ---- pump from injector files (LPSE laser.E_<c>.loadInjector.<side>.<axis>.filename; plan 2 L.4c)
    injector_keys = {
        k: v
        for k, v in parms.items()
        if re.fullmatch(r"(laser|raman)\.E_[xyz]\.loadInjector\.(min|max)\.[xyz]\.filename", k) and v
    }
    if injector_keys:
        faces = set()
        files = {}
        for key, value in injector_keys.items():
            field, comp, _, side, axis, _ = key.split(".")
            if field != "laser" or axis != "x":
                report["unsupported"].append(f"{key}: adept loads pump (laser) injectors on the x faces only")
                continue
            path = Path(value)
            if not path.is_absolute() and parms.get("_deck_dir"):
                path = Path(parms["_deck_dir"]) / path
            faces.add(side)
            files[comp[-1]] = str(path)
        if len(faces) > 1:
            report["unsupported"].append("laser loadInjector files on both x faces: adept injects from one face")
        elif files:
            side = faces.pop()
            if not laser_evolves:
                report["unsupported"].append("laser loadInjector files need laser.solver = fd (LPSE refuses otherwise)")
            if n_beams > 1 or "laser.1.intensity" in parms:
                report["unsupported"].append("laser loadInjector files together with laser beams (LPSE refuses it)")
            # LPSE's primary injector node is the edge of the light absorber, int(Labc / h) nodes in
            # from the face (SchrodingerSolver3::completeBoundaryConditions); the file's first plane
            # sits on it. adept's drivers.E0.offset names the plane's last scattered-field row
            # (rightward: the row before the first injected row; leftward: the first injected row).
            n_abc = injector_cells("laser.evolution.", side)
            offset_cells = n_abc - 0.5 if side == "min" else n_abc + 0.5
            drivers["E0"]["offset"] = f"{offset_cells * dx}um"
            drivers["E0"]["injector_file"] = {"side": f"{side}.x", "files": files}
            # the nominal intensity (units.laser intensity, the flux metrics' normalisation) is the
            # peak |E|^2 of the files, e E / (m_e w0 c) -> W/cm^2 as electronOscillationVelocity.m
            peak = 0.0
            for path in files.values():
                try:
                    _, planes = read_injector_file(path, ny)
                    peak = max(peak, float(np.max(np.abs(planes[:, 0]))))
                except (OSError, ValueError) as err:
                    report["notes"].append(f"injector file {path} not read at translation: {err}")
            w0 = 2.0 * np.pi * C_CGS / (wavelength_um * 1e-4)
            intensity = (peak * ME_CGS * w0 * C_CGS / QE_CGS) ** 2 * C_CGS / (8.0 * np.pi) * 1e-7
            drivers["E0"].pop("beams", None)
    if raman_on and epw_solver != "combined" and not _bool(g("raman.sourceTerm.lw.enable"), True):
        report["unsupported"].append(
            "raman.sourceTerm.lw.enable = false: adept's Raman light is always driven by the EPW"
        )
    n_seed = 0
    if raman_on and int(float(g("raman.nBeams", "0"))) > 0:
        # the seed from x-max at LPSE's plane, with raman.evolution.riseTime (default 30 fs,
        # LightSolver.cpp:1288) as its turn-on
        if str(g("raman.1.evolution.source", "max.x")).lower() != "max.x":
            report["unsupported"].append("raman.1.evolution.source != max.x: adept's seed enters from x-max")
        n_seed = injector_cells("raman.evolution.", "max")
        drivers["E1"] = {
            "intensity": f"{float(g('raman.1.intensity', '0'))}W/cm^2",
            "polarization": float(g("raman.1.polarization", "0")),
            "offset": f"{(n_seed + 0.5) * dx}um",
            "turn_on_time": f"{float(g('raman.evolution.riseTime', '30'))}fs",
        }

    # ---- IAW
    iaw = None
    if _bool(g("iaw.enable")):
        iaw_solver = "spectral" if g("iaw.solver", "spectral").lower() == "spectral" else "fd"
        iaw = {
            "active": True,
            "solver": iaw_solver,
            "damping": {
                "landau": float(g("iaw.landauDampingRate", g("iaw.dampingRate", "0"))),
                "collisions": float(g("iaw.collisionalDampingRate", "0")),
                # the full Krall-Trivelpiece rate only on the fd path (iaw.fd.damping.isSimplified;
                # the spectral step always uses nu |k|, ZakharovSolver.cpp:2083)
                "landau_form": "full"
                if iaw_solver == "fd" and not _bool(g("iaw.fd.damping.isSimplified"), True)
                else "simplified",
            },
            "max_density_perturbation": float(g("iaw.amplitudeClamp", "0.9")),
            # LPSE iaw.maxWavenumber (IawSolver.cpp:232-234; default none), separate from lw.maxWavenumber
            "max_wavenumber": float(parms["iaw.maxWavenumber"]) if "iaw.maxWavenumber" in parms else None,
            "noise": _bool(g("iaw.noise.enable")),
            # LPSE iaw.Labc (default 0: no IAW layer) at iaw.abc.maxDampingRate (default 100)
            "boundary_width": f"{widths['iaw'][0]}um",
            "boundary_max_rate": float(g("iaw.abc.maxDampingRate", "100")),
        }
        if iaw_solver == "fd":
            # iaw.fd.* (IawSolver::readParameters; plan 2 I.1)
            iaw["super_samples"] = int(float(g("iaw.fd.superSamples", "2")))
            iaw["dt_fraction"] = float(g("iaw.fd.dtFraction", "0.95"))
            iaw["landau_update"] = int(float(g("iaw.fd.numStepsPerLandauDampingUpdate", "1")))
            if str(g("iaw.fd.advectionSolver", "ppm")).lower() not in ("ppm",):
                report["notes"].append("iaw.fd.advectionSolver != ppm: adept's fd IAW solver advects with PPM")
        # iaw.velocityProfile.* (plan 2 I.2): the plasma-flow profile of the fd solver
        if _bool(g("iaw.velocityProfile.enable")):
            shape = str(g("iaw.velocityProfile.shape", "linear")).lower()
            if iaw_solver != "fd":
                # LPSE's spectral IAW takes the mean flow and needs from.speed == to.speed
                # (IawSolver.cpp:340-356)
                v_from = float(g("iaw.velocityProfile.from.speed", "0"))
                v_to = float(g("iaw.velocityProfile.to.speed", "0"))
                if shape == "file" or v_from != v_to:
                    raise ValueError(
                        "iaw.velocityProfile with spectral IAWs needs from.speed == to.speed (LPSE refuses)"
                    )
                a = _floats(g("iaw.velocityProfile.from.location", "0 0"))
                b = _floats(g("iaw.velocityProfile.to.location", "1 0"))
                d = np.array([b[0] - a[0], (b[1] if len(b) > 1 else 0.0) - (a[1] if len(a) > 1 else 0.0)])
                d = d / np.linalg.norm(d)
                iaw["flow"] = [float(d[0] * v_from), float(d[1] * v_from)]
            elif shape == "file":
                report["unsupported"].append(
                    "iaw.velocityProfile.shape = file (convert the LPSE binary to an .npz with ux, uy)"
                )
            else:
                from_loc = _floats(g("iaw.velocityProfile.from.location", "0 0"))
                to_loc = _floats(g("iaw.velocityProfile.to.location", "1 0"))
                iaw["flow"] = {
                    "shape": shape,
                    "geometry": str(g("iaw.velocityProfile.geometry", "cartesian")).lower(),
                    "from_location": [f"{from_loc[0]}um", f"{from_loc[1] if len(from_loc) > 1 else 0.0}um"],
                    "to_location": [f"{to_loc[0]}um", f"{to_loc[1] if len(to_loc) > 1 else 0.0}um"],
                    "from_mach": float(g("iaw.velocityProfile.from.speed", "0")),
                    "to_mach": float(g("iaw.velocityProfile.to.speed", "0")),
                    "sg_order": float(
                        g(
                            "iaw.velocityProfile.sgOrder",
                            g("iaw.velocityProfile.sgPower", g("iaw.velocityProfile.power", "2")),
                        )
                    ),
                    "temporal_slope": float(g("iaw.velocityProfile.temporalSlope", "0")),
                }
        if iaw_stride > 1:
            iaw["stride"] = iaw_stride  # LPSE's IAW step in EPW steps (lpse_time_steps)
        if _bool(g("iaw.restrictSourceRange.enable")):
            iaw["source_window"] = _source_window(g, "iaw.restrictSourceRange")
        if g("iaw.startEvolvingTime") is not None:
            iaw["t_start"] = float(g("iaw.startEvolvingTime"))
        if g("iaw.stopEvolvingTime") is not None:
            iaw["t_stop"] = float(g("iaw.stopEvolvingTime"))
        tf = {w: _bool(g(f"thermalFil.{w}.enable")) for w in ("laser", "raman", "lw")}
        if any(tf.values()):
            tf["nonlocal"] = _bool(g("thermalFil.isNonlocal"))
            tf["conductivity_multiplier"] = float(g("thermalFil.conductivityMultiplier", "1"))
            iaw["thermal_filamentation"] = tf
            report["notes"].append(
                "thermalFil: LPSE's source form with adept's own normalization (heat + momentum equations)"
            )
        # which waves see the IAW density and which drive it (LPSE defaults all six off:
        # {lw|laser|raman}.ionAcousticPerturbations.enable, ParameterManager.cpp:333 / 370 / 734;
        # iaw.sourceTerm.{lw|laser|raman}.enable, IawSolver.cpp:203-209)
        pairs = (("epw", "lw"), ("pump", "laser"), ("raman", "raman"))
        perturbs = {w: _bool(g(f"{p}.ionAcousticPerturbations.enable")) for w, p in pairs}
        drive = {w: _bool(g(f"iaw.sourceTerm.{p}.enable")) for w, p in pairs}
        if epw_solver == "combined" and perturbs["epw"] != perturbs["raman"]:
            raise ValueError("lw/raman.ionAcousticPerturbations.enable differ with lw.solver = combined (LPSE refuses)")
        if epw_solver == "combined" and drive["epw"] != drive["raman"]:
            raise ValueError("iaw.sourceTerm.lw/raman.enable differ with lw.solver = combined (LPSE refuses)")
        if perturbs["pump"] and not laser_evolves and _bool(g("laser.enable")):
            raise ValueError("laser.ionAcousticPerturbations.enable with a static laser (LPSE refuses)")
        if not any(drive.values()) and not any(v for k, v in tf.items() if k in ("laser", "raman", "lw")):
            report["notes"].append("iaw: no iaw.sourceTerm.* enabled -- LPSE's IAW is driven by noise only")
        iaw["perturbs"], iaw["drive"] = perturbs, drive
        if "fluid.velocity" in parms:
            report["unsupported"].append("fluid.velocity: not an LPSE deck key (LPSE sets it from iaw.velocityProfile)")
            vel = _floats(parms["fluid.velocity"])
            iaw["flow"] = [vel[0], vel[1] if len(vel) > 1 else 0.0]

    # ---- HPE
    hpe = None
    if _bool(g("hpe.enable")):
        # LPSE hpe.landauDampingEvolution.enable (ParameterManager.cpp:272-279, default false): the
        # particles' histogram replaces the Maxwellian Landau rate only when it is on
        # (ElectronTracker.cu:335 makeLandauDamping); off, the particles are diagnostics
        hpe = {"active": True, "feedback": _bool(g("hpe.landauDampingEvolution.enable"), False)}
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
        # the tracked tail: LPSE Vmin = VminOverVminPhase * wpe / (2 pi / h), default 0 -- the whole
        # Maxwellian (ElectronTracker.cu:3082-3083, 3375-3382); adept's v_min is in vte
        ratio = float(g("hpe.VminOverVminPhase", "0"))
        hpe["v_min"] = float(ratio * dx / (2.0 * np.pi * zak.debye_length_cm * 1.0e4))
        # LPSE thermalizationProbability default (1, 0, 0): x walls thermalize, y wraps (ElectronTracker.cu:3014)
        tp = _floats(g("hpe.thermalizationProbability", "1 0 0"))
        hpe["thermalization_probability"] = [tp[0], tp[1] if len(tp) > 1 else 0.0]
        if "hpe.magneticField" in parms:
            b = _floats(parms["hpe.magneticField"])
            b = (list(b) + [0.0, 0.0, 0.0])[:3]
            # B_z alone keeps the (p_x, p_y) tracker; any in-plane component adds p_z (plan 2 F.4)
            hpe["magnetic_field"] = b if any(abs(v) > 0 for v in b[:2]) else b[2]
        # LPSE hpe.gammaLimit.{damping, growth} are ZAK rates, default 1e6 -- effectively no limit
        # (ElectronTracker.cu:3038-3043); converted to 1/ps
        hpe["gamma_limit_damping"] = float(float(g("hpe.gammaLimit.damping", "1e6")) * zak.zak_per_ps)
        hpe["gamma_limit_growth"] = float(float(g("hpe.gammaLimit.growth", "1e6")) * zak.zak_per_ps)
        if _bool(g("hpe.allowGrowth")):
            hpe["allow_growth"] = True
        if _bool(g("hpe.enforceEnergyConservation")):
            hpe["energy_conservation"] = True
            hpe["energy_conservation_steps"] = float(g("hpe.numStepsToAverageEnergyChange", "10"))  # LPSE default
        n_flux = int(float(g("hpe.metrics.nFluxMetrics", "0")))
        if n_flux > 0:
            # LPSE hpe.metrics.fluxMetric.N.energy.{min,max} in keV (the decks bin a 2 keV plasma at
            # 0.01, 0.5, 1, 2, 4, 6, 8, 20); the adept instrument bins on one sorted edge list (keV)
            edges = set()
            for i in range(1, n_flux + 1):
                for end, default in (("min", "0"), ("max", "1e9")):
                    key = f"hpe.metrics.fluxMetric.{i}.energy.{end}"
                    key = key if key in parms else f"hpe.metrics.flux.{i}.energy.{end}"
                    edges.add(float(g(key, default)))
            hpe["flux_bins"] = sorted(edges)
        if int(float(g("hpe.metrics.nPowerMetrics", "0"))) > 0:
            prefix = "hpe.metrics.powerMetric.1"
            if f"{prefix}.angle" not in parms:
                prefix = "hpe.metrics.power.1"
            hpe["cone_angle"] = float(g(f"{prefix}.angle", "30"))
            direction = _floats(g(f"{prefix}.direction", "1 0 0"))
            hpe["cone_direction"] = [direction[0], direction[1] if len(direction) > 1 else 0.0]
        report["notes"].append(
            "hpe: dt/dtFields/stepsPerLandauUpdate/vdf.*/blend.gamma are LPSE tracker internals without an adept "
            "equivalent (adept sub-cycles at substep_courant and blends with v_blend_buffer)"
        )

    # ---- qle (QuasilinearEvolution.cpp; plan 2 K.1)
    qle = None
    if _bool(g("qle.enable")):
        p_therm = _floats(g("qle.thermalizationProbability", "1 0 0"))
        qle = {
            "active": True,
            "nv": int(float(g("qle.velocityGrid", "100"))),
            "v_max": float(g("qle.VmaxOverC", "0.5")),
            "update_every": int(float(g("qle.numLwStepsPerUpdate", "1"))),
            "t_start": float(g("qle.startEvolutionAt", "0")),
            "thermal_correction": _bool(g("qle.includeThermalCorrection"), True),
            "derivative_in_tensor": _bool(g("qle.includeDerivativeInDiffusionTensor"), True),
            "landau_evolution": _bool(g("qle.landauDampingEvolution.enable")),
            "thermalization_probability": [p_therm[0], p_therm[1] if len(p_therm) > 1 else 0.0],
            "subcycling": int(float(g("qle.additionalSubcycling", "1"))),
            "multiplier": float(g("qle.coefficientMultiplier", "1")),
        }
        if str(g("qle.solver", "spectral")).lower() == "implicit":
            qle["solver"] = "implicit"
        if hpe is not None:
            report["unsupported"].append("qle with hpe (LPSE refuses the pair as well)")
            qle = None

    # ---- initialPerturbation (InitialPerturbation.cpp): plane wave in one field at t = 0
    initial_perturbation = None
    if _bool(g("initialPerturbation.enable")):
        ip_field = str(g("initialPerturbation.field", "pots")).lower()
        ip_type = str(g("initialPerturbation.type", "planewave")).lower()
        if ip_type != "planewave":
            report["unsupported"].append(f"initialPerturbation.type = {ip_type} (only planewave)")
        if ip_field == "pots":
            target, component = "epw", "y"
        elif ip_field[:2] in ("e0", "e1") and ip_field[-1] in "xyz":
            target, component = ip_field[:2].upper(), ip_field[-1]
        else:
            target, component = "epw", "y"
            report["unsupported"].append(f"initialPerturbation.field = {ip_field}")
        direction = _floats(g("initialPerturbation.direction", "1 0 0"))
        size = _floats(g("initialPerturbation.envelopeSize", "0 0 0"))
        offset = _floats(g("initialPerturbation.envelopeOffset", "0 0 0"))
        initial_perturbation = {
            "field": target,
            "component": component,
            "amplitude": float(g("initialPerturbation.amplitude", "1")),
            "wavelength": f"{float(g('initialPerturbation.wavelength', '1'))}um",
            "direction": direction[:2],
            "envelope_size": [f"{v}um" for v in (list(size) + [0.0, 0.0])[:2]],
            "envelope_offset": [f"{v}um" for v in (list(offset) + [0.0, 0.0])[:2]],
            "envelope_sg_order": float(g("initialPerturbation.envelopeSgOrder", "4")),
        }

    cfg = {
        "solver": "envelope-2d",
        "units": {
            "atomic number": mi_over_me / 1836.15,
            "envelope density": n_env,
            "ionization state": z,  # LPSE's Z is a float (ParameterManager.cpp:487)
            "laser intensity": f"{intensity:.6g}W/cm^2",
            "laser_wavelength": f"{wavelength_um}um",
            "reference electron temperature": f"{te}keV",
            "reference ion temperature": f"{ti}keV",
        },
        "density": density,
        "grid": {
            "boundary_abs_coeff": 200.0,
            "boundary_width": f"{boundary_width}um",
            "smooth_fft_size": False,  # LPSE's grid.nodes, not a 5-smooth size
            "boundary_profile": "exp",
            "boundary_max_rate": float(g("lw.abc.maxDampingRate", "200")),
            "boundary_lambda": lam,
            # the flux probes 4 cells inside the deepest injector plane and every active layer
            "probe_offset": f"{max(max(n_pump, n_seed) * dx, deepest_layer) + 4 * dx}um",
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
    light_spectrum = _light_spectrum_probes(parms, g, report, laser_evolves, raman_on)
    if light_spectrum:
        cfg["save"]["light_spectrum"] = light_spectrum
    if iaw is not None:
        cfg["terms"]["iaw"] = iaw
    if hpe is not None:
        cfg["terms"]["hpe"] = hpe
    if qle is not None:
        cfg["terms"]["qle"] = qle
    if initial_perturbation is not None:
        cfg["initial_perturbation"] = initial_perturbation
    if _bool(g("absoluteThreshold.isFind")):
        # the search's parameters (threshold.find_threshold_lpse); the deck's intensity is its start
        noise_range = _floats(g("absoluteThreshold.noiseTimeRange", "0.01 0.1"))
        cfg["threshold"] = {
            "gain": float(g("absoluteThreshold.gain", "23.026")),
            "dI_fract": float(g("absoluteThreshold.dI_fract", "0.3333")),
            "n_iter": int(float(g("absoluteThreshold.numIterations", "5"))),
            "noise_time_range": [noise_range[0], noise_range[1] if len(noise_range) > 1 else 0.1],
            # LPSE's statistic: max |rho| with a Langmuir-wave solver, max |Nelf| without one
            "statistic": "max_rho" if _bool(g("lw.enable")) else "max_nelf",
        }
        report["notes"].append(
            "absoluteThreshold.isFind: run adept._lpse2d.threshold.find_threshold_lpse on this config"
        )
    for item in report["unsupported"]:
        print(f"lpse_deck: not translated -- {item}")
    return cfg, report


# ---------------------------------------------------------------- outputs --


def _light_spectrum_probes(parms: dict, g, report: dict, laser_evolves: bool, raman_on: bool) -> list[dict]:
    """``spectrum.N.{laser|raman}.*`` (``LightSpectrum::readParameters``; ``laserSpectrum.nLaserSpectrum``
    / ``ramanSpectrum.nRamanSpectrum`` give the counts) -> ``save.light_spectrum`` entries: the
    field (E0 / E1), ``startTime`` -> ``tmin``, ``interval``, ``location.min / .max`` (um from
    the box centre) -> ``x`` / ``y``, any ``file.S0.*`` -> ``poynting``. A probe with no output
    file, or on a field that is not evolved, is disabled in LPSE and skipped here."""
    probes = []
    for kind, field, active in (("laser", "E0", laser_evolves), ("raman", "E1", raman_on)):
        count = int(float(g(f"{kind}Spectrum.n{kind.capitalize()}Spectrum", "0")))
        for n in range(1, count + 1):
            prefix = f"spectrum.{n}.{kind}"
            if not _bool(g(f"{prefix}.enable"), True):
                continue
            has_e = any(k.startswith(f"{prefix}.file.E0.") for k in parms)
            has_s = any(k.startswith(f"{prefix}.file.S0.") for k in parms)
            if not (has_e or has_s):
                continue
            if not active:
                report["notes"].append(f"{prefix}: the {kind} field is not evolved; probe skipped")
                continue
            lo = _floats(g(f"{prefix}.location.min", "0"))
            hi = _floats(g(f"{prefix}.location.max", "0"))
            probe = {
                "field": field,
                "interval": f"{float(g(f'{prefix}.interval', '0.05'))}ps",
                "tmin": f"{float(g(f'{prefix}.startTime', '0'))}ps",
                "x": [f"{lo[0]}um", f"{hi[0]}um"],
                "poynting": has_s,
            }
            if len(lo) > 1 and len(hi) > 1:
                probe["y"] = [f"{lo[1]}um", f"{hi[1]}um"]
            probes.append(probe)
    return probes


def _source_window(g, prefix: str) -> dict:
    """``<prefix>.{width, center, edgeWidth}`` (um, from the box centre) -> ``source_window``."""
    width = _floats(g(f"{prefix}.width", "0 0 0"))
    center = _floats(g(f"{prefix}.center", "0 0 0"))
    return {
        "width": [f"{v}um" for v in (list(width) + [0.0, 0.0])[:2]],
        "center": [f"{v}um" for v in (list(center) + [0.0, 0.0])[:2]],
        "edge_width": f"{float(g(f'{prefix}.edgeWidth', '0'))}um",
    }


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
    array; ``Nz > 1`` gives ``(Nx, Ny, Nz)``). Header values are parsed to float where
    possible.

    The values are what LPSE wrote: ``lw.save.pots`` frames are scaled by
    ``potentialNormalizationFactor`` (``saveInNormalizedUnits`` is forced on), so a potential
    frame holds ``e phi / (m_e c^2)``; divide by ``ZakUnits.potential_normalization_factor``
    for the ZAK potential (``rho`` frames: ``rho_normalization_factor``).

    LPSE writes the field in C order ``(Nx, Ny, Nz)`` — the last index varies fastest — so
    a 2-D frame is ``reshape((Nx, Ny))`` with *no* transpose. Verified on
    ``runs/test_025``: the ``lpse.pots.downSample_4`` frame equals ``pots[::4, ::4]`` of
    the full frame under this layout (normalized inner product 1.0000) and not under the
    x-fastest one (0.49); the 2 um ``lw.Labc`` skirts along x show up as a 17 % dip of
    ``<|phi|>`` in the outer 36 cells of axis 0, and the field is isotropically smooth."""
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
        arr = arr.reshape((nx, ny, nz))
        if nz == 1:
            arr = arr[:, :, 0]
        frames.append((header, arr))
        pos = start + 4 * count
    return frames


def read_injector_file(path: str | Path, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Read an LPSE light-injector file (``laser.E_<c>.loadInjector.<side>.<axis>.filename``,
    written by ``matlab/m201902_createLpseInjector_v02.m``): ``(times_ps (T,), planes (T, 2,
    n_points))`` with the planes complex in LPSE's light-field unit ``e E / (m_e w0 c)``.

    The format is ``LightSolver::readInjectorFiles``: single-precision floats, per time the
    time in ps, then the ``n_points`` complex values (re, im interleaved) of the injector plane,
    then those of the plane one cell further into the box. ``n_points`` is the number of grid
    points on the injector face (``Ny`` for an x face in 2-D). LPSE requires the first time to
    be 0 and the times to increase."""
    data = np.fromfile(Path(path), dtype="<f4").astype(np.float64)
    per_time = 1 + 2 * 2 * n_points
    if data.size == 0 or data.size % per_time:
        raise ValueError(
            f"injector file {path} holds {data.size} floats, not a multiple of 1 + 4 * {n_points} "
            "(time + two planes of complex values per time)"
        )
    data = data.reshape(-1, per_time)
    times = data[:, 0].copy()
    planes = (data[:, 1::2] + 1j * data[:, 2::2]).reshape(-1, 2, n_points)
    if times[0] != 0.0:
        raise ValueError(f"injector file {path}: the first time must be 0 (LPSE), got {times[0]}")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"injector file {path}: the times must increase (LPSE)")
    return times, planes


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
