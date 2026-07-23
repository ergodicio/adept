"""Spitzer-Härm and SNB heat-flux diagnostics for VFP-1D.

These are pure-numpy post-processing diagnostics that compute reduced-model heat
fluxes from the fluid moments of a VFP-1D run so they can be compared against the
kinetic heat flux carried by f10. Everything is evaluated in code-normalized units
(v in c, t in 1/w0, n in n0) so the comparison with the kinetic ``q`` is direct.

References:

1. Schurtz, G. P., Nicolaï, Ph. D. & Busquet, M. A nonlocal electron conduction
   model for multidimensional radiation hydrodynamics codes. Phys. Plasmas 7,
   4238 (2000).
2. Brodrick, J. P. et al. Testing nonlocal models of electron thermal conduction
   for magnetic and inertial confinement fusion applications. Phys. Plasmas 24,
   092309 (2017).  [The "separated" SNB variant with r = 2 is implemented here.]
3. Epperlein, E. M. & Haines, M. G. Plasma transport coefficients in a magnetic
   field by direct numerical solution of the Fokker-Planck equation.
   Physics of Fluids 29, 1029 (1986).
"""

import os

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.linalg import solve_banded

from adept.vfp1d.storage import calc_EH

DEFAULT_SNB_CFG = {"ngroups": 32, "beta_max": 30.0, "r": 2.0}


def _geometry_factors(x: np.ndarray, dx: float, geometry: str) -> tuple[np.ndarray, np.ndarray]:
    """Face areas (nx+1,) and cell volumes (nx,) for the 1D divergence operator."""
    x_edge = np.concatenate([[x[0] - dx / 2], 0.5 * (x[1:] + x[:-1]), [x[-1] + dx / 2]])
    if geometry == "spherical":
        area = x_edge**2
        vol = np.diff(x_edge**3) / 3.0
    else:
        area = np.ones_like(x_edge)
        vol = np.full_like(x, dx)
    return area, vol


def _interp_c2e(f: np.ndarray) -> np.ndarray:
    """Cell centers (nx,) to edges (nx+1,) with clamped ends."""
    return np.concatenate([f[:1], 0.5 * (f[1:] + f[:-1]), f[-1:]])


def _ddx_c2e(f: np.ndarray, dx: float) -> np.ndarray:
    """Gradient at interior edges, zero at the boundary edges (reflective/symmetry)."""
    return np.concatenate([[0.0], np.diff(f) / dx, [0.0]])


def local_transport_coefficients(n: np.ndarray, T: np.ndarray, Z: np.ndarray, cfg: dict) -> dict:
    """Local (x-dependent) transport quantities in code-normalized units.

    The local e-i momentum-relaxation rate is scaled from the reference
    ``nuei_epphaines_norm`` (evaluated at the reference n, T, Z) using
    nu_ei ~ Z * n * T^(-3/2). Spatial variation of log-Lambda is neglected.

    Args:
        n: electron density in units of n0 (nx,)
        T: electron temperature in code units, T = P/n with v in c (nx,)
        Z: physical ionization state (nx,)
        cfg: full config with cfg["units"]["derived"] populated
    """
    derived = cfg["units"]["derived"]
    n_ref = float((derived["ne"] / derived["n0"]).to("").magnitude)
    T_ref = float(derived["vth_norm"]) ** 2 / 2.0
    Z_ref = cfg["units"]["Z"]
    nuei_ref = float(derived["nuei_epphaines_norm"].magnitude)

    nu_ei = nuei_ref * (Z / Z_ref) * (n / n_ref) * (T / T_ref) ** -1.5
    return {"nu_ei": nu_ei, "kappa_eh": calc_EH(Z, 0.0)}


def spitzer_harm_heat_flux(
    x: np.ndarray, dx: float, n: np.ndarray, T: np.ndarray, Z: np.ndarray, cfg: dict
) -> np.ndarray:
    """Local Spitzer-Härm(-Epperlein-Haines) heat flux at cell edges (nx+1,).

    q_SH = -kappa_EH(Z) * n * T / nu_ei(n, T, Z) * dT/dx, which by construction is
    consistent with the ``kappa`` metric logged by the VFP-1D post-processing
    (kappa = -q / (n T dT/dx) * nuei_epphaines_norm -> kappa_EH in the local limit).
    """
    coeffs = local_transport_coefficients(n, T, Z, cfg)
    chi_c = coeffs["kappa_eh"] * n * T / coeffs["nu_ei"]
    return -_interp_c2e(chi_c) * _ddx_c2e(T, dx)


def snb_heat_flux(
    x: np.ndarray,
    dx: float,
    n: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    ni: np.ndarray,
    cfg: dict,
    geometry: str = "cartesian",
    snb_cfg: dict | None = None,
) -> dict:
    """Multigroup SNB heat flux at cell edges (nx+1,).

    Implements the "separated" SNB variant recommended by Brodrick et al. (2017):
    for each energy group g (beta = v^2 / 2T, group weights from the Spitzer-Härm
    spectrum W(beta) = beta^4 exp(-beta) / 24),

        H_g / (lambda_g^ee / r) - div( (xi * lambda_g^ei / 3) grad H_g ) = -div(W_g q_SH)

    with r = 2, xi(Z) = (Z + 0.24)/(Z + 4.2), and

        q_SNB = q_SH - sum_g (xi * lambda_g^ei / 3) grad H_g.

    The mean free paths use the same normalized collision rates as the kinetic
    solver: nu_ei(v) = nuee_coeff * logLam_ratio * Z^2 * ni / v^3 (the l=1 rate in
    FLMCollisions) and nu_ee(v) = nuee_coeff * n / v^3, so lambda = v^4 / (...).

    Boundary conditions are zero flux at both ends, matching the reflective /
    r=0 symmetry boundaries of the spherical solver.

    Returns:
        dict with "q_snb" (nx+1,), "q_sh" (nx+1,), and "H" (ngroups, nx)
    """
    snb = {**DEFAULT_SNB_CFG, **(snb_cfg or {})}
    ngroups, beta_max, r_snb = int(snb["ngroups"]), float(snb["beta_max"]), float(snb["r"])

    derived = cfg["units"]["derived"]
    nuee_coeff = float(derived["nuee_coeff"])
    logLam_ratio = derived["logLam_ratio"]
    logLam_ratio = float(logLam_ratio.magnitude if hasattr(logLam_ratio, "magnitude") else logLam_ratio)

    area, vol = _geometry_factors(x, dx, geometry)
    q_sh = spitzer_harm_heat_flux(x, dx, n, T, Z, cfg)

    # group edges in beta = v^2/(2T); logarithmic spacing resolves the low-energy
    # groups while still covering the suprathermal tail that carries q_SH
    beta_edges = np.geomspace(beta_max / 400.0, beta_max, ngroups + 1)
    beta_edges[0] = 0.0

    def _cdf(b):
        # int_0^b beta^4 exp(-beta)/24 dbeta = 1 - exp(-b)(1 + b + b^2/2 + b^3/6 + b^4/24)
        return 1.0 - np.exp(-b) * (1.0 + b + b**2 / 2.0 + b**3 / 6.0 + b**4 / 24.0)

    weights = np.diff(_cdf(beta_edges))
    beta_c = np.sqrt(beta_edges[:-1] * beta_edges[1:])
    beta_c[0] = 0.5 * beta_edges[1]

    xi = (Z + 0.24) / (Z + 4.2)
    T_edge = _interp_c2e(T)

    q_nonlocal = np.zeros_like(q_sh)
    H_all = np.zeros((ngroups, x.size))

    for g in range(ngroups):
        v2_c = 2.0 * beta_c[g] * T  # v_g^2 at centers
        v2_e = 2.0 * beta_c[g] * T_edge
        lam_ee_c = v2_c**2 / (nuee_coeff * n)
        lam_ei_e = v2_e**2 / (nuee_coeff * logLam_ratio * _interp_c2e(Z**2 * ni))
        D_e = _interp_c2e(xi) * lam_ei_e / 3.0

        # source: -div(W_g q_SH), finite volume with zero flux through boundary edges
        q_g = weights[g] * q_sh
        src = -np.diff(area * q_g) / vol

        # tridiagonal finite-volume diffusion operator
        w = area[1:-1] * D_e[1:-1] / dx  # interior face conductances (nx-1,)
        lower = np.concatenate([-w / vol[1:], [0.0]])
        upper = np.concatenate([[0.0], -w / vol[:-1]])
        diag = r_snb / lam_ee_c
        diag[:-1] += w / vol[:-1]
        diag[1:] += w / vol[1:]

        ab = np.zeros((3, x.size))
        ab[0, :] = upper
        ab[1, :] = diag
        ab[2, :] = lower
        H_g = solve_banded((1, 1), ab, src)
        H_all[g] = H_g

        dH_e = _ddx_c2e(H_g, dx)
        q_nonlocal += D_e * dH_e

    return {"q_snb": q_sh - q_nonlocal, "q_sh": q_sh, "H": H_all}


def compare_heat_flux(fields_xr: xr.Dataset, cfg: dict, td: str) -> dict:
    """Computes Spitzer-Härm and SNB heat fluxes from the saved fluid moments and
    compares them against the kinetic heat flux for every saved timestep.

    Writes a netCDF with the three fluxes and a comparison plot at the final saved
    time. Returns a dict with the xr.Dataset and summary metrics.
    """
    x = np.asarray(cfg["grid"]["x"])
    dx = float(cfg["grid"]["dx"])
    geometry = cfg["grid"].get("geometry", "cartesian")
    Z_ref = cfg["units"]["Z"]
    snb_cfg = cfg.get("diagnostics", {}).get("snb", None)

    n_t = np.asarray(fields_xr["fields-n n_c"].data)
    T_t = np.asarray(fields_xr["fields-T a.u."].data)
    q_kin_t = np.asarray(fields_xr["fields-q a.u."].data)
    ni_t = np.asarray(fields_xr["fields-ni a.u."].data)
    if "fields-Z a.u." in fields_xr:
        Z_t = np.asarray(fields_xr["fields-Z a.u."].data) * Z_ref
    else:
        Z_t = Z_ref * np.ones_like(n_t)

    nt = n_t.shape[0]
    q_sh_t = np.zeros_like(q_kin_t)
    q_snb_t = np.zeros_like(q_kin_t)
    for it in range(nt):
        res = snb_heat_flux(x, dx, n_t[it], T_t[it], Z_t[it], ni_t[it], cfg, geometry=geometry, snb_cfg=snb_cfg)
        # fluxes live at edges; average to centers to match the saved kinetic q
        q_sh_t[it] = 0.5 * (res["q_sh"][1:] + res["q_sh"][:-1])
        q_snb_t[it] = 0.5 * (res["q_snb"][1:] + res["q_snb"][:-1])

    tax = fields_xr.coords["t (ps)"].data
    xax = fields_xr.coords["x (um)"].data
    coords = (("t (ps)", tax), ("x (um)", xax))
    ds = xr.Dataset(
        {
            "q_kinetic": xr.DataArray(q_kin_t, coords=coords),
            "q_spitzer_harm": xr.DataArray(q_sh_t, coords=coords),
            "q_snb": xr.DataArray(q_snb_t, coords=coords),
        }
    )
    ds.to_netcdf(os.path.join(td, "binary", "heat-flux-comparison.nc"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)
    ax.plot(xax, q_kin_t[-1], label="kinetic (VFP)")
    ax.plot(xax, q_sh_t[-1], "--", label="Spitzer-Härm")
    ax.plot(xax, q_snb_t[-1], "-.", label="SNB")
    xlabel = "r (um)" if geometry == "spherical" else "x (um)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("q (code units)")
    ax.set_title(f"Heat flux comparison at t = {tax[-1]:.3f} ps")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(td, "plots", "fields", "heat-flux-comparison.png"), bbox_inches="tight")
    plt.close(fig)

    peak = np.argmax(np.abs(q_kin_t[-1]))
    q_sh_peak = q_sh_t[-1][peak]

    def _safe_ratio(num):
        # avoid inf metrics when there is no temperature gradient (q_SH = 0)
        return float(num / q_sh_peak) if abs(q_sh_peak) > 1e-30 else 0.0

    metrics = {
        "q_ratio_kinetic_over_sh": _safe_ratio(q_kin_t[-1][peak]),
        "q_ratio_snb_over_sh": _safe_ratio(q_snb_t[-1][peak]),
    }

    return {"heat_flux": ds, "metrics": metrics}
