"""Exact integrals of the saved, post-remesh biquadratic representation.

These are not marker/material quadrature invariants. Three-point Gauss in each
coordinate exactly integrates both a biquadratic polynomial and its square on
each rectangular leaf, up to floating-point arithmetic. No claim is made for
advected/deformed panels, positivity, or resolution of the physical solution.
"""

from __future__ import annotations

import numpy as np


def integrate_rectangles(f, areas):
    """Return integral(f) and integral(f²) for disjoint 3×3 nodal panels."""
    from .movies import _lagrange_basis

    f, areas = np.asarray(f, dtype=float), np.asarray(areas, dtype=float)
    if f.ndim != 3 or f.shape[1:] != (3, 3) or areas.shape != (f.shape[0],):
        raise ValueError("Expected f(panel,3,3) and areas(panel)")
    if not np.isfinite(f).all() or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("Panels must be finite with positive areas")
    points, weights = np.polynomial.legendre.leggauss(3)
    basis = _lagrange_basis(points + 1)
    values = np.einsum("ai,pij,bj->pab", basis, f, basis)
    quadrature = areas[:, None, None] * weights[None, :, None] * weights[None, None, :] / 4
    return float(np.sum(quadrature * values)), float(np.sum(quadrature * values**2))


def representation_integrals(dataset, config):
    """Validate saved geometry and compute exact polynomial mass/C2 per frame."""
    from .movies import iter_validated_amr_frames, reconstruct_fixed_panels

    grid = config["grid"]
    masses, c2s = [], []
    if config.get("amr", {}).get("enabled", False):
        frames = iter_validated_amr_frames(dataset, grid, config["amr"])
        for x, v, f, _ in frames:
            areas = (x[:, 8] - x[:, 0]) * (v[:, 8] - v[:, 0])
            mass, c2 = integrate_rectangles(f.reshape(-1, 3, 3), areas)
            masses.append(mass)
            c2s.append(c2)
    else:
        # This performs complete fixed-grid geometry/seam/finite validation;
        # only the throwaway target raster is small.
        reconstruct_fixed_panels(dataset, [grid["xmin"]], [grid["vmin"]], grid)
        nx, nv = grid["nx"], grid["nv"]
        ix = (2 * np.arange(nx // 2)[:, None] + np.arange(3)).astype(int)
        iv = (2 * np.arange(nv // 2)[:, None] + np.arange(3)).astype(int)
        area = 4 * (grid["xmax"] - grid["xmin"]) * (grid["vmax"] - grid["vmin"]) / (nx * nv)
        areas = np.full((nx // 2) * (nv // 2), area)
        for frame in np.asarray(dataset["f"]):
            panels = frame[ix[:, None, :, None], iv[None, :, None, :]].reshape(-1, 3, 3)
            mass, c2 = integrate_rectangles(panels, areas)
            masses.append(mass)
            c2s.append(c2)
    masses, c2s = np.asarray(masses), np.asarray(c2s)
    if not len(masses) or masses[0] <= 0 or c2s[0] <= 0:
        raise ValueError("Representation must have positive initial mass and C2")
    return {
        "t": np.asarray(dataset["t"]).tolist(),
        "mass": masses.tolist(),
        "c2": c2s.tolist(),
        "relative_mass": ((masses - masses[0]) / masses[0]).tolist(),
        "relative_c2": ((c2s - c2s[0]) / c2s[0]).tolist(),
        "method": "3x3 Gauss per disjoint rectilinear leaf; exact for the saved biquadratic polynomial and its square",
        "caveat": "Not native/material quadrature, a positivity test, or a physical convergence estimate.",
    }
