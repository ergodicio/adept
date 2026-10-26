"""Initialization and post-processing helpers for Vlasov-1D2V simulations."""

import os
from time import time

import numpy as np
import xarray
from diffrax import Solution
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adept._vlasov1d.helpers import _initialize_supergaussian_distribution_
from adept._vlasov1d.simulation import Vlasov1DSimulation
from adept._vlasov1d.storage import store_fields
from adept._vlasov1d2v.storage import store_f

from .. import patched_mlflow as mlflow


def perp_grid(nvperp: int, vperp_max: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Return the cylindrical v_perp grid: cell-centered points, spacing, weights.

    The integration weight is w_perp = 2*pi*v_perp*dv_perp so that
    int f d3v = sum_j f(., ., j) w_perp[j] dv_par.
    """
    dvperp = vperp_max / nvperp
    vperp = np.linspace(dvperp / 2.0, vperp_max - dvperp / 2.0, nvperp)
    wperp = 2.0 * np.pi * vperp * dvperp
    return vperp, dvperp, wperp


def _initialize_total_distribution_2v_(cfg, simulation: Vlasov1DSimulation, nvperp: int, vperp_max: float):
    """Initialize f(x, v_par, v_perp) for all species.

    The v_par part of each density component reuses the 1D initializer
    (including super-Gaussian order, drift, and noise); the v_perp part is a
    Maxwellian at the component's T0 (super-Gaussian shapes apply to v_par
    only), normalized so that sum_j M_perp[j] w_perp[j] = 1. Hence the
    marginal of the initialized f equals the corresponding 1D initialization
    exactly, which underpins the 1D-limit equivalence test.

    Returns:
        dict mapping species_name -> (n_prof, f_s[nx, nv, nvperp], v_ax, vperp_ax)
    """
    vperp, _, wperp = perp_grid(nvperp, vperp_max)

    species_distributions = {}
    species_found = False

    grid = simulation.grid

    for species_cfg in simulation.species:
        species_name = species_cfg.name
        density_components = simulation.species_distributions[species_name]
        vmax = species_cfg.vmax
        vmin = species_cfg.vmin
        nv = species_cfg.nv
        mass = species_cfg.mass

        n_prof_species = np.zeros([grid.nx])
        dv = (vmax - vmin) / nv
        vax = np.linspace(vmin + dv / 2.0, vmax - dv / 2.0, nv)
        f_species = np.zeros([grid.nx, nv, nvperp])

        for distribution_spec in density_components:
            nprof = np.array(distribution_spec.density_profile(grid.x))
            n_prof_species += nprof

            temp_f, _ = _initialize_supergaussian_distribution_(
                nx=grid.nx,
                nv=nv,
                v0=distribution_spec.v0,
                supergaussian_order=distribution_spec.supergaussian_order,
                T0=distribution_spec.T0,
                mass=mass,
                vmax=vmax,
                vmin=vmin,
                n_prof=nprof,
            )

            m_perp = np.exp(-(vperp**2.0) / (2.0 * distribution_spec.T0 / mass))
            m_perp = m_perp / np.sum(m_perp * wperp)

            f_species += temp_f[:, :, None] * m_perp[None, None, :]
            species_found = True

        species_distributions[species_name] = (n_prof_species, f_species, vax, vperp)

    if not species_found:
        raise ValueError("No species found! Check the config")

    return species_distributions


def post_process(result: Solution, cfg: dict, td: str, args: dict):
    """Write binary output and diagnostic plots from a completed Vlasov-1D2V solve."""
    t0 = time()

    species_names = list(cfg["grid"]["species_grids"].keys())

    os.makedirs(os.path.join(td, "plots"), exist_ok=True)
    fields_base_dir = os.path.join(td, "plots", "fields")
    os.makedirs(fields_base_dir, exist_ok=True)
    os.makedirs(os.path.join(fields_base_dir, "logplots"), exist_ok=True)
    for species_name in species_names:
        os.makedirs(os.path.join(fields_base_dir, species_name), exist_ok=True)
    scalars_base_dir = os.path.join(td, "plots", "scalars")
    os.makedirs(scalars_base_dir, exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "dists"), exist_ok=True)

    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir, exist_ok=True)

    fields_result = {}
    scalars_xr = None

    for k in result.ys:
        if k.startswith("field"):
            fields_dict = store_fields(cfg, binary_dir, result.ys[k], result.ts[k], k)

            for species_name in species_names:
                if species_name not in fields_dict:
                    continue
                species_dir = os.path.join(fields_base_dir, species_name)
                for nm, fld in fields_dict[species_name].items():
                    field_name = nm.split("-", 1)[1] if "-" in nm else nm
                    fld.plot(figsize=(12, 8))
                    plt.savefig(os.path.join(species_dir, f"spacetime_{field_name}.png"), bbox_inches="tight", dpi=150)
                    plt.close()

            if "fields" in fields_dict:
                for nm, fld in fields_dict["fields"].items():
                    field_name = nm.split("-", 1)[1] if "-" in nm else nm
                    fld.plot()
                    plt.savefig(os.path.join(fields_base_dir, f"spacetime_{field_name}.png"), bbox_inches="tight")
                    plt.close()
                    np.log10(np.abs(fld)).plot()
                    plt.savefig(
                        os.path.join(fields_base_dir, "logplots", f"spacetime_log_{field_name}.png"),
                        bbox_inches="tight",
                    )
                    plt.close()

            fields_result = fields_dict

        elif k.startswith("default"):
            scalars_xr = xarray.Dataset(
                {
                    kk: xarray.DataArray(v, coords=(("t", result.ts["default"]),))
                    for kk, v in result.ys["default"].items()
                }
            )
            scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(scalars_xr.coords['t'].data[-1], 4)}.nc"))
            for nm, srs in scalars_xr.items():
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                srs.plot(ax=ax[0])
                ax[0].grid()
                np.log10(np.abs(srs)).plot(ax=ax[1])
                ax[1].grid()
                ax[1].set_ylabel("$log_{10}$(|" + nm + "|)")
                fig.savefig(os.path.join(scalars_base_dir, f"{nm}.png"), bbox_inches="tight")
                plt.close()

    f_result = store_f(cfg, result.ts, td, result.ys)

    # Marginal phase-space snapshots: f(t=0) then f(t) - f(t=0)
    for save_key, f_xr in f_result.items():
        save_cfg = cfg["save"][save_key]
        if "_species_name" not in save_cfg:
            continue
        da = f_xr[save_key]
        if da.ndim != 3:
            continue  # rank-4 full-f snapshots: stored, not plotted here
        species_dist_dir = os.path.join(td, "plots", "dists", save_key)
        os.makedirs(species_dist_dir, exist_ok=True)

        v_dim = next(d for d in da.dims if d.startswith("v"))
        t_skip = max(int(da.coords["t"].data.size // 8), 1)
        f_sliced = da[slice(0, -1, t_skip)]
        f0 = da.isel(t=0)
        n_panels = f_sliced.sizes["t"]
        ncols = min(4, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, constrained_layout=True)
        axes_flat = axes.flatten()
        for i in range(n_panels):
            data = f_sliced.isel(t=i) - (f0 if i > 0 else 0.0)
            im = axes_flat[i].pcolormesh(
                data.coords["x"].values, data.coords[v_dim].values, data.values.T, cmap="RdBu_r" if i > 0 else None
            )
            axes_flat[i].set_title(f"t = {f_sliced.coords['t'].values[i]:.2f}" + (" (f - f0)" if i > 0 else ""))
            fig.colorbar(im, ax=axes_flat[i])
        for i in range(n_panels, len(axes_flat)):
            axes_flat[i].set_visible(False)
        plt.savefig(os.path.join(species_dist_dir, "phase_space.png"), bbox_inches="tight")
        plt.close()

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})

    return {"fields": fields_result, "dists": f_result, "scalars": scalars_xr}
