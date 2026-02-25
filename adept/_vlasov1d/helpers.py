#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import os
from time import time

import numpy as np
import xarray
from diffrax import Solution
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.special import gamma

from adept._vlasov1d.simulation import SubspeciesDistributionSpec, Vlasov1DSimulation
from adept._vlasov1d.storage import store_f, store_fields
from adept.normalization import PlasmaNormalization

from .. import patched_mlflow as mlflow

# gamma_da = xarray.open_dataarray(os.path.join(os.path.dirname(__file__), "gamma_func_for_sg.nc"))
# m_ax = gamma_da.coords["m"].data
# g_3_m = np.squeeze(gamma_da.loc[{"gamma": "3/m"}].data)
# g_5_m = np.squeeze(gamma_da.loc[{"gamma": "5/m"}].data)


def gamma_3_over_m(m):
    return gamma(3.0 / m)  # np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m):
    return gamma(5.0 / m)  # np.interp(m, m_ax, g_5_m)


def _initialize_supergaussian_distribution_(
    nx: int,
    nv: int,
    v0=0.0,
    supergaussian_order=2.0,
    T0=1.0,
    mass=1.0,
    vmax=6.0,
    n_prof=np.ones(1),
):
    """
    Initialize a supergaussian distribution function.

    For supergaussian_order=2, this gives a Maxwell-Boltzmann distribution.

    Args:
        nx: size of grid in x
        nv: size of grid in v
        v0: drift velocity
        supergaussian_order: order of supergaussian (2 = Maxwell-Boltzmann)
        T0: temperature
        mass: species mass for thermal velocity calculation
        vmax: maximum absolute value of v
        n_prof: density profile (noise should already be applied)

    Returns:
        Tuple of (f[nx, nv], vax[nv])
    """
    dv = 2.0 * vmax / nv
    vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    # Thermal velocity: v_t = sqrt(T/m)
    v_thermal = np.sqrt(T0 / mass)

    # Alpha factor for supergaussian normalization
    alpha = np.sqrt(3.0 * gamma_3_over_m(supergaussian_order) / gamma_5_over_m(supergaussian_order))

    single_dist = -(np.power(np.abs((vax[None, :] - v0) / (alpha * v_thermal)), supergaussian_order))

    single_dist = np.exp(single_dist)

    f = np.repeat(single_dist, nx, axis=0)
    # normalize
    f = f / np.trapz(f, dx=dv, axis=1)[:, None]

    if n_prof.size > 1:
        # scale by density profile
        f = n_prof[:, None] * f

    return f, vax


def _initialize_total_distribution_(cfg, simulation: Vlasov1DSimulation):
    """
    Initialize distribution functions for all species using domain models.

    The species config is normalized in modules.py:get_derived_quantities() so that
    a species config always exists (for backward compatibility with single-species
    config files, a default electron species is generated).

    Args:
        cfg: Configuration dictionary
        grid: Grid object with spatial coordinates
        species_distribution_specs: Dictionary from species name to the list of SubspeciesDistributionSpecs for it.
        norm: PlasmaNormalization for unit conversion (required for linear/exponential profiles)

    Returns:
        dict mapping species_name -> (n_prof, f_s, v_ax)
    """
    species_distributions = {}
    species_found = False

    norm = simulation.plasma_norm
    grid = simulation.grid
    species_configs = simulation.species
    species_distribution_specs = simulation.species_distributions

    for species_cfg in species_configs:
        species_name = species_cfg.name
        density_components = species_distribution_specs[species_name]
        vmax = species_cfg.vmax
        nv = species_cfg.nv
        mass = species_cfg.mass

        # Initialize arrays for this species
        n_prof_species = np.zeros([grid.nx])
        dv = 2.0 * vmax / nv
        vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
        f_species = np.zeros([grid.nx, nv])

        # Sum contributions from all density components
        for distribution_spec in density_components:
            # Evaluate density profile (noise is applied by the domain model)
            nprof = np.array(distribution_spec.density_profile(grid.x))
            n_prof_species += nprof

            # Initialize distribution
            temp_f, _ = _initialize_supergaussian_distribution_(
                nx=grid.nx,
                nv=nv,
                v0=distribution_spec.v0,
                supergaussian_order=distribution_spec.supergaussian_order,
                T0=distribution_spec.T0,
                mass=mass,
                vmax=vmax,
                n_prof=nprof,
            )
            f_species += temp_f
            species_found = True

        species_distributions[species_name] = (n_prof_species, f_species, vax)

    if not species_found:
        raise ValueError("No species found! Check the config")

    return species_distributions


def post_process(result: Solution, cfg: dict, td: str, args: dict):
    t0 = time()

    # Get species names for directory creation
    species_names = list(cfg["grid"]["species_grids"].keys())

    # Create base plot directories
    os.makedirs(os.path.join(td, "plots"), exist_ok=True)

    # Create fields directory structure
    # - fields/ (shared EM fields at top level)
    # - fields/{species}/ (species-specific moments)
    os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "logplots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "lineouts"), exist_ok=True)
    for species_name in species_names:
        species_dir = os.path.join(td, "plots", "fields", species_name)
        os.makedirs(species_dir, exist_ok=True)
        os.makedirs(os.path.join(species_dir, "logplots"), exist_ok=True)
        os.makedirs(os.path.join(species_dir, "lineouts"), exist_ok=True)

    # Create scalars directory structure
    # - scalars/ (shared field scalars at top level)
    # - scalars/{species}/ (species-specific scalars)
    os.makedirs(os.path.join(td, "plots", "scalars"), exist_ok=True)
    for species_name in species_names:
        os.makedirs(os.path.join(td, "plots", "scalars", species_name), exist_ok=True)

    # Create dists directory for distribution function snapshots
    os.makedirs(os.path.join(td, "plots", "dists"), exist_ok=True)
    for species_name in species_names:
        os.makedirs(os.path.join(td, "plots", "dists", species_name), exist_ok=True)

    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir)

    fields_result = {}
    fields_base_dir = os.path.join(td, "plots", "fields")

    for k in result.ys.keys():
        if k.startswith("field"):
            # store_fields now returns dict with species names and "fields" keys
            fields_dict = store_fields(cfg, binary_dir, result.ys[k], result.ts[k], k)

            # Plot species-specific moments in fields/{species}/
            for species_name in species_names:
                if species_name not in fields_dict:
                    continue

                species_xr = fields_dict[species_name]
                species_dir = os.path.join(fields_base_dir, species_name)

                t_skip = int(species_xr.coords["t"].data.size // 8)
                t_skip = t_skip if t_skip > 1 else 1
                tslice = slice(0, -1, t_skip)

                for nm, fld in species_xr.items():
                    # Strip prefix (e.g., "fields-n" -> "n")
                    field_name = nm.split("-", 1)[1] if "-" in nm else nm

                    # Spacetime plot
                    fld.plot()
                    plt.savefig(os.path.join(species_dir, f"spacetime_{field_name}.png"), bbox_inches="tight")
                    plt.close()

                    # Log plot
                    np.log10(np.abs(fld)).plot()
                    plt.savefig(
                        os.path.join(species_dir, "logplots", f"spacetime_log_{field_name}.png"), bbox_inches="tight"
                    )
                    plt.close()

                    # Lineouts
                    fld[tslice].T.plot(col="t", col_wrap=4)
                    plt.savefig(os.path.join(species_dir, "lineouts", f"{field_name}.png"), bbox_inches="tight")
                    plt.close()

            # Plot shared field data (e, de, a, pond) in fields/ (top level)
            if "fields" in fields_dict:
                shared_xr = fields_dict["fields"]

                t_skip = int(shared_xr.coords["t"].data.size // 8)
                t_skip = t_skip if t_skip > 1 else 1
                tslice = slice(0, -1, t_skip)

                for nm, fld in shared_xr.items():
                    field_name = nm.split("-", 1)[1] if "-" in nm else nm

                    fld.plot()
                    plt.savefig(os.path.join(fields_base_dir, f"spacetime_{field_name}.png"), bbox_inches="tight")
                    plt.close()

                    np.log10(np.abs(fld)).plot()
                    log_path = os.path.join(fields_base_dir, "logplots", f"spacetime_log_{field_name}.png")
                    plt.savefig(log_path, bbox_inches="tight")
                    plt.close()

                    fld[tslice].T.plot(col="t", col_wrap=4)
                    plt.savefig(os.path.join(fields_base_dir, "lineouts", f"{field_name}.png"), bbox_inches="tight")
                    plt.close()

            fields_result = fields_dict

        elif k.startswith("default"):
            scalars_xr = xarray.Dataset(
                {k: xarray.DataArray(v, coords=(("t", result.ts["default"]),)) for k, v in result.ys["default"].items()}
            )
            scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(scalars_xr.coords['t'].data[-1], 4)}.nc"))

            scalars_base_dir = os.path.join(td, "plots", "scalars")
            for nm, srs in scalars_xr.items():
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                srs.plot(ax=ax[0])
                ax[0].grid()
                np.log10(np.abs(srs)).plot(ax=ax[1])
                ax[1].grid()
                ax[1].set_ylabel("$log_{10}$(|" + nm + "|)")

                # Determine if this is a species-specific or shared scalar
                # Species-specific scalars have format: mean_X_{species_name}
                scalar_species = None
                for species_name in species_names:
                    if nm.endswith(f"_{species_name}"):
                        scalar_species = species_name
                        break

                if scalar_species:
                    # Save to scalars/{species}/
                    fig.savefig(os.path.join(scalars_base_dir, scalar_species, f"{nm}.png"), bbox_inches="tight")
                else:
                    # Shared field scalar (e.g., mean_e2, mean_de2, mean_pond)
                    fig.savefig(os.path.join(scalars_base_dir, f"{nm}.png"), bbox_inches="tight")
                plt.close()

    f_xr = store_f(cfg, result.ts, td, result.ys)

    # Plot velocity space distributions for each species
    for species_name in species_names:
        if species_name not in f_xr.data_vars:
            continue

        f_species = f_xr[species_name]
        species_dist_dir = os.path.join(td, "plots", "dists", species_name)

        # Select ~8 time snapshots for facet plot
        t_skip = int(f_species.coords["t"].data.size // 8)
        t_skip = t_skip if t_skip > 1 else 1
        tslice = slice(0, -1, t_skip)

        # Create f(x,v) phase space plot
        # First panel: f(t=0), remaining panels: f(t) - f(t=0) with separate color scales
        f_sliced = f_species[tslice]
        f0 = f_species.isel(t=0)
        f_diff = f_sliced.isel(t=slice(1, None)) - f0
        n_diff = f_diff.sizes["t"]
        n_total = 1 + n_diff
        ncols = min(4, n_total)
        nrows = (n_total + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, constrained_layout=True)
        axes_flat = axes.flatten()

        # Plot f(t=0) in first panel with its own colorbar
        im0 = axes_flat[0].pcolormesh(f0.coords["x"].values, f0.coords[f"v_{species_name}"].values, f0.values.T)
        axes_flat[0].set_xlabel("x")
        axes_flat[0].set_ylabel("v")
        axes_flat[0].set_title(f"t = {f_sliced.coords['t'].values[0]:.2f}")
        fig.colorbar(im0, ax=axes_flat[0], label="f")

        # Plot f(t) - f(t=0) in remaining panels with shared color scale
        if n_diff > 0:
            vmin, vmax = float(f_diff.min()), float(f_diff.max())
            vabs = max(abs(vmin), abs(vmax))
            for i in range(n_diff):
                ax = axes_flat[i + 1]
                data = f_diff.isel(t=i)
                im = ax.pcolormesh(
                    data.coords["x"].values,
                    data.coords[f"v_{species_name}"].values,
                    data.values.T,
                    vmin=-vabs,
                    vmax=vabs,
                    cmap="RdBu_r",
                )
                ax.set_xlabel("x")
                ax.set_ylabel("v")
                ax.set_title(f"t = {f_diff.coords['t'].values[i]:.2f}")
            # Add shared colorbar for difference panels
            fig.colorbar(im, ax=axes_flat[1 : n_diff + 1].tolist(), label="f - f(t=0)")

        # Hide unused axes
        for i in range(n_total, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.savefig(os.path.join(species_dist_dir, "phase_space.png"), bbox_inches="tight")
        plt.close()

    diags_dict = {}
    for k in ["diag-vlasov-dfdt", "diag-fp-dfdt"]:
        if cfg["diagnostics"][k]:
            diags_dict[k] = xarray.DataArray(
                result.ys[k], coords=(("t", result.ts[k]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"]))
            )

    if len(diags_dict.keys()):
        diags_xr = xarray.Dataset(diags_dict)
        diags_xr.to_netcdf(os.path.join(td, "binary", "diags.nc"))

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})

    return {"fields": fields_result, "dists": f_xr, "scalars": scalars_xr}
