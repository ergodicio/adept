#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import os
from time import time

import numpy as np
import pint
import xarray
from diffrax import Solution
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.special import gamma

from adept._base_ import get_envelope
from adept._vlasov1d.storage import store_f, store_fields

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
    noise_val=0.0,
    noise_seed=42,
    noise_type="Uniform",
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
        n_prof: density profile
        noise_val: noise amplitude
        noise_seed: random seed for noise
        noise_type: type of noise ("Uniform")

    Returns:
        Tuple of (f[nx, nv], vax[nv])
    """
    noise_generator = np.random.default_rng(seed=noise_seed)

    dv = 2.0 * vmax / nv
    vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    # Thermal velocity: v_t = sqrt(T/m)
    v_thermal = np.sqrt(T0 / mass)

    # Alpha factor for supergaussian normalization
    alpha = np.sqrt(3.0 * gamma_3_over_m(supergaussian_order) / gamma_5_over_m(supergaussian_order))

    single_dist = -(np.power(np.abs((vax[None, :] - v0) / (alpha * v_thermal)), supergaussian_order))

    single_dist = np.exp(single_dist)
    # single_dist = np.exp(-(vaxs[0][None, None, :, None]**2.+vaxs[1][None, None, None, :]**2.)/2/T0)

    # for ix in range(nx):
    f = np.repeat(single_dist, nx, axis=0)
    # normalize
    f = f / np.trapz(f, dx=dv, axis=1)[:, None]

    if n_prof.size > 1:
        # scale by density profile
        f = n_prof[:, None] * f

    if noise_type.casefold() == "uniform":
        f = (1.0 + noise_generator.uniform(-noise_val, noise_val, nx)[:, None]) * f
    # elif noise_type.casefold() == "gaussian":
    #     f = (1.0 + noise_generator.normal(-noise_val, noise_val, nx)[:, None]) * f

    return f, vax


def _initialize_total_distribution_(cfg, cfg_grid):
    """
    Initialize distribution functions for all species.

    The species config is normalized in modules.py:get_derived_quantities() so that
    a species config always exists (for backward compatibility with single-species
    config files, a default electron species is generated).

    Returns:
        dict mapping species_name -> (n_prof, f_s, v_ax)
    """
    species_configs = cfg["terms"]["species"]
    species_distributions = {}
    species_found = False

    for species_cfg in species_configs:
        species_name = species_cfg["name"]
        density_components = species_cfg["density_components"]
        vmax = species_cfg["vmax"]
        nv = species_cfg["nv"]

        # Initialize arrays for this species
        n_prof_species = np.zeros([cfg_grid["nx"]])
        dv = 2.0 * vmax / nv
        vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
        f_species = np.zeros([cfg_grid["nx"], nv])

        # Sum contributions from all density components
        for component_name in density_components:
            if component_name not in cfg["density"]:
                raise ValueError(f"Density component '{component_name}' not found in config")

            species_params = cfg["density"][component_name]
            v0 = species_params["v0"]
            T0 = species_params["T0"]

            # Supergaussian order from density config (default 2.0 = Maxwell-Boltzmann)
            supergaussian_order = species_params.get("m", 2.0)

            # Mass from species config for thermal velocity calculation
            mass = species_cfg["mass"]

            # Get density profile
            nprof = _get_density_profile(species_params, cfg, cfg_grid)
            n_prof_species += nprof

            # Distribution function for this component
            temp_f, _ = _initialize_supergaussian_distribution_(
                nx=int(cfg_grid["nx"]),
                nv=nv,
                v0=v0,
                supergaussian_order=supergaussian_order,
                T0=T0,
                mass=mass,
                vmax=vmax,
                n_prof=nprof,
                noise_val=species_params["noise_val"],
                noise_seed=int(species_params["noise_seed"]),
                noise_type=species_params["noise_type"],
            )
            f_species += temp_f
            species_found = True

        species_distributions[species_name] = (n_prof_species, f_species, vax)

    if not species_found:
        raise ValueError("No species found! Check the config")

    return species_distributions


def _get_density_profile(species_params, cfg, cfg_grid):
    """Extract density profile generation logic into a helper function."""
    if species_params["basis"] == "uniform":
        nprof = np.ones([cfg_grid["nx"]])

    elif species_params["basis"] == "linear":
        left = species_params["center"] - species_params["width"] * 0.5
        right = species_params["center"] + species_params["width"] * 0.5
        rise = species_params["rise"]
        mask = get_envelope(rise, rise, left, right, cfg_grid["x"])

        ureg = pint.UnitRegistry()
        _Q = ureg.Quantity

        L = (
            _Q(species_params["gradient scale length"]).to("nm").magnitude
            / cfg["units"]["derived"]["x0"].to("nm").magnitude
        )
        nprof = species_params["val at center"] + (cfg_grid["x"] - species_params["center"]) / L
        nprof = mask * nprof

    elif species_params["basis"] == "exponential":
        left = species_params["center"] - species_params["width"] * 0.5
        right = species_params["center"] + species_params["width"] * 0.5
        rise = species_params["rise"]
        mask = get_envelope(rise, rise, left, right, cfg_grid["x"])

        ureg = pint.UnitRegistry()
        _Q = ureg.Quantity

        L = (
            _Q(species_params["gradient scale length"]).to("nm").magnitude
            / cfg["units"]["derived"]["x0"].to("nm").magnitude
        )
        nprof = species_params["val at center"] * np.exp((cfg_grid["x"] - species_params["center"]) / L)
        nprof = mask * nprof

    elif species_params["basis"] == "tanh":
        left = species_params["center"] - species_params["width"] * 0.5
        right = species_params["center"] + species_params["width"] * 0.5
        rise = species_params["rise"]
        nprof = get_envelope(rise, rise, left, right, cfg_grid["x"])

        if species_params["bump_or_trough"] == "trough":
            nprof = 1 - nprof
        nprof = species_params["baseline"] + species_params["bump_height"] * nprof

    elif species_params["basis"] == "sine":
        baseline = species_params["baseline"]
        amp = species_params["amplitude"]
        kk = species_params["wavenumber"]
        nprof = baseline * (1.0 + amp * jnp.sin(kk * cfg_grid["x"]))
    else:
        raise NotImplementedError

    return nprof


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
