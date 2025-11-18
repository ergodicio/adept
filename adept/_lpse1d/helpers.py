import os
from functools import partial

import interpax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.units import Quantity as _Q
from jax import numpy as jnp

from adept._base_ import get_envelope
from adept._lpse2d.helpers import next_smooth_fft_size


def get_derived_quantities_1d(cfg: dict) -> dict:
    """
    Calculate derived grid quantities for 1D simulations.

    This is the 1D version of the 2D get_derived_quantities.
    Sets up grid spacing, time stepping, and validates parameters.

    Args:
        cfg: Configuration dictionary

    Returns:
        Updated configuration dictionary
    """
    cfg_grid = cfg["grid"]

    # Set up x-domain
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

    if "x" in cfg.get("save", {}):
        cfg["save"]["x"]["xmax"] = cfg_grid["xmax"]

    dx = cfg_grid["dx"] = _Q(cfg_grid["dx"]).to("um").value

    # Calculate optimal nx for FFT performance
    cfg_grid["nx"] = int((xmax - xmin) / dx)
    cfg_grid["nx"] = next_smooth_fft_size(cfg_grid["nx"], max_prime=5)
    cfg_grid["dx"] = dx = (xmax - xmin) / cfg_grid["nx"]  # Recalculate dx based on optimal nx

    # Calculate dt based on stability conditions if SRS is enabled
    # This matches MATLAB lines 496-520
    if cfg["terms"]["epw"]["source"].get("srs", False):
        # Get derived units (already calculated by write_units)
        c = float(cfg["units"]["derived"]["c"])
        w0 = float(cfg["units"]["derived"]["w0"])
        wp0 = float(cfg["units"]["derived"]["wp0"])
        w1 = float(cfg["units"]["derived"]["w1"])

        # Calculate plasma frequency at maximum density
        nmax = cfg["density"]["max"]
        wpe_max = w0 * np.sqrt(nmax)

        # Stability condition for pump (w0) field
        # MATLAB line 499: dt_max_pump = 1/(c^2/(dx^2*w0) - (w0^2 - max(wpe(:))^2)/(4*w0))
        dt_max_pump = 1.0 / (c**2 / (dx**2 * w0) - (w0**2 - wpe_max**2) / (4 * w0))

        # Stability condition for seed (w1) field
        # MATLAB line 500: dt_max_seed = 1/(c^2/(dx^2*w1) - (w1^2 - max(wpe(:))^2)/(4*w1))
        dt_max_seed = 1.0 / (c**2 / (dx**2 * w1) - (w1**2 - wpe_max**2) / (4 * w1))

        # Take minimum for stability
        # MATLAB line 501: dt_max = min([dt_max_pump,dt_max_seed])
        dt_max = min(dt_max_pump, dt_max_seed)

        # Apply fraction
        # MATLAB line 519: dt = dtFraction * dt_max
        dtFraction = cfg_grid.get("dtFraction", 0.9)
        cfg_grid["dt"] = dtFraction * dt_max
        print(f"Calculated dt = {cfg_grid['dt']:.6g} ps (dtFraction = {dtFraction}, dt_max = {dt_max:.6g} ps)")
    else:
        cfg_grid["dt"] = _Q(cfg_grid["dt"]).to("ps").value

    cfg_grid["tmax"] = _Q(cfg_grid["tmax"]).to("ps").value
    cfg_grid["nt"] = int(cfg_grid["tmax"] / cfg_grid["dt"] + 1)
    cfg_grid["tmax"] = cfg_grid["dt"] * cfg_grid["nt"]

    cfg_grid["max_steps"] = cfg_grid["nt"] + 2048

    cfg["grid"] = cfg_grid

    return cfg


def get_solver_quantities_1d(cfg: dict) -> dict:
    """
    Set up 1D solver quantities (k-space grids, filters, boundaries).

    Adapted from 2D version for 1D grids.

    Args:
        cfg: Configuration dictionary

    Returns:
        Updated grid configuration
    """
    cfg_grid = cfg["grid"]

    # 1D spatial grid
    cfg_grid["x"] = np.linspace(
        cfg_grid["xmin"] + cfg_grid["dx"] / 2,
        cfg_grid["xmax"] - cfg_grid["dx"] / 2,
        cfg_grid["nx"],
    )

    # Time grid
    cfg_grid["t"] = np.linspace(0, cfg_grid["tmax"], cfg_grid["nt"])

    # K-space grid (1D)
    cfg_grid["kx"] = np.fft.fftfreq(cfg_grid["nx"], d=cfg_grid["dx"] / 2.0 / np.pi)

    # 1/k utilities
    one_over_kx = np.zeros_like(cfg_grid["kx"])
    one_over_kx[1:] = 1.0 / cfg_grid["kx"][1:]
    cfg_grid["one_over_kx"] = np.array(one_over_kx)

    # 1/k^2
    one_over_ksq = np.array(1.0 / (cfg_grid["kx"] ** 2.0))
    one_over_ksq[0] = 0.0
    cfg_grid["one_over_ksq"] = np.array(one_over_ksq)

    # Absorbing boundaries (1D)
    boundary_width = cfg_grid.get("boundary_width", 3.0)  # in um
    rise = boundary_width / 5

    if cfg["terms"]["epw"]["boundary"]["x"] == "absorbing":
        left = cfg["grid"]["xmin"] + boundary_width
        right = cfg["grid"]["xmax"] - boundary_width
        envelope_x = get_envelope(rise, rise, left, right, cfg_grid["x"])
    else:
        envelope_x = np.ones((cfg_grid["nx"],))

    cfg_grid["absorbing_boundaries"] = np.exp(
        -float(cfg_grid.get("boundary_abs_coeff", 10.0)) * cfg_grid["dt"] * (1.0 - envelope_x)
    )

    # Zero mask for k=0
    cfg_grid["zero_mask"] = np.where(cfg_grid["kx"] == 0, 0, 1) if cfg["terms"]["zero_mask"] else 1

    # Low-pass filter
    k_mag = np.abs(cfg_grid["kx"])
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

    return cfg_grid


def get_save_quantities_1d(cfg: dict) -> dict:
    """
    Set up save/diagnostic functions for 1D.

    Args:
        cfg: Configuration dictionary

    Returns:
        Updated configuration with save functions
    """
    # Set up time axis for field saves
    tmin = _Q(cfg["save"]["fields"]["t"]["tmin"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    tmax = _Q(cfg["save"]["fields"]["t"]["tmax"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    dt = _Q(cfg["save"]["fields"]["t"]["dt"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    nt = int((tmax - tmin) / dt) + 1

    cfg["save"]["fields"]["t"]["dt"] = dt
    cfg["save"]["fields"]["t"]["ax"] = jnp.linspace(tmin, tmax, nt)

    # Set up spatial interpolation if different resolution requested
    if "x" in cfg["save"]["fields"]:
        xmin = cfg["grid"]["xmin"]
        xmax = cfg["grid"]["xmax"]
        dx = _Q(cfg["save"]["fields"]["x"]["dx"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
        nx = int((xmax - xmin) / dx)
        cfg["save"]["fields"]["x"]["dx"] = dx
        cfg["save"]["fields"]["x"]["ax"] = jnp.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
        cfg["save"]["fields"]["kx"] = np.fft.fftfreq(nx, d=dx / 2.0 / np.pi)

        # Create interpolator
        xq = cfg["save"]["fields"]["x"]["ax"]
        interpolator = partial(
            interpax.interp1d,
            xq=xq,
            x=cfg["grid"]["x"],
            method="linear",
        )

        def fields_save_func(t, y, args):
            save_y = {}
            for k, v in y.items():
                if k in ["E0", "E1"]:
                    # For laser fields (complex scalars in 1D)
                    cmplx_fld = v.view(jnp.complex128)
                    save_y[k] = interpolator(f=cmplx_fld).view(jnp.float64)
                elif k == "epw":
                    # For EPW (complex in k-space)
                    cmplx_fld = v.view(jnp.complex128)
                    # Transform to real space for interpolation
                    real_space = jnp.fft.ifft(cmplx_fld)
                    interpolated = interpolator(f=real_space)
                    # Transform back to k-space at new resolution
                    save_y[k] = jnp.fft.fft(interpolated).view(jnp.float64)
                else:
                    save_y[k] = interpolator(f=v)
            return save_y

    else:
        # No interpolation, save at native resolution
        def fields_save_func(t, y, args):
            return y

    # Default save function (energy diagnostics)
    def default_save_func(t, y, args):
        phi_k = y["epw"].view(jnp.complex128)
        e = -1j * cfg["grid"]["kx"] * phi_k
        e = jnp.fft.ifft(e)
        e_sq = jnp.abs(e) ** 2

        E0 = y["E0"].view(jnp.complex128)
        E1 = y["E1"].view(jnp.complex128)
        E0_sq = jnp.abs(E0) ** 2
        E1_sq = jnp.abs(E1) ** 2

        return {
            "epw_energy": jnp.sum(e_sq * cfg["grid"]["dx"]),
            "E0_energy": jnp.sum(E0_sq * cfg["grid"]["dx"]),
            "E1_energy": jnp.sum(E1_sq * cfg["grid"]["dx"]),
            "max_phi": jnp.max(jnp.abs(phi_k)),
            "max_E0": jnp.max(jnp.abs(E0)),
            "max_E1": jnp.max(jnp.abs(E1)),
        }

    cfg["save"] = {
        "default": {"t": {"ax": cfg["grid"]["t"]}, "func": default_save_func},
        "fields": {**cfg["save"]["fields"], "func": fields_save_func},
    }

    return cfg


def post_process_1d(result, cfg: dict, td: str) -> dict:
    """
    Post-process 1D simulation results.

    Creates xarray datasets for:
    - Time series (energies, maxima)
    - Fields in real space (phi, E0, E1, e)
    - Fields in k-space (phi_k, e_k)

    Args:
        result: Solver result from diffeqsolve
        cfg: Configuration dictionary
        td: Target directory

    Returns:
        Dictionary with processed data
    """
    os.makedirs(td, exist_ok=True)
    os.makedirs(os.path.join(td, "binary"), exist_ok=True)

    # ========================================================================
    # Extract time series
    # ========================================================================
    this_t = result.ts["default"]
    state = result.ys["default"]

    series_xr = xr.Dataset(
        {
            "epw_energy": xr.DataArray(state["epw_energy"], coords=(("t (ps)", this_t),)),
            "E0_energy": xr.DataArray(state["E0_energy"], coords=(("t (ps)", this_t),)),
            "E1_energy": xr.DataArray(state["E1_energy"], coords=(("t (ps)", this_t),)),
            "max_phi": xr.DataArray(state["max_phi"], coords=(("t (ps)", this_t),)),
            "max_E0": xr.DataArray(state["max_E0"], coords=(("t (ps)", this_t),)),
            "max_E1": xr.DataArray(state["max_E1"], coords=(("t (ps)", this_t),)),
        }
    )

    series_xr.to_netcdf(os.path.join(td, "binary", "series.nc"), engine="h5netcdf", invalid_netcdf=True)

    # ========================================================================
    # Extract and process field data
    # ========================================================================
    if "fields" in result.ts:
        field_t = result.ts["fields"]
        field_state = result.ys["fields"]

        # Determine which grid we're using
        if "x" in cfg["save"]["fields"]:
            kx = cfg["save"]["fields"]["kx"]
            xax = cfg["save"]["fields"]["x"]["ax"]
            nx = cfg["save"]["fields"]["x"]["ax"].size
        else:
            kx = cfg["grid"]["kx"]
            xax = cfg["grid"]["x"]
            nx = cfg["grid"]["nx"]

        # Convert to numpy for processing
        phi_k_np = np.array(field_state["epw"]).view(np.complex64)
        E0_np = np.array(field_state["E0"]).view(np.complex64)
        E1_np = np.array(field_state["E1"]).view(np.complex64)

        # Calculate electric field from phi_k
        e_k_np = -1j * kx[None, :] * phi_k_np

        # Transform to real space
        phi_vs_t = np.fft.ifft(phi_k_np, axis=1)
        e_vs_t = np.fft.ifft(e_k_np, axis=1)

        # Create coordinate tuples
        tax_tuple = ("t (ps)", field_t)
        xax_tuple = ("x (um)", xax)
        shift_kx = np.fft.fftshift(kx) * cfg["units"]["derived"]["c"] / cfg["units"]["derived"]["w0"]
        kax_tuple = (r"kx ($kc\omega_0^{-1}$)", shift_kx)

        # ====================================================================
        # K-space fields
        # ====================================================================
        kfields = xr.Dataset(
            {
                "phi": xr.DataArray(np.fft.fftshift(phi_k_np, axes=1), coords=(tax_tuple, kax_tuple)),
                "e": xr.DataArray(np.fft.fftshift(e_k_np, axes=1), coords=(tax_tuple, kax_tuple)),
            }
        )
        kfields.to_netcdf(os.path.join(td, "binary", "k-fields.nc"), engine="h5netcdf", invalid_netcdf=True)

        # ====================================================================
        # Real space fields
        # ====================================================================
        # Get density profile for saving
        from adept._lpse2d.helpers import get_density_profile

        density_2d = get_density_profile(cfg)
        density_1d = density_2d[:, 0]  # Take first column

        # Interpolate density to save grid if needed
        if "x" in cfg["save"]["fields"]:
            from scipy import interpolate

            density_interpolator = interpolate.interp1d(
                cfg["grid"]["x"], density_1d, kind="linear", bounds_error=False, fill_value=0.0
            )
            density_on_save_grid = density_interpolator(xax)
        else:
            density_on_save_grid = density_1d

        background_density = xr.DataArray(
            np.repeat(density_on_save_grid[None, ...], repeats=len(field_t), axis=0),
            coords=(tax_tuple, xax_tuple),
        )

        fields = xr.Dataset(
            {
                "phi": xr.DataArray(phi_vs_t, coords=(tax_tuple, xax_tuple)),
                "e": xr.DataArray(e_vs_t, coords=(tax_tuple, xax_tuple)),
                "E0": xr.DataArray(E0_np, coords=(tax_tuple, xax_tuple)),
                "E1": xr.DataArray(E1_np, coords=(tax_tuple, xax_tuple)),
                "background_density": background_density,
            }
        )
        fields.to_netcdf(os.path.join(td, "binary", "fields.nc"), engine="h5netcdf", invalid_netcdf=True)

        # ====================================================================
        # Create plots
        # ====================================================================
        try:
            plot_fields_1d(fields, kfields, series_xr, td)
        except Exception as e:
            print(f"Warning: Plotting failed with error: {e}")

        return {"series": series_xr, "fields": fields, "kfields": kfields}
    else:
        return {"series": series_xr}


def plot_fields_1d(fields, kfields, series, td: str):
    """
    Create diagnostic plots for 1D simulation.

    Args:
        fields: xarray Dataset with real-space fields
        kfields: xarray Dataset with k-space fields
        series: xarray Dataset with time series
        td: Target directory
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(td, "plots"), exist_ok=True)

    # ========================================================================
    # Time series plots
    # ========================================================================
    for key in series.data_vars:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        series[key].plot(ax=ax[0])
        ax[0].set_title(f"{key} vs time")
        series[key].plot(ax=ax[1])
        ax[1].set_yscale("log")
        ax[1].set_title(f"{key} vs time (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(td, "plots", f"{key}_vs_t.png"), dpi=150)
        plt.close()

    # ========================================================================
    # Space-time plots
    # ========================================================================
    for key in ["phi", "e", "E0", "E1"]:
        if key in fields:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Absolute value
            np.abs(fields[key]).plot(ax=axes[0, 0])
            axes[0, 0].set_title(f"|{key}(x,t)|")

            # Log absolute value
            np.log10(np.abs(fields[key]) + 1e-20).plot(ax=axes[0, 1], vmin=-15)
            axes[0, 1].set_title(f"log10|{key}(x,t)|")

            # Real part
            np.real(fields[key]).plot(ax=axes[1, 0])
            axes[1, 0].set_title(f"Re[{key}(x,t)]")

            # Imaginary part
            np.imag(fields[key]).plot(ax=axes[1, 1])
            axes[1, 1].set_title(f"Im[{key}(x,t)]")

            plt.tight_layout()
            plt.savefig(os.path.join(td, "plots", f"{key}_spacetime.png"), dpi=150)
            plt.close()

    # ========================================================================
    # K-space plots
    # ========================================================================
    for key in ["phi", "e"]:
        if key in kfields:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            np.log10(np.abs(kfields[key]) + 1e-20).plot(ax=ax, vmin=-15)
            ax.set_title(f"log10|{key}(k,t)|")
            plt.tight_layout()
            plt.savefig(os.path.join(td, "plots", f"{key}_k_spacetime.png"), dpi=150)
            plt.close()

    print(f"Plots saved to {os.path.join(td, 'plots')}")
