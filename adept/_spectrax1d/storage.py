"""
Storage functions for spectrax-1d module

This module handles saving spectrax output to disk in standard formats:
- netCDF for data (xarray datasets)
- PNG/PDF for plots
"""

import os

import numpy as np
import xarray as xr
from jax import numpy as jnp


def store_fields_timeseries(cfg: dict, fields_dict: dict, t_array: np.ndarray, binary_dir: str) -> xr.Dataset:
    """
    Store field diagnostic time series from diffeqsolve SubSaveAt output.

    This handles the output from the fields save function, which includes
    scalar diagnostics like energy and density moments.

    Args:
        cfg: Configuration dict with grid info
        fields_dict: Dict of arrays from fields save function
        t_array: Time axis for the saved data
        binary_dir: Directory path for saving netCDF files

    Returns:
        xr.Dataset: Scalar field diagnostics with time coordinate
    """
    # Create DataArrays for each saved quantity
    das = {k: xr.DataArray(v, coords=[("t", t_array)], name=k) for k, v in fields_dict.items()}

    # Combine into Dataset
    fields_xr = xr.Dataset(das)

    # Save to netCDF
    fields_xr.to_netcdf(os.path.join(binary_dir, f"field_diagnostics-t={round(t_array[-1], 4)}.nc"))

    return fields_xr


def store_scalars(cfg: dict, scalars_dict: dict, t_array: np.ndarray, binary_dir: str) -> xr.Dataset:
    """
    Store scalar diagnostics time series.

    Args:
        cfg: Configuration dict
        scalars_dict: Dict of scalar arrays (energy, amplitudes, etc.)
        t_array: Time axis for the saved data
        binary_dir: Directory path for saving netCDF files

    Returns:
        xr.Dataset: Scalar diagnostics with time coordinate
    """
    # Create DataArrays for each scalar quantity
    das = {k: xr.DataArray(v, coords=[("t", t_array)], name=k) for k, v in scalars_dict.items()}

    # Combine into Dataset
    scalars_xr = xr.Dataset(das)

    # Save to netCDF
    scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(t_array[-1], 4)}.nc"))

    return scalars_xr


def store_species_distribution_timeseries(
    cfg: dict, species_name: str, Ck_array: np.ndarray, t_array: np.ndarray, binary_dir: str
) -> xr.Dataset:
    """
    Store distribution function (Hermite-Fourier coefficients) for a single species.

    Saves the Hermite-Fourier coefficients Ck over time for one species from SubSaveAt output.

    Args:
        cfg: Configuration dict with grid info
        species_name: Name of the species ("electrons" or "ions")
        Ck_array: Array of shape (nt, Np, Nm, Nn, Ny, Nx, Nz) - Single species
        t_array: Time axis for the saved data
        binary_dir: Directory path for saving netCDF files

    Returns:
        xr.Dataset: Hermite coefficient data with time and spatial coordinates
    """
    # Get grid dimensions (species-specific for Hermite modes)
    Nn = int(cfg["grid"][f"Nn_{species_name}"])
    Nm = int(cfg["grid"][f"Nm_{species_name}"])
    Np = int(cfg["grid"][f"Np_{species_name}"])
    Nx = int(cfg["grid"]["Nx"])
    Ny = int(cfg["grid"]["Ny"])
    Nz = int(cfg["grid"]["Nz"])

    # Reshape from 6D to 4D for storage: (nt, Np, Nm, Nn, Ny, Nx, Nz) -> (nt, Np*Nm*Nn, Ny, Nx, Nz)
    nt = Ck_array.shape[0]
    Ck_array_4d = Ck_array.reshape(nt, Np * Nm * Nn, Ny, Nx, Nz)

    # Create coordinate arrays
    hermite_modes = np.arange(Nn * Nm * Np)
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0 / Nx))  # Sort kx in increasing order
    ky = np.arange(Ny)
    kz = np.arange(Nz)

    # Shift the data to match the sorted kx coordinate
    Ck_array_shifted = np.fft.fftshift(Ck_array_4d, axes=-2)  # Shift along kx dimension

    # Create Dataset with Hermite coefficients
    Ck_ds = xr.Dataset(
        {
            f"Ck_{species_name}": (["t", "hermite_mode", "ky", "kx", "kz"], Ck_array_shifted),
        },
        coords={
            "t": t_array,
            "hermite_mode": hermite_modes,
            "ky": ky,
            "kx": kx,
            "kz": kz,
        },
    )

    # Add metadata
    Ck_ds[f"Ck_{species_name}"].attrs["long_name"] = f"{species_name.capitalize()} Hermite-Fourier coefficients"
    Ck_ds[f"Ck_{species_name}"].attrs["description"] = f"Distribution function for {species_name} in Hermite-Fourier basis"
    Ck_ds[f"Ck_{species_name}"].attrs["species"] = species_name

    # Save to netCDF
    Ck_ds.to_netcdf(os.path.join(binary_dir, f"distribution_{species_name}_timeseries-t={round(t_array[-1], 4)}.nc"))

    return Ck_ds


def store_distribution_timeseries(cfg: dict, Ck_array: np.ndarray, t_array: np.ndarray, binary_dir: str) -> xr.Dataset:
    """
    Store distribution function (Hermite-Fourier coefficients) time series.

    Saves the Hermite-Fourier coefficients Ck over time from SubSaveAt output.

    Args:
        cfg: Configuration dict with grid info
        Ck_array: Array of shape (nt, Ns, Np, Nm, Nn, Ny, Nx, Nz)
        t_array: Time axis for the saved data
        binary_dir: Directory path for saving netCDF files

    Returns:
        xr.Dataset: Hermite coefficient data with time and spatial coordinates
    """
    # Get grid dimensions
    Nn = int(cfg["grid"]["Nn"])
    Nm = int(cfg["grid"]["Nm"])
    Np = int(cfg["grid"]["Np"])
    Ns = int(cfg["grid"]["Ns"])
    Nx = int(cfg["grid"]["Nx"])
    Ny = int(cfg["grid"]["Ny"])
    Nz = int(cfg["grid"]["Nz"])

    # Reshape from 7D to 4D for storage: (nt, Ns, Np, Nm, Nn, Ny, Nx, Nz) -> (nt, Ns*Np*Nm*Nn, Ny, Nx, Nz)
    nt = Ck_array.shape[0]
    Ck_array_4d = Ck_array.reshape(nt, Ns * Np * Nm * Nn, Ny, Nx, Nz)

    # Create coordinate arrays
    hermite_modes = np.arange(Ns * Nn * Nm * Np)
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0 / Nx))  # Sort kx in increasing order
    ky = np.arange(Ny)
    kz = np.arange(Nz)

    # Shift the data to match the sorted kx coordinate
    Ck_array_shifted = np.fft.fftshift(Ck_array_4d, axes=-2)  # Shift along kx dimension

    # Create Dataset with Hermite coefficients
    Ck_ds = xr.Dataset(
        {
            "Ck": (["t", "hermite_mode", "ky", "kx", "kz"], Ck_array_shifted),
        },
        coords={
            "t": t_array,
            "hermite_mode": hermite_modes,
            "ky": ky,
            "kx": kx,
            "kz": kz,
        },
    )

    # Add metadata
    Ck_ds["Ck"].attrs["long_name"] = "Hermite-Fourier coefficients"
    Ck_ds["Ck"].attrs["description"] = "Distribution function in Hermite-Fourier basis"

    # Save to netCDF
    Ck_ds.to_netcdf(os.path.join(binary_dir, f"distribution_timeseries-t={round(t_array[-1], 4)}.nc"))

    return Ck_ds


def get_field_save_func(cfg, Nx, Ny, Nz, Nn, Nm, Np, Ns):
    """
    Create a save function for electromagnetic fields and moments.

    This function computes:
    - Field energy (electric and magnetic)
    - Density moments from distribution function
    - Velocity moments

    Args:
        cfg: Configuration dict
        Nx, Ny, Nz: Fourier mode dimensions
        Nn, Nm, Np, Ns: Hermite mode dimensions

    Returns:
        Callable save function
    """

    def fields_save_func(t, y, args):
        """Extract field quantities and moments from state."""
        # Extract per-species Ck and Fk from state dictionary (convert from float64 views to complex128)
        Ck_electrons = y["Ck_electrons"].view(jnp.complex128)
        Ck_ions = y["Ck_ions"].view(jnp.complex128)
        Fk = y["Fk"].view(jnp.complex128)

        # Compute electromagnetic field energy
        # E^2 + B^2 (sum over all Fourier modes)
        E_energy = jnp.sum(jnp.abs(Fk[0:3, :, :, :]) ** 2.0)
        B_energy = jnp.sum(jnp.abs(Fk[3:6, :, :, :]) ** 2.0)
        EM_energy = E_energy + B_energy

        # Get density from n=0 Hermite mode (equilibrium)
        # Ck shape: (Np, Nm, Nn, Ny, Nx, Nz) - Per species
        # Electron density ((p,m,n)=(0,0,0) Hermite mode at k=0)
        ne_k0 = Ck_electrons[0, 0, 0, int((Ny - 1) / 2), int((Nx - 1) / 2), int((Nz - 1) / 2)]
        # Ion density ((p,m,n)=(0,0,0) Hermite mode at k=0)
        ni_k0 = Ck_ions[0, 0, 0, int((Ny - 1) / 2), int((Nx - 1) / 2), int((Nz - 1) / 2)]

        return {
            "EM_energy": EM_energy,
            "E_energy": E_energy,
            "B_energy": B_energy,
            "ne_k0": jnp.abs(ne_k0),
            "ni_k0": jnp.abs(ni_k0),
        }

    return fields_save_func


def get_distribution_save_func(Nx, Ny, Nz, Nn, Nm, Np, Ns):
    """
    Create a save function for distribution function (Hermite coefficients).

    Args:
        Nx, Ny, Nz: Fourier mode dimensions
        Nn, Nm, Np, Ns: Hermite mode dimensions

    Returns:
        Callable save function
    """

    def dist_save_func(t, y, args):
        """Extract Hermite-Fourier coefficients from state as dict of species."""
        # Extract per-species Ck from state dictionary (convert from float64 views to complex128)
        return {
            "electrons": y["Ck_electrons"].view(jnp.complex128),
            "ions": y["Ck_ions"].view(jnp.complex128),
        }

    return dist_save_func


def get_fields_only_save_func(Nx, Ny, Nz):
    """
    Create a save function for electromagnetic fields only.

    Args:
        Nx, Ny, Nz: Fourier mode dimensions

    Returns:
        Callable save function
    """

    def fields_only_save_func(t, y, args):
        """Extract electromagnetic field Fourier coefficients from state."""
        # Extract Fk from state dictionary (convert from float64 view to complex128)
        return y["Fk"].view(jnp.complex128)

    return fields_only_save_func


def get_default_save_func(Nx, Ny, Nz, Nn, Nm, Np, Ns):
    """
    Create a default save function for scalar diagnostics.

    This tracks key scalar quantities over time:
    - Total field energy
    - Field energy by component
    - Peak mode amplitudes

    Args:
        Nx, Ny, Nz: Fourier mode dimensions
        Nn, Nm, Np, Ns: Hermite mode dimensions

    Returns:
        Callable save function
    """

    def save_func(t, y, args):
        """Compute scalar diagnostics from state."""
        # Extract per-species Ck and Fk from state dictionary (convert from float64 views to complex128)
        # (Distribution functions not needed for these scalar diagnostics, but kept for consistency)
        Fk = y["Fk"].view(jnp.complex128)

        # Compute electromagnetic field energy
        E_energy = jnp.sum(jnp.abs(Fk[0:3, :, :, :]) ** 2.0)
        B_energy = jnp.sum(jnp.abs(Fk[3:6, :, :, :]) ** 2.0)
        EM_energy = E_energy + B_energy

        # Get peak field amplitudes (useful for tracking mode growth/damping)
        Ex_max = jnp.max(jnp.abs(Fk[0, :, :, :]))
        Ey_max = jnp.max(jnp.abs(Fk[1, :, :, :]))
        Ez_max = jnp.max(jnp.abs(Fk[2, :, :, :]))

        return {
            "total_EM_energy": EM_energy,
            "E_energy": E_energy,
            "B_energy": B_energy,
            "Ex_max": Ex_max,
            "Ey_max": Ey_max,
            "Ez_max": Ez_max,
        }

    return save_func


def get_save_quantities(cfg: dict) -> dict:
    """
    Configure save times and functions for diagnostics during simulation.

    This processes the cfg["save"] dictionary to create time axes and
    save functions that will be used with diffrax.SubSaveAt during the
    ODE integration.

    Reference: adept/_vlasov1d/storage.py lines 178-213

    Args:
        cfg: Configuration dict with save specification

    Returns:
        dict: Updated cfg with save functions and time axes added
    """
    # Get grid dimensions needed for save functions
    Nx = int(cfg["grid"]["Nx"])
    Ny = int(cfg["grid"]["Ny"])
    Nz = int(cfg["grid"]["Nz"])

    # Get mode counts - handle both legacy and per-species formats
    # For save functions that need these, use electron modes (typically larger)
    if "Nn_electrons" in cfg["grid"]:
        Nn = int(cfg["grid"]["Nn_electrons"])
        Nm = int(cfg["grid"]["Nm_electrons"])
        Np = int(cfg["grid"]["Np_electrons"])
    else:
        Nn = int(cfg["grid"]["Nn"])
        Nm = int(cfg["grid"]["Nm"])
        Np = int(cfg["grid"]["Np"])

    Ns = int(cfg["grid"]["Ns"])

    # Process each save type in the configuration
    for save_type, save_config in cfg["save"].items():
        if not isinstance(save_config, dict):
            continue

        # Process time dimension to create time axis
        if "t" in save_config and isinstance(save_config["t"], dict):
            t_cfg = save_config["t"]
            tmin = t_cfg.get("tmin", 0.0)
            tmax = t_cfg.get("tmax", cfg["grid"]["tmax"])
            nt = t_cfg.get("nt", cfg["grid"]["nt"])

            # Create time axis for this save type
            t_cfg["ax"] = np.linspace(tmin, tmax, nt)

        # Assign the appropriate save function based on save type
        if save_type == "fields":
            save_config["func"] = get_field_save_func(cfg, Nx, Ny, Nz, Nn, Nm, Np, Ns)
        elif save_type == "hermite" or save_type == "distribution":
            save_config["func"] = get_distribution_save_func(Nx, Ny, Nz, Nn, Nm, Np, Ns)
        elif save_type == "fields_only":
            save_config["func"] = get_fields_only_save_func(Nx, Ny, Nz)

    # Always add a default save for scalar time series
    cfg["save"]["default"] = {
        "t": {"ax": cfg["grid"]["t"]},
        "func": get_default_save_func(Nx, Ny, Nz, Nn, Nm, Np, Ns),
    }

    # Always add fields_only save for electromagnetic fields (for spacetime plots)
    # Use the same time axis as default
    if "fields_only" not in cfg["save"]:
        cfg["save"]["fields_only"] = {
            "t": {"ax": cfg["grid"]["t"]},
            "func": get_fields_only_save_func(Nx, Ny, Nz),
        }

    return cfg
