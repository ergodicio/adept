"""
Storage functions for spectrax-1d module

This module handles saving spectrax output to disk in standard formats:
- netCDF for data (xarray datasets)
- PNG/PDF for plots
"""

import os

import numpy as np
import xarray as xr
from interpax import interp1d
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
    Ck_ds[f"Ck_{species_name}"].attrs["description"] = (
        f"Distribution function for {species_name} in Hermite-Fourier basis"
    )
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
        # k=0 mode is at index [0, 0, 0] in standard FFT ordering
        ne_k0 = Ck_electrons[0, 0, 0, 0, 0, 0]
        # Ion density ((p,m,n)=(0,0,0) Hermite mode at k=0)
        ni_k0 = Ck_ions[0, 0, 0, 0, 0, 0]

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


def get_fields_output_save_func(cfg, Nx, Ny, Nz, save_config):
    """
    Create a save function for electromagnetic fields with optional x/kx interpolation.
    """
    Lx = float(cfg["physics"]["Lx"])
    dx = Lx / Nx
    x_base = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    kx_base = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d=dx)) * 2.0 * jnp.pi

    x_ax = save_config.get("x", {}).get("ax", None)
    kx_ax = save_config.get("kx", {}).get("ax", None)

    def _interp_real(xq, x, f):
        return interp1d(xq=xq, x=x, f=f, extrap=True)

    def _interp_complex(xq, x, f):
        real = interp1d(xq=xq, x=x, f=jnp.real(f), extrap=True)
        imag = interp1d(xq=xq, x=x, f=jnp.imag(f), extrap=True)
        return real + 1j * imag

    def fields_output_save_func(t, y, args):
        del t, args
        Fk = y["Fk"].view(jnp.complex128)

        if x_ax is not None:
            if Ny != 1 or Nz != 1:
                raise ValueError("x interpolation for fields_only currently supports Ny=Nz=1 only")
            F_real = jnp.real(jnp.fft.ifftn(Fk, axes=(-3, -2, -1), norm="forward"))
            F1d = F_real[:, 0, :, 0]
            F_interp = jnp.stack([_interp_real(x_ax, x_base, F1d[i]) for i in range(6)], axis=0)
            return {
                "Ex": F_interp[0],
                "Ey": F_interp[1],
                "Ez": F_interp[2],
                "Bx": F_interp[3],
                "By": F_interp[4],
                "Bz": F_interp[5],
            }

        if kx_ax is not None:
            if Ny != 1 or Nz != 1:
                raise ValueError("kx interpolation for fields_only currently supports Ny=Nz=1 only")
            F1d = jnp.fft.fftshift(Fk, axes=-2)[:, 0, :, 0]
            F_interp = jnp.stack([_interp_complex(kx_ax, kx_base, F1d[i]) for i in range(6)], axis=0)
            return {
                "Ex": F_interp[0],
                "Ey": F_interp[1],
                "Ez": F_interp[2],
                "Bx": F_interp[3],
                "By": F_interp[4],
                "Bz": F_interp[5],
            }

        return Fk

    return fields_output_save_func


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


def get_moments_save_func(
    Nx,
    Ny,
    Nz,
    Nn_e,
    Nm_e,
    Np_e,
    Nn_i,
    Nm_i,
    Np_i,
    x_base=None,
    kx_base=None,
    x_ax=None,
    kx_ax=None,
):
    """
    Create a save function for low-order distribution moments in real space.

    Computes per-species:
    - density (n)
    - current components (Jx, Jy, Jz)
    - temperature proxy (T) from second Hermite coefficient
    """

    def _ifft_to_real(arr):
        return jnp.fft.ifftn(arr, axes=(-3, -2, -1), norm="forward")

    def _interp_real(xq, x, f):
        return interp1d(xq=xq, x=x, f=f, extrap=True)

    def _interp_complex(xq, x, f):
        real = interp1d(xq=xq, x=x, f=jnp.real(f), extrap=True)
        imag = interp1d(xq=xq, x=x, f=jnp.imag(f), extrap=True)
        return real + 1j * imag

    def _moments_species(Ck, alpha, u, Nn, Nm, Np, prefix):
        C0 = Ck[0, 0, 0]
        C100 = Ck[0, 0, 1] if Nn > 1 else jnp.zeros_like(C0)
        C010 = Ck[0, 1, 0] if Nm > 1 else jnp.zeros_like(C0)
        C001 = Ck[1, 0, 0] if Np > 1 else jnp.zeros_like(C0)

        C0_r = _ifft_to_real(C0)
        C100_r = _ifft_to_real(C100)
        C010_r = _ifft_to_real(C010)
        C001_r = _ifft_to_real(C001)

        a0, a1, a2 = alpha[0], alpha[1], alpha[2]
        u0, u1, u2 = u[0], u[1], u[2]
        pre = a0 * a1 * a2

        n = pre * C0_r
        Jx = pre * (u0 * C0_r + (1.0 / jnp.sqrt(2.0)) * a0 * C100_r)
        Jy = pre * (u1 * C0_r + (1.0 / jnp.sqrt(2.0)) * a1 * C010_r)
        Jz = pre * (u2 * C0_r + (1.0 / jnp.sqrt(2.0)) * a2 * C001_r)

        # Temperature proxy from second Hermite coefficient (C200/C020/C002)
        # This matches the Hermite basis convention: C2=0 -> T = alpha^2 / 2.
        eps = 1e-20
        C0_safe = jnp.where(jnp.abs(C0_r) < eps, eps + 0j, C0_r)
        if Nn > 2:
            C200 = _ifft_to_real(Ck[0, 0, 2])
            Tx = 0.5 * a0**2 * (1.0 + jnp.sqrt(2.0) * jnp.real(C200 / C0_safe))
        else:
            Tx = 0.5 * a0**2 * jnp.ones_like(C0_r)

        if Nm > 2:
            C020 = _ifft_to_real(Ck[0, 2, 0])
            Ty = 0.5 * a1**2 * (1.0 + jnp.sqrt(2.0) * jnp.real(C020 / C0_safe))
        else:
            Ty = 0.5 * a1**2 * jnp.ones_like(C0_r)

        if Np > 2:
            C002 = _ifft_to_real(Ck[2, 0, 0])
            Tz = 0.5 * a2**2 * (1.0 + jnp.sqrt(2.0) * jnp.real(C002 / C0_safe))
        else:
            Tz = 0.5 * a2**2 * jnp.ones_like(C0_r)

        T = (Tx + Ty + Tz) / 3.0

        out = {
            f"n_{prefix}": jnp.real(n),
            f"Jx_{prefix}": jnp.real(Jx),
            f"Jy_{prefix}": jnp.real(Jy),
            f"Jz_{prefix}": jnp.real(Jz),
            f"T_{prefix}": jnp.real(T),
        }

        if x_ax is not None and x_base is not None:
            for key, val in out.items():
                out[key] = _interp_real(x_ax, x_base, val)

        if kx_ax is not None and kx_base is not None:
            for key, val in out.items():
                f_k = jnp.fft.fftshift(jnp.fft.fftn(val[None, :, None], axes=(-3, -2, -1), norm="forward"), axes=-2)[
                    0, :, 0
                ]
                out[key] = _interp_complex(kx_ax, kx_base, f_k)

        return out

    def moments_save_func(t, y, args):
        del t
        Ck_electrons = y["Ck_electrons"].view(jnp.complex128)
        Ck_ions = y["Ck_ions"].view(jnp.complex128)

        alpha_s = args["alpha_s"].reshape(2, 3)
        u_s = args["u_s"].reshape(2, 3)

        moments = {}
        moments.update(_moments_species(Ck_electrons, alpha_s[0], u_s[0], Nn_e, Nm_e, Np_e, "e"))
        moments.update(_moments_species(Ck_ions, alpha_s[1], u_s[1], Nn_i, Nm_i, Np_i, "i"))
        moments["Jx_total"] = moments["Jx_e"] + moments["Jx_i"]
        moments["Jy_total"] = moments["Jy_e"] + moments["Jy_i"]
        moments["Jz_total"] = moments["Jz_e"] + moments["Jz_i"]

        return moments

    return moments_save_func


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

    Lx = float(cfg["physics"]["Lx"])
    dx = Lx / Nx
    x_base = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    kx_base = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d=dx)) * 2.0 * np.pi

    # Process each save type in the configuration
    for save_type, save_config in cfg["save"].items():
        if not isinstance(save_config, dict):
            continue

        # Process time dimension to create time axis
        if "t" in save_config and isinstance(save_config["t"], dict):
            t_cfg = save_config["t"]
            if "ax" not in t_cfg:
                tmin = t_cfg.get("tmin", 0.0)
                tmax = t_cfg.get("tmax", cfg["grid"]["tmax"])
                nt = t_cfg.get("nt", cfg["grid"]["nt"])

                # Create time axis for this save type
                t_cfg["ax"] = np.linspace(tmin, tmax, nt)

        # Optional x/kx interpolation axes
        for dim_key in ["x", "kx"]:
            if dim_key in save_config and isinstance(save_config[dim_key], dict):
                dim_cfg = save_config[dim_key]
                dmin = dim_cfg.get(f"{dim_key}min", None)
                dmax = dim_cfg.get(f"{dim_key}max", None)
                dn = dim_cfg.get(f"n{dim_key}", None)
                if dmin is None or dmax is None or dn is None:
                    raise ValueError(f"save.{save_type}.{dim_key} requires {dim_key}min, {dim_key}max, n{dim_key}")
                if dim_key == "x":
                    dd = (dmax - dmin) / dn
                    dim_cfg["ax"] = np.linspace(dmin + dd / 2.0, dmax - dd / 2.0, dn)
                else:
                    dim_cfg["ax"] = np.linspace(dmin, dmax, dn)

        # Assign the appropriate save function based on save type
        if save_type == "fields":
            save_config["func"] = get_field_save_func(cfg, Nx, Ny, Nz, Nn, Nm, Np, Ns)
        elif save_type == "hermite" or save_type == "distribution":
            save_config["func"] = get_distribution_save_func(Nx, Ny, Nz, Nn, Nm, Np, Ns)
        elif save_type == "moments":
            # Use per-species Hermite mode counts if available
            if "Nn_electrons" in cfg["grid"]:
                Nn_e = int(cfg["grid"]["Nn_electrons"])
                Nm_e = int(cfg["grid"]["Nm_electrons"])
                Np_e = int(cfg["grid"]["Np_electrons"])
                Nn_i = int(cfg["grid"]["Nn_ions"])
                Nm_i = int(cfg["grid"]["Nm_ions"])
                Np_i = int(cfg["grid"]["Np_ions"])
            else:
                Nn_e = Nn_i = Nn
                Nm_e = Nm_i = Nm
                Np_e = Np_i = Np
            save_config["func"] = get_moments_save_func(
                Nx,
                Ny,
                Nz,
                Nn_e,
                Nm_e,
                Np_e,
                Nn_i,
                Nm_i,
                Np_i,
                x_base=x_base,
                kx_base=kx_base,
                x_ax=save_config.get("x", {}).get("ax", None),
                kx_ax=save_config.get("kx", {}).get("ax", None),
            )
        elif save_type == "fields_only":
            if "x" in save_config or "kx" in save_config:
                save_config["func"] = get_fields_output_save_func(cfg, Nx, Ny, Nz, save_config)
            else:
                save_config["func"] = get_fields_only_save_func(Nx, Ny, Nz)

    # Always add a default save for scalar time series (respect user override if provided)
    if "default" not in cfg["save"]:
        cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}}
    elif "t" not in cfg["save"]["default"]:
        cfg["save"]["default"]["t"] = {"ax": cfg["grid"]["t"]}
    if "ax" not in cfg["save"]["default"]["t"]:
        cfg["save"]["default"]["t"]["ax"] = cfg["grid"]["t"]
    cfg["save"]["default"]["func"] = get_default_save_func(Nx, Ny, Nz, Nn, Nm, Np, Ns)

    # Always add fields_only save for electromagnetic fields (for spacetime plots)
    # Use the same time axis as default
    if "fields_only_plot" not in cfg["save"]:
        cfg["save"]["fields_only_plot"] = {"t": {"ax": cfg["grid"]["t"]}}
    elif "t" not in cfg["save"]["fields_only_plot"]:
        cfg["save"]["fields_only_plot"]["t"] = {"ax": cfg["grid"]["t"]}
    if "ax" not in cfg["save"]["fields_only_plot"]["t"]:
        cfg["save"]["fields_only_plot"]["t"]["ax"] = cfg["grid"]["t"]
    cfg["save"]["fields_only_plot"]["func"] = get_fields_only_save_func(Nx, Ny, Nz)

    return cfg
