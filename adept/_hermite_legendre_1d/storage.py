"""
Storage / save functions for the 1D mixed Hermite-Legendre solver.

State keys: Ck (Nh, Nx) complex viewed as float64, Bk (Nl, Nx) complex viewed as
float64, e (Nx,), phi (Nx,).

Supported cfg["save"] keys:
  "fields"   -> {e, phi} spacetime arrays
  "hermite"  -> Ck Hermite-Fourier coefficient timeseries
  "legendre" -> Bk Legendre-Fourier coefficient timeseries
  "default"  -> scalar invariants (mass, momentum, energy) + field energy. Always added.

The scalar invariants follow the analytic definitions of the mixed method (paper
eqns 26, 28, 30-31). With the conservation constraint J_{Nh,0..2}=0 enforced they
are conserved to solver tolerance, which is the primary correctness gate.
"""

import os

import numpy as np
import xarray as xr
from jax import numpy as jnp

# ---------------------------------------------------------------------------
# Save functions (called by diffrax SubSaveAt at specified timesteps)
# ---------------------------------------------------------------------------


def get_fields_save_func():
    """Save electric field e, potential phi, and external Ex driver field de."""

    def fields_save_func(t, y, args):
        out = {"e": y["e"], "phi": y["phi"]}
        if "de" in y:
            out["de"] = y["de"]
        return out

    return fields_save_func


def get_hermite_save_func():
    def hermite_save_func(t, y, args):
        return {"Ck": y["Ck"].view(jnp.complex128)}

    return hermite_save_func


def get_legendre_save_func():
    def legendre_save_func(t, y, args):
        return {"Bk": y["Bk"].view(jnp.complex128)}

    return legendre_save_func


def get_default_save_func(
    alpha: float, u: float, width: float, sigma1: float, sigma2: float, sigma_bar: float, Lx: float
):
    """Save scalar invariants: total mass, momentum, energy, and field energy.

    Spatial integrals reduce to the domain length times the k=0 Fourier mode, e.g.
    integral C_n dx = Lx * Re(Ck[n, 0]) since norm="forward" puts the mean in [0].
    """
    alpha = float(alpha)
    u = float(u)
    width = float(width)
    sigma1 = float(sigma1)
    sigma2 = float(sigma2)
    sigma_bar = float(sigma_bar)
    Lx = float(Lx)

    def default_save_func(t, y, args):
        Ck = y["Ck"].view(jnp.complex128)
        Bk = y["Bk"].view(jnp.complex128)
        e = y["e"]
        nx = e.shape[0]
        dx_local = Lx / nx

        def intg(arr_k, n):
            return Lx * jnp.real(arr_k[n, 0]) if arr_k.shape[0] > n else 0.0

        C0, C1, C2 = intg(Ck, 0), intg(Ck, 1), intg(Ck, 2)
        B0, B1, B2 = intg(Bk, 0), intg(Bk, 1), intg(Bk, 2)

        mass = alpha * C0 + width * B0
        momentum = (alpha**2 / jnp.sqrt(2.0)) * C1 + u * alpha * C0 + width * sigma1 * B1 + sigma_bar * width * B0
        e_kin = (alpha / 2.0) * (
            (alpha**2 / jnp.sqrt(2.0)) * C2 + jnp.sqrt(2.0) * u * alpha * C1 + (alpha**2 / 2.0 + u**2) * C0
        ) + (width / 2.0) * (sigma2 * sigma1 * B2 + 2.0 * sigma1 * sigma_bar * B1 + (sigma1**2 + sigma_bar**2) * B0)
        e_pot = 0.5 * jnp.sum(e**2) * dx_local
        energy = e_kin + e_pot

        # density extrema for blow-up monitoring
        n_f0 = alpha * jnp.fft.ifft(Ck[0], norm="forward").real
        n_df = width * jnp.fft.ifft(Bk[0], norm="forward").real
        n_e = n_f0 + n_df

        return {
            "mass": mass,
            "momentum": momentum,
            "energy": energy,
            "e_energy": e_pot,
            "n_e_max": jnp.max(n_e),
            "n_e_min": jnp.min(n_e),
            "Bk_max": jnp.max(jnp.abs(Bk)),
            "Ck_max": jnp.max(jnp.abs(Ck)),
        }

    return default_save_func


# ---------------------------------------------------------------------------
# Configure save axes and attach save functions
# ---------------------------------------------------------------------------


def get_save_quantities(cfg: dict) -> dict:
    """Attach time axes and save functions to cfg["save"]. Modifies cfg in place."""
    grid = cfg["grid"]
    physics = cfg["physics"]

    tmax = float(grid["tmax"])
    nt = int(grid["nt"])

    alpha = float(physics["alpha"])
    u = float(physics.get("u", 0.0))
    v_a = float(physics["v_a"])
    v_b = float(physics["v_b"])
    width = v_b - v_a
    sigma1 = (width / 2.0) * 1.0 / np.sqrt(3.0 * 1.0)  # n=1
    sigma2 = (width / 2.0) * 2.0 / np.sqrt(5.0 * 3.0)  # n=2
    sigma_bar = 0.5 * (v_a + v_b)
    Lx = float(physics["Lx"])

    for save_key, save_cfg in cfg.get("save", {}).items():
        if not isinstance(save_cfg, dict):
            continue
        if "t" in save_cfg and isinstance(save_cfg["t"], dict):
            t_cfg = save_cfg["t"]
            if "ax" not in t_cfg:
                t_cfg["ax"] = np.linspace(
                    float(t_cfg.get("tmin", 0.0)), float(t_cfg.get("tmax", tmax)), int(t_cfg.get("nt", nt))
                )
        if "func" in save_cfg:
            continue
        if save_key == "fields":
            save_cfg["func"] = get_fields_save_func()
        elif save_key == "hermite":
            save_cfg["func"] = get_hermite_save_func()
        elif save_key == "legendre":
            save_cfg["func"] = get_legendre_save_func()

    if "save" not in cfg:
        cfg["save"] = {}
    if "default" not in cfg["save"]:
        cfg["save"]["default"] = {"t": {"ax": np.linspace(0.0, tmax, nt)}}
    elif "t" not in cfg["save"]["default"]:
        cfg["save"]["default"]["t"] = {"ax": np.linspace(0.0, tmax, nt)}
    elif "ax" not in cfg["save"]["default"]["t"]:
        cfg["save"]["default"]["t"]["ax"] = np.linspace(0.0, tmax, nt)

    cfg["save"]["default"]["func"] = get_default_save_func(alpha, u, width, sigma1, sigma2, sigma_bar, Lx)
    return cfg


# ---------------------------------------------------------------------------
# Post-processing storage helpers
# ---------------------------------------------------------------------------


def store_fields_timeseries(fields_dict: dict, t_array: np.ndarray, binary_dir: str, x: np.ndarray) -> xr.Dataset:
    das = {
        k: xr.DataArray(np.asarray(v), coords=[("t", t_array), ("x", x)], name=k)
        for k, v in fields_dict.items()
        if np.asarray(v).ndim == 2
    }
    ds = xr.Dataset(das)
    ds.to_netcdf(os.path.join(binary_dir, f"fields-t={round(float(t_array[-1]), 4)}.nc"))
    return ds


def store_coeff_timeseries(name: str, arr: np.ndarray, t_array: np.ndarray, binary_dir: str) -> xr.Dataset:
    """Save (nt, Nmodes, Nx) complex coefficient timeseries to netCDF."""
    Nx = arr.shape[-1]
    Nmodes = arr.shape[-2]
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0 / Nx))
    arr_shifted = np.fft.fftshift(arr, axes=-1)
    ds = xr.Dataset(
        {name: (["t", "mode", "kx"], arr_shifted)},
        coords={"t": t_array, "mode": np.arange(Nmodes), "kx": kx},
    )
    ds.to_netcdf(os.path.join(binary_dir, f"{name}-t={round(float(t_array[-1]), 4)}.nc"))
    return ds
