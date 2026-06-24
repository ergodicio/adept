"""
Storage / save functions for the 1D Hermite-Poisson module.

State keys: Ck_electrons (Nn_e, Nx) complex viewed as float64,
            Ck_ions (Nn_i, Nx) complex viewed as float64,
            a (Nx+2,), prev_a (Nx+2,), e (Nx,), da (Nx+2,), de (Nx,).

Supported cfg["save"] keys:
  "fields"  → {e, a, da, de} spacetime arrays
  "hermite" → {electrons, ions} Hermite-Fourier coefficient timeseries
  "default" → scalar energy diagnostics (always added automatically)
"""

import os

import numpy as np
import xarray as xr
from jax import numpy as jnp

# ---------------------------------------------------------------------------
# Save functions (called by diffrax SubSaveAt at specified timesteps)
# ---------------------------------------------------------------------------


def get_fields_save_func():
    """Save electrostatic field e, vector potential a interior, and drivers da/de."""

    def fields_save_func(t, y, args):
        out = {
            "e": y["e"],
            "a": y["a"][1:-1],
            "da": y["da"][1:-1],
        }
        if "de" in y:
            out["de"] = y["de"]  # external longitudinal (ex) driver field, (Nx,)
        return out

    return fields_save_func


def get_hermite_save_func():
    """Save Hermite-Fourier coefficients for both species."""

    def hermite_save_func(t, y, args):
        return {
            "electrons": y["Ck_electrons"].view(jnp.complex128),
            "ions": y["Ck_ions"].view(jnp.complex128),
        }

    return hermite_save_func


def get_default_save_func(alpha_e: float, alpha_i: float):
    """Save scalar diagnostics: field energies and density extrema."""
    alpha_e = float(alpha_e)
    alpha_i = float(alpha_i)

    def default_save_func(t, y, args):
        Ck_e = y["Ck_electrons"].view(jnp.complex128)
        Ck_i = y["Ck_ions"].view(jnp.complex128)
        n_e = (alpha_e**3) * jnp.fft.ifft(Ck_e[0], norm="forward").real
        n_i = (alpha_i**3) * jnp.fft.ifft(Ck_i[0], norm="forward").real
        e = y["e"]
        a = y["a"][1:-1]
        return {
            "e_energy": jnp.sum(e**2),
            "a_energy": jnp.sum(a**2),
            "n_e_max": jnp.max(n_e),
            "n_e_min": jnp.min(n_e),
            "n_i_max": jnp.max(n_i),
            "Ck_e_max": jnp.max(jnp.abs(Ck_e)),
        }

    return default_save_func


# ---------------------------------------------------------------------------
# Configure save axes and attach save functions
# ---------------------------------------------------------------------------


def get_save_quantities(cfg: dict) -> dict:
    """Attach save axes and save functions to cfg["save"].

    Called by BaseHermitePoisson1D.init_diffeqsolve() before building
    SubSaveAt objects. Modifies cfg in-place and returns it.

    Supported save keys: "fields", "hermite", "default".
    All others are passed through unchanged.
    """
    grid = cfg["grid"]
    physics = cfg.get("physics", {})

    tmax = float(grid["tmax"])
    nt = int(grid["nt"])
    alpha_e = float(physics.get("alpha_e", physics.get("alpha_s", [0.05])[0]))
    alpha_i = float(
        physics.get(
            "alpha_i",
            physics.get("alpha_s", [0.05, 0.05, 0.05, 0.001])[3] if len(physics.get("alpha_s", [])) > 3 else 0.001,
        )
    )

    for save_key, save_cfg in cfg.get("save", {}).items():
        if not isinstance(save_cfg, dict):
            continue
        # Build time axis from t sub-dict
        if "t" in save_cfg and isinstance(save_cfg["t"], dict):
            t_cfg = save_cfg["t"]
            if "ax" not in t_cfg:
                t_cfg["ax"] = np.linspace(
                    float(t_cfg.get("tmin", 0.0)),
                    float(t_cfg.get("tmax", tmax)),
                    int(t_cfg.get("nt", nt)),
                )

        if "func" in save_cfg:
            continue  # already set by caller (e.g. probe diagnostics)

        if save_key == "fields":
            save_cfg["func"] = get_fields_save_func()
        elif save_key in ("hermite", "distribution"):
            save_cfg["func"] = get_hermite_save_func()

    # Always ensure a "default" scalar-diagnostics save
    if "save" not in cfg:
        cfg["save"] = {}
    if "default" not in cfg["save"]:
        cfg["save"]["default"] = {"t": {"ax": np.linspace(0.0, tmax, nt)}}
    elif "t" not in cfg["save"]["default"]:
        cfg["save"]["default"]["t"] = {"ax": np.linspace(0.0, tmax, nt)}
    elif "ax" not in cfg["save"]["default"]["t"]:
        cfg["save"]["default"]["t"]["ax"] = np.linspace(0.0, tmax, nt)

    cfg["save"]["default"]["func"] = get_default_save_func(alpha_e, alpha_i)

    return cfg


# ---------------------------------------------------------------------------
# Post-processing storage helpers
# ---------------------------------------------------------------------------


def store_fields_timeseries(
    cfg: dict, fields_dict: dict, t_array: np.ndarray, binary_dir: str, x: np.ndarray
) -> xr.Dataset:
    """Save {e, a, da} spacetime data to netCDF."""
    das = {
        k: xr.DataArray(np.asarray(v), coords=[("t", t_array), ("x", x)], name=k)
        for k, v in fields_dict.items()
        if np.asarray(v).ndim == 2
    }
    ds = xr.Dataset(das)
    ds.to_netcdf(os.path.join(binary_dir, f"fields-t={round(float(t_array[-1]), 4)}.nc"))
    return ds


def store_ck_timeseries(
    cfg: dict, species: str, Ck_array: np.ndarray, t_array: np.ndarray, binary_dir: str
) -> xr.Dataset:
    """Save Hermite-Fourier coefficients (Nn, Nx) complex over time to netCDF.

    Ck_array shape: (nt, Nn, Nx).
    """
    Nx = Ck_array.shape[-1]
    Nn = Ck_array.shape[-2]
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0 / Nx))
    Ck_shifted = np.fft.fftshift(Ck_array, axes=-1)
    ds = xr.Dataset(
        {f"Ck_{species}": (["t", "n", "kx"], Ck_shifted)},
        coords={"t": t_array, "n": np.arange(Nn), "kx": kx},
    )
    ds.to_netcdf(os.path.join(binary_dir, f"Ck_{species}-t={round(float(t_array[-1]), 4)}.nc"))
    return ds
