"""Save-function construction and netCDF writers for Vlasov-1D output."""

import os
import warnings

import numpy as np
import xarray as xr
from jax import numpy as jnp

from adept._vlasov1d.observations import (
    DistributionObservation,
    FieldsObservation,
    InterpolatedDistributionObservation,
    ScalarsObservation,
)


def field_datasets(cfg: dict, fields: dict, this_t: np.ndarray, prefix: str) -> dict:
    """
    Construct field datasets in memory, handling multispecies data.

    :param prefix:
    :param cfg:
    :param fields: dict with species names as keys (each containing moment dicts) and shared field keys at top level
    :param this_t:
    :return: dict mapping species names to xr.Dataset of moments, plus "fields" key for shared fields
    """
    result = {}
    # Shared field keys at top level
    shared_field_keys = {"e", "de", "a", "prev_a", "pond", "ep", "em"}
    species_names = [k for k in fields.keys() if k not in shared_field_keys]

    # Store species-specific moments
    for species_name in species_names:
        species_moments = fields[species_name]
        das = {}
        for k, v in species_moments.items():
            das[f"{prefix}-{k}"] = xr.DataArray(v, coords=(("t", this_t), ("x", cfg["grid"]["x"])))

        species_xr = xr.Dataset(das)
        result[species_name] = species_xr

    # Store shared field data (at top level of fields dict)
    das = {}
    for k in ["e", "de", "a", "prev_a", "pond"]:
        if k in fields:
            v = fields[k]
            das[f"{prefix}-{k}"] = xr.DataArray(
                v[:, 1:-1] if k in ["a", "prev_a"] else v, coords=(("t", this_t), ("x", cfg["grid"]["x"]))
            )

    if len(cfg["drivers"]["ey"].keys()) > 0 and "a" in fields and "prev_a" in fields:
        # ey = -dA/dt computed from (a - prev_a)/dt gives E_y at time t - dt/2
        ey = -(fields["a"][:, 1:-1] - fields["prev_a"][:, 1:-1]) / cfg["grid"]["dt"]

        # bz = dA/dx must be computed at the same time t - dt/2 for correct em/ep split
        # Average the gradient at t and t-dt to get the value at t - dt/2
        bz_t = jnp.gradient(fields["a"], cfg["grid"]["dx"], axis=1)[:, 1:-1]
        bz_prev = jnp.gradient(fields["prev_a"], cfg["grid"]["dx"], axis=1)[:, 1:-1]
        bz = 0.5 * (bz_t + bz_prev)

        c_light = cfg["units"]["derived"]["c_light"]
        ep = ey + c_light * bz
        em = ey - c_light * bz

        das[f"{prefix}-ep"] = xr.DataArray(ep, coords=(("t", this_t), ("x", cfg["grid"]["x"])))
        das[f"{prefix}-em"] = xr.DataArray(em, coords=(("t", this_t), ("x", cfg["grid"]["x"])))

    fields_xr = xr.Dataset(das)
    result["fields"] = fields_xr

    return result


def store_fields(cfg: dict, binary_dir: str, fields: dict, this_t: np.ndarray, prefix: str) -> dict:
    """Write field datasets for the legacy artifact pipeline."""
    result = field_datasets(cfg, fields, this_t, prefix)
    for name, dataset in result.items():
        label = "shared" if name == "fields" else name
        dataset.to_netcdf(os.path.join(binary_dir, f"{prefix}-{label}-t={round(this_t[-1], 4)}.nc"))
    return result


def distribution_datasets(cfg: dict, this_t: dict, ys: dict) -> dict:
    """
    Construct distribution datasets in memory.

    Handles species dist saves (keyed by "_species_name") and diagnostic dist saves
    (keyed by "_diag"), returning one dataset per save key.

    :param cfg:
    :param this_t:
    :param ys:
    :return: dict mapping save_key -> xr.Dataset
    """
    dist_save_keys = [
        k for k in ys.keys() if "_species_name" in cfg["save"].get(k, {}) or "_diag" in cfg["save"].get(k, {})
    ]

    result = {}
    for save_key in dist_save_keys:
        spc_save_cfg = cfg["save"][save_key]

        if "_species_name" in spc_save_cfg:
            species_name = spc_save_cfg["_species_name"]
            v_dim = f"v_{species_name}"
            full_v = cfg["grid"]["species_grids"][species_name]["v"]
            meta_keys = {"t", "func", "_species_name"}
        else:
            v_dim = "v"
            full_v = cfg["grid"]["species_grids"]["electron"]["v"]
            meta_keys = {"t", "func", "_diag"}

        save_keys = set(spc_save_cfg.keys()) - meta_keys
        if {"x", "v"} <= save_keys:
            coords = (("t", this_t[save_key]), ("x", spc_save_cfg["x"]["ax"]), (v_dim, spc_save_cfg["v"]["ax"]))
        elif {"kx", "v"} <= save_keys:
            coords = (("t", this_t[save_key]), ("kx", spc_save_cfg["kx"]["ax"]), (v_dim, spc_save_cfg["v"]["ax"]))
        else:
            warnings.warn(f"Saving distribution for '{save_key}' at full resolution.", stacklevel=2)
            coords = (("t", this_t[save_key]), ("x", cfg["grid"]["x"]), (v_dim, full_v))

        f_store = xr.Dataset({save_key: xr.DataArray(ys[save_key], coords=coords)})
        result[save_key] = f_store

    return result


def store_f(cfg: dict, this_t: dict, td: str, ys: dict) -> dict:
    """Write distribution datasets for the legacy artifact pipeline."""
    result = distribution_datasets(cfg, this_t, ys)
    for name, dataset in result.items():
        dataset.to_netcdf(os.path.join(td, "binary", f"dist-{name}.nc"))
    return result


def get_field_save_func(cfg):
    """Build the explicit field and species-moment observation."""
    if {"t"} != set(cfg["save"]["fields"].keys()):
        raise NotImplementedError
    return FieldsObservation(cfg["grid"]["species_grids"], cfg["grid"]["dx"])


def get_dist_save_func(axes, dist_save_config, dist_key):
    """Build a full or interpolated distribution observation with explicit arrays."""
    keys = set(dist_save_config)
    if keys == {"t"}:
        return DistributionObservation(dist_key)
    if keys not in ({"t", "x", "v"}, {"t", "kx", "v"}):
        raise NotImplementedError(f"Unsupported distribution save axes: {sorted(keys)}")
    spatial = "kx" if "kx" in keys else "x"
    xq, vq = jnp.meshgrid(dist_save_config[spatial]["ax"], dist_save_config["v"]["ax"], indexing="ij")
    return InterpolatedDistributionObservation(
        dist_key, spatial == "kx", xq.shape, axes[spatial], axes["v"], xq.ravel(), vq.ravel()
    )


def _add_dim_axes(save_config: dict) -> None:
    """Add 'ax' numpy array to each dimension sub-dict in a save config."""
    for dim_key, dim_config in save_config.items():
        if not isinstance(dim_config, dict) or f"n{dim_key}" not in dim_config:
            continue
        dim_min = float(dim_config[f"{dim_key}min"])
        dim_max = float(dim_config[f"{dim_key}max"])
        dim_n = int(dim_config[f"n{dim_key}"])
        if dim_key == "x":
            dx = (dim_max - dim_min) / dim_n
        else:
            dx = 0.0
        dim_config["ax"] = np.linspace(
            dim_min + dx / 2.0,
            dim_max - dx / 2.0,
            dim_n,
        )


def get_save_quantities(cfg: dict) -> dict:
    """
    Expand the save config into a flat dict keyed by save identifier and attach
    JAX-callable save functions.

    Species distribution saves use a nested YAML structure::

        save:
          electron:
            main:
              t: {nt: 11}
            full:
              t: {nt: 5}
              x: {xmin: 0.0, xmax: 20.94, nx: 32}
              v: {vmin: -6.4, vmax: 6.4, nv: 512}

    Each ``<species>/<label>`` pair becomes a flat key in the internal save dict,
    e.g. ``"electron.main"``, ``"electron.full"``.  Field and diagnostic saves are
    kept as-is.
    """
    species_names = list(cfg["grid"]["species_grids"].keys())
    diag_types = ["diag-vlasov-dfdt", "diag-fp-dfdt"]

    new_save: dict = {}

    for save_type, save_config in cfg["save"].items():
        if save_type.startswith("fields"):
            _add_dim_axes(save_config)
            save_config["func"] = get_field_save_func(cfg)
            new_save[save_type] = save_config

        elif save_type in species_names:
            # Nested: {label: {t: {...}, x: {...} (optional), v: {...} (optional)}}
            species_grid = cfg["grid"]["species_grids"][save_type]
            for label, label_config in save_config.items():
                _add_dim_axes(label_config)
                label_config["func"] = get_dist_save_func(
                    axes={"x": cfg["grid"]["x"], "v": species_grid["v"], "kx": cfg["grid"]["kxr"]},
                    dist_save_config=label_config,
                    dist_key=save_type,
                )
                # Set after func so it doesn't interfere with key-set matching inside get_dist_save_func
                label_config["_species_name"] = save_type
                new_save[f"{save_type}.{label}"] = label_config

        elif save_type in diag_types:
            _add_dim_axes(save_config)
            electron_grid = cfg["grid"]["species_grids"]["electron"]
            save_config["func"] = get_dist_save_func(
                axes={"x": cfg["grid"]["x"], "v": electron_grid["v"], "kx": cfg["grid"]["kxr"]},
                dist_save_config=save_config,
                dist_key=save_type,
            )
            save_config["_diag"] = True
            new_save[save_type] = save_config

        else:
            raise NotImplementedError(f"Unknown save type: {save_type}")

    cfg["save"] = new_save
    cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}, "func": get_default_save_func(cfg)}
    return cfg


def get_default_save_func(cfg):
    """Build the explicit scalar moment and field-energy observation."""
    return ScalarsObservation(cfg["grid"]["species_grids"], cfg["grid"]["species_params"], cfg["grid"]["dx"])
