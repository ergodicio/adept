import os
import warnings

import numpy as np
import xarray as xr
from interpax import interp2d
from jax import numpy as jnp


def store_fields(cfg: dict, binary_dir: str, fields: dict, this_t: np.ndarray, prefix: str) -> dict:
    """
    Stores fields to netcdf, handling multispecies data.

    :param prefix:
    :param cfg:
    :param binary_dir:
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
        species_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-{species_name}-t={round(this_t[-1], 4)}.nc"))
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
        ey = -(fields["a"][:, 1:-1] - fields["prev_a"][:, 1:-1]) / cfg["grid"]["dt"]
        bz = jnp.gradient(fields["a"], cfg["grid"]["dx"], axis=1)[:, 1:-1]

        ep = ey + cfg["units"]["derived"]["c_light"].magnitude * bz
        em = ey - cfg["units"]["derived"]["c_light"].magnitude * bz

        das[f"{prefix}-ep"] = xr.DataArray(ep, coords=(("t", this_t), ("x", cfg["grid"]["x"])))
        das[f"{prefix}-em"] = xr.DataArray(em, coords=(("t", this_t), ("x", cfg["grid"]["x"])))

    fields_xr = xr.Dataset(das)
    fields_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-shared-t={round(this_t[-1], 4)}.nc"))
    result["fields"] = fields_xr

    return result


def store_f(cfg: dict, this_t: dict, td: str, ys: dict) -> dict:
    """
    Stores distribution function saves to netcdf.

    Handles species dist saves (keyed by "_species_name") and diagnostic dist saves
    (keyed by "_diag"), writing one netcdf file per save key.

    :param cfg:
    :param this_t:
    :param td:
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
        f_store.to_netcdf(os.path.join(td, "binary", f"dist-{save_key}.nc"))
        result[save_key] = f_store

    return result


def get_field_save_func(cfg):
    if {"t"} == set(cfg["save"]["fields"].keys()):
        species_grids = cfg["grid"]["species_grids"]
        species_names = list(species_grids.keys())

        def fields_save_func(t, y, args):
            result = {}

            # Compute moments for each species
            for species_name in species_names:
                v = species_grids[species_name]["v"]
                dv = species_grids[species_name]["dv"]

                def _calc_moment_(inp, _dv=dv):
                    return jnp.sum(inp, axis=1) * _dv

                f = y[species_name]
                species_moments = {}
                species_moments["n"] = _calc_moment_(f)
                species_moments["v"] = _calc_moment_(f * v[None, :])
                v_m_vbar = v[None, :] - species_moments["v"][:, None]
                species_moments["p"] = _calc_moment_(f * v_m_vbar**2.0)
                species_moments["q"] = _calc_moment_(f * v_m_vbar**3.0)
                species_moments["-flogf"] = _calc_moment_(f * jnp.log(jnp.abs(f)))
                species_moments["f^2"] = _calc_moment_(f * f)

                result[species_name] = species_moments

            # Store shared field data at top level for backward compatibility
            result["e"] = y["e"]
            result["de"] = y["de"]
            result["a"] = y["a"]
            result["prev_a"] = y["prev_a"]
            result["pond"] = -0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1]

            return result

    else:
        raise NotImplementedError

    return fields_save_func


def get_dist_save_func(axes, dist_save_config, dist_key):
    if {"t"} == set(dist_save_config.keys()):

        def dist_save_func(t, y, args):
            return y[dist_key]

    elif {"t", "x", "v"} == set(dist_save_config.keys()):
        xq, vq = jnp.meshgrid(dist_save_config["x"]["ax"], dist_save_config["v"]["ax"], indexing="ij")
        xq_flat, vq_flat = xq.ravel(), vq.ravel()
        out_shape = xq.shape

        def dist_save_func(t, y, args):
            return interp2d(xq_flat, vq_flat, axes["x"], axes["v"], y[dist_key], method="linear").reshape(out_shape)

    elif {"t", "kx", "v"} == set(dist_save_config.keys()):
        kxq, vq = jnp.meshgrid(dist_save_config["kx"]["ax"], dist_save_config["v"]["ax"], indexing="ij")
        kxq_flat, vq_flat = kxq.ravel(), vq.ravel()
        out_shape = kxq.shape

        def dist_save_func(t, y, args):
            fkx = jnp.abs(jnp.fft.rfft(y[dist_key], axes=0))
            return interp2d(kxq_flat, vq_flat, axes["kx"], axes["v"], fkx, method="linear").reshape(out_shape)

    elif {"t", "x", "kv"} == set(dist_save_config.keys()):
        pass

    elif {"t", "kx", "kv"} == set(dist_save_config.keys()):
        pass
    else:
        raise NotImplementedError

    return dist_save_func


def _add_dim_axes(save_config: dict) -> None:
    """Add 'ax' numpy array to each dimension sub-dict in a save config."""
    for dim_key, dim_config in save_config.items():
        if not isinstance(dim_config, dict) or f"n{dim_key}" not in dim_config:
            continue
        if dim_key == "x":
            dx = (dim_config[f"{dim_key}max"] - dim_config[f"{dim_key}min"]) / dim_config[f"n{dim_key}"]
        else:
            dx = 0.0
        dim_config["ax"] = np.linspace(
            dim_config[f"{dim_key}min"] + dx / 2.0,
            dim_config[f"{dim_key}max"] - dx / 2.0,
            dim_config[f"n{dim_key}"],
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
                    axes={"x": cfg["grid"]["x"], "v": species_grid["v"], "kx": cfg["grid"]["kx"]},
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
                axes={"x": cfg["grid"]["x"], "v": electron_grid["v"], "kx": cfg["grid"]["kx"]},
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
    species_grids = cfg["grid"]["species_grids"]
    species_names = list(species_grids.keys())

    def save(t, y, args):
        scalars = {}

        # Compute scalars for each species
        for species_name in species_names:
            v = species_grids[species_name]["v"][None, :]
            dv = species_grids[species_name]["dv"]

            def _calc_mean_moment_(inp, _dv=dv):
                return jnp.mean(jnp.sum(inp, axis=1) * _dv)

            f = y[species_name]
            scalars[f"mean_P_{species_name}"] = _calc_mean_moment_(f * v**2.0)
            scalars[f"mean_j_{species_name}"] = _calc_mean_moment_(f * v)
            scalars[f"mean_n_{species_name}"] = _calc_mean_moment_(f)
            scalars[f"mean_q_{species_name}"] = _calc_mean_moment_(f * v**3.0)
            scalars[f"mean_-flogf_{species_name}"] = _calc_mean_moment_(-jnp.log(jnp.abs(f)) * jnp.abs(f))
            scalars[f"mean_f2_{species_name}"] = _calc_mean_moment_(f * f)

        # Shared field scalars (not species-specific)
        scalars["mean_de2"] = jnp.mean(y["de"] ** 2.0)
        scalars["mean_e2"] = jnp.mean(y["e"] ** 2.0)
        scalars["mean_pond"] = jnp.mean(-0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1])

        return scalars

    return save
