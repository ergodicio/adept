import os

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


def store_f(cfg: dict, this_t: dict, td: str, ys: dict) -> xr.Dataset:
    """
    Stores f to netcdf

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    # Find which species distributions were saved
    species_names = list(cfg["grid"]["species_grids"].keys())
    species_to_save = [spc for spc in species_names if spc in ys]

    data_vars = {}
    for spc in species_to_save:
        v = cfg["grid"]["species_grids"][spc]["v"]
        data_vars[spc] = xr.DataArray(ys[spc], coords=(("t", this_t[spc]), ("x", cfg["grid"]["x"]), ("v", v)))

    f_store = xr.Dataset(data_vars)
    f_store.to_netcdf(os.path.join(td, "binary", "dist.nc"))

    return f_store


def store_diags(cfg: dict, this_t: dict, td: str, ys: dict) -> xr.Dataset:
    """
    Stores diagnostics to netcdf

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    assert cfg["save"]["diag-vlasov-dfdt"] == cfg["diagnostics"]["diag-fp-dfdt"]

    if {"t"} == set(cfg["save"]["diags"].keys()):
        axes = (("t", this_t["diag-vlasov-dfdt"]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"]))
    elif {"t", "x", "v"} == set(cfg["save"]["diags"].keys()):
        axes = (("t", this_t["diag-vlasov-dfdt"]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"]))
    elif {"t", "kx", "v"} == set(cfg["save"]["diags"].keys()):
        axes = (("t", this_t["diag-vlasov-dfdt"]), ("kx", cfg["grid"]["kx"]), ("v", cfg["grid"]["v"]))
    elif {"t", "x", "kv"} == set(cfg["save"]["diags"].keys()):
        axes = (("t", this_t["diag-vlasov-dfdt"]), ("x", cfg["grid"]["x"]), ("kv", cfg["grid"]["kv"]))
    elif {"t", "kx", "kv"} == set(cfg["save"]["diags"].keys()):
        axes = (("t", this_t["diag-vlasov-dfdt"]), ("kx", cfg["grid"]["kx"]), ("kv", cfg["grid"]["kv"]))
    else:
        raise NotImplementedError

    diags_store = xr.Dataset({spc: xr.DataArray(ys[spc], coords=axes) for spc in ["diag-vlasov-dfdt", "diag-fp-dfdt"]})
    diags_store.to_netcdf(os.path.join(td, "binary", "diagnostics.nc"))

    return diags_store


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

        def dist_save_func(t, y, args):
            fkx = jnp.fft.rfft(y[dist_key], axes=0)
            return interp2d(
                dist_save_config["x"]["ax"],
                dist_save_config["v"]["ax"],
                axes["x"],
                axes["v"],
                fkx,
                kind="linear",
            )

    elif {"t", "kx", "v"} == set(dist_save_config.keys()):

        def dist_save_func(t, y, args):
            fkx = jnp.fft.rfft(y[dist_key], axes=0)
            return interp2d(
                dist_save_config["kx"]["ax"],
                dist_save_config["v"]["ax"],
                axes["kx"],
                axes["v"],
                fkx,
                kind="linear",
            )

    elif {"t", "x", "kv"} == set(dist_save_config.keys()):
        pass

    elif {"t", "kx", "kv"} == set(dist_save_config.keys()):
        pass
    else:
        raise NotImplementedError

    return dist_save_func


def get_save_quantities(cfg: dict) -> dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """
    # Get species names from config for dynamic handling
    species_names = list(cfg["grid"]["species_grids"].keys())
    diag_types = ["diag-vlasov-dfdt", "diag-fp-dfdt"]

    for save_type, save_config in cfg["save"].items():  # this can be fields or species name or diags
        for dim_key, dim_config in save_config.items():  # this can be t, x, y, kx, ky (eventually)
            if dim_key == "x":
                dx = (dim_config[f"{dim_key}max"] - dim_config[f"{dim_key}min"]) / dim_config[f"n{dim_key}"]
            else:
                dx = 0.0

            dim_config["ax"] = np.linspace(
                dim_config[f"{dim_key}min"] + dx / 2.0,
                dim_config[f"{dim_key}max"] - dx / 2.0,
                dim_config[f"n{dim_key}"],
            )

        if save_type.startswith("fields"):
            save_config["func"] = get_field_save_func(cfg)

        elif save_type.casefold() in [s.casefold() for s in species_names] or save_type in diag_types:
            save_config["func"] = get_dist_save_func(
                axes={dim: cfg["grid"][dim] for dim in ("x", "v", "kx")},
                dist_save_config=save_config,
                dist_key=save_type,
            )

        else:
            raise NotImplementedError(f"Unknown save type: {save_type}")

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
