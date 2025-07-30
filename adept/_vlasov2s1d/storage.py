import os

import numpy as np
import xarray as xr
from interpax import interp2d
from jax import numpy as jnp


def store_fields(cfg: dict, binary_dir: str, fields: dict, this_t: np.ndarray, prefix: str) -> xr.Dataset:
    """
    Stores fields to netcdf

    :param prefix:
    :param cfg:
    :param td:
    :param fields:
    :param this_t:
    :return:
    """

    if any(x in ["x", "kx"] for x in cfg["save"][prefix].keys()):
        crds = set(cfg["save"][prefix].keys()) - {"t", "func"}
        if {"x"} == crds:
            xnm = "x"
        elif {"kx"} == crds:
            xnm = "kx"
        else:
            raise NotImplementedError

        xx = cfg["save"][prefix][xnm]["ax"]

        das = {f"{prefix}-{k}": xr.DataArray(v, coords=(("t", this_t), (xnm, xx))) for k, v in fields.items()}
    else:
        das = {}
        for k, v in fields.items():
            das[f"{prefix}-{k}"] = xr.DataArray(
                v[:, 1:-1] if k in ["a", "prev_a"] else v, coords=(("t", this_t), ("x", cfg["grid"]["x"]))
            )

        if len(cfg["drivers"]["ey"].keys()) > 0:
            ey = -(fields["a"][:, 1:-1] - fields["prev_a"][:, 1:-1]) / cfg["grid"]["dt"]
            bz = jnp.gradient(fields["a"], cfg["grid"]["dx"], axis=1)[:, 1:-1]

            ep = ey + cfg["units"]["derived"]["c_light"].magnitude * bz
            em = ey - cfg["units"]["derived"]["c_light"].magnitude * bz

            das[f"{prefix}-ep"] = xr.DataArray(ep, coords=(("t", this_t), ("x", cfg["grid"]["x"])))
            das[f"{prefix}-em"] = xr.DataArray(em, coords=(("t", this_t), ("x", cfg["grid"]["x"])))

    fields_xr = xr.Dataset(das)
    fields_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-t={round(this_t[-1], 4)}.nc"))

    return fields_xr


def store_f(cfg: dict, this_t: dict, td: str, ys: dict) -> xr.Dataset:
    """
    Stores f to netcdf for 2-species solver

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    f_store_dict = {}

    # Store electron distribution
    if "electron" in ys:
        f_store_dict["electron"] = xr.DataArray(
            ys["electron"], coords=(("t", this_t["electron"]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"]))
        )

    # Store ion distribution
    if "ion" in ys:
        f_store_dict["ion"] = xr.DataArray(
            ys["ion"], coords=(("t", this_t["ion"]), ("x", cfg["grid"]["x"]), ("v_i", cfg["grid"]["v_i"]))
        )

    f_store = xr.Dataset(f_store_dict)
    f_store.to_netcdf(os.path.join(td, "binary", "dist.nc"))

    return f_store


def store_diags(cfg: dict, this_t: dict, td: str, ys: dict) -> xr.Dataset:
    """
    Stores diagnostics to netcdf for 2-species solver

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    # Determine the axes for diagnostics storage
    if {"t"} == set(cfg["save"]["diags"].keys()):
        axes_e = (("t", this_t["diag-vlasov-dfdt-e"]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"]))
        axes_i = (("t", this_t["diag-vlasov-dfdt-i"]), ("x", cfg["grid"]["x"]), ("v_i", cfg["grid"]["v_i"]))
    elif {"t", "x", "v"} == set(cfg["save"]["diags"].keys()):
        axes_e = (("t", this_t["diag-vlasov-dfdt-e"]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"]))
        axes_i = (("t", this_t["diag-vlasov-dfdt-i"]), ("x", cfg["grid"]["x"]), ("v_i", cfg["grid"]["v_i"]))
    elif {"t", "kx", "v"} == set(cfg["save"]["diags"].keys()):
        axes_e = (("t", this_t["diag-vlasov-dfdt-e"]), ("kx", cfg["grid"]["kx"]), ("v", cfg["grid"]["v"]))
        axes_i = (("t", this_t["diag-vlasov-dfdt-i"]), ("kx", cfg["grid"]["kx"]), ("v_i", cfg["grid"]["v_i"]))
    elif {"t", "x", "kv"} == set(cfg["save"]["diags"].keys()):
        axes_e = (("t", this_t["diag-vlasov-dfdt-e"]), ("x", cfg["grid"]["x"]), ("kv", cfg["grid"]["kv"]))
        axes_i = (("t", this_t["diag-vlasov-dfdt-i"]), ("x", cfg["grid"]["x"]), ("kv_i", cfg["grid"]["kv_i"]))
    elif {"t", "kx", "kv"} == set(cfg["save"]["diags"].keys()):
        axes_e = (("t", this_t["diag-vlasov-dfdt-e"]), ("kx", cfg["grid"]["kx"]), ("kv", cfg["grid"]["kv"]))
        axes_i = (("t", this_t["diag-vlasov-dfdt-i"]), ("kx", cfg["grid"]["kx"]), ("kv_i", cfg["grid"]["kv_i"]))
    else:
        raise NotImplementedError

    diags_dict = {}

    # Store electron diagnostics
    if "diag-vlasov-dfdt-e" in ys:
        diags_dict["diag-vlasov-dfdt-e"] = xr.DataArray(ys["diag-vlasov-dfdt-e"], coords=axes_e)
    if "diag-fp-dfdt-e" in ys:
        diags_dict["diag-fp-dfdt-e"] = xr.DataArray(ys["diag-fp-dfdt-e"], coords=axes_e)

    # Store ion diagnostics
    if "diag-vlasov-dfdt-i" in ys:
        diags_dict["diag-vlasov-dfdt-i"] = xr.DataArray(ys["diag-vlasov-dfdt-i"], coords=axes_i)
    if "diag-fp-dfdt-i" in ys:
        diags_dict["diag-fp-dfdt-i"] = xr.DataArray(ys["diag-fp-dfdt-i"], coords=axes_i)

    diags_store = xr.Dataset(diags_dict)
    diags_store.to_netcdf(os.path.join(td, "binary", "diagnostics.nc"))

    return diags_store


def get_field_save_func(cfg):
    if {"t"} == set(cfg["save"]["fields"].keys()):

        def _calc_moment_e_(inp):
            return jnp.sum(inp, axis=1) * cfg["grid"]["dv"]

        def _calc_moment_i_(inp):
            return jnp.sum(inp, axis=1) * cfg["grid"]["dv_i"]

        def fields_save_func(t, y, args):
            temp = {}

            # Electron moments
            if "electron" in y:
                n_e = _calc_moment_e_(y["electron"])
                v_e = _calc_moment_e_(y["electron"] * cfg["grid"]["v"][None, :])
                v_m_vbar_e = cfg["grid"]["v"][None, :] - v_e[:, None]
                temp["n_e"] = n_e
                temp["v_e"] = v_e
                temp["p_e"] = _calc_moment_e_(y["electron"] * v_m_vbar_e**2.0)
                temp["q_e"] = _calc_moment_e_(y["electron"] * v_m_vbar_e**3.0)
                temp["-flogf_e"] = _calc_moment_e_(y["electron"] * jnp.log(jnp.abs(y["electron"])))
                temp["f2_e"] = _calc_moment_e_(y["electron"] * y["electron"])

            # Ion moments
            if "ion" in y:
                n_i = _calc_moment_i_(y["ion"])
                v_i = _calc_moment_i_(y["ion"] * cfg["grid"]["v_i"][None, :])
                v_m_vbar_i = cfg["grid"]["v_i"][None, :] - v_i[:, None]
                temp["n_i"] = n_i
                temp["v_i"] = v_i
                temp["p_i"] = _calc_moment_i_(y["ion"] * v_m_vbar_i**2.0)
                temp["q_i"] = _calc_moment_i_(y["ion"] * v_m_vbar_i**3.0)
                temp["-flogf_i"] = _calc_moment_i_(y["ion"] * jnp.log(jnp.abs(y["ion"])))
                temp["f2_i"] = _calc_moment_i_(y["ion"] * y["ion"])

            # Field quantities
            temp["e"] = y["e"]
            temp["de"] = y["de"]
            temp["a"] = y["a"]
            temp["prev_a"] = y["prev_a"]
            temp["pond"] = -0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1]

            return temp

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
    for save_type, save_config in cfg["save"].items():  # this can be fields or electron or diags?
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

        elif save_type.casefold() in [
            "electron",
            "ion",
            # "diag-vlasov-dfdt-e",
            # "diag-fp-dfdt-e",
            # "diag-vlasov-dfdt-i",
            # "diag-fp-dfdt-i",
        ]:
            # Determine which velocity grid to use
            if save_type.casefold() in ["ion", "diag-vlasov-dfdt-i", "diag-fp-dfdt-i"]:
                v_grid = "v_i"
                kv_grid = "kv_i"
            else:
                v_grid = "v"
                kv_grid = "kv"

            save_config["func"] = get_dist_save_func(
                axes={dim: cfg["grid"][dim] for dim in ("x", v_grid, "kx", kv_grid)},
                dist_save_config=save_config,
                dist_key=save_type,
            )

        else:
            raise NotImplementedError(f"Unknown save type: {save_type}")

    cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}, "func": get_default_save_func(cfg)}
    return cfg


def get_default_save_func(cfg):
    v_e = cfg["grid"]["v"][None, :]
    dv_e = cfg["grid"]["dv"]

    # Check if ion grid exists
    has_ions = "v_i" in cfg["grid"] and "dv_i" in cfg["grid"]
    if has_ions:
        v_i = cfg["grid"]["v_i"][None, :]
        dv_i = cfg["grid"]["dv_i"]

    def _calc_mean_moment_e_(inp):
        return jnp.mean(jnp.sum(inp, axis=1) * dv_e)

    def _calc_mean_moment_i_(inp):
        return jnp.mean(jnp.sum(inp, axis=1) * dv_i)

    def save(t, y, args):
        scalars = {}

        # Electron scalars
        if "electron" in y:
            scalars.update(
                {
                    "mean_P_e": _calc_mean_moment_e_(y["electron"] * v_e**2.0),
                    "mean_j_e": _calc_mean_moment_e_(y["electron"] * v_e),
                    "mean_n_e": _calc_mean_moment_e_(y["electron"]),
                    "mean_q_e": _calc_mean_moment_e_(y["electron"] * v_e**3.0),
                    "mean_-flogf_e": _calc_mean_moment_e_(-jnp.log(jnp.abs(y["electron"])) * jnp.abs(y["electron"])),
                    "mean_f2_e": _calc_mean_moment_e_(y["electron"] * y["electron"]),
                }
            )

        # Ion scalars
        if has_ions and "ion" in y:
            scalars.update(
                {
                    "mean_P_i": _calc_mean_moment_i_(y["ion"] * v_i**2.0),
                    "mean_j_i": _calc_mean_moment_i_(y["ion"] * v_i),
                    "mean_n_i": _calc_mean_moment_i_(y["ion"]),
                    "mean_q_i": _calc_mean_moment_i_(y["ion"] * v_i**3.0),
                    "mean_-flogf_i": _calc_mean_moment_i_(-jnp.log(jnp.abs(y["ion"])) * jnp.abs(y["ion"])),
                    "mean_f2_i": _calc_mean_moment_i_(y["ion"] * y["ion"]),
                }
            )

        # Field scalars
        scalars.update(
            {
                "mean_de2": jnp.mean(y["de"] ** 2.0),
                "mean_e2": jnp.mean(y["e"] ** 2.0),
                "mean_pond": jnp.mean(-0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1]),
            }
        )

        return scalars

    return save
