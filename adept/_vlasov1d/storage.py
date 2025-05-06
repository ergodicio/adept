from typing import Dict
import os

from jax import numpy as jnp
import numpy as np
import xarray as xr
from interpax import interp2d


def store_fields(cfg: Dict, binary_dir: str, fields: Dict, this_t: np.ndarray, prefix: str) -> xr.Dataset:
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
    fields_xr.to_netcdf(os.path.join(binary_dir, f"{prefix}-t={round(this_t[-1],4)}.nc"))

    return fields_xr


def store_f(cfg: Dict, this_t: Dict, td: str, ys: Dict) -> xr.Dataset:
    """
    Stores f to netcdf

    :param cfg:
    :param this_t:
    :param td:
    :param ys:
    :return:
    """
    f_store = xr.Dataset(
        {
            spc: xr.DataArray(ys[spc], coords=(("t", this_t[spc]), ("x", cfg["grid"]["x"]), ("v", cfg["grid"]["v"])))
            for spc in ["electron"]
        }
    )
    f_store.to_netcdf(os.path.join(td, "binary", "dist.nc"))

    return f_store


def store_diags(cfg: Dict, this_t: Dict, td: str, ys: Dict) -> xr.Dataset:
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

        def _calc_moment_(inp):
            return jnp.sum(inp, axis=1) * cfg["grid"]["dv"]

        def fields_save_func(t, y, args):
            temp = {"n": _calc_moment_(y["electron"]), "v": _calc_moment_(y["electron"] * cfg["grid"]["v"][None, :])}
            v_m_vbar = cfg["grid"]["v"][None, :] - temp["v"][:, None]
            temp["p"] = _calc_moment_(y["electron"] * v_m_vbar**2.0)
            temp["q"] = _calc_moment_(y["electron"] * v_m_vbar**3.0)
            temp["-flogf"] = _calc_moment_(y["electron"] * jnp.log(jnp.abs(y["electron"])))
            temp["f^2"] = _calc_moment_(y["electron"] * y["electron"])
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


def get_save_quantities(cfg: Dict) -> Dict:
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

        elif save_type.casefold() in ["electron", "diag-vlasov-dfdt", "diag-fp-dfdt"]:
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
    v = cfg["grid"]["v"][None, :]
    dv = cfg["grid"]["dv"]

    def _calc_mean_moment_(inp):
        return jnp.mean(jnp.sum(inp, axis=1) * dv)

    def save(t, y, args):
        scalars = {
            "mean_P": _calc_mean_moment_(y["electron"] * v**2.0),
            "mean_j": _calc_mean_moment_(y["electron"] * v),
            "mean_n": _calc_mean_moment_(y["electron"]),
            "mean_q": _calc_mean_moment_(y["electron"] * v**3.0),
            "mean_-flogf": _calc_mean_moment_(-jnp.log(jnp.abs(y["electron"])) * jnp.abs(y["electron"])),
            "mean_f2": _calc_mean_moment_(y["electron"] * y["electron"]),
            "mean_de2": jnp.mean(y["de"] ** 2.0),
            "mean_e2": jnp.mean(y["e"] ** 2.0),
            "mean_pond": jnp.mean(-0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1]),
        }

        return scalars

    return save
