"""Storage / save functions for the Vlasov-2D solver."""

import os
import warnings

import numpy as np
import xarray as xr
from interpax import interp2d
from jax import numpy as jnp


def store_fields(cfg: dict, binary_dir: str, fields: dict, this_t: np.ndarray, prefix: str) -> dict:
    """Persist saved fields/moments to netCDF and return them as xr.Datasets."""
    result = {}
    shared_keys = {"ex", "ey", "bz", "jx_driver", "jy_driver"}
    species_names = [k for k in fields.keys() if k not in shared_keys]

    x_coord = cfg["grid"]["x"]
    y_coord = cfg["grid"]["y"]

    for s in species_names:
        moments = fields[s]
        das = {}
        for k, v in moments.items():
            das[f"{prefix}-{k}"] = xr.DataArray(v, coords=(("t", this_t), ("x", x_coord), ("y", y_coord)))
        ds = xr.Dataset(das)
        ds.to_netcdf(os.path.join(binary_dir, f"{prefix}-{s}-t={round(this_t[-1], 4)}.nc"))
        result[s] = ds

    das = {}
    for k in shared_keys:
        if k in fields:
            das[f"{prefix}-{k}"] = xr.DataArray(fields[k], coords=(("t", this_t), ("x", x_coord), ("y", y_coord)))
    ds = xr.Dataset(das)
    ds.to_netcdf(os.path.join(binary_dir, f"{prefix}-shared-t={round(this_t[-1], 4)}.nc"))
    result["fields"] = ds

    return result


def store_f(cfg: dict, this_t: dict, td: str, ys: dict) -> dict:
    """Persist distribution snapshots to netCDF."""
    save_keys = [k for k in ys.keys() if "_species_name" in cfg["save"].get(k, {})]
    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir, exist_ok=True)

    result = {}
    for key in save_keys:
        sc = cfg["save"][key]
        s = sc["_species_name"]
        meta_keys = {"t", "func", "_species_name"}
        present = set(sc.keys()) - meta_keys
        v_dim_x = f"vx_{s}"
        v_dim_y = f"vy_{s}"
        if {"x", "y", "vx", "vy"} <= present:
            coords = (
                ("t", this_t[key]),
                ("x", sc["x"]["ax"]),
                ("y", sc["y"]["ax"]),
                (v_dim_x, sc["vx"]["ax"]),
                (v_dim_y, sc["vy"]["ax"]),
            )
        else:
            warnings.warn(f"Saving full-resolution dist for '{key}'.", stacklevel=2)
            sg = cfg["grid"]["species_grids"][s]
            coords = (
                ("t", this_t[key]),
                ("x", cfg["grid"]["x"]),
                ("y", cfg["grid"]["y"]),
                (v_dim_x, sg["vx"]),
                (v_dim_y, sg["vy"]),
            )
        ds = xr.Dataset({key: xr.DataArray(ys[key], coords=coords)})
        ds.to_netcdf(os.path.join(binary_dir, f"dist-{key}.nc"))
        result[key] = ds
    return result


def get_field_save_func(cfg):
    if {"t"} != set(cfg["save"]["fields"].keys()):
        raise NotImplementedError("Vlasov-2D fields save currently only supports {t}.")

    species_grids = cfg["grid"]["species_grids"]
    species_names = list(species_grids.keys())

    def fields_save_func(t, y, args):
        result = {}
        for s in species_names:
            vx = species_grids[s]["vx"]
            vy = species_grids[s]["vy"]
            dvx = species_grids[s]["dvx"]
            dvy = species_grids[s]["dvy"]

            f = y[s]
            n = jnp.sum(f, axis=(2, 3)) * (dvx * dvy)

            jx = jnp.sum(vx[None, None, :, None] * f, axis=(2, 3)) * (dvx * dvy)
            jy = jnp.sum(vy[None, None, None, :] * f, axis=(2, 3)) * (dvx * dvy)

            ux = jx / jnp.where(n > 0, n, 1.0)
            uy = jy / jnp.where(n > 0, n, 1.0)

            v_m_ux = vx[None, None, :, None] - ux[:, :, None, None]
            v_m_uy = vy[None, None, None, :] - uy[:, :, None, None]
            Txx = jnp.sum(v_m_ux**2 * f, axis=(2, 3)) * (dvx * dvy)
            Tyy = jnp.sum(v_m_uy**2 * f, axis=(2, 3)) * (dvx * dvy)
            T = 0.5 * (Txx + Tyy) / jnp.where(n > 0, n, 1.0)

            result[s] = {"n": n, "ux": ux, "uy": uy, "Txx": Txx, "Tyy": Tyy, "T": T}

        result["ex"] = y["ex"]
        result["ey"] = y["ey"]
        result["bz"] = y["bz"]
        if "jx_driver" in y:
            result["jx_driver"] = y["jx_driver"]
            result["jy_driver"] = y["jy_driver"]
        return result

    return fields_save_func


def get_dist_save_func(axes, dist_save_config, dist_key):
    """Save f(x, y, vx, vy) — either full or interpolated onto a coarser grid.

    Supported configs: {t}, or {t, x, y, vx, vy}.
    """
    keys = set(dist_save_config.keys())
    if keys == {"t"} or keys == {"t", "func"} or keys == {"t", "func", "_species_name"}:

        def dist_save_func(t, y, args):
            return y[dist_key]

        return dist_save_func

    needed = {"t", "x", "y", "vx", "vy"}
    if not needed.issubset(keys):
        raise NotImplementedError(
            f"Vlasov-2D dist save needs subset of {{t,x,y,vx,vy}}, got {keys - {'func', '_species_name'}}."
        )

    xq = jnp.asarray(dist_save_config["x"]["ax"])
    yq = jnp.asarray(dist_save_config["y"]["ax"])
    vxq = jnp.asarray(dist_save_config["vx"]["ax"])
    vyq = jnp.asarray(dist_save_config["vy"]["ax"])

    full_x = axes["x"]
    full_y = axes["y"]
    full_vx = axes["vx"]
    full_vy = axes["vy"]

    # Use linear interpolation since 4D is expensive; users who want full-res
    # can omit the spatial/velocity reduction keys.
    def dist_save_func(t, y, args):
        f = y[dist_key]  # (nx, ny, nvx, nvy)
        # Reduce (vx, vy) by interpolation per (x, y), then reduce (x, y) by interpolation per (vx, vy).
        # 2D over (vx, vy):
        from jax import vmap

        vxq_flat, vyq_flat = jnp.meshgrid(vxq, vyq, indexing="ij")
        vxq_flat = vxq_flat.ravel()
        vyq_flat = vyq_flat.ravel()

        def _vv(f_xy):
            return interp2d(vxq_flat, vyq_flat, full_vx, full_vy, f_xy, method="linear", extrap=0.0).reshape(
                vxq.size, vyq.size
            )

        f_vv = vmap(vmap(_vv))(f)  # (nx, ny, nvxq, nvyq)

        # Then over (x, y):
        xq_flat, yq_flat = jnp.meshgrid(xq, yq, indexing="ij")
        xq_flat = xq_flat.ravel()
        yq_flat = yq_flat.ravel()

        def _xy(f_vv_for_v):
            return interp2d(xq_flat, yq_flat, full_x, full_y, f_vv_for_v, method="linear", extrap=0.0).reshape(
                xq.size, yq.size
            )

        # bring (vxq, vyq) to leading axes
        f_vv_t = jnp.transpose(f_vv, (2, 3, 0, 1))
        f_red = vmap(vmap(_xy))(f_vv_t)  # (nvxq, nvyq, nxq, nyq)
        return jnp.transpose(f_red, (2, 3, 0, 1))

    return dist_save_func


def _add_dim_axes(save_config: dict) -> None:
    """Attach 'ax' arrays to each dim sub-dict in a save config."""
    for dim_key, dim_config in save_config.items():
        if not isinstance(dim_config, dict) or f"n{dim_key}" not in dim_config:
            continue
        dim_min = float(dim_config[f"{dim_key}min"])
        dim_max = float(dim_config[f"{dim_key}max"])
        dim_n = int(dim_config[f"n{dim_key}"])
        dx = (dim_max - dim_min) / dim_n if dim_key in ("x", "y") else 0.0
        dim_config["ax"] = np.linspace(dim_min + dx / 2.0, dim_max - dx / 2.0, dim_n)


def get_save_quantities(cfg: dict) -> dict:
    """Expand cfg['save'] into the flat layout consumed by diffrax."""
    species_names = list(cfg["grid"]["species_grids"].keys())
    new_save: dict = {}

    for save_type, save_config in cfg["save"].items():
        if save_type.startswith("fields"):
            _add_dim_axes(save_config)
            save_config["func"] = get_field_save_func(cfg)
            new_save[save_type] = save_config

        elif save_type in species_names:
            sg = cfg["grid"]["species_grids"][save_type]
            for label, label_cfg in save_config.items():
                _add_dim_axes(label_cfg)
                label_cfg["func"] = get_dist_save_func(
                    axes={"x": cfg["grid"]["x"], "y": cfg["grid"]["y"], "vx": sg["vx"], "vy": sg["vy"]},
                    dist_save_config=label_cfg,
                    dist_key=save_type,
                )
                label_cfg["_species_name"] = save_type
                new_save[f"{save_type}.{label}"] = label_cfg

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
        for s in species_names:
            sg = species_grids[s]
            dvx = sg["dvx"]
            dvy = sg["dvy"]
            f = y[s]
            n = jnp.sum(f, axis=(2, 3)) * (dvx * dvy)
            scalars[f"mean_n_{s}"] = jnp.mean(n)
            v2 = sg["vx"][None, None, :, None] ** 2 + sg["vy"][None, None, None, :] ** 2
            scalars[f"mean_KE_{s}"] = jnp.mean(jnp.sum(0.5 * v2 * f, axis=(2, 3)) * (dvx * dvy))
            scalars[f"mean_f2_{s}"] = jnp.mean(jnp.sum(f * f, axis=(2, 3)) * (dvx * dvy))

        scalars["mean_ex2"] = jnp.mean(y["ex"] ** 2)
        scalars["mean_ey2"] = jnp.mean(y["ey"] ** 2)
        scalars["mean_bz2"] = jnp.mean(y["bz"] ** 2)
        return scalars

    return save
