"""Save-function construction and netCDF writers for Vlasov-1D2V output.

Field/moment saves and scalars reuse the vlasov-1d layout (species moment
dicts + shared fields) so `_vlasov1d.storage.store_fields` writes them
verbatim. Species distribution saves come in two kinds:

- ``{t, x, v}`` label: the MARGINAL F(x, v_par), optionally interpolated onto
  the requested (x, v) sample points — rank-3, identical to a vlasov-1d dist
  save, so the whole 1D analysis stack (Hilbert omega(t), calc_deps, nlfs
  metrics) applies unchanged.
- ``{t}`` label: the full f(x, v_par, v_perp) at full resolution — rank-4,
  intended for sparse snapshot cadences.

Diagnostic saves (diag-vlasov-dfdt / diag-fp-dfdt) are already marginal
arrays in the state, so the 1D save machinery is reused directly.
"""

import os

import numpy as np
import xarray as xr
from interpax import interp2d
from jax import numpy as jnp

from adept._vlasov1d.storage import _add_dim_axes, get_dist_save_func


def get_field_save_func(cfg):
    """Build the save callback for field and species moment snapshots."""
    if {"t"} != set(cfg["save"]["fields"].keys()):
        raise NotImplementedError
    species_grids = cfg["grid"]["species_grids"]
    species_names = list(species_grids.keys())

    def fields_save_func(t, y, args):
        """Compute field, moment, and ponderomotive quantities for one save time."""
        result = {}

        for species_name in species_names:
            g = species_grids[species_name]
            v = g["v"]
            dv = g["dv"]
            wperp = g["wperp"]
            vperp = g["vperp"]

            f = y[species_name]
            F = jnp.einsum("xvp,p->xv", f, wperp)  # marginal

            def _marg_moment_(inp, _dv=dv):
                return jnp.sum(inp, axis=1) * _dv

            m = {}
            m["n"] = _marg_moment_(F)
            m["j"] = _marg_moment_(F * v[None, :])
            m["v"] = m["j"] / m["n"]
            v_m_vbar = v[None, :] - m["v"][:, None]
            m["p"] = _marg_moment_(F * v_m_vbar**2.0)
            m["q"] = _marg_moment_(F * v_m_vbar**3.0)
            # perpendicular pressure: int f vperp^2 w dvperp dvpar (2 perp DOF => Tperp = pperp/(2n))
            m["pperp"] = jnp.einsum("xvp,p->x", f, wperp * vperp**2) * dv
            m["-flogf"] = jnp.einsum("xvp,p->x", -jnp.abs(f) * jnp.log(jnp.abs(f)), wperp) * dv
            m["f^2"] = jnp.einsum("xvp,p->x", f * f, wperp) * dv

            result[species_name] = m

        result["e"] = y["e"]
        result["de"] = y["de"]
        result["a"] = y["a"]
        result["prev_a"] = y["prev_a"]
        result["pond"] = -0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1]

        return result

    return fields_save_func


def get_dist_save_func_2v(axes, dist_save_config, dist_key, wperp):
    """Build a save callback for marginal or full distribution output."""
    if {"t"} == set(dist_save_config.keys()):

        def dist_save_func(t, y, args):
            """Return the full f(x, v_par, v_perp) for this save point."""
            return y[dist_key]

    elif {"t", "x", "v"} == set(dist_save_config.keys()):
        xq, vq = jnp.meshgrid(dist_save_config["x"]["ax"], dist_save_config["v"]["ax"], indexing="ij")
        xq_flat, vq_flat = xq.ravel(), vq.ravel()
        out_shape = xq.shape

        def dist_save_func(t, y, args):
            """Return the marginal interpolated on configured x-v sample points.

            extrap=True is required, not cosmetic: interpax returns NaN outside
            the knot range, and a save axis built from a rounded config value
            (e.g. xmax: 20.94 against a box of 2*pi/0.3 = 20.94395) puts the
            first sample ~3e-5 below the first node. That silently NaNs a whole
            column, and any x-average of the save then propagates NaN
            everywhere downstream.
            """
            F = jnp.einsum("xvp,p->xv", y[dist_key], wperp)
            return interp2d(
                xq_flat, vq_flat, axes["x"], axes["v"], F, method="linear", extrap=True
            ).reshape(out_shape)

    else:
        raise NotImplementedError(f"Unsupported 2V dist save axes: {set(dist_save_config.keys())}")

    return dist_save_func


def get_default_save_func(cfg):
    """Build the default scalar save callback for moments and field energies."""
    species_grids = cfg["grid"]["species_grids"]
    species_names = list(species_grids.keys())

    def save(t, y, args):
        """Compute scalar diagnostics at one solver save point."""
        scalars = {}

        mean_kinetic_energy = 0.0
        for species_name in species_names:
            g = species_grids[species_name]
            v = g["v"][None, :]
            dv = g["dv"]
            wperp = g["wperp"]
            vperp = g["vperp"]
            mass = cfg["grid"]["species_params"][species_name]["mass"]

            f = y[species_name]
            F = jnp.einsum("xvp,p->xv", f, wperp)

            def _mean_marg_moment_(inp, _dv=dv):
                return jnp.mean(jnp.sum(inp, axis=1) * _dv)

            scalars[f"mean_P_{species_name}"] = _mean_marg_moment_(F * v**2.0)
            scalars[f"mean_Pperp_{species_name}"] = jnp.mean(jnp.einsum("xvp,p->x", f, wperp * vperp**2) * dv)
            scalars[f"mean_j_{species_name}"] = _mean_marg_moment_(F * v)
            scalars[f"mean_n_{species_name}"] = _mean_marg_moment_(F)
            scalars[f"mean_q_{species_name}"] = _mean_marg_moment_(F * v**3.0)
            scalars[f"mean_-flogf_{species_name}"] = jnp.mean(
                jnp.einsum("xvp,p->x", -jnp.log(jnp.abs(f)) * jnp.abs(f), wperp) * dv
            )
            scalars[f"mean_f2_{species_name}"] = jnp.mean(jnp.einsum("xvp,p->x", f * f, wperp) * dv)
            mean_kinetic_energy += (
                0.5 * mass * (scalars[f"mean_P_{species_name}"] + scalars[f"mean_Pperp_{species_name}"])
            )

        scalars["mean_de2"] = jnp.mean(y["de"] ** 2.0)
        scalars["mean_e2"] = jnp.mean(y["e"] ** 2.0)
        scalars["mean_pond"] = jnp.mean(-0.5 * jnp.gradient(y["a"] ** 2.0, cfg["grid"]["dx"])[1:-1])

        scalars["mean_kinetic_energy"] = mean_kinetic_energy
        scalars["mean_field_energy"] = 0.5 * scalars["mean_e2"]
        scalars["mean_total_energy"] = mean_kinetic_energy + 0.5 * scalars["mean_e2"]

        return scalars

    return save


def get_save_quantities(cfg: dict) -> dict:
    """Expand the save config into flat keys with attached JAX save functions."""
    species_names = list(cfg["grid"]["species_grids"].keys())
    diag_types = ["diag-vlasov-dfdt", "diag-fp-dfdt"]

    new_save: dict = {}

    for save_type, save_config in cfg["save"].items():
        if save_type.startswith("fields"):
            _add_dim_axes(save_config)
            save_config["func"] = get_field_save_func(cfg)
            new_save[save_type] = save_config

        elif save_type in species_names:
            species_grid = cfg["grid"]["species_grids"][save_type]
            for label, label_config in save_config.items():
                _add_dim_axes(label_config)
                label_config["func"] = get_dist_save_func_2v(
                    axes={"x": cfg["grid"]["x"], "v": species_grid["v"]},
                    dist_save_config=label_config,
                    dist_key=save_type,
                    wperp=species_grid["wperp"],
                )
                label_config["_species_name"] = save_type
                new_save[f"{save_type}.{label}"] = label_config

        elif save_type in diag_types:
            # diag state arrays are already marginal (nx, nv) -> reuse 1D machinery
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


def store_f(cfg: dict, this_t: dict, td: str, ys: dict) -> dict:
    """Store marginal (rank-3) and full (rank-4) distribution saves to netCDF."""
    dist_save_keys = [k for k in ys if "_species_name" in cfg["save"].get(k, {}) or "_diag" in cfg["save"].get(k, {})]

    result = {}
    for save_key in dist_save_keys:
        spc_save_cfg = cfg["save"][save_key]

        if "_species_name" in spc_save_cfg:
            species_name = spc_save_cfg["_species_name"]
            grid = cfg["grid"]["species_grids"][species_name]
            v_dim = f"v_{species_name}"
        else:
            species_name = "electron"
            grid = cfg["grid"]["species_grids"]["electron"]
            v_dim = "v"

        data = np.asarray(ys[save_key])
        if data.ndim == 4:
            coords = (
                ("t", this_t[save_key]),
                ("x", np.asarray(cfg["grid"]["x"])),
                (v_dim, np.asarray(grid["v"])),
                (f"vperp_{species_name}", np.asarray(grid["vperp"])),
            )
        elif {"x", "v"} <= set(spc_save_cfg.keys()):
            coords = (("t", this_t[save_key]), ("x", spc_save_cfg["x"]["ax"]), (v_dim, spc_save_cfg["v"]["ax"]))
        else:
            coords = (("t", this_t[save_key]), ("x", np.asarray(cfg["grid"]["x"])), (v_dim, np.asarray(grid["v"])))

        f_store = xr.Dataset({save_key: xr.DataArray(data, coords=coords)})
        f_store.to_netcdf(os.path.join(td, "binary", f"dist-{save_key}.nc"))
        result[save_key] = f_store

    return result
