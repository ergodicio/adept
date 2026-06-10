"""Initialization and post-processing helpers for Vlasov-2D."""

import os
from time import time

import numpy as np
import xarray as xr
from diffrax import Solution
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.special import gamma

from adept._vlasov2d.distributed import create_distribution_sharding, make_sharded_array
from adept._vlasov2d.simulation import Vlasov2DSimulation
from adept._vlasov2d.storage import store_f, store_fields

from .. import patched_mlflow as mlflow


def _gamma_ratio_alpha(m: float) -> float:
    return float(np.sqrt(3.0 * gamma(3.0 / m) / gamma(5.0 / m)))


def _initialize_bi_supergaussian(
    nx: int,
    ny: int,
    nvx: int,
    nvy: int,
    v0x: float,
    v0y: float,
    T0: float,
    mass: float,
    supergaussian_order: float,
    vmax: float,
    n_prof: np.ndarray,
):
    """Initialize a bi-supergaussian f(x, y, vx, vy) shaped by n_prof[nx, ny]."""
    dvx = 2.0 * vmax / nvx
    dvy = 2.0 * vmax / nvy
    vxax = np.linspace(-vmax + dvx / 2.0, vmax - dvx / 2.0, nvx)
    vyax = np.linspace(-vmax + dvy / 2.0, vmax - dvy / 2.0, nvy)
    f = _evaluate_bi_supergaussian_on_axes(vxax, vyax, v0x, v0y, T0, mass, supergaussian_order, vmax, n_prof)
    return f, vxax, vyax


def _evaluate_bi_supergaussian_on_axes(
    vxax: np.ndarray,
    vyax: np.ndarray,
    v0x: float,
    v0y: float,
    T0: float,
    mass: float,
    supergaussian_order: float,
    vmax: float,
    n_prof: np.ndarray,
):
    """Evaluate a normalized separable bi-supergaussian on preselected velocity axes."""
    nvx = vxax.size
    nvy = vyax.size
    dvx = 2.0 * vmax / nvx
    dvy = 2.0 * vmax / nvy
    v_th = np.sqrt(T0 / mass)
    alpha = _gamma_ratio_alpha(supergaussian_order)
    scale = alpha * v_th

    fvx = np.exp(-np.power(np.abs((vxax - v0x) / scale), supergaussian_order))
    fvy = np.exp(-np.power(np.abs((vyax - v0y) / scale), supergaussian_order))

    fvx /= np.sum(fvx) * dvx
    fvy /= np.sum(fvy) * dvy

    fv = fvx[:, None] * fvy[None, :]
    return n_prof[:, :, None, None] * fv[None, None, :, :]


def _slice_axis(axis, indexer) -> np.ndarray:
    """Return a NumPy view/copy of a JAX axis for a make_array_from_callback index."""
    return np.atleast_1d(np.asarray(axis[indexer]))


def _component_density_on_slice(spec, x_slice: np.ndarray, y_slice: np.ndarray) -> np.ndarray:
    """Evaluate one density component on an x-y shard."""
    return np.asarray(spec.density_profile(jnp.asarray(x_slice), jnp.asarray(y_slice)))


def _initialize_sharded_species_distribution(
    species_cfg,
    distribution_specs,
    grid,
    sharding,
    dtype,
):
    """Create a sharded f(x, y, vx, vy) array without allocating the global array."""
    for spec in distribution_specs:
        noise = spec.density_profile.noise_profile
        if noise.noise_type.casefold() != "none" and noise.noise_val != 0.0:
            raise NotImplementedError(
                "Distributed Vlasov-2D initialization currently requires deterministic density profiles "
                "(set noise_type: none or noise_val: 0.0)."
            )

    vx_full = np.linspace(
        -species_cfg.vmax + species_cfg.vmax / species_cfg.nvx,
        species_cfg.vmax - species_cfg.vmax / species_cfg.nvx,
        species_cfg.nvx,
    )
    vy_full = np.linspace(
        -species_cfg.vmax + species_cfg.vmax / species_cfg.nvy,
        species_cfg.vmax - species_cfg.vmax / species_cfg.nvy,
        species_cfg.nvy,
    )
    shape = (grid.nx, grid.ny, species_cfg.nvx, species_cfg.nvy)

    def _callback(index):
        x_index, y_index, vx_index, vy_index = index
        x_slice = _slice_axis(grid.x, x_index)
        y_slice = _slice_axis(grid.y, y_index)
        vx_slice = vx_full[vx_index]
        vy_slice = vy_full[vy_index]

        f_shard = np.zeros((x_slice.size, y_slice.size, vx_slice.size, vy_slice.size), dtype=dtype)
        for spec in distribution_specs:
            nprof = _component_density_on_slice(spec, x_slice, y_slice)
            f_shard += _evaluate_bi_supergaussian_on_axes(
                vx_slice,
                vy_slice,
                v0x=spec.v0x,
                v0y=spec.v0y,
                T0=spec.T0,
                mass=species_cfg.mass,
                supergaussian_order=spec.supergaussian_order,
                vmax=species_cfg.vmax,
                n_prof=nprof,
            ).astype(dtype, copy=False)
        return f_shard

    return make_sharded_array(shape, sharding, _callback, dtype=dtype), vx_full, vy_full


def _initialize_total_distribution_(cfg: dict, simulation: Vlasov2DSimulation):
    """Build f(x,y,vx,vy) for each species by summing density-component contributions."""
    grid = simulation.grid
    species_distributions = {}
    species_found = False
    sharding_cfg = cfg.get("grid", {}).get("distribution_sharding") or cfg.get("grid", {}).get("distribution-sharding")
    dist_sharding = create_distribution_sharding(sharding_cfg)

    for species_cfg in simulation.species:
        name = species_cfg.name
        nvx = species_cfg.nvx
        nvy = species_cfg.nvy
        vmax = species_cfg.vmax
        mass = species_cfg.mass

        n_prof_species = np.zeros([grid.nx, grid.ny])
        f_species = None
        vxax = np.zeros(nvx)
        vyax = np.zeros(nvy)

        # Compute each component's density profile once and reuse below.
        component_specs = simulation.species_distributions[name]
        component_nprofs = []
        for spec in component_specs:
            nprof = np.array(spec.density_profile(grid.x, grid.y))
            component_nprofs.append(nprof)
            n_prof_species += nprof
            species_found = True

        if dist_sharding is None:
            f_species = np.zeros([grid.nx, grid.ny, nvx, nvy])
            for spec, nprof in zip(component_specs, component_nprofs, strict=True):
                f_one, vxax, vyax = _initialize_bi_supergaussian(
                    nx=grid.nx,
                    ny=grid.ny,
                    nvx=nvx,
                    nvy=nvy,
                    v0x=spec.v0x,
                    v0y=spec.v0y,
                    T0=spec.T0,
                    mass=mass,
                    supergaussian_order=spec.supergaussian_order,
                    vmax=vmax,
                    n_prof=nprof,
                )
                f_species += f_one
        else:
            f_species, vxax, vyax = _initialize_sharded_species_distribution(
                species_cfg,
                component_specs,
                grid,
                dist_sharding.sharding,
                dtype=n_prof_species.dtype,
            )

        species_distributions[name] = (n_prof_species, f_species, vxax, vyax)

    if not species_found:
        raise ValueError("No species found! Check the config.")

    return species_distributions


def _save_xy_facet(fld, path: str, n_panels: int = 8) -> None:
    """Save an `n_panels`-panel 2D (x, y) facet grid of `fld(t, x, y)`.

    Picks `n_panels` evenly-spaced time indices that include t[0] and t[-1].
    Uses a shared symmetric color scale across panels so amplitude evolution
    is visible.
    """
    nt = fld.sizes["t"]
    n_panels = min(n_panels, nt)
    idx = np.linspace(0, nt - 1, n_panels).round().astype(int)
    panels = fld.isel(t=idx)

    vmax = float(np.abs(panels).max())
    if vmax == 0.0:
        vmax = 1.0
    vmin = -vmax

    ncols = 4 if n_panels >= 4 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    g = panels.plot(
        col="t",
        col_wrap=ncols,
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
        figsize=(3.2 * ncols, 2.8 * nrows),
    )
    g.fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(g.fig)


def post_process(result: Solution, cfg: dict, td: str, args: dict):
    t0 = time()
    species_names = list(cfg["grid"]["species_grids"].keys())

    os.makedirs(os.path.join(td, "plots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
    for s in species_names:
        os.makedirs(os.path.join(td, "plots", "fields", s), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "scalars"), exist_ok=True)
    for s in species_names:
        os.makedirs(os.path.join(td, "plots", "scalars", s), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "dists"), exist_ok=True)

    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir, exist_ok=True)

    fields_result = {}
    for k in result.ys.keys():
        if k.startswith("field"):
            fields_dict = store_fields(cfg, binary_dir, result.ys[k], result.ts[k], k)

            for s in species_names:
                if s not in fields_dict:
                    continue
                sx = fields_dict[s]
                sdir = os.path.join(td, "plots", "fields", s)
                for nm, fld in sx.items():
                    short = nm.split("-", 1)[1] if "-" in nm else nm
                    _save_xy_facet(fld, os.path.join(sdir, f"xy_facet_{short}.png"))

            if "fields" in fields_dict:
                fx = fields_dict["fields"]
                fdir = os.path.join(td, "plots", "fields")
                for nm, fld in fx.items():
                    short = nm.split("-", 1)[1] if "-" in nm else nm
                    _save_xy_facet(fld, os.path.join(fdir, f"xy_facet_{short}.png"))

            fields_result = fields_dict

        elif k.startswith("default"):
            scalars_xr = xr.Dataset(
                {kk: xr.DataArray(vv, coords=(("t", result.ts["default"]),)) for kk, vv in result.ys["default"].items()}
            )
            scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(scalars_xr.coords['t'].data[-1], 4)}.nc"))
            scalars_dir = os.path.join(td, "plots", "scalars")
            for nm, srs in scalars_xr.items():
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                srs.plot(ax=ax[0])
                ax[0].grid()
                np.log10(np.abs(srs)).plot(ax=ax[1])
                ax[1].grid()
                ax[1].set_ylabel(f"log10(|{nm}|)")
                scalar_species = None
                for s in species_names:
                    if nm.endswith(f"_{s}"):
                        scalar_species = s
                        break
                if scalar_species:
                    fig.savefig(os.path.join(scalars_dir, scalar_species, f"{nm}.png"), bbox_inches="tight")
                else:
                    fig.savefig(os.path.join(scalars_dir, f"{nm}.png"), bbox_inches="tight")
                plt.close(fig)

    f_result = store_f(cfg, result.ts, td, result.ys)

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})
    return {"fields": fields_result, "dists": f_result, "scalars": scalars_xr}
