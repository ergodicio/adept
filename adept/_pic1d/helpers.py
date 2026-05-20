"""PIC-1D particle initialization and post-processing.

The loaders produce three arrays per species — ``x``, ``v``, ``w`` — sized
``(nx * ppc,)``. They mirror the supergaussian / non-uniform density profile
used by the Vlasov-1D loader so that the same input deck reproduces the same
initial moments up to PIC sampling noise.
"""

import os
from time import time

import numpy as np
import xarray as xr
from diffrax import Solution
from matplotlib import pyplot as plt
from scipy.special import gamma

from adept._pic1d.simulation import PIC1DSimulation
from adept._vlasov1d.helpers import gamma_3_over_m, gamma_5_over_m

from .. import patched_mlflow as mlflow


def _inverse_cdf_supergaussian(
    n_particles: int, T0: float, mass: float, supergaussian_order: float, vmax: float, v0: float
) -> np.ndarray:
    """Return velocity samples drawn quietly via the inverse CDF.

    A high-resolution velocity grid is built on ``[-vmax, vmax]``, the CDF is
    integrated by the trapezoidal rule, and ``n_particles`` equal-probability
    quantiles are taken at ``(k + 0.5) / n_particles``. This gives a deterministic
    sampling that exactly reproduces the requested moments in expectation.
    """
    v_thermal = np.sqrt(T0 / mass)
    alpha = np.sqrt(3.0 * gamma_3_over_m(supergaussian_order) / gamma_5_over_m(supergaussian_order))

    nv = max(8192, 8 * n_particles)
    v_grid = np.linspace(-vmax, vmax, nv)
    pdf = np.exp(-np.power(np.abs((v_grid - v0) / (alpha * v_thermal)), supergaussian_order))
    pdf = pdf / np.trapezoid(pdf, v_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(v_grid))])
    cdf = cdf / cdf[-1]

    quantiles = (np.arange(n_particles) + 0.5) / n_particles
    return np.interp(quantiles, cdf, v_grid)


def _random_supergaussian(
    rng: np.random.Generator,
    n_particles: int,
    T0: float,
    mass: float,
    supergaussian_order: float,
    vmax: float,
    v0: float,
) -> np.ndarray:
    """Random velocity sampling via rejection on a bounded supergaussian."""
    v_thermal = np.sqrt(T0 / mass)
    alpha = np.sqrt(3.0 * gamma_3_over_m(supergaussian_order) / gamma_5_over_m(supergaussian_order))

    out = np.empty(n_particles)
    filled = 0
    batch = max(n_particles, 1024)
    while filled < n_particles:
        candidates = rng.uniform(-vmax, vmax, size=batch) + v0
        p = np.exp(-np.power(np.abs((candidates - v0) / (alpha * v_thermal)), supergaussian_order))
        accept = rng.uniform(size=batch) < p
        keep = candidates[accept]
        take = min(keep.size, n_particles - filled)
        out[filled : filled + take] = keep[:take]
        filled += take
    return out


def _initialize_particles_(cfg: dict, simulation: PIC1DSimulation) -> dict:
    """Build particle arrays for every species.

    Returns a dict ``{species_name: (x, v, w, n_prof, vax)}`` where ``n_prof``
    is the requested density profile sampled on the grid (used as the
    neutralizing ion background) and ``vax`` is a synthetic velocity axis used
    for downstream plots.
    """
    grid = simulation.grid
    ppc = simulation.ppc
    n_particles = int(grid.nx * ppc)
    L = float(grid.xmax - grid.xmin)

    out = {}
    for species in simulation.species:
        # Combine density profile contributions; same convention as Vlasov-1D loader.
        n_prof = np.zeros(grid.nx)
        # The per-subspecies thermal parameters are needed when sampling v.
        # If multiple subspecies exist we treat them as additive populations of
        # particles, each carrying a fraction of ``n_particles`` weighted by
        # their integrated density. This mirrors the Vlasov "sum of fs".
        sub_specs = simulation.species_distributions[species.name]
        sub_profiles = []
        for spec in sub_specs:
            nprof = np.array(spec.density_profile(grid.x))
            n_prof = n_prof + nprof
            sub_profiles.append(nprof)

        integrated = np.array([np.sum(p) * grid.dx for p in sub_profiles])
        total_density = integrated.sum()
        if total_density <= 0:
            raise ValueError(f"Species '{species.name}' has non-positive integrated density")
        share = integrated / total_density
        np_per_sub = np.maximum(np.round(share * n_particles).astype(int), 1)
        # Adjust last entry so totals match exactly.
        np_per_sub[-1] += n_particles - int(np_per_sub.sum())

        x_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []

        rng = np.random.default_rng(2025)  # fallback; per-subspecies seeds set below
        for spec, sub_np, nprof in zip(sub_specs, np_per_sub, sub_profiles, strict=True):
            sub_np = int(sub_np)
            seed = spec.density_profile.noise_profile.noise_seed
            rng = np.random.default_rng(int(seed))

            if species.loading == "quiet":
                # Uniform x slots; weight encodes the density profile so all
                # subspecies contributions sum into the same ion background.
                x_local = grid.xmin + (np.arange(sub_np) + 0.5) * (L / sub_np)
                # Per-particle weight: integrate the local density across its
                # mini-cell of width L/sub_np.
                # Evaluate density at particle positions (cheap interp).
                density_at_x = np.interp(x_local, grid.x, nprof)
                w_local = density_at_x * (L / sub_np)
                v_local = _inverse_cdf_supergaussian(
                    sub_np,
                    T0=spec.T0,
                    mass=species.mass,
                    supergaussian_order=spec.supergaussian_order,
                    vmax=species.vmax_load,
                    v0=spec.v0,
                )
                # De-correlate x and v ordering so the quiet start doesn't
                # alias structured modes onto specific positions.
                v_local = rng.permutation(v_local)
            else:  # random
                # Sample x from the density profile via inverse CDF.
                pdf = nprof / np.trapezoid(nprof, grid.x)
                cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid.x))])
                cdf = cdf / cdf[-1]
                u = rng.uniform(size=sub_np)
                x_local = np.interp(u, cdf, grid.x)
                # Uniform weight per particle = (integrated density) / sub_np.
                w_local = np.full(sub_np, integrated[sub_specs.index(spec)] / sub_np)
                v_local = _random_supergaussian(
                    rng,
                    sub_np,
                    T0=spec.T0,
                    mass=species.mass,
                    supergaussian_order=spec.supergaussian_order,
                    vmax=species.vmax_load,
                    v0=spec.v0,
                )

            x_parts.append(x_local)
            v_parts.append(v_local)
            w_parts.append(w_local)

        x_arr = np.concatenate(x_parts)
        v_arr = np.concatenate(v_parts)
        w_arr = np.concatenate(w_parts)

        # Pad/truncate to the exact n_particles target so all species have
        # statically-sized arrays for JIT.
        if x_arr.size != n_particles:
            x_arr = np.resize(x_arr, n_particles)
            v_arr = np.resize(v_arr, n_particles)
            w_arr = np.resize(w_arr, n_particles)

        # Synthetic v-axis purely for post-process plots: matches Vlasov vmax.
        vax = np.linspace(-species.vmax_load, species.vmax_load, 128)

        out[species.name] = (x_arr, v_arr, w_arr, n_prof, vax)

    return out


def _spacetime_plots(ds: xr.Dataset, base_dir: str) -> None:
    """For every DataArray in ``ds`` write the three Vlasov-style plots:

    1. ``<base_dir>/spacetime_<name>.png`` — linear pcolormesh.
    2. ``<base_dir>/logplots/spacetime_log_<name>.png`` — log10(|·|).
    3. ``<base_dir>/lineouts/<name>.png`` — facet of ~8 time snapshots.
    """
    logs_dir = os.path.join(base_dir, "logplots")
    lineouts_dir = os.path.join(base_dir, "lineouts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(lineouts_dir, exist_ok=True)

    nt = ds.coords["t"].data.size
    t_skip = max(1, nt // 8)
    tslice = slice(0, -1, t_skip)

    for nm, fld in ds.items():
        # Drop a "fields-" prefix if present so filenames stay clean.
        plain = nm.split("-", 1)[1] if "-" in nm else nm

        fld.plot(figsize=(12, 8))
        plt.savefig(os.path.join(base_dir, f"spacetime_{plain}.png"), bbox_inches="tight", dpi=150)
        plt.close()

        np.log10(np.abs(fld)).plot(figsize=(12, 8))
        plt.savefig(
            os.path.join(logs_dir, f"spacetime_log_{plain}.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

        fld[tslice].T.plot(col="t", col_wrap=4)
        plt.savefig(os.path.join(lineouts_dir, f"{plain}.png"), bbox_inches="tight")
        plt.close()


def post_process(result: Solution, cfg: dict, td: str, args: dict) -> dict:
    """Vlasov-1D-style post-process: per-species and shared field plot trees.

    Reads the nested ``fields`` save dict produced by
    :func:`adept._pic1d.storage.get_field_save_func`. For each species the
    moments ``n``, ``j``, ``P`` are written to
    ``plots/fields/<species>/{,logplots/,lineouts/}`` and netcdf'd separately;
    shared quantities ``e``, ``de``, ``a``, ``prev_a``, ``pond`` (and the
    ``ep``/``em`` decomposition when an ``ey`` driver is configured) live at
    the top of ``plots/fields/``. Scalars split similarly between
    ``plots/scalars/<species>/`` and ``plots/scalars/``.
    """
    t0 = time()

    species_names = list(cfg["grid"]["species_params"].keys())

    # Build directory tree up front, matching Vlasov-1D.
    plots_root = os.path.join(td, "plots")
    fields_root = os.path.join(plots_root, "fields")
    scalars_root = os.path.join(plots_root, "scalars")
    dists_root = os.path.join(plots_root, "dists")
    binary_dir = os.path.join(td, "binary")
    for d in (
        os.path.join(fields_root, "logplots"),
        os.path.join(fields_root, "lineouts"),
        scalars_root,
        dists_root,
        binary_dir,
    ):
        os.makedirs(d, exist_ok=True)
    for sp in species_names:
        os.makedirs(os.path.join(fields_root, sp, "logplots"), exist_ok=True)
        os.makedirs(os.path.join(fields_root, sp, "lineouts"), exist_ok=True)
        os.makedirs(os.path.join(scalars_root, sp), exist_ok=True)

    x_axis = np.asarray(cfg["grid"]["x"])
    dt = float(cfg["grid"]["dt"])
    dx = float(cfg["grid"]["dx"])
    has_ey = len(cfg["drivers"].get("ey", {})) > 0

    fields_out: dict = {}
    scalars_xr = None

    for key, ys_for_key in result.ys.items():
        ts = np.asarray(result.ts[key])

        if key.startswith("fields"):
            # Per-species moment datasets.
            species_datasets: dict[str, xr.Dataset] = {}
            for sp in species_names:
                if sp not in ys_for_key:
                    continue
                das = {}
                for moment_name, arr in ys_for_key[sp].items():
                    das[f"{key}-{moment_name}"] = xr.DataArray(
                        np.asarray(arr), coords=(("t", ts), ("x", x_axis))
                    )
                ds = xr.Dataset(das)
                ds.to_netcdf(
                    os.path.join(binary_dir, f"{key}-{sp}-t={round(ts[-1], 4)}.nc")
                )
                species_datasets[sp] = ds
                _spacetime_plots(ds, os.path.join(fields_root, sp))

            # Shared fields (e, de, pond live on nx; a / prev_a on nx+2 — strip
            # the ghost cells when projecting onto the x-axis).
            shared_das = {}
            for nm in ("e", "de", "pond"):
                if nm in ys_for_key:
                    shared_das[f"{key}-{nm}"] = xr.DataArray(
                        np.asarray(ys_for_key[nm]), coords=(("t", ts), ("x", x_axis))
                    )
            for nm in ("a", "prev_a"):
                if nm in ys_for_key:
                    arr = np.asarray(ys_for_key[nm])
                    shared_das[f"{key}-{nm}"] = xr.DataArray(
                        arr[:, 1:-1] if arr.shape[-1] == x_axis.size + 2 else arr,
                        coords=(("t", ts), ("x", x_axis)),
                    )

            # ep / em decomposition for the right- and left-going light waves,
            # only meaningful when an ey driver is configured.
            if has_ey and "a" in ys_for_key and "prev_a" in ys_for_key:
                a_full = np.asarray(ys_for_key["a"])
                prev_a_full = np.asarray(ys_for_key["prev_a"])
                ey = -(a_full[:, 1:-1] - prev_a_full[:, 1:-1]) / dt
                bz_t = np.gradient(a_full, dx, axis=1)[:, 1:-1]
                bz_prev = np.gradient(prev_a_full, dx, axis=1)[:, 1:-1]
                bz = 0.5 * (bz_t + bz_prev)
                c_light = float(cfg["units"]["derived"]["c_light"])
                shared_das[f"{key}-ep"] = xr.DataArray(
                    ey + c_light * bz, coords=(("t", ts), ("x", x_axis))
                )
                shared_das[f"{key}-em"] = xr.DataArray(
                    ey - c_light * bz, coords=(("t", ts), ("x", x_axis))
                )

            shared_xr = xr.Dataset(shared_das)
            shared_xr.to_netcdf(
                os.path.join(binary_dir, f"{key}-shared-t={round(ts[-1], 4)}.nc")
            )
            _spacetime_plots(shared_xr, fields_root)

            fields_out = {**species_datasets, "fields": shared_xr}

        elif key == "default":
            scalars_xr = xr.Dataset(
                {sk: xr.DataArray(np.asarray(sv), coords=(("t", ts),)) for sk, sv in ys_for_key.items()}
            )
            scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(ts[-1], 4)}.nc"))
            for nm, srs in scalars_xr.items():
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                srs.plot(ax=ax[0])
                ax[0].grid()
                np.log10(np.abs(srs)).plot(ax=ax[1])
                ax[1].grid()
                ax[1].set_ylabel(f"$\\log_{{10}}$(|{nm}|)")

                # Route species-tagged scalars to plots/scalars/<species>/.
                scalar_species = None
                for sp in species_names:
                    if nm.endswith(f"_{sp}"):
                        scalar_species = sp
                        break
                target_dir = (
                    os.path.join(scalars_root, scalar_species)
                    if scalar_species
                    else scalars_root
                )
                fig.savefig(os.path.join(target_dir, f"{nm}.png"), bbox_inches="tight")
                plt.close(fig)

        elif "_species_name" in cfg["save"].get(key, {}):
            species_name = cfg["save"][key]["_species_name"]
            sub_dir = os.path.join(dists_root, key)
            os.makedirs(sub_dir, exist_ok=True)
            x_arr = np.asarray(ys_for_key["x"])
            v_arr = np.asarray(ys_for_key["v"])
            n_snap = x_arr.shape[0]
            step = max(1, n_snap // 8)
            for it in range(0, n_snap, step):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(x_arr[it], v_arr[it], s=0.5, alpha=0.3)
                ax.set_xlabel("x")
                ax.set_ylabel("v")
                ax.set_title(f"{species_name} t={ts[it]:.2f}")
                fig.savefig(os.path.join(sub_dir, f"phase_t={ts[it]:.3f}.png"), bbox_inches="tight")
                plt.close(fig)

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})
    return {"fields": fields_out, "scalars": scalars_xr}
