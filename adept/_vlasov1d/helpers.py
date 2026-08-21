"""Initialization and post-processing helpers for Vlasov-1D simulations."""

#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import math
import os
from time import time

import numpy as np
import xarray
from diffrax import Solution
from jax import numpy as jnp
from jax import tree_util as jtu
import equinox as eqx
from matplotlib import pyplot as plt
from scipy.special import gamma

from adept._vlasov1d.simulation import SubspeciesDistributionSpec, Vlasov1DSimulation
from adept._vlasov1d.storage import store_f, store_fields
from adept.normalization import UREG, PlasmaNormalization, normalize

from .. import patched_mlflow as mlflow

# gamma_da = xarray.open_dataarray(os.path.join(os.path.dirname(__file__), "gamma_func_for_sg.nc"))
# m_ax = gamma_da.coords["m"].data
# g_3_m = np.squeeze(gamma_da.loc[{"gamma": "3/m"}].data)
# g_5_m = np.squeeze(gamma_da.loc[{"gamma": "5/m"}].data)


def gamma_1_over_m(m):
    """Evaluate Gamma(1 / m) for super-Gaussian normalization."""
    return gamma(1.0 / m)


def gamma_3_over_m(m):
    """Evaluate Gamma(3 / m) for super-Gaussian normalization."""
    return gamma(3.0 / m)  # np.interp(m, m_ax, g_3_m)


def gamma_5_over_m(m):
    """Evaluate Gamma(5 / m) for super-Gaussian normalization."""
    return gamma(5.0 / m)  # np.interp(m, m_ax, g_5_m)

def _initialize_supergaussian_distribution_(
    nx: int,
    nv: int,
    v0=0.0,
    supergaussian_order=2.0,
    T0=1.0,
    mass=1.0,
    vmax=6.0,
    vmin=None,
    n_prof=np.ones(1),
):
    """
    Initialize a supergaussian distribution function.

    For supergaussian_order=2, this gives a Maxwell-Boltzmann distribution.

    Args:
        nx: size of grid in x
        nv: size of grid in v
        v0: drift velocity
        supergaussian_order: order of supergaussian (2 = Maxwell-Boltzmann)
        T0: temperature
        mass: species mass for thermal velocity calculation
        vmax: upper bound of the velocity grid
        vmin: lower bound of the velocity grid (defaults to ``-vmax``)
        n_prof: density profile (noise should already be applied)

    Returns:
        Tuple of (f[nx, nv], vax[nv])
    """
    if vmin is None:
        vmin = -vmax
    dv = (vmax - vmin) / nv
    vax = np.linspace(vmin + dv / 2.0, vmax - dv / 2.0, nv)

    # Thermal velocity: v_t = sqrt(T/m)
    v_thermal = np.sqrt(T0 / mass)

    # 1D super-Gaussian width normalization: alpha = sqrt(Gamma(1/m)/Gamma(3/m))
    # makes the realized VARIANCE equal T0/mass for every order m, i.e. a species
    # labeled T0 is at temperature T0 for any m (alpha = sqrt(2) at m=2, unchanged).
    # This is also the convention the Krook target (variance T0/mass) and the
    # SuperGaussianDougherty temperature relation D = beta^(-2/m)*G(3/m)/G(1/m)
    # already use. The previous alpha = sqrt(3*G(3/m)/G(5/m)) is the 3D-isotropic
    # (Matte/DLM) normalization -- it fixes <v^2>_3D = 3*T0/mass -- and on a 1D
    # axis it inflates the variance by F(m) = 3*G(3/m)^2/(G(5/m)*G(1/m))
    # (x1.24 at m=3, x1.37 at m=4). See docs config.md.
    alpha = np.sqrt(gamma_1_over_m(supergaussian_order) / gamma_3_over_m(supergaussian_order))

    single_dist = -(np.power(np.abs((vax[None, :] - v0) / (alpha * v_thermal)), supergaussian_order))

    single_dist = np.exp(single_dist)

    f = np.repeat(single_dist, nx, axis=0)
    # normalize
    f = f / np.sum(f, axis=1)[:, None] / dv

    if n_prof.size > 1:
        # scale by density profile
        f = n_prof[:, None] * f

    return f, vax


def _initialize_total_distribution_(cfg, simulation: Vlasov1DSimulation):
    """
    Initialize distribution functions for all species using domain models.

    The species config is normalized in modules.py:get_derived_quantities() so that
    a species config always exists (for backward compatibility with single-species
    config files, a default electron species is generated).

    Args:
        cfg: Configuration dictionary
        grid: Grid object with spatial coordinates
        species_distribution_specs: Dictionary from species name to the list of SubspeciesDistributionSpecs for it.
        norm: PlasmaNormalization for unit conversion (required for linear/exponential profiles)

    Returns:
        dict mapping species_name -> (n_prof, f_s, v_ax)
    """
    species_distributions = {}
    species_found = False

    norm = simulation.plasma_norm
    grid = simulation.grid
    species_configs = simulation.species
    species_distribution_specs = simulation.species_distributions

    for species_cfg in species_configs:
        species_name = species_cfg.name
        density_components = species_distribution_specs[species_name]
        vmax = species_cfg.vmax
        vmin = species_cfg.vmin
        nv = species_cfg.nv
        mass = species_cfg.mass

        # Initialize arrays for this species
        n_prof_species = np.zeros([grid.nx])
        dv = (vmax - vmin) / nv
        vax = np.linspace(vmin + dv / 2.0, vmax - dv / 2.0, nv)
        f_species = np.zeros([grid.nx, nv])

        # Sum contributions from all density components
        for distribution_spec in density_components:
            # Evaluate density profile (noise is applied by the domain model)
            nprof = np.array(distribution_spec.density_profile(grid.x))
            n_prof_species += nprof

            # Initialize distribution
            temp_f, _ = _initialize_supergaussian_distribution_(
                nx=grid.nx,
                nv=nv,
                v0=distribution_spec.v0,
                supergaussian_order=distribution_spec.supergaussian_order,
                T0=distribution_spec.T0,
                mass=mass,
                vmax=vmax,
                vmin=vmin,
                n_prof=nprof,
            )
            f_species += temp_f
            species_found = True

        species_distributions[species_name] = (n_prof_species, f_species, vax)

    if not species_found:
        raise ValueError("No species found! Check the config")

    return species_distributions


def get_akw_from_intensity_wavelength(intensity, wavelength, leftgoing, norm: PlasmaNormalization | None = None):
    '''getting amplitude (a), wave number (k) and angular frequency (w)
    from intensity and wavelength (passed in as args to this function) defined in
    'intensity_wavelength' type of configs'''

    intensity = UREG.Quantity(intensity).to("W/m^2")
    wavelength = UREG.Quantity(wavelength).to("nm")

    e = UREG.e
    m_e = UREG.m_e
    eps0 = UREG.epsilon_0
    c = UREG.c

    # Standard a0 = eE0/(m_e c w0) — identical to HermiteSRS1D formula
    a0_std = ((e * wavelength / (m_e * math.pi)) * (intensity / (2 * eps0 * c**5)) ** 0.5).to("").magnitude
    # Vlasov normalization: a0_vlasov = a0_std / β  (β = v0/c)
    a0 = a0_std * norm.speed_of_light_norm()

    # k0 in Debye-length units: k0_vlasov = k_phys x v0/wp0
    k0_phys = (2 * math.pi / wavelength).to("1/m")
    k_sign = -1.0 if leftgoing else 1.0
    k0 = k_sign * float((k0_phys * norm.L0).to("").magnitude)

    # w0 normalized to wp0 (same normalization as Hermite)
    w0_phys = (2 * math.pi * c / wavelength).to("1/s")
    w0 = float((w0_phys * norm.tau).to("").magnitude)

    return a0, k0, w0


def plot_driver_spectra(cfg: dict, td: str, args: dict):
    """Per-line intensity and phase vs frequency offset, for each multi-line driver.

    Reads the LIVE driver objects out of `args["drivers"]` rather than re-deriving
    the line set from the config's init/seed. That matters twice over: the plot cannot
    drift if BroadbandDriver's construction changes, and it shows optimized line sets
    from a backward pass (which never appear in the config) automatically. A
    `BroadbandDriver` contributes its per-line arrays (`amplitudes`/`delta_omega`/
    `phases`); plain `EMDriver`s contribute their scalars, so a hand-built list of
    mono drivers still plots.

    Per-line intensity needs no normalization constant. The driver builds amplitudes as
    A_j = a0 * sqrt(w_j / sum_k w_k), so

        I_j / I_base = A_j^2 / sum_k A_k^2

    exactly, independent of a0 and of the plasma normalization. `base_intensity` is
    read from the config only to put an absolute scale on the axis.

    Panels: I_j (linear), log10 I_j (dynamic range -- the informative one once an
    optimizer produces structure, since I ~ A^2 doubles the span), and phi_j.
    Single-line (monochromatic) drivers are skipped.
    """
    drivers = (args or {}).get("drivers")
    if drivers is None:
        return
    out_dir = os.path.join(td, "plots", "drivers")

    for field in ("ex", "ey"):
        dlist = getattr(drivers, field, None)
        if not dlist:
            continue

        amp_list, dw_list, phase_list = [], [], []
        for d in dlist:
            if hasattr(d, "amplitudes"):  # BroadbandDriver: (N,) array leaves
                amp_list.extend(np.asarray(d.amplitudes, dtype=float))
                dw_list.extend(np.asarray(d.delta_omega, dtype=float) / float(d.w0))
                phase_list.extend(np.asarray(d.phases, dtype=float))
            else:  # plain EMDriver: scalar leaves
                amp_list.append(float(d.a0))
                dw_list.append(float(d.dw0) / float(d.w0))
                phase_list.append(float(d.phase))
        if len(amp_list) < 2:
            continue  # monochromatic -> no spectrum to show

        amp = np.asarray(amp_list)
        dw = np.asarray(dw_list)  # dw_j/w0
        phases = np.asarray(phase_list)

        power = amp**2
        frac = power / power.sum() if power.sum() > 0 else power
        # absolute scale: base_intensity of the broadband driver in this field (whichever
        # key it was given under -- a driver keyed '1' must not lose the axis scale)
        base = None
        for dcfg in (cfg.get("drivers", {}).get(field, {}) or {}).values():
            ints = ((dcfg or {}).get("params", {}) or {}).get("intensities")
            if isinstance(ints, dict) and ints.get("base_intensity") is not None:
                base = ints["base_intensity"]
                break

        I_j, unit = frac, ""
        if isinstance(base, str) and base.split():  # "2.378e+14 W/cm^2"
            val, _, u = base.partition(" ")
            try:
                I_j, unit = frac * float(val), u.strip()
            except ValueError:
                pass
        elif isinstance(base, (int, float)):
            I_j = frac * float(base)

        order = np.argsort(dw)
        dw, I_j, phases = dw[order], I_j[order], phases[order]

        # constrained_layout sizes the suptitle band to the actual text; do NOT pair
        # it with tight_layout(rect=...) + suptitle(y=...), which reserve a fixed band
        # and leave a gap when the title is shorter than the reservation.
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True, constrained_layout=True)
        bw_pct = (dw.max() - dw.min()) * 100.0
        spacing = float(np.diff(np.sort(dw)).mean()) if len(dw) > 1 else 0.0
        run_name = ((cfg.get("mlflow") or {}).get("run")) or ""

        title = f"{field} driver — broadband line spectrum"
        if run_name:
            title += f"\n{run_name}"
        title += (
            f"\n{len(dlist)} lines   |   full width $\\Delta\\omega/\\omega_0$ = "
            f"{bw_pct:.3g}%   |   spacing $\\delta\\omega/\\omega_0$ = {spacing:.3g}"
        )
        if base is not None:
            title += f"\n$I_{{base}}$ = {base}"
            if unit:
                title += f"   |   $I_j$ = {I_j.mean():.4g} {unit} mean per line"
        fig.suptitle(title, fontsize=9.5, linespacing=1.4)

        axes[0].plot(dw, I_j, "o", ms=4, color="#1f77b4")
        axes[0].set_ylabel("line intensity $I_j$" + (f"  [{unit}]" if unit else "  [$I_j/I_{base}$]"))
        axes[0].set_ylim(bottom=0)
        axes[0].annotate(
            rf"$\Sigma_j I_j$ = {I_j.sum():.4g}", xy=(0.02, 0.06), xycoords="axes fraction", fontsize=8, color="0.35"
        )

        pos = I_j > 0
        if pos.any():
            axes[1].plot(dw[pos], np.log10(I_j[pos]), "o", ms=4, color="#d62728")
            lo_, hi_ = np.log10(I_j[pos]).min(), np.log10(I_j[pos]).max()
            if hi_ - lo_ < 0.1:
                axes[1].set_ylim(lo_ - 0.5, hi_ + 0.5)
            axes[1].annotate(
                f"dynamic range: {10 ** (hi_ - lo_):.3g}x",
                xy=(0.02, 0.88),
                xycoords="axes fraction",
                fontsize=8,
                color="0.35",
            )
        if (~pos).any():  # an optimizer can drive lines to zero; don't hide them
            floor = np.log10(I_j[pos]).min() if pos.any() else 0.0
            axes[1].plot(dw[~pos], np.full(int((~pos).sum()), floor), "x", ms=6, color="0.5")
            axes[1].annotate(
                f"{int((~pos).sum())} line(s) at I=0 (x)",
                xy=(0.02, 0.06),
                xycoords="axes fraction",
                fontsize=8,
                color="0.4",
            )
        axes[1].set_ylabel(r"$\log_{10} I_j$")

        axes[2].plot(dw, phases, "o", ms=4, color="#2ca02c")
        axes[2].set_ylabel(r"phase $\phi_j$ [rad]")
        axes[2].set_xlabel(r"$\delta\omega_j/\omega_0$")
        axes[2].set_ylim(-0.25, 2 * np.pi + 0.25)
        axes[2].set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        axes[2].set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

        for ax in axes:
            ax.grid(alpha=0.3)
            ax.axvline(0.0, color="0.6", lw=0.8, ls="--")

        os.makedirs(out_dir, exist_ok=True)
        # no tight_layout here — constrained_layout (set on the figure) already sized
        # the title band; calling both would re-reserve a fixed strip and reopen the gap
        fig.savefig(os.path.join(out_dir, f"{field}-lines.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)


def post_process(result: Solution, cfg: dict, td: str, args: dict):
    """Write binary output and diagnostic plots from a completed Vlasov-1D solve."""
    t0 = time()

    # Driver line spectra (multi-line drivers only). Guarded: a diagnostics failure
    # here must never cost a completed solve its binary output and field/dist plots.
    try:
        plot_driver_spectra(cfg, td, args)
    except Exception as exc:
        print(f"[post_process] driver spectrum plot skipped: {type(exc).__name__}: {exc}", flush=True)

    # Get species names for directory creation
    species_names = list(cfg["grid"]["species_grids"].keys())

    # Create base plot directories
    os.makedirs(os.path.join(td, "plots"), exist_ok=True)

    # Create fields directory structure
    # - fields/ (shared EM fields at top level)
    # - fields/{species}/ (species-specific moments)
    os.makedirs(os.path.join(td, "plots", "fields"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "logplots"), exist_ok=True)
    os.makedirs(os.path.join(td, "plots", "fields", "lineouts"), exist_ok=True)
    for species_name in species_names:
        species_dir = os.path.join(td, "plots", "fields", species_name)
        os.makedirs(species_dir, exist_ok=True)
        os.makedirs(os.path.join(species_dir, "logplots"), exist_ok=True)
        os.makedirs(os.path.join(species_dir, "lineouts"), exist_ok=True)

    # Create scalars directory structure
    # - scalars/ (shared field scalars at top level)
    # - scalars/{species}/ (species-specific scalars)
    os.makedirs(os.path.join(td, "plots", "scalars"), exist_ok=True)
    for species_name in species_names:
        os.makedirs(os.path.join(td, "plots", "scalars", species_name), exist_ok=True)

    # Create dists directory for distribution function snapshots (one subdir per dist save key)
    os.makedirs(os.path.join(td, "plots", "dists"), exist_ok=True)
    for save_key, save_cfg in cfg["save"].items():
        if "_species_name" in save_cfg:
            os.makedirs(os.path.join(td, "plots", "dists", save_key), exist_ok=True)

    binary_dir = os.path.join(td, "binary")
    os.makedirs(binary_dir)

    fields_result = {}
    fields_base_dir = os.path.join(td, "plots", "fields")

    for k in result.ys.keys():
        if k.startswith("field"):
            # store_fields now returns dict with species names and "fields" keys
            fields_dict = store_fields(cfg, binary_dir, result.ys[k], result.ts[k], k)

            # Plot species-specific moments in fields/{species}/
            for species_name in species_names:
                if species_name not in fields_dict:
                    continue

                species_xr = fields_dict[species_name]
                species_dir = os.path.join(fields_base_dir, species_name)

                t_skip = int(species_xr.coords["t"].data.size // 8)
                t_skip = t_skip if t_skip > 1 else 1
                tslice = slice(0, -1, t_skip)

                for nm, fld in species_xr.items():
                    # Strip prefix (e.g., "fields-n" -> "n")
                    field_name = nm.split("-", 1)[1] if "-" in nm else nm

                    # Spacetime plot
                    fld.plot(figsize=(12, 8))
                    plt.savefig(os.path.join(species_dir, f"spacetime_{field_name}.png"), bbox_inches="tight", dpi=150)
                    plt.close()

                    # Log plot
                    np.log10(np.abs(fld)).plot(figsize=(12, 8))
                    plt.savefig(
                        os.path.join(species_dir, "logplots", f"spacetime_log_{field_name}.png"),
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close()

                    # Lineouts
                    fld[tslice].T.plot(col="t", col_wrap=4)
                    plt.savefig(os.path.join(species_dir, "lineouts", f"{field_name}.png"), bbox_inches="tight")
                    plt.close()

            # Plot shared field data (e, de, a, pond) in fields/ (top level)
            if "fields" in fields_dict:
                shared_xr = fields_dict["fields"]

                t_skip = int(shared_xr.coords["t"].data.size // 8)
                t_skip = t_skip if t_skip > 1 else 1
                tslice = slice(0, -1, t_skip)

                for nm, fld in shared_xr.items():
                    field_name = nm.split("-", 1)[1] if "-" in nm else nm

                    fld.plot()
                    plt.savefig(os.path.join(fields_base_dir, f"spacetime_{field_name}.png"), bbox_inches="tight")
                    plt.close()

                    np.log10(np.abs(fld)).plot()
                    log_path = os.path.join(fields_base_dir, "logplots", f"spacetime_log_{field_name}.png")
                    plt.savefig(log_path, bbox_inches="tight")
                    plt.close()

                    fld[tslice].T.plot(col="t", col_wrap=4)
                    plt.savefig(os.path.join(fields_base_dir, "lineouts", f"{field_name}.png"), bbox_inches="tight")
                    plt.close()

            fields_result = fields_dict

        elif k.startswith("default"):
            scalars_xr = xarray.Dataset(
                {k: xarray.DataArray(v, coords=(("t", result.ts["default"]),)) for k, v in result.ys["default"].items()}
            )
            scalars_xr.to_netcdf(os.path.join(binary_dir, f"scalars-t={round(scalars_xr.coords['t'].data[-1], 4)}.nc"))

            scalars_base_dir = os.path.join(td, "plots", "scalars")
            for nm, srs in scalars_xr.items():
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
                srs.plot(ax=ax[0])
                ax[0].grid()
                np.log10(np.abs(srs)).plot(ax=ax[1])
                ax[1].grid()
                ax[1].set_ylabel("$log_{10}$(|" + nm + "|)")

                # Determine if this is a species-specific or shared scalar
                # Species-specific scalars have format: mean_X_{species_name}
                scalar_species = None
                for species_name in species_names:
                    if nm.endswith(f"_{species_name}"):
                        scalar_species = species_name
                        break

                if scalar_species:
                    # Save to scalars/{species}/
                    fig.savefig(os.path.join(scalars_base_dir, scalar_species, f"{nm}.png"), bbox_inches="tight")
                else:
                    # Shared field scalar (e.g., mean_e2, mean_de2, mean_pond)
                    fig.savefig(os.path.join(scalars_base_dir, f"{nm}.png"), bbox_inches="tight")
                plt.close()

    f_result = store_f(cfg, result.ts, td, result.ys)

    # Plot velocity space distributions for each species dist save key (skip diag saves)
    for save_key, f_xr in f_result.items():
        if "_species_name" not in cfg["save"][save_key]:
            continue
        species_name = cfg["save"][save_key]["_species_name"]
        f_species = f_xr[save_key]
        species_dist_dir = os.path.join(td, "plots", "dists", save_key)

        # Select ~8 time snapshots for facet plot
        t_skip = int(f_species.coords["t"].data.size // 8)
        t_skip = t_skip if t_skip > 1 else 1
        tslice = slice(0, -1, t_skip)

        # Create f(x,v) phase space plot
        # First panel: f(t=0), remaining panels: f(t) - f(t=0) with separate color scales
        f_sliced = f_species[tslice]
        f0 = f_species.isel(t=0)
        f_diff = f_sliced.isel(t=slice(1, None)) - f0
        n_diff = f_diff.sizes["t"]
        n_total = 1 + n_diff
        ncols = min(4, n_total)
        nrows = (n_total + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False, constrained_layout=True)
        axes_flat = axes.flatten()

        # Plot f(t=0) in first panel with its own colorbar
        im0 = axes_flat[0].pcolormesh(f0.coords["x"].values, f0.coords[f"v_{species_name}"].values, f0.values.T)
        axes_flat[0].set_xlabel("x")
        axes_flat[0].set_ylabel("v")
        axes_flat[0].set_title(f"t = {f_sliced.coords['t'].values[0]:.2f}")
        fig.colorbar(im0, ax=axes_flat[0], label="f")

        # Plot f(t) - f(t=0) in remaining panels with shared color scale
        if n_diff > 0:
            vmin, vmax = float(f_diff.min()), float(f_diff.max())
            vabs = max(abs(vmin), abs(vmax))
            for i in range(n_diff):
                ax = axes_flat[i + 1]
                data = f_diff.isel(t=i)
                im = ax.pcolormesh(
                    data.coords["x"].values,
                    data.coords[f"v_{species_name}"].values,
                    data.values.T,
                    vmin=-vabs,
                    vmax=vabs,
                    cmap="RdBu_r",
                )
                ax.set_xlabel("x")
                ax.set_ylabel("v")
                ax.set_title(f"t = {f_diff.coords['t'].values[i]:.2f}")
            # Add shared colorbar for difference panels
            fig.colorbar(im, ax=axes_flat[1 : n_diff + 1].tolist(), label="f - f(t=0)")

        # Hide unused axes
        for i in range(n_total, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.savefig(os.path.join(species_dist_dir, "phase_space.png"), bbox_inches="tight")
        plt.close()

    mlflow.log_metrics({"postprocess_time_min": round((time() - t0) / 60, 3)})

    return {"fields": fields_result, "dists": f_result, "scalars": scalars_xr}
