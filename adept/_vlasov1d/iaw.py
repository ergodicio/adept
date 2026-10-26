"""Problem-specific ADEPT module for driven ion-acoustic turbulence.

Kinetic ions + Boltzmann electrons (``terms.field: poisson-boltzmann``) driven
by stochastic box-scale forcing. This module adds the diagnostics that IAW
turbulence studies need on every run, so they don't have to be recomputed from
raw moment output after the fact.
"""

import os

import numpy as np
import xarray as xr
from diffrax import SubSaveAt
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adept._vlasov1d.modules import BaseVlasov1D


class IAWTurbulence1D(BaseVlasov1D):
    """Vlasov-1D specialized for ion-acoustic turbulence (``solver: vlasov-1d-iaw``).

    On top of :class:`BaseVlasov1D` this module

    - saves an ``nk`` stream: the low-``|k|`` complex spectrum of the total
      kinetic-species charge density, densely sampled in time. Each sample is
      only ``2 * (nk_modes + 1)`` floats, so the spectrum can be saved at a
      cadence far beyond what full moment profiles allow.
    - post-processes IAW-specific plots: the late-window-averaged density
      spectrum P(m) with a k*lambda_De axis and m^-2 / m^-3/2 guides, a
      P(m, t) spectrogram, and delta-f phase-space panels (f - <f>_x) for
      every configured distribution save.

    Configuration (all optional) lives under a top-level ``iaw_diagnostics``
    block::

        iaw_diagnostics:
          nk_modes: 1024            # box modes to save (clipped to nx // 2)
          nk_nt: 2001               # nk time samples
          spectrum_window: [0.5, 1.0]   # late-window average, fractions of tmax
    """

    def __init__(self, cfg) -> None:
        """Validate configuration and stash the IAW diagnostics options."""
        super().__init__(cfg)
        iaw_cfg = cfg.get("iaw_diagnostics") or {}
        self.nk_modes = int(iaw_cfg.get("nk_modes", 1024))
        self.nk_nt = int(iaw_cfg.get("nk_nt", 2001))
        self.spectrum_window = tuple(iaw_cfg.get("spectrum_window", (0.5, 1.0)))

    def screening_length(self) -> float:
        """Return the Boltzmann-electron screening length in code units.

        Falls back to sqrt(Te) (i.e. n0 = 1) when ``lambda_De`` is not given
        explicitly, and to 1.0 when the field solver is not poisson-boltzmann.
        """
        if self.cfg["terms"]["field"] != "poisson-boltzmann":
            return 1.0
        boltzmann_cfg = self.cfg["terms"].get("boltzmann_electrons") or {}
        lambda_De = boltzmann_cfg.get("lambda_De")
        if lambda_De is None:
            lambda_De = float(np.sqrt(boltzmann_cfg.get("Te", 1.0)))
        return float(lambda_De)

    def init_diffeqsolve(self):
        """Assemble the base solver quantities, then attach the nk save stream."""
        super().init_diffeqsolve()

        grid = self.simulation.grid
        nx = grid.nx
        nk_modes = min(self.nk_modes, nx // 2)
        t_ax = np.linspace(grid.tmin, grid.tmax, self.nk_nt)

        species_grids = self.cfg["grid"]["species_grids"]
        species_params = self.cfg["grid"]["species_params"]

        def nk_save(t, y, args):
            """Save the low-|k| complex charge-density spectrum as (re, im) rows."""
            rho = jnp.zeros(nx)
            for name, species_grid in species_grids.items():
                rho = rho + species_params[name]["charge"] * jnp.sum(y[name], axis=1) * species_grid["dv"]
            rho_k = jnp.fft.rfft(rho)[: nk_modes + 1] / nx
            return jnp.stack([jnp.real(rho_k), jnp.imag(rho_k)])

        # Registered directly on the Diffrax saveat (not through cfg["save"], which
        # was already expanded by get_save_quantities and whose downstream consumers
        # only understand field/dist/default saves).
        self.diffeqsolve_quants["saveat"]["subs"]["nk"] = SubSaveAt(ts=t_ax, fn=nk_save)

    def post_process(self, run_output: dict, td: str):
        """Run the base post-processing, then the IAW-specific spectra and plots."""
        datasets = super().post_process(run_output, td)
        result = run_output["solver result"]

        iaw_dir = os.path.join(td, "plots", "iaw")
        os.makedirs(iaw_dir, exist_ok=True)

        if "nk" in result.ys:
            datasets["nk"] = self._process_nk(result, td, iaw_dir)

        self._plot_delta_f_panels(datasets.get("dists", {}), td)

        return datasets

    def _process_nk(self, result, td: str, iaw_dir: str) -> xr.Dataset:
        """Write the nk netcdf and plot the spectrogram + late-window spectrum."""
        grid = self.simulation.grid
        length = grid.xmax - grid.xmin
        lambda_De = self.screening_length()

        t_ax = np.asarray(result.ts["nk"])
        arr = np.asarray(result.ys["nk"])  # (nt, 2, nk_modes + 1)
        power = arr[:, 0] ** 2 + arr[:, 1] ** 2
        m_ax = np.arange(arr.shape[-1])

        nk_ds = xr.Dataset(
            {
                "nk_real": xr.DataArray(arr[:, 0], coords=(("t", t_ax), ("m", m_ax))),
                "nk_imag": xr.DataArray(arr[:, 1], coords=(("t", t_ax), ("m", m_ax))),
                "P": xr.DataArray(power, coords=(("t", t_ax), ("m", m_ax))),
            },
            attrs={"length": float(length), "lambda_De": lambda_De},
        )
        nk_ds.to_netcdf(os.path.join(td, "binary", "nk.nc"))

        # --- P(m, t) spectrogram ---
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        with np.errstate(divide="ignore"):
            log_power = np.log10(power[:, 1:])
        pcm = ax.pcolormesh(m_ax[1:], t_ax, log_power, cmap="viridis", rasterized=True)
        ax.set_xscale("log")
        ax.set_xlabel("mode number $m$")
        ax.set_ylabel("$t$")
        fig.colorbar(pcm, ax=ax, label=r"$\log_{10} |n_k|^2$")
        fig.savefig(os.path.join(iaw_dir, "nk_spectrogram.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

        # --- late-window-averaged spectrum ---
        t0 = t_ax[0] + self.spectrum_window[0] * (t_ax[-1] - t_ax[0])
        t1 = t_ax[0] + self.spectrum_window[1] * (t_ax[-1] - t_ax[0])
        window = (t_ax >= t0) & (t_ax <= t1)
        p_avg = power[window].mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
        ax.loglog(m_ax[1:], p_avg[1:] / p_avg[1], "k", lw=1.2, label=rf"$t \in [{t0:.0f}, {t1:.0f}]$")

        # power-law guides anchored to the curve
        for slope, m_anchor, label in [(-2.0, 4, r"$m^{-2}$"), (-1.5, 40, r"$m^{-3/2}$")]:
            if m_anchor >= len(p_avg):
                continue
            mm = np.array([m_anchor, min(m_anchor * 10, len(p_avg) - 1)])
            y0 = 3.0 * p_avg[m_anchor] / p_avg[1]
            ax.loglog(mm, y0 * (mm / mm[0]) ** slope, color="gray", lw=1, alpha=0.9)
            ax.annotate(label, (mm[1], y0 * (mm[1] / mm[0]) ** slope), fontsize=11, color="gray")

        m_debye = length / (2 * np.pi * lambda_De)
        if m_debye < m_ax[-1]:
            ax.axvline(m_debye, color="k", ls=":", lw=1)
            ax.annotate(r"$k\lambda_{De}=1$", (m_debye * 1.08, 0.3), rotation=90, fontsize=10)

        ax.set_xlabel("mode number $m$  ($k = 2\\pi m/L$)")
        ax.set_ylabel(r"$\langle |n_k|^2\rangle_t / \langle |n_1|^2\rangle_t$")
        ax.legend(frameon=False)
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda m, _c=2 * np.pi * lambda_De / length: m * _c,
                lambda k, _c=2 * np.pi * lambda_De / length: k / _c,
            ),
        )
        secax.set_xlabel(r"$k\lambda_{De}$")
        fig.savefig(os.path.join(iaw_dir, "density_spectrum.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

        return nk_ds

    def _plot_delta_f_panels(self, dists: dict, td: str):
        """Plot f - <f>_x facet panels for every species distribution save."""
        for save_key, f_xr in dists.items():
            save_cfg = self.cfg["save"].get(save_key, {})
            if "_species_name" not in save_cfg:
                continue
            species_name = save_cfg["_species_name"]
            v_dim = f"v_{species_name}"

            f_species = f_xr[save_key]
            delta_f = f_species - f_species.mean("x")

            t_skip = max(int(delta_f.coords["t"].data.size // 8), 1)
            delta_f = delta_f[slice(0, -1, t_skip)]
            n_panels = delta_f.sizes["t"]
            ncols = min(4, n_panels)
            nrows = (n_panels + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False, constrained_layout=True
            )
            axes_flat = axes.flatten()
            for i in range(n_panels):
                data = delta_f.isel(t=i)
                vabs = float(np.abs(data).max())
                vabs = vabs if vabs > 0 else 1.0
                im = axes_flat[i].pcolormesh(
                    data.coords["x"].values,
                    data.coords[v_dim].values,
                    data.values.T,
                    vmin=-vabs,
                    vmax=vabs,
                    cmap="RdBu_r",
                    rasterized=True,
                )
                axes_flat[i].set_title(f"t = {data.coords['t'].values:.1f}")
                axes_flat[i].set_xlabel("x")
                axes_flat[i].set_ylabel("v")
                fig.colorbar(im, ax=axes_flat[i])
            for i in range(n_panels, len(axes_flat)):
                axes_flat[i].set_visible(False)

            fig.suptitle(f"{save_key}: f - <f>_x")
            fig.savefig(os.path.join(td, "plots", "dists", save_key, "phase_space_dfx.png"), bbox_inches="tight")
            plt.close(fig)
