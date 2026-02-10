"""Specialized ADEPTModule for 1D Electron Plasma Wave (EPW) analysis."""

import os

import numpy as np
from jax import Array

from adept._spectrax1d.base_module import BaseSpectrax1D


class EPW1D(BaseSpectrax1D):
    """
    Specialized ADEPTModule for 1D Electron Plasma Wave (EPW) analysis.

    Extends BaseSpectrax1D with EPW-specific postprocessing:
    - Plots first mode amplitude and frequency of EPW over time
    - Enhanced visualization of Hermite coefficients for the first spatial mode
    - 2x2 plots with ions and electrons on linear and log scale
    - Time axis vertical, Hermite mode axis horizontal
    """

    def _compute_epw_frequency(self, Ex_k1: Array, t_array: Array) -> tuple[Array, Array]:
        """
        Compute instantaneous frequency of the first EPW mode from phase evolution.

        Args:
            Ex_k1: Complex amplitude of Ex at k=1 mode as function of time (nt,)
            t_array: Time array (nt,)

        Returns:
            tuple: (frequencies, times_for_frequency)
                frequencies: Instantaneous frequency dφ/dt at each time
                times_for_frequency: Time points for frequency (centered difference)
        """
        # Extract phase
        phase = np.unwrap(np.angle(Ex_k1))

        # Compute instantaneous frequency using centered differences
        dt = t_array[1] - t_array[0]
        frequencies = np.gradient(phase, dt)

        return frequencies, t_array

    def _plot_epw_diagnostics(self, Fk: Array, t_array: Array, Nx: int, Ny: int, Nz: int, td: str) -> None:
        """
        Plot EPW amplitude and frequency for the first mode over time.

        Creates a 2-panel figure:
        - Left: Amplitude |Ex(k=1)| vs time (log scale) - full time range
        - Right: Frequency ω(k=1) vs time - zoomed to driver-on period, excluding final 50 ωpe^-1

        Args:
            Fk: Field array of shape (nt, 6, Ny, Nx, Nz)
            t_array: Time array
            Nx, Ny, Nz: Grid dimensions
            td: Temporary directory for plots
        """
        import matplotlib.pyplot as plt

        # k=1 mode at index 1 in standard FFT ordering
        idx_k1 = 1
        idx_k0 = 0  # k=0 for other dimensions (1D problem in x)

        # Extract Ex at k=1 mode
        if idx_k1 < Nx:
            Ex_k1 = Fk[:, 0, idx_k0, idx_k1, idx_k0]

            # Compute amplitude
            amplitude = np.abs(Ex_k1)

            # Compute instantaneous frequency
            frequencies, freq_times = self._compute_epw_frequency(Ex_k1, t_array)

            # Determine time window for frequency plot
            # Get driver config if available
            driver_config = self.cfg.get("drivers", {})
            ex_drivers = driver_config.get("ex", {})

            if ex_drivers:
                # Use first driver config to determine when driver is on
                first_driver = next(iter(ex_drivers.values()))
                t_center = first_driver.get("t_center", 0.0)
                t_width = first_driver.get("t_width", 0.0)
                t_rise = first_driver.get("t_rise", 0.0)
                # Driver is fully on after t_center - t_width/2 + t_rise
                t_start = t_center - t_width / 2 + t_rise
            else:
                # No driver config, use heuristic: skip first 25% of simulation
                t_start = 0.25 * t_array[-1]

            # Omit final 50 ωpe^-1
            t_end = t_array[-1] - 50.0

            # Create time mask for frequency plot
            freq_mask = (freq_times >= t_start) & (freq_times <= t_end)
            freq_times_zoomed = freq_times[freq_mask]
            frequencies_zoomed = frequencies[freq_mask]

            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
            fig.suptitle("EPW First Mode Diagnostics (k=1)", fontsize=14, fontweight="bold")

            # Left panel: Amplitude (full time range)
            axes[0].semilogy(t_array, amplitude, "b-", linewidth=2)
            axes[0].set_xlabel(r"Time ($\omega_{pe}^{-1}$)", fontsize=12)
            axes[0].set_ylabel(r"$|E_x(k=1)|$", fontsize=12)
            axes[0].set_title("EPW Amplitude (First Mode)", fontsize=12)
            axes[0].grid(True, alpha=0.3)

            # Right panel: Frequency (zoomed to driver-on period, excluding final 50)
            axes[1].plot(freq_times_zoomed, frequencies_zoomed, "r-", linewidth=2)
            axes[1].set_xlabel(r"Time ($\omega_{pe}^{-1}$)", fontsize=12)
            axes[1].set_ylabel(r"$\omega$ ($\omega_{pe}$)", fontsize=12)
            axes[1].set_title(f"EPW Frequency (t={t_start:.1f} to {t_end:.1f})", fontsize=12)
            axes[1].grid(True, alpha=0.3)

            plt.savefig(os.path.join(td, "plots", "epw_mode1_diagnostics.png"), bbox_inches="tight")
            plt.close()

    def _plot_hermite_coefficients_enhanced(
        self,
        Ck_electrons: Array,
        Ck_ions: Array,
        t_array: Array,
        Nn_e: int,
        Nm_e: int,
        Np_e: int,
        Nn_i: int,
        Nm_i: int,
        Np_i: int,
        Nx: int,
        Ny: int,
        Nz: int,
        td: str,
    ) -> None:
        """
        Enhanced visualization of Hermite coefficients at k=1 mode.

        Creates a 2x2 plot grid:
        - Top row: Electron species (linear and log scale)
        - Bottom row: Ion species (linear and log scale)
        - Time axis: vertical (y-axis)
        - Hermite mode axis: horizontal (x-axis)

        For 1D1V problems, only one Hermite mode dimension is active and visible.

        Args:
            Ck_electrons: Electron distribution function coefficients (nt, Np_e, Nm_e, Nn_e, Ny, Nx, Nz)
            Ck_ions: Ion distribution function coefficients (nt, Np_i, Nm_i, Nn_i, Ny, Nx, Nz)
            t_array: Time array
            Nn_e, Nm_e, Np_e: Electron Hermite mode dimensions
            Nn_i, Nm_i, Np_i: Ion Hermite mode dimensions
            Nx, Ny, Nz: Spatial grid dimensions
            td: Temporary directory for plots
        """
        import matplotlib.pyplot as plt
        import xarray as xr

        # k=1 mode at index 1 in standard FFT ordering
        idx_k1 = 1
        idx_k0 = 0  # k=0 for other dimensions (1D problem in x)

        if idx_k1 < Nx:
            # Extract k=1 mode for all time steps and all Hermite modes
            # Ck shape per species: (nt, Np, Nm, Nn, Ny, Nx, Nz)
            # Extract at kx=1: Ck[:, :, :, :, idx_k0, idx_k1, idx_k0]
            # Result shape: electrons (nt, Np_e, Nm_e, Nn_e), ions (nt, Np_i, Nm_i, Nn_i)
            kx1_electrons = Ck_electrons[:, :, :, :, idx_k0, idx_k1, idx_k0]
            kx1_ions = Ck_ions[:, :, :, :, idx_k0, idx_k1, idx_k0]

            # Flatten Hermite modes with per-species dimensions
            nt = kx1_electrons.shape[0]
            electron_modes = kx1_electrons.reshape(nt, Np_e * Nm_e * Nn_e)
            ion_modes = kx1_ions.reshape(nt, Np_i * Nm_i * Nn_i)

            # Create per-species Hermite mode index arrays
            hermite_mode_indices_e = np.arange(Nn_e * Nm_e * Np_e)
            hermite_mode_indices_i = np.arange(Nn_i * Nm_i * Np_i)

            # Create xarray DataArrays for easier plotting
            electron_amp_da = xr.DataArray(
                np.abs(electron_modes),
                coords={"t": t_array, "m": hermite_mode_indices_e},
                dims=["t", "m"],
                name="electron_amplitude",
            )

            ion_amp_da = xr.DataArray(
                np.abs(ion_modes),
                coords={"t": t_array, "m": hermite_mode_indices_i},
                dims=["t", "m"],
                name="ion_amplitude",
            )

            # Create 2x2 figure with time vertical, hermite mode horizontal
            fig, axes = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
            fig.suptitle("Hermite Coefficients at EPW First Mode (k=1)", fontsize=14, fontweight="bold")

            # Electron - Linear scale (time vertical, m horizontal)
            electron_amp_da.plot(
                ax=axes[0, 0], y="t", x="m", cmap="viridis", cbar_kwargs={"label": r"$|C_{m}|$"}, add_colorbar=True
            )
            axes[0, 0].set_ylabel(r"Time ($\omega_{pe}^{-1}$)", fontsize=11)
            axes[0, 0].set_xlabel("Hermite Mode Index (m)", fontsize=11)
            axes[0, 0].set_title("Electrons - Linear Scale", fontsize=12)

            # Electron - Log scale (time vertical, m horizontal)
            log_electron = np.log10(electron_amp_da + 1e-20)
            log_electron.plot(
                ax=axes[0, 1],
                y="t",
                x="m",
                cmap="viridis",
                vmin=-10,
                vmax=0,
                cbar_kwargs={"label": r"$\log_{10}(|C_{m}|)$"},
                add_colorbar=True,
            )
            axes[0, 1].set_ylabel(r"Time ($\omega_{pe}^{-1}$)", fontsize=11)
            axes[0, 1].set_xlabel("Hermite Mode Index (m)", fontsize=11)
            axes[0, 1].set_title("Electrons - Log Scale", fontsize=12)

            # Ion - Linear scale (time vertical, m horizontal)
            ion_amp_da.plot(
                ax=axes[1, 0], y="t", x="m", cmap="viridis", cbar_kwargs={"label": r"$|C_{m}|$"}, add_colorbar=True
            )
            axes[1, 0].set_ylabel(r"Time ($\omega_{pe}^{-1}$)", fontsize=11)
            axes[1, 0].set_xlabel("Hermite Mode Index (m)", fontsize=11)
            axes[1, 0].set_title("Ions - Linear Scale", fontsize=12)

            # Ion - Log scale (time vertical, m horizontal)
            log_ion = np.log10(ion_amp_da + 1e-20)
            log_ion.plot(
                ax=axes[1, 1],
                y="t",
                x="m",
                cmap="viridis",
                vmin=-10,
                vmax=0,
                cbar_kwargs={"label": r"$\log_{10}(|C_{m}|)$"},
                add_colorbar=True,
            )
            axes[1, 1].set_ylabel(r"Time ($\omega_{pe}^{-1}$)", fontsize=11)
            axes[1, 1].set_xlabel("Hermite Mode Index (m)", fontsize=11)
            axes[1, 1].set_title("Ions - Log Scale", fontsize=12)

            plt.savefig(os.path.join(td, "plots", "epw_hermite_coefficients_2x2.png"), dpi=150, bbox_inches="tight")
            plt.close()

    def post_process(self, run_output: dict, td: str) -> dict:
        """
        EPW-specific post-processing with enhanced diagnostics.

        Extends base post-processing with:
        - EPW amplitude and frequency plots for the first mode
        - Enhanced 2x2 Hermite coefficient visualization

        Args:
            run_output: Dict containing {"solver result": <diffrax Solution>}
            td: Temporary directory path for saving artifacts

        Returns:
            dict: {"metrics": {...}, "datasets": {...}} with EPW-specific metrics
        """
        # Call base class post_process first
        result = super().post_process(run_output, td)

        # Extract solution
        sol = run_output["solver result"]

        # Get grid parameters
        Nx = int(self.cfg["grid"]["Nx"])
        Ny = int(self.cfg["grid"]["Ny"])
        Nz = int(self.cfg["grid"]["Nz"])
        Nn_e = int(self.cfg["grid"]["Nn_electrons"])
        Nm_e = int(self.cfg["grid"]["Nm_electrons"])
        Np_e = int(self.cfg["grid"]["Np_electrons"])
        Nn_i = int(self.cfg["grid"]["Nn_ions"])
        Nm_i = int(self.cfg["grid"]["Nm_ions"])
        Np_i = int(self.cfg["grid"]["Np_ions"])

        # Process EPW-specific diagnostics if field data is available
        if "fields" in sol.ys and not isinstance(sol.ys["fields"], dict):
            Fk_array = np.asarray(sol.ys["fields"])
            t_array = sol.ts["fields"]

            # Create EPW diagnostics plot
            self._plot_epw_diagnostics(Fk_array, t_array, Nx, Ny, Nz, td)

            # Add EPW-specific metrics
            idx_k1 = 1  # k=1 mode at index 1 in standard FFT ordering
            idx_k0 = 0  # k=0 for other dimensions (1D problem in x)

            if idx_k1 < Nx:
                Ex_k1 = Fk_array[:, 0, idx_k0, idx_k1, idx_k0]

                # Compute time-averaged amplitude and final frequency
                avg_amplitude = float(np.mean(np.abs(Ex_k1)))
                final_amplitude = float(np.abs(Ex_k1[-1]))

                frequencies, _ = self._compute_epw_frequency(Ex_k1, t_array)
                avg_frequency = float(np.mean(frequencies))
                final_frequency = float(frequencies[-1])

                # Compute average electrostatic energy over final 50 ωₚt
                # Electrostatic energy density ∝ |E|² (summed over all spatial modes)
                Ex_all = Fk_array[:, 0, :, :, :]  # Shape: (nt, Ny, Nx, Nz)
                electrostatic_energy = np.sum(np.abs(Ex_all) ** 2, axis=(1, 2, 3))  # Sum over spatial modes

                # Find time indices for final 50 ωₚt
                t_final_window = 50.0
                time_mask = t_array >= (t_array[-1] - t_final_window)
                avg_es_energy_final_50 = float(np.mean(electrostatic_energy[time_mask]))

                # Add to metrics
                result["metrics"]["epw_avg_amplitude_k1"] = avg_amplitude
                result["metrics"]["epw_final_amplitude_k1"] = final_amplitude
                result["metrics"]["epw_avg_frequency_k1"] = avg_frequency
                result["metrics"]["epw_final_frequency_k1"] = final_frequency
                result["metrics"]["avg_electrostatic_energy_final_50"] = avg_es_energy_final_50

        # Process enhanced Hermite coefficient visualization if distribution data is available
        if "hermite" in sol.ys or "distribution" in sol.ys:
            key = "hermite" if "hermite" in sol.ys else "distribution"
            species_dict = sol.ys[key]
            t_array = sol.ts[key]

            # Extract per-species arrays
            Ck_electrons = np.asarray(species_dict["electrons"])
            Ck_ions = np.asarray(species_dict["ions"])

            # Create enhanced Hermite coefficient plot
            self._plot_hermite_coefficients_enhanced(
                Ck_electrons, Ck_ions, t_array, Nn_e, Nm_e, Np_e, Nn_i, Nm_i, Np_i, Nx, Ny, Nz, td
            )

        return result
