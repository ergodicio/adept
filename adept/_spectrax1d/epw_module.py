"""Specialized ADEPTModule for 1D Electron Plasma Wave (EPW) analysis."""

import os

import numpy as np
from jax import Array

from adept._spectrax1d.base_module import BaseSpectrax1D
from adept._spectrax1d.plotting import plot_epw_diagnostics, plot_hermite_coefficients_enhanced


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
                frequencies: Instantaneous frequency ω at each time (positive)
                times_for_frequency: Time points for frequency (centered difference)
        """
        # Extract phase
        phase = np.unwrap(np.angle(Ex_k1))

        # Compute instantaneous frequency using centered differences
        # For wave e^(i(kx - ωt)), phase φ = kx - ωt, so dφ/dt = -ω
        # Negate to get positive frequency ω
        dt = t_array[1] - t_array[0]
        frequencies = -np.gradient(phase, dt)

        return frequencies, t_array

    def _compute_damping_rate(self, Ex_k1: Array, t_array: Array, fraction: float = 2 / 3) -> float:
        """
        Compute damping rate from exponential decay of EPW amplitude.

        Fits exponential decay A(t) = A0 * exp(gamma * t) to the latter portion
        of the time series to measure the damping rate gamma.

        Args:
            Ex_k1: Complex amplitude of Ex at k=1 mode as function of time (nt,)
            t_array: Time array (nt,)
            fraction: Fraction of time series to use for fitting (default: 2/3)
                     Uses the last 'fraction' of data to avoid initial transients

        Returns:
            float: Measured damping rate gamma (negative for damping)
        """
        # Get amplitude
        amplitude = np.abs(Ex_k1)

        # Use latter portion of simulation to avoid initial transients
        n_points = len(t_array)
        start_idx = int(n_points * (1 - fraction))
        t_fit = t_array[start_idx:]
        A_fit = amplitude[start_idx:]

        # Filter out very small values to avoid log issues
        valid_mask = A_fit > 1e-20
        t_fit = t_fit[valid_mask]
        A_fit = A_fit[valid_mask]

        if len(t_fit) < 10:
            return np.nan

        # Fit exponential: A(t) = A0 * exp(gamma * t)
        # Taking log: ln(A) = ln(A0) + gamma * t
        log_A = np.log(A_fit)

        # Linear fit: log(A) = log(A0) + gamma * t
        coeffs = np.polyfit(t_fit, log_A, 1)
        damping_rate = float(coeffs[0])

        return damping_rate

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

        # Plot output directory (already created by base class)
        plots_dir = os.path.join(td, "plots")

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
            plot_epw_diagnostics(Fk_array, t_array, Nx, Ny, Nz, plots_dir, driver_config=self.cfg.get("drivers", {}))

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

                # Compute damping rate from exponential decay
                damping_rate = self._compute_damping_rate(Ex_k1, t_array)

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
                result["metrics"]["epw_damping_rate_k1"] = damping_rate
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
            plot_hermite_coefficients_enhanced(
                Ck_electrons, Ck_ions, t_array, Nn_e, Nm_e, Np_e, Nn_i, Nm_i, Np_i, Nx, Ny, Nz, plots_dir
            )

        return result
