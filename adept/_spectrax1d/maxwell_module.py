"""Specialized ADEPTModule for 1D electromagnetic wave dispersion and absorption analysis.

This module extends BaseSpectrax1D with EM-specific post-processing:
  - Measures the frequency of the k=1 transverse EM mode (Ey component) from
    phase evolution, returning em_avg_frequency_k1 as a metric.
  - Computes an absorption ratio (final EM energy / peak EM energy) to verify
    sponge boundary layer effectiveness.

Physics background
------------------
In the normalized units used by this code (time in ωpe^{-1}, length in c/ωpe,
speed of light c = 1, ωpe = 1), the cold-plasma transverse EM dispersion relation is

    ω² = ξ² + ωpe²  =  ξ² + 1

where ξ = kx / Lx is the normalized wavenumber returned by MaxwellExponential.
For the k=1 Fourier mode with domain length Lx:

    ξ = 2π / Lx    →    ω_EM = sqrt((2π/Lx)² + 1)

This follows from the Hermite-Fourier Vlasov-Maxwell equations:
  - Maxwell curl terms handled exactly by MaxwellExponential (vacuum: ω_vac = |ξ|)
  - Cold plasma current: Jy = i q² Omega_cs[0] / ω * Ey  (high-ω limit)
  - Ampère law: dEy/dt = -i ξ Bz - Jy/Omega_cs[0]
  Combined → ω² = ξ² + q² = ξ² + 1  (for q = -1, Omega_cs[0] = 1)
"""

import os

import numpy as np

from adept._spectrax1d.base_module import BaseSpectrax1D
from adept._spectrax1d.plotting import plot_em_diagnostics


class Maxwell1D(BaseSpectrax1D):
    """ADEPTModule for 1D EM wave dispersion and sponge-absorption tests."""

    def _compute_em_frequency(self, field_k1: np.ndarray, t_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute instantaneous frequency from phase evolution of a complex mode amplitude.

        Uses phase unwrapping and centred finite differences.  For a wave
        Ey_k1(t) ~ A(t) exp(-iω t), the phase advances at rate -ω, so
        ω = -dφ/dt with φ = unwrap(angle(Ey_k1)).

        Args:
            field_k1: Complex time series (nt,) for the k=1 mode.
            t_array: Corresponding time array (nt,).

        Returns:
            (frequencies, t_array): instantaneous-frequency array and same t_array.
        """
        phase = np.unwrap(np.angle(field_k1))
        dt = t_array[1] - t_array[0]
        return -np.gradient(phase, dt), t_array

    def post_process(self, run_output: dict, td: str) -> dict:
        """
        Maxwell-specific post-processing: EM frequency and absorption metrics.

        Extends BaseSpectrax1D.post_process() with two additional metrics:

        em_avg_frequency_k1
            Average instantaneous frequency of Ey at the k=1 Fourier mode,
            measured over the latter 75% of the saved-fields time series.
            Expected value: sqrt((2π/Lx)² + 1) for the cold-plasma dispersion.

        em_absorption_ratio
            Ratio of mean EM energy in the final 20% of the simulation to the
            peak EM energy.  Values << 1 indicate the wave has been absorbed by
            the sponge boundary layer.

        em_peak_energy, em_final_energy
            Scalar values used to compute em_absorption_ratio.

        Args:
            run_output: Dict from ergoExo.__call__() containing "solver result".
            td: Temporary directory path for saving plots and binaries.

        Returns:
            dict with "metrics" and "datasets" keys (from base class, extended here).
        """
        # Run base-class post-processing first (plots, netCDF, scalar metrics)
        result = super().post_process(run_output, td)

        sol = run_output["solver result"]

        # Plot output directory (already created by base class)
        plots_dir = os.path.join(td, "plots")

        Nx = int(self.cfg["grid"]["Nx"])
        Ny = int(self.cfg["grid"]["Ny"])
        Nz = int(self.cfg["grid"]["Nz"])

        # ------------------------------------------------------------------ #
        # EM frequency from the saved fields time series                      #
        # ------------------------------------------------------------------ #
        if "fields" in sol.ys and not isinstance(sol.ys["fields"], dict):
            Fk_array = np.asarray(sol.ys["fields"])  # (nt, 6, Ny, Nx, Nz)
            t_array = np.asarray(sol.ts["fields"])

            plot_em_diagnostics(Fk_array, t_array, Nx, Ny, Nz, plots_dir)

            idx_k1 = 1
            idx_k0 = 0

            if idx_k1 < Nx:
                Ey_k1 = Fk_array[:, 1, idx_k0, idx_k1, idx_k0]  # complex (nt,)

                # Use only the LAST 20% of the time series.
                # The driver is typically off after the first ~5% of tmax; using the
                # very end ensures we measure the natural EM frequency long after any
                # transient or driving effects have dispersed.
                n_start = max(1, len(t_array) * 8 // 10)  # start at 80% mark
                Ey_late = Ey_k1[n_start:]
                t_late = t_array[n_start:]

                if np.max(np.abs(Ey_late)) > 1e-30:
                    frequencies, _ = self._compute_em_frequency(Ey_late, t_late)
                    em_avg_freq = float(np.mean(np.abs(frequencies)))
                else:
                    # Wave was absorbed (absorption test) – frequency is meaningless
                    em_avg_freq = float("nan")

                result["metrics"]["em_avg_frequency_k1"] = em_avg_freq

        # ------------------------------------------------------------------ #
        # Absorption ratio from scalar diagnostics                            #
        # ------------------------------------------------------------------ #
        if "default" in sol.ys:
            em_energy = np.asarray(sol.ys["default"]["total_EM_energy"])  # (nt_default,)

            peak_em_energy = float(np.max(em_energy))
            n_final = max(1, len(em_energy) // 5)
            final_em_energy = float(np.mean(em_energy[-n_final:]))

            if peak_em_energy > 0:
                absorption_ratio = final_em_energy / peak_em_energy
            else:
                absorption_ratio = 1.0  # No wave was created → trivially "not absorbed"

            result["metrics"]["em_peak_energy"] = peak_em_energy
            result["metrics"]["em_final_energy"] = final_em_energy
            result["metrics"]["em_absorption_ratio"] = absorption_ratio

        return result
