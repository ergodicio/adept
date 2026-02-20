"""Plotting functions for spectrax1d module."""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_hermite_modes_at_k(Ck, t_array, Nn, Nm, Np, Nx, Ny, Nz, td: str) -> None:
    """Plot Hermite mode amplitudes at kx=1 Fourier mode."""
    # k=1 mode at index 1 in standard FFT ordering
    idx_k1 = 1
    idx_k0 = 0  # k=0 for other dimensions (1D problem in x)

    if idx_k1 < Nx:
        # Get kx=1 mode for all time steps and Hermite modes
        kx1_mode = Ck[:, :, idx_k0, idx_k1, idx_k0]

        # Separate electron and ion contributions
        electron_modes = kx1_mode[:, : Nn * Nm * Np]
        ion_modes = kx1_mode[:, Nn * Nm * Np :]

        # Create xarray DataArrays for plotting
        hermite_mode_indices = np.arange(Nn * Nm * Np)

        electron_amp_da = xr.DataArray(
            np.abs(electron_modes),
            coords={"t": t_array, "hermite_mode": hermite_mode_indices},
            dims=["t", "hermite_mode"],
            name="electron_amplitude",
        )

        ion_amp_da = xr.DataArray(
            np.abs(ion_modes),
            coords={"t": t_array, "hermite_mode": hermite_mode_indices},
            dims=["t", "hermite_mode"],
            name="ion_amplitude",
        )

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 4), tight_layout=True)
        fig.suptitle("Hermite Mode Amplitudes at kx=1", fontsize=14)

        # Electron - Linear scale (time vertical)
        electron_amp_da.plot(
            ax=axes[0, 0], y="t", x="hermite_mode", cmap="viridis", cbar_kwargs={"label": r"$|C_{n,m,p}|$"}
        )
        axes[0, 0].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
        axes[0, 0].set_xlabel("Hermite Mode Index")
        axes[0, 0].set_title("Electron Species (Linear Scale)")

        # Electron - Log scale (time vertical)
        (np.log10(electron_amp_da + 1e-20)).plot(
            ax=axes[0, 1],
            y="t",
            x="hermite_mode",
            cmap="viridis",
            vmin=-10,
            vmax=0,
            cbar_kwargs={"label": r"$\log_{10}(|C_{n,m,p}|)$"},
        )
        axes[0, 1].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
        axes[0, 1].set_xlabel("Hermite Mode Index")
        axes[0, 1].set_title("Electron Species (Log Scale)")

        # Ion - Linear scale (time vertical)
        ion_amp_da.plot(ax=axes[1, 0], y="t", x="hermite_mode", cmap="viridis", cbar_kwargs={"label": r"$|C_{n,m,p}|$"})
        axes[1, 0].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
        axes[1, 0].set_xlabel("Hermite Mode Index")
        axes[1, 0].set_title("Ion Species (Linear Scale)")

        # Ion - Log scale (time vertical)
        (np.log10(ion_amp_da + 1e-20)).plot(
            ax=axes[1, 1],
            y="t",
            x="hermite_mode",
            cmap="viridis",
            vmin=-10,
            vmax=0,
            cbar_kwargs={"label": r"$\log_{10}(|C_{n,m,p}|)$"},
        )
        axes[1, 1].set_ylabel(r"Time ($\omega_{pe}^{-1}$)")
        axes[1, 1].set_xlabel("Hermite Mode Index")
        axes[1, 1].set_title("Ion Species (Log Scale)")

        plt.tight_layout()
        plt.savefig(os.path.join(td, "plots", "hermite_mode_amplitudes_kx1.png"), bbox_inches="tight")
        plt.close()


def plot_field_mode_amplitudes(Fk, t_array, Nx, Ny, Nz, td: str) -> None:
    """Plot field mode amplitudes for k=1...5."""
    # k=0 for other dimensions (1D problem in x)
    idx_k0 = 0

    # Plot electric field modes
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
    fig.suptitle("Electric Field Mode Amplitudes (k=1...5)", fontsize=14)

    field_components = ["Ex", "Ey", "Ez"]
    for comp_idx, (ax, comp_name) in enumerate(zip(axes, field_components, strict=True)):
        for k_mode in range(1, 6):
            if k_mode < Nx:
                mode_amplitude = np.abs(Fk[:, comp_idx, idx_k0, k_mode, idx_k0])
                ax.plot(t_array, mode_amplitude, label=f"k={k_mode}", linewidth=2)

        ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
        ax.set_ylabel(f"|{comp_name}|")
        ax.set_yscale("log")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{comp_name} Mode Amplitudes")

    plt.tight_layout()
    plt.savefig(os.path.join(td, "plots", "field_mode_amplitudes.png"), bbox_inches="tight")
    plt.close()

    # Plot magnetic field modes
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
    fig.suptitle("Magnetic Field Mode Amplitudes (k=1...5)", fontsize=14)

    field_components = ["Bx", "By", "Bz"]
    for comp_idx, (ax, comp_name) in enumerate(zip(axes, field_components, strict=True)):
        for k_mode in range(1, 6):
            if k_mode < Nx:
                mode_amplitude = np.abs(Fk[:, 3 + comp_idx, idx_k0, k_mode, idx_k0])
                ax.plot(t_array, mode_amplitude, label=f"k={k_mode}", linewidth=2)

        ax.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
        ax.set_ylabel(f"|{comp_name}|")
        ax.set_yscale("log")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{comp_name} Mode Amplitudes")

    plt.tight_layout()
    plt.savefig(os.path.join(td, "plots", "magnetic_field_mode_amplitudes.png"), bbox_inches="tight")
    plt.close()


def plot_fields_spacetime(fields_xr: xr.Dataset, td: str) -> None:
    """
    Create spacetime plots for electromagnetic fields.

    Args:
        fields_xr: xarray Dataset containing field data with (t, x) coordinates
        td: Temporary directory path
    """
    plots_dir = os.path.join(td, "plots", "fields")

    for field_name, field_data in fields_xr.items():
        # Spacetime plot
        field_data.plot()
        plt.title(f"{field_name} Spacetime")
        plt.savefig(os.path.join(plots_dir, f"spacetime-{field_name}.png"), bbox_inches="tight")
        plt.close()


def plot_fields_lineouts(fields_xr: xr.Dataset, td: str, n_slices: int = 6) -> None:
    """
    Create facet grid plots showing field snapshots at multiple times.

    Args:
        fields_xr: xarray Dataset containing field data with (t, x) coordinates
        td: Temporary directory path
        n_slices: Number of time slices to show (default: 6)
    """
    plots_dir = os.path.join(td, "plots", "fields", "lineouts")

    for field_name, field_data in fields_xr.items():
        # Calculate time slice indices
        nt = field_data.coords["t"].size
        t_skip = max(1, nt // n_slices)
        tslice = slice(0, None, t_skip)

        # Create facet plot
        field_data[tslice].T.plot(col="t", col_wrap=3)
        plt.savefig(os.path.join(plots_dir, f"{field_name}.png"), bbox_inches="tight")
        plt.close()


def plot_moments_spacetime(moments_xr: xr.Dataset, td: str) -> None:
    """
    Create spacetime plots for distribution moments.

    Args:
        moments_xr: xarray Dataset containing moments with (t, x) coordinates
        td: Temporary directory path
    """
    plots_dir = os.path.join(td, "plots", "moments")
    os.makedirs(plots_dir, exist_ok=True)

    for name, data in moments_xr.items():
        if "x" not in data.dims:
            continue
        data.plot()
        plt.title(f"{name} Spacetime")
        plt.savefig(os.path.join(plots_dir, f"spacetime-{name}.png"), bbox_inches="tight")
        plt.close()


def plot_moments_lineouts(moments_xr: xr.Dataset, td: str, n_slices: int = 6) -> None:
    """
    Create facet grid plots of moments at multiple times.

    Args:
        moments_xr: xarray Dataset containing moments with (t, x) coordinates
        td: Temporary directory path
        n_slices: Number of time slices to show
    """
    plots_dir = os.path.join(td, "plots", "moments", "lineouts")
    os.makedirs(plots_dir, exist_ok=True)

    for name, data in moments_xr.items():
        if "x" not in data.dims:
            continue
        nt = data.coords["t"].size
        t_skip = max(1, nt // n_slices)
        tslice = slice(0, None, t_skip)

        data[tslice].T.plot(col="t", col_wrap=3)
        plt.savefig(os.path.join(plots_dir, f"{name}.png"), bbox_inches="tight")
        plt.close()


def plot_distribution_facets(dist_xr: xr.Dataset, td: str, n_timesteps: int = 6) -> None:
    """
    Create facet plots of distribution in (kx, hermite_mode) space for multiple timesteps.

    Args:
        dist_xr: xarray Dataset containing Ck with dimensions (t, hermite_mode, ky, kx, kz)
        td: Temporary directory path
        n_timesteps: Number of timesteps to show (default: 6)
    """
    plots_dir = os.path.join(td, "plots", "distributions")

    Ck = dist_xr["Ck"]

    # Get center indices for y and z (assume 1D in x)
    center_y = int((Ck.coords["ky"].size - 1) / 2)
    center_z = int((Ck.coords["kz"].size - 1) / 2)

    # Slice to get (t, hermite_mode, kx) at center ky, kz
    Ck_slice = Ck[:, :, center_y, :, center_z]

    # Select timesteps
    nt = Ck_slice.coords["t"].size
    t_indices = np.linspace(0, nt - 1, n_timesteps, dtype=int)
    Ck_selected = Ck_slice.isel(t=t_indices)

    # Plot amplitude (log scale)
    amplitude = np.log10(np.abs(Ck_selected) + 1e-20)
    amplitude.plot(
        x="kx", y="hermite_mode", col="t", col_wrap=3, cmap="viridis", cbar_kwargs={"label": r"$\log_{10}(|C_k|)$"}
    )
    plt.savefig(os.path.join(plots_dir, "Ck_amplitude_facets.png"), bbox_inches="tight")
    plt.close()


def plot_epw_diagnostics(Fk, t_array, Nx: int, Ny: int, Nz: int, td: str, driver_config: dict | None = None) -> None:
    """
    Plot EPW amplitude and frequency for the first mode over time.

    Creates a 2-panel figure:
    - Left: Amplitude |Ex(k=1)| vs time (log scale) - full time range
    - Right: Frequency omega(k=1) vs time - zoomed to driver-on period, excluding final 50 omega_pe^-1

    Args:
        Fk: Field array of shape (nt, 6, Ny, Nx, Nz)
        t_array: Time array
        Nx, Ny, Nz: Grid dimensions
        td: Temporary directory for plots
        driver_config: Driver configuration dict (optional)
    """
    # k=1 mode at index 1 in standard FFT ordering
    idx_k1 = 1
    idx_k0 = 0  # k=0 for other dimensions (1D problem in x)

    # Extract Ex at k=1 mode
    if idx_k1 < Nx:
        Ex_k1 = Fk[:, 0, idx_k0, idx_k1, idx_k0]

        # Compute amplitude
        amplitude = np.abs(Ex_k1)

        # Compute instantaneous frequency
        frequencies, freq_times = _compute_epw_frequency(Ex_k1, t_array)

        # Determine time window for frequency plot
        # Get driver config if available
        if driver_config is None:
            driver_config = {}
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

        # Omit final 50 omega_pe^-1
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


def plot_hermite_coefficients_enhanced(
    Ck_electrons,
    Ck_ions,
    t_array,
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

        plt.savefig(os.path.join(td, "plots", "epw_hermite_coefficients_2x2.png"), bbox_inches="tight")
        plt.close()


def plot_em_diagnostics(
    Fk,
    t_array,
    Nx: int,
    Ny: int,
    Nz: int,
    td: str,
) -> None:
    """
    Plot Ey amplitude (log scale) and instantaneous frequency for the k=1 mode.

    Creates a 2-panel figure saved as em_wave_k1_diagnostics.png.

    Args:
        Fk: Field array of shape (nt, 6, Ny, Nx, Nz)
        t_array: Time array
        Nx, Ny, Nz: Grid dimensions
        td: Temporary directory for plots
    """
    idx_k1 = 1
    idx_k0 = 0
    if idx_k1 >= Nx:
        return

    Ey_k1 = Fk[:, 1, idx_k0, idx_k1, idx_k0]
    amplitude = np.abs(Ey_k1)
    frequencies, _ = _compute_em_frequency(Ey_k1, t_array)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    fig.suptitle("EM Wave k=1 Diagnostics (Ey component)", fontsize=14)

    # Left: amplitude on log scale
    safe_amplitude = np.maximum(amplitude, 1e-30)
    axes[0].semilogy(t_array, safe_amplitude, "b-", linewidth=2)
    axes[0].set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
    axes[0].set_ylabel(r"$|E_y(k=1)|$")
    axes[0].set_title("EM Wave Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Right: instantaneous frequency
    axes[1].plot(t_array, frequencies, "r-", linewidth=2)
    axes[1].axhline(y=np.sqrt(2.0), color="k", linestyle="--", alpha=0.5, label=r"$\sqrt{2}$ (ξ=1, ωpe=1)")
    axes[1].set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
    axes[1].set_ylabel(r"$\omega$ ($\omega_{pe}$)")
    axes[1].set_title("EM Wave Instantaneous Frequency")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.savefig(os.path.join(td, "plots", "em_wave_k1_diagnostics.png"), bbox_inches="tight")
    plt.close()


def _compute_em_frequency(field_k1, t_array):
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


def _compute_epw_frequency(Ex_k1, t_array):
    """
    Compute instantaneous frequency of the first EPW mode from phase evolution.

    Args:
        Ex_k1: Complex amplitude of Ex at k=1 mode as function of time (nt,)
        t_array: Time array (nt,)

    Returns:
        tuple: (frequencies, times_for_frequency)
            frequencies: Instantaneous frequency omega at each time (positive)
            times_for_frequency: Time points for frequency (centered difference)
    """
    # Extract phase
    phase = np.unwrap(np.angle(Ex_k1))

    # Compute instantaneous frequency using centered differences
    # For wave e^(i(kx - omega*t)), phase phi = kx - omega*t, so dphi/dt = -omega
    # Negate to get positive frequency omega
    dt = t_array[1] - t_array[0]
    frequencies = -np.gradient(phase, dt)

    return frequencies, t_array
