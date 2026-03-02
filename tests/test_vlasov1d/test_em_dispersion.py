# Copyright (c) Ergodic LLC 2023
# research@ergodic.io
"""Test EM wave dispersion relation for Vlasov-1D.

Validates that ey-driven light waves propagate with the correct dispersion relation
through a plasma density gradient: ω² = ωpe²(x) + c²k²(x)
"""

import math

import numpy as np
import yaml
from adept._vlasov1d.normalization import UREG
from scipy import signal

from adept import ergoExo
from adept.normalization import UREG


def compute_critical_wavelength(n0: UREG.Quantity) -> UREG.Quantity:
    """Compute the critical wavelength for EM waves at density n0.

    At this wavelength, ω = ωpe, so the wave is at cutoff.

    λ_crit = 2πc / ωpe where ωpe = sqrt(n0 * e² / (ε₀ * m_e))
    """
    e = UREG.e
    m_e = UREG.m_e
    eps0 = UREG.epsilon_0
    c = UREG.c

    omega_pe = ((n0 * e**2 / (eps0 * m_e)) ** 0.5).to("rad/s")
    lambda_crit = (2 * math.pi * c / omega_pe).to("um")

    return lambda_crit


def build_config() -> dict:
    """Load YAML config and replace placeholders with computed values."""
    # 1. Choose normalizing density (this sets the plasma frequency scale)
    n0 = UREG.Quantity("1e21/cc")

    # 2. Compute critical wavelength for this density
    lambda_crit = compute_critical_wavelength(n0)

    # 3. Choose driver wavelength as fraction of critical
    #    λ_driver < λ_crit means ω_driver > ωpe (wave propagates)
    lambda_driver = 0.8 * lambda_crit

    # 4. Load YAML config and replace placeholders
    with open("tests/test_vlasov1d/configs/em_dispersion.yaml") as f:
        config = yaml.safe_load(f)

    # Replace placeholders
    config["units"]["laser_wavelength"] = f"{lambda_crit.magnitude:.4f}um"
    config["units"]["normalizing_density"] = f"{n0.to('1/cc').magnitude:.2e}/cc"
    config["drivers"]["ey"]["0"]["params"]["wavelength"] = f"{lambda_driver.magnitude:.4f}um"

    return config


def get_local_wavenumber(field_spatial: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract local wavenumber k(x) from spatial field profile using Hilbert transform.

    Similar to get_nlfs but operates on spatial data.

    Args:
        field_spatial: 1D array of field values E_y(x) at fixed time
        dx: Spatial grid spacing

    Returns:
        amplitude: Envelope amplitude at each position
        wavenumber: Instantaneous wavenumber k(x) from phase gradient
    """
    # Hilbert transform to get analytic signal
    analytic_signal = signal.hilbert(np.real(field_spatial))

    # Envelope (amplitude)
    amplitude = np.abs(analytic_signal)

    # Unwrapped phase
    phase = np.unwrap(np.angle(analytic_signal))

    # Wavenumber = d(phase)/dx
    wavenumber = np.gradient(phase, dx)

    # Smooth the wavenumber (similar to get_nlfs smoothing)
    midpt = len(wavenumber) // 2
    if midpt > 8:
        b, a = signal.butter(8, 0.125)
        wavenumber_smooth = signal.filtfilt(b, a, wavenumber, padlen=min(midpt, len(wavenumber) - 1))
    else:
        wavenumber_smooth = wavenumber

    return amplitude, wavenumber_smooth


def get_local_frequency(field_time_series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract instantaneous frequency ω(t) from time series using Hilbert transform.

    Args:
        field_time_series: 1D array of field values at a fixed position over time
        dt: Time step

    Returns:
        amplitude: Envelope amplitude at each time
        frequency: Instantaneous frequency ω(t) from phase gradient
    """
    # Hilbert transform to get analytic signal
    analytic_signal = signal.hilbert(np.real(field_time_series))

    # Envelope (amplitude)
    amplitude = np.abs(analytic_signal)

    # Unwrapped phase
    phase = np.unwrap(np.angle(analytic_signal))

    # Frequency = d(phase)/dt
    frequency = np.gradient(phase, dt)

    # Smooth the frequency
    midpt = len(frequency) // 2
    if midpt > 8:
        b, a = signal.butter(8, 0.125)
        frequency_smooth = signal.filtfilt(b, a, frequency, padlen=min(midpt, len(frequency) - 1))
    else:
        frequency_smooth = frequency

    return amplitude, frequency_smooth


def test_em_wave_dispersion():
    """Verify EM wave propagation follows the dispersion relation."""
    # Build config with computed wavelengths
    config = build_config()

    # Run simulation
    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    # Extract field data
    print(f"\nFields available: {list(solver_result.ys['fields'].keys())}")
    # The EM field is stored as 'a' (vector potential) in the solver
    a_field = solver_result.ys["fields"]["a"]  # Shape: (nt, nx+2) - includes ghost cells
    t_axis = np.array(solver_result.ts["fields"])

    # Get simulation parameters
    sim = exo.adept_module.simulation
    grid = sim.grid
    x_axis = np.array(grid.x)  # Interior grid points (nx,)
    dx = grid.dx
    dt = t_axis[1] - t_axis[0]
    c_light = sim.plasma_norm.speed_of_light_norm()  # Normalized c = c/v0

    # Strip ghost cells from field (first and last points)
    a_field_interior = np.array(a_field[:, 1:-1])  # Shape: (nt, nx)

    # Check driver parameters
    if sim.drivers.ey:
        driver = sim.drivers.ey[0]
        print("\nDriver info:")
        print(f"  a0 = {driver.a0:.6e}")
        print(f"  k0 = {driver.k0:.4f}")
        print(f"  w0 = {driver.w0:.4f}")

    # Compute density profile n(x) / n0
    density_profile = np.zeros_like(x_axis)
    for spec in sim.species:
        subspecies_spec = sim.species_distributions[spec.name]
        for subspec in subspecies_spec:
            density_profile += np.array(subspec.density_profile(x_axis))

    # Compute local plasma frequency: ωpe(x) = sqrt(n(x)/n0) * ωpe0
    # In normalized units, ωpe0 = 1
    omega_pe_x = np.sqrt(np.maximum(density_profile, 0.0))

    # Measure local frequency ω(x) from time series at each position
    # Use the middle portion of the time series to avoid transients
    t_start_idx = int(0.3 * len(t_axis))
    t_end_idx = int(0.9 * len(t_axis))

    omega_measured = np.zeros(len(x_axis))
    amplitude_t = np.zeros(len(x_axis))

    for i in range(len(x_axis)):
        time_series = a_field_interior[t_start_idx:t_end_idx, i]
        if np.std(time_series) > 1e-10:  # Only process if there's signal
            amp, freq = get_local_frequency(time_series, dt)
            # Use the mean frequency in the middle of the time window
            mid_start = len(freq) // 4
            mid_end = 3 * len(freq) // 4
            omega_measured[i] = np.mean(freq[mid_start:mid_end])
            amplitude_t[i] = np.mean(amp[mid_start:mid_end])
        else:
            omega_measured[i] = np.nan
            amplitude_t[i] = 0.0

    # Measure local wavenumber k(x) from spatial profile at a fixed time
    # Use only the right portion of the domain where the plasma is
    t_idx = int(0.9 * len(t_axis))
    plasma_start_idx = len(x_axis) // 2  # Right half of domain
    field_slice = a_field_interior[t_idx, plasma_start_idx:]
    amplitude_x, k_measured_partial = get_local_wavenumber(field_slice, dx)
    # Pad k_measured to full domain size (left side with NaN)
    k_measured = np.full(len(x_axis), np.nan)
    k_measured[plasma_start_idx:] = k_measured_partial

    # Find valid region: where we have significant amplitude, valid measurements, and k is measured
    amp_threshold = 0.1 * np.max(amplitude_t)
    valid_region = (
        (amplitude_t > amp_threshold) & np.isfinite(omega_measured) & (omega_measured > 0) & np.isfinite(k_measured)
    )

    # Trim edges to avoid Hilbert transform boundary artifacts
    # The Hilbert transform has edge effects that cause spurious k values
    valid_indices = np.where(valid_region)[0]
    if len(valid_indices) > 0:
        edge_trim = max(20, len(valid_indices) // 20)  # Trim at least 20 points or 5%
        trim_start = valid_indices[0] + edge_trim
        trim_end = valid_indices[-1] - edge_trim
        valid_region = valid_region & (np.arange(len(x_axis)) >= trim_start) & (np.arange(len(x_axis)) <= trim_end)

    if not np.any(valid_region):
        raise AssertionError("No valid region found for dispersion analysis")

    # Extract values in valid region
    omega_valid = omega_measured[valid_region]
    k_valid = np.abs(k_measured[valid_region])
    omega_pe_valid = omega_pe_x[valid_region]

    # Verify dispersion relation: ω² = ωpe² + c²k²
    # Compute residual using measured ω and k
    omega_sq = omega_valid**2
    omega_pe_sq = omega_pe_valid**2
    c_sq = c_light**2

    # Dispersion relation residual: should be close to 0
    # residual = ω² - ωpe² - c²k²
    residual = omega_sq - omega_pe_sq - c_sq * k_valid**2

    # Relative error normalized by ω²
    relative_error = np.abs(residual) / omega_sq

    # Print diagnostics
    print("\nEM Dispersion Relation Test")
    print(f"  Speed of light c = {c_light:.4f}")
    print(f"  Valid region: {np.sum(valid_region)} points")
    print(f"  Mean ω_measured = {np.mean(omega_valid):.4f}")
    print(f"  Mean |k_measured| = {np.mean(k_valid):.4f}")
    print(f"  Mean ωpe = {np.mean(omega_pe_valid):.4f}")
    print(f"  Mean relative error = {np.mean(relative_error):.4f}")
    print(f"  Max relative error = {np.max(relative_error):.4f}")

    # Assert dispersion relation is satisfied within tolerance
    mean_rel_error = np.mean(relative_error)
    max_rel_error = np.max(relative_error)
    assert mean_rel_error < 0.02, f"Dispersion relation not satisfied: mean relative error {mean_rel_error:.4f} > 0.02"
    assert max_rel_error < 0.15, f"Dispersion relation not satisfied: max relative error {max_rel_error:.4f} > 0.15"


if __name__ == "__main__":
    test_em_wave_dispersion()
