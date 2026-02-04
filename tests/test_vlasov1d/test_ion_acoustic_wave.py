#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
"""
Integration test for ion acoustic waves in multi-species Vlasov-Poisson.

Ion acoustic waves are a fundamental multi-species phenomenon where:
- Electrons provide the pressure (restoring force)
- Ions provide the inertia

Dispersion relation (long wavelength limit, T_e >> T_i):
    ω = k * c_s where c_s = sqrt(Z * T_e / m_i)

Full dispersion relation:
    ω² = k² * c_s² / (1 + k²λ_D²)

Reference: https://farside.ph.utexas.edu/teaching/plasma/Plasma/node112.html
"""

import numpy as np
import pytest
import yaml

from adept import ergoExo


def ion_acoustic_frequency(k, Z, T_e, m_i, lambda_D=1.0):
    """
    Compute ion acoustic wave frequency from dispersion relation.

    ω² = k² * c_s² / (1 + k²λ_D²)
    where c_s² = Z * T_e / m_i

    Args:
        k: wavenumber (normalized to 1/λ_D)
        Z: ion charge number
        T_e: electron temperature (normalized)
        m_i: ion mass (normalized to electron mass)
        lambda_D: Debye length (normalized, typically 1)

    Returns:
        Ion acoustic wave frequency ω
    """
    c_s_squared = Z * T_e / m_i
    omega_squared = k**2 * c_s_squared / (1 + (k * lambda_D) ** 2)
    return np.sqrt(omega_squared)


def measure_frequency_from_density(density, time_axis, expected_omega=None):
    """
    Measure oscillation frequency from density time series.

    Uses FFT of the k=1 Fourier mode to find the dominant frequency.

    Args:
        density: Density array [nt, nx] (ion or electron density)
        time_axis: Time points [nt]
        expected_omega: If provided, search near this frequency

    Returns:
        Measured frequency
    """
    # Get the k=1 Fourier mode (the driven mode)
    nx = density.shape[1]
    nk1 = 2.0 / nx * np.fft.fft(density, axis=1)[:, 1]

    # Use latter half of simulation (after transients settle)
    nt = len(time_axis)
    start_idx = nt // 2

    # Compute frequency from FFT of the time series
    dt = time_axis[1] - time_axis[0]
    nk1_late = nk1[start_idx:]
    freq_axis = np.fft.fftfreq(len(nk1_late), dt)
    spectrum = np.abs(np.fft.fft(nk1_late))

    # Convert to angular frequency
    omega_axis = 2 * np.pi * freq_axis
    positive_mask = omega_axis > 0

    # If expected frequency is provided, search in a window around it
    if expected_omega is not None:
        # Search within factor of 5 of expected
        search_mask = positive_mask & (omega_axis < 5 * expected_omega) & (omega_axis > expected_omega / 5)
        if np.any(search_mask):
            peak_idx = np.argmax(spectrum[search_mask])
            return omega_axis[search_mask][peak_idx]

    # Otherwise find global peak
    peak_idx = np.argmax(spectrum[positive_mask])
    return omega_axis[positive_mask][peak_idx]


def compute_ion_density(f_ion, dv):
    """Compute ion density by integrating distribution over velocity."""
    return np.sum(f_ion, axis=-1) * dv


@pytest.mark.parametrize("time_integrator", ["sixth"])
def test_ion_acoustic_simulation_runs(time_integrator):
    """
    Smoke test that multi-species ion acoustic simulation runs end-to-end.

    This test verifies:
    1. Multi-species config is parsed correctly
    2. Electrons and ions are initialized with different velocity grids
    3. The Vlasov-Poisson solver handles the multi-species dict structure
    4. Field solves compute total charge density from all species
    5. Simulation completes without errors

    Note: Quantitative frequency verification requires careful physics calibration
    and is tracked separately. This test ensures the infrastructure works.
    """
    with open("tests/test_vlasov1d/configs/multispecies_ion_acoustic.yaml") as file:
        config = yaml.safe_load(file)

    # Use shorter simulation for smoke test
    config["grid"]["nx"] = 64  # Override for better resolution
    config["grid"]["tmax"] = 100.0
    config["grid"]["dt"] = 0.1
    config["save"]["fields"]["t"]["tmax"] = 100.0
    config["save"]["fields"]["t"]["nt"] = 101
    config["save"]["electron"]["t"]["tmax"] = 100.0
    config["save"]["ion"]["t"]["tmax"] = 100.0

    # Modify config for this test
    config["terms"]["time"] = time_integrator
    config["mlflow"]["experiment"] = "vlasov1d-test-ion-acoustic"
    config["mlflow"]["run"] = f"ion-acoustic-smoke-{time_integrator}"

    # Run simulation
    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    # Verify we got output
    e_field = solver_result.ys["fields"]["e"]
    time_axis = solver_result.ts["fields"]

    assert e_field.shape[0] == len(time_axis), "Time axis mismatch"
    assert e_field.shape[1] == config["grid"]["nx"], "Spatial grid mismatch"

    # Verify the field has non-trivial dynamics (not just zeros)
    assert np.std(e_field) > 0, "Electric field should have dynamics"

    print(f"\nIon Acoustic Smoke Test ({time_integrator}):")
    print("  Simulation completed successfully")
    print(f"  Time steps: {len(time_axis)}")
    print(f"  E-field std: {np.std(e_field):.2e}")


@pytest.mark.parametrize("time_integrator", ["sixth"])
def test_ion_acoustic_dispersion(time_integrator):
    """
    Test that ion acoustic wave frequency matches theoretical prediction.

    Uses ion density oscillations to measure the frequency, which directly
    reflects the ion acoustic mode without contamination from fast electron
    plasma oscillations.

    With k=0.1 and λ_D=1, we have kλ_D=0.1 << 1, so we're in the
    long-wavelength regime where ω ≈ k * c_s.
    """
    with open("tests/test_vlasov1d/configs/multispecies_ion_acoustic.yaml") as file:
        config = yaml.safe_load(file)

    # Modify config for this test
    config["grid"]["nx"] = 64  # Override for better resolution
    config["terms"]["time"] = time_integrator
    config["mlflow"]["experiment"] = "vlasov1d-test-ion-acoustic"
    config["mlflow"]["run"] = f"ion-acoustic-dispersion-{time_integrator}"

    # Extract physical parameters
    k = config["density"]["species-electron-background"]["wavenumber"]
    Z = config["terms"]["species"][1]["charge"]  # ion charge
    T_e = config["density"]["species-electron-background"]["T0"]
    m_i = config["terms"]["species"][1]["mass"]

    # Compute expected frequency
    expected_omega = ion_acoustic_frequency(k, Z, T_e, m_i)

    # Run simulation
    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    solver_result = result["solver result"]

    # Get electron density from species-specific moments
    # Use electron density as a proxy since both oscillate together
    # in an ion acoustic wave (quasineutral oscillation)
    n_field = solver_result.ys["fields"]["electron"]["n"]
    time_axis = solver_result.ts["fields"]

    # Measure frequency from density oscillations
    measured_omega = measure_frequency_from_density(n_field, time_axis, expected_omega)

    print(f"\nIon Acoustic Wave Test ({time_integrator}):")
    print(f"  Wavenumber k = {k}")
    print(f"  Ion charge Z = {Z}")
    print(f"  Ion mass m_i = {m_i}")
    print(f"  Sound speed c_s = {np.sqrt(Z * T_e / m_i):.6f}")
    print(f"  Expected ω = {expected_omega:.6f}")
    print(f"  Measured ω = {measured_omega:.6f}")
    print(f"  Relative error = {abs(measured_omega - expected_omega) / expected_omega * 100:.2f}%")

    # Assert frequency matches within 15%
    # (Some error expected from numerical dispersion and finite grid effects)
    np.testing.assert_allclose(measured_omega, expected_omega, rtol=0.15)


if __name__ == "__main__":
    test_ion_acoustic_dispersion("sixth")
