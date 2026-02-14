#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
"""
Test Landau damping for Spectrax-1D using Hermite-Fourier decomposition.

This test verifies that the spectrax-1d ADEPTModule correctly reproduces
driven EPW dynamics for klambda_D = 0.266.
"""

from jax import config

config.update("jax_enable_x64", True)
import numpy as np
import pytest
import yaml

from adept import electrostatic, ergoExo


@pytest.mark.skip(reason="Testing driven case only")
def test_landau_damping_initial_value(klambda_D: float | None = None):
    """
    Test Landau damping for arbitrary klambda_D using spectrax-1d.

    Args:
        klambda_D: Normalized wavenumber k*lambda_D. If None, randomly chosen from [0.26, 0.4].

    This test:
    1. Loads a simplified test config
    2. Configures thermal velocity for desired klambda_D
    3. Runs the simulation using the spectrax-1d ADEPTModule
    4. Extracts the electric field amplitude from the default save
    5. Measures the damping rate from the exponential decay
    6. Compares to the theoretical damping rate from the electrostatic dispersion relation
    """
    # Random selection if not specified
    if klambda_D is None:
        klambda_D = np.random.uniform(0.26, 0.4)
        print(f"\n{'=' * 60}")
        print(f"Randomly selected klambda_D = {klambda_D:.4f}")
        print(f"{'=' * 60}")

    # Load simplified test config
    with open("tests/test_spectrax1d/configs/landau-damping-test.yaml") as file:
        config = yaml.safe_load(file)

    # Use EPW1D class for EPW-specific analysis and postprocessing
    config["solver"] = "hermite-epw-1d"

    # Calculate required thermal velocity for desired klambdaD
    Lx = config["physics"]["Lx"]
    k_fundamental = 2.0 * np.pi / Lx
    required_alpha_e = klambda_D * np.sqrt(2) / k_fundamental

    # Update config with calculated thermal velocity
    config["physics"]["alpha_e"] = [required_alpha_e, required_alpha_e, required_alpha_e]
    config["physics"]["alpha_s"][:3] = [required_alpha_e, required_alpha_e, required_alpha_e]

    # Keep ion thermal velocity ratio consistent (Ti/Te ≈ 0.001 for cold ions)
    Ti_Te_ratio = 0.001
    required_alpha_i = required_alpha_e * np.sqrt(Ti_Te_ratio)
    config["physics"]["alpha_s"][3:6] = [required_alpha_i, required_alpha_i, required_alpha_i]

    # Verify configuration
    vth_e = required_alpha_e
    k0 = k_fundamental
    calculated_klambdaD = k0 * vth_e / np.sqrt(2)

    print("\nTest configuration:")
    print(f"  Target klambda_D = {klambda_D:.4f}")
    print(f"  alpha_e[0] (vth_e) = {vth_e:.6f}")
    print(f"  k_fundamental = {k0:.6f}")
    print(f"  Calculated klambda_D = {calculated_klambdaD:.4f}")

    # Verify calculation is correct
    np.testing.assert_allclose(
        calculated_klambdaD, klambda_D, rtol=1e-10, err_msg="Config update failed: klambdaD mismatch"
    )

    # Get theoretical complex frequency
    # Pass klambda_D directly since the dispersion function expects k*lambda_D
    theoretical_root = electrostatic.get_roots_to_electrostatic_dispersion(
        wp_e=1.0,
        vth_e=1.0,  # Normalized to 1 since klambda_D already includes vth scaling
        k0=klambda_D,
        maxwellian_convention_factor=2.0,
    )

    expected_damping_rate = np.imag(theoretical_root)
    expected_frequency = np.real(theoretical_root)

    print("\nTheoretical values:")
    print(f"  Frequency: {expected_frequency:.6f}")
    print(f"  Damping rate: {expected_damping_rate:.6f}")

    # Run simulation
    print("\nRunning simulation...")
    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)

    # Extract results
    sol = result["solver result"]

    # Get scalar diagnostics from the default save
    # This includes Ex_max and Ex_k1 which we can use to measure damping
    assert "default" in sol.ys, "No 'default' save data found in simulation results"

    scalar_data = sol.ys["default"]
    t_array = sol.ts["default"]

    print(f"\nAvailable scalar diagnostics: {list(scalar_data.keys())}")

    # Use Ex_k1 (electron density perturbation at k=1) for damping measurement
    # This is directly proportional to the electric field amplitude
    if "Ex_k1" in scalar_data:
        field_amplitude = np.abs(scalar_data["Ex_k1"])
        print("Using Ex_k1 for damping rate measurement")
    elif "Ex_max" in scalar_data:
        field_amplitude = np.abs(scalar_data["Ex_max"])
        print("Using Ex_max for damping rate measurement")
    else:
        pytest.fail("No suitable field amplitude data found (expected Ex_k1 or Ex_max)")

    # Measure damping rate from the exponential decay
    # Use the latter 2/3 of the simulation to avoid initial transients
    n_points = len(t_array)
    start_idx = n_points // 3
    fit_slice = slice(start_idx, n_points)

    # Fit exponential: A(t) = A0 * exp(gamma * t)
    # Taking log: ln(A) = ln(A0) + gamma * t
    t_fit = t_array[fit_slice]
    A_fit = field_amplitude[fit_slice]

    # Filter out very small values to avoid log issues
    valid_idx = A_fit > 1e-20
    t_fit = t_fit[valid_idx]
    A_fit = A_fit[valid_idx]

    assert len(t_fit) > 10, "Not enough valid data points for damping rate measurement"

    # Take log and fit linear trend
    log_A = np.log(A_fit)

    # Linear fit: log(A) = log(A0) + gamma * t
    # Use polyfit for robust linear regression
    coeffs = np.polyfit(t_fit, log_A, 1)
    measured_damping_rate = coeffs[0]

    print("\nDamping rate measurement:")
    print(f"  Measured: {measured_damping_rate:.6f}")
    print(f"  Expected: {expected_damping_rate:.6f}")
    print(
        f"  Relative error: {100 * abs(measured_damping_rate - expected_damping_rate) / abs(expected_damping_rate):.2f}%"  # noqa: E501
    )

    # Assert that measured damping rate is close to expected
    # Allow 25% relative error since:
    # 1. This is a challenging kinetic regime (klambda_D = 0.266)
    # 2. Hermite truncation effects can impact damping rate
    # 3. We're using a simplified config with fewer modes for speed
    np.testing.assert_allclose(
        measured_damping_rate,
        expected_damping_rate,
        rtol=0.01,
        err_msg=f"Damping rate mismatch: measured={measured_damping_rate:.6f}, expected={expected_damping_rate:.6f}",
    )

    print("\n✓ Test passed!")


def test_driven_epw(klambda_D: float | None = None):
    """
    Test driven EPW with external field using spectrax-1d for arbitrary klambda_D.

    Args:
        klambda_D: Normalized wavenumber k*lambda_D. If None, randomly chosen from [0.26, 0.4].

    This test:
    1. Loads the full config from configs/spectrax-1d/landau-damping.yaml
    2. Configures thermal velocity and driver frequency for desired klambda_D
    3. Runs the simulation with the external Ex driver
    4. Measures the frequency during driven phase using Hilbert transform
    5. Measures the damping rate after driver turns off
    6. Compares both to theoretical values
    """
    # Random selection if not specified
    if klambda_D is None:
        klambda_D = np.random.uniform(0.26, 0.34)
        print(f"\n{'=' * 60}")
        print(f"Randomly selected klambda_D = {klambda_D:.4f}")
        print(f"{'=' * 60}")

    # Load the full config with driver
    with open("configs/spectrax-1d/landau-damping.yaml") as file:
        config = yaml.safe_load(file)

    # Use EPW1D class for EPW-specific analysis and postprocessing
    config["solver"] = "hermite-epw-1d"
    # Use per-species hermite modes: high resolution for electrons, lower for ions
    config["grid"]["hermite_modes"] = {"electrons": {"Nn": 512, "Nm": 1, "Np": 1}, "ions": {"Nn": 32, "Nm": 1, "Np": 1}}

    # Calculate required thermal velocity for desired klambdaD
    Lx = config["physics"]["Lx"]
    k_fundamental = 2.0 * np.pi / Lx
    required_alpha_e = klambda_D * np.sqrt(2) / k_fundamental

    # Update config with calculated thermal velocity
    config["physics"]["alpha_e"] = [required_alpha_e, required_alpha_e, required_alpha_e]
    config["physics"]["alpha_s"][:3] = [required_alpha_e, required_alpha_e, required_alpha_e]

    # Keep ion thermal velocity ratio consistent (Ti/Te ≈ 0.001 for cold ions)
    Ti_Te_ratio = 0.001
    required_alpha_i = required_alpha_e * np.sqrt(Ti_Te_ratio)
    config["physics"]["alpha_s"][3:6] = [required_alpha_i, required_alpha_i, required_alpha_i]

    # Calculate expected frequency and damping rate from dispersion relation
    theoretical_root = electrostatic.get_roots_to_electrostatic_dispersion(
        wp_e=1.0, vth_e=1.0, k0=klambda_D, maxwellian_convention_factor=2.0
    )
    expected_frequency = np.real(theoretical_root)
    expected_damping_rate = np.imag(theoretical_root)

    # Update driver frequency to match theoretical EPW frequency
    assert "drivers" in config, "No drivers in config"
    assert "ex" in config["drivers"], "No ex driver in config"
    assert "0" in config["drivers"]["ex"], "No driver '0' in ex drivers"

    config["drivers"]["ex"]["0"]["w0"] = expected_frequency

    # Get driver configuration for info
    driver_config = config["drivers"]["ex"]["0"]
    k0 = driver_config["k0"]
    w0 = driver_config["w0"]
    a0 = driver_config["a0"]
    t_center = driver_config.get("t_center", 0.0)
    t_width = driver_config.get("t_width", 1e10)
    t_rise = driver_config.get("t_rise", 10.0)

    # Verify configuration
    vth_e = required_alpha_e
    calculated_klambdaD = k_fundamental * vth_e / np.sqrt(2)

    print(f"\nConfiguration for klambda_D = {klambda_D:.4f}:")
    print(f"  alpha_e[0] (vth_e) = {vth_e:.6f}")
    print(f"  k_fundamental = {k_fundamental:.6f}")
    print(f"  Calculated klambda_D = {calculated_klambdaD:.4f}")

    # Verify calculation is correct
    np.testing.assert_allclose(
        calculated_klambdaD, klambda_D, rtol=1e-10, err_msg="Config update failed: klambdaD mismatch"
    )

    print("\nDriver configuration:")
    print(f"  Wavenumber k0: {k0:.6f}")
    print(f"  Driver frequency w0: {w0:.6f} (updated to match dispersion)")
    print(f"  Driver amplitude a0: {a0}")
    print(f"  Driver temporal envelope: t_center={t_center}, t_width={t_width}, t_rise={t_rise}")

    print("\nTheoretical values from dispersion relation:")
    print(f"  Expected frequency: {expected_frequency:.6f}")
    print(f"  Expected damping rate: {expected_damping_rate:.6f}")

    # Verify that k0 matches the fundamental mode
    np.testing.assert_allclose(
        k0, k_fundamental, rtol=1e-6, err_msg=f"Driver k0={k0} doesn't match fundamental mode k={k_fundamental}"
    )

    print(f"✓ Configuration verified and updated for klambda_D = {klambda_D:.4f}")

    # Override MLflow experiment name for testing
    config["mlflow"]["experiment"] = "epw1d-test-driven"
    config["mlflow"]["run"] = f"epw1d-klambda-{klambda_D:.3f}-driven-test"

    # Run simulation
    print("\nRunning driven EPW simulation...")
    exo = ergoExo()
    exo.setup(config)
    result, post_processing_output, run_id = exo(None)

    # Extract EPW metrics computed by EPW1D module during post-processing
    # Metrics are in post_processing_output, not result
    metrics = post_processing_output.get("metrics", {})

    print("EPW Metrics:", metrics)

    # Verify metrics were computed
    assert "epw_avg_frequency_k1" in metrics, "EPW frequency metric not computed"
    assert "epw_damping_rate_k1" in metrics, "EPW damping rate metric not computed"

    measured_frequency = metrics["epw_avg_frequency_k1"]
    measured_damping_rate = metrics["epw_damping_rate_k1"]

    print("\nEPW Metrics (from EPW1D post-processing):")
    print(f"  Average frequency (k=1): {measured_frequency:.6f}")
    print(f"  Damping rate (k=1): {measured_damping_rate:.6f}")
    print(f"  Average amplitude (k=1): {metrics['epw_avg_amplitude_k1']:.2e}")
    print(f"  Final amplitude (k=1): {metrics['epw_final_amplitude_k1']:.2e}")

    # --- 1. Validate frequency against theory ---
    print("\n--- Frequency Validation ---")
    print(f"  Measured frequency: {measured_frequency:.6f}")
    print(f"  Expected frequency: {expected_frequency:.6f}")
    rel_err_freq = 100 * abs(measured_frequency - expected_frequency) / expected_frequency
    print(f"  Relative error: {rel_err_freq:.2f}%")

    # Assert frequency matches (allow 10% error)
    np.testing.assert_allclose(
        measured_frequency,
        expected_frequency,
        rtol=0.1,
        err_msg=f"Frequency mismatch: measured={measured_frequency:.6f}, expected={expected_frequency:.6f}",
    )
    print("  ✓ Frequency test passed!")

    # --- 2. Validate damping rate against theory ---
    print("\n--- Damping Rate Validation ---")
    print(f"  Measured damping rate: {measured_damping_rate:.6f}")
    print(f"  Expected damping rate: {expected_damping_rate:.6f}")
    rel_err_damp = 100 * abs(measured_damping_rate - expected_damping_rate) / abs(expected_damping_rate)
    print(f"  Relative error: {rel_err_damp:.2f}%")

    # Assert damping rate matches (allow 10% error for kinetic regime)
    np.testing.assert_allclose(
        measured_damping_rate,
        expected_damping_rate,
        rtol=0.1,
        err_msg=f"Damping rate mismatch: measured={measured_damping_rate:.6f}, expected={expected_damping_rate:.6f}",
    )
    print("  ✓ Damping rate test passed!")

    # Print energy summary from metrics
    if "avg_electrostatic_energy_final_50" in metrics:
        print(f"\nElectrostatic energy (avg over final 50 ωₚt): {metrics['avg_electrostatic_energy_final_50']:.2e}")

    print("\n✓ All tests passed - driven EPW validated successfully!")


if __name__ == "__main__":
    # Run the driven EPW test with random klambdaD
    test_driven_epw()
