#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
"""
Test Landau damping with the Lawson-RK4 exponential integrator.

Runs the same driven EPW test as test_landau_damping.py but using the
exponential integrator instead of explicit Dopri8.
"""

from jax import config

config.update("jax_enable_x64", True)
import numpy as np
import yaml

from adept import electrostatic, ergoExo


def test_driven_epw_exponential(klambda_D: float | None = None):
    """
    Test driven EPW with exponential integrator (Lawson-RK4).

    Same physics as test_driven_epw in test_landau_damping.py, but using
    the Lawson-RK4 exponential integrator instead of Dopri8.
    """
    if klambda_D is None:
        klambda_D = np.random.uniform(0.26, 0.34)
        print(f"\n{'=' * 60}")
        print(f"Randomly selected klambda_D = {klambda_D:.4f}")
        print(f"{'=' * 60}")

    # Load the full config with driver
    with open("configs/spectrax-1d/landau-damping.yaml") as file:
        cfg = yaml.safe_load(file)

    # Use EPW1D class for EPW-specific analysis and postprocessing
    cfg["solver"] = "hermite-epw-1d"
    cfg["grid"]["hermite_modes"] = {
        "electrons": {"Nn": 512, "Nm": 1, "Np": 1},
        "ions": {"Nn": 32, "Nm": 1, "Np": 1},
    }

    # --- Switch to exponential integrator ---
    cfg["grid"]["integrator"] = "exponential"
    cfg["grid"]["adaptive_time_step"] = False

    # Calculate required thermal velocity for desired klambdaD
    Lx = cfg["physics"]["Lx"]
    k_fundamental = 2.0 * np.pi / Lx
    required_alpha_e = klambda_D * np.sqrt(2) / k_fundamental

    cfg["physics"]["alpha_e"] = [required_alpha_e] * 3
    cfg["physics"]["alpha_s"][:3] = [required_alpha_e] * 3

    Ti_Te_ratio = 0.001
    required_alpha_i = required_alpha_e * np.sqrt(Ti_Te_ratio)
    cfg["physics"]["alpha_s"][3:6] = [required_alpha_i] * 3

    # Get theoretical frequency and damping rate
    theoretical_root = electrostatic.get_roots_to_electrostatic_dispersion(
        wp_e=1.0, vth_e=1.0, k0=klambda_D, maxwellian_convention_factor=2.0
    )
    expected_frequency = np.real(theoretical_root)
    expected_damping_rate = np.imag(theoretical_root)

    # Update driver frequency to match theory
    cfg["drivers"]["ex"]["0"]["w0"] = expected_frequency

    print(f"\nConfiguration for klambda_D = {klambda_D:.4f}:")
    print(f"  alpha_e = {required_alpha_e:.6f}")
    print(f"  Expected frequency: {expected_frequency:.6f}")
    print(f"  Expected damping rate: {expected_damping_rate:.6f}")
    print(f"  Integrator: exponential (Lawson-RK4)")
    print(f"  dt: {cfg['grid']['dt']}")

    cfg["mlflow"]["experiment"] = "epw1d-exponential-test"
    cfg["mlflow"]["run"] = f"exponential-klambda-{klambda_D:.3f}"

    # Run simulation
    print("\nRunning driven EPW simulation with exponential integrator...")
    exo = ergoExo()
    exo.setup(cfg)
    result, post_processing_output, run_id = exo(None)

    # Extract EPW metrics
    metrics = post_processing_output.get("metrics", {})
    print("EPW Metrics:", metrics)

    assert "epw_avg_frequency_k1" in metrics, "EPW frequency metric not computed"
    assert "epw_damping_rate_k1" in metrics, "EPW damping rate metric not computed"

    measured_frequency = metrics["epw_avg_frequency_k1"]
    measured_damping_rate = metrics["epw_damping_rate_k1"]

    print(f"\n--- Frequency ---")
    print(f"  Measured:  {measured_frequency:.6f}")
    print(f"  Expected:  {expected_frequency:.6f}")
    rel_err_freq = 100 * abs(measured_frequency - expected_frequency) / expected_frequency
    print(f"  Rel error: {rel_err_freq:.2f}%")

    print(f"\n--- Damping Rate ---")
    print(f"  Measured:  {measured_damping_rate:.6f}")
    print(f"  Expected:  {expected_damping_rate:.6f}")
    rel_err_damp = 100 * abs(measured_damping_rate - expected_damping_rate) / abs(expected_damping_rate)
    print(f"  Rel error: {rel_err_damp:.2f}%")

    # Assert frequency matches (allow 10% error)
    np.testing.assert_allclose(
        measured_frequency,
        expected_frequency,
        rtol=0.1,
        err_msg=f"Frequency mismatch: measured={measured_frequency:.6f}, expected={expected_frequency:.6f}",
    )
    print("  Frequency test PASSED")

    # Assert damping rate matches (allow 10% error)
    np.testing.assert_allclose(
        measured_damping_rate,
        expected_damping_rate,
        rtol=0.1,
        err_msg=f"Damping rate mismatch: measured={measured_damping_rate:.6f}, expected={expected_damping_rate:.6f}",
    )
    print("  Damping rate test PASSED")

    print("\nAll tests passed - exponential integrator validated!")


if __name__ == "__main__":
    test_driven_epw_exponential()
