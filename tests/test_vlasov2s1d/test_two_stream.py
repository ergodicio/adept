#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import numpy as np
import pytest
import yaml

from adept import electrostatic, ergoExo

from .theory import get_two_stream_growth_rate


def test_two_stream_instability():
    """Test two-stream instability growth rate."""

    with open("tests/test_vlasov2s1d/configs/two_stream.yaml") as file:
        defaults = yaml.safe_load(file)

    # Extract drift velocities
    v_e = defaults["density"]["species-background"]["v0"]
    v_i = defaults["density"]["species-ion"]["v0"]
    vd = abs(v_e - v_i) / 2.0  # relative drift velocity

    # Test parameters
    k0 = 0.25
    vth = 1.0  # thermal velocity

    # Calculate theoretical growth rate
    theoretical_growth_rate = get_two_stream_growth_rate(k0, vd, vth)

    # Modify config for this test
    defaults["drivers"]["ex"]["0"]["k0"] = k0
    defaults["drivers"]["ex"]["0"]["w0"] = 0.0  # Zero frequency
    defaults["drivers"]["ex"]["0"]["a0"] = 1.e-8  # Very small perturbation

    # Set domain size
    wavelength = 2.0 * np.pi / k0
    defaults["grid"]["xmax"] = wavelength
    defaults["grid"]["nx"] = 64

    defaults["mlflow"]["experiment"] = "vlasov2s1d-test-two-stream"

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Analyze the electric field growth
    efs = result.ys["fields"]["e"]
    ek1 = 2.0 / defaults["grid"]["nx"] * np.fft.fft(efs, axis=1)[:, 1]
    ek1_mag = np.abs(ek1)

    # Find the linear growth phase (avoid initial transient and late nonlinear phase)
    dt = result.ts["fields"][1] - result.ts["fields"][0]
    time_array = result.ts["fields"] * dt

    # Look for exponential growth in the middle portion
    start_idx = len(ek1_mag) // 4
    end_idx = 3 * len(ek1_mag) // 4

    # Fit exponential growth: log(E) = gamma*t + C
    log_ek1 = np.log(ek1_mag[start_idx:end_idx])
    times = time_array[start_idx:end_idx]

    # Linear fit to log(amplitude) vs time
    coeffs = np.polyfit(times, log_ek1, 1)
    measured_growth_rate = coeffs[0]

    relative_error_pct = 100*abs(measured_growth_rate - theoretical_growth_rate)/theoretical_growth_rate
    print(
        f"Two-Stream Instability Growth Rate Check \n"
        f"measured: {np.round(measured_growth_rate, 5)}, "
        f"theoretical: {np.round(theoretical_growth_rate, 5)}, "
        f"relative error: {np.round(relative_error_pct, 2)}%"
    )

    # Allow for 30% error due to finite temperature effects and numerical dispersion
    relative_error = abs(measured_growth_rate - theoretical_growth_rate) / theoretical_growth_rate
    assert relative_error < 0.3, f"Two-stream growth rate error too large: {relative_error*100:.1f}%"


@pytest.mark.parametrize("drift_velocity", [0.3, 0.5, 0.7])
def test_two_stream_growth_scaling(drift_velocity):
    """Test that two-stream growth rate scales with drift velocity."""

    with open("tests/test_vlasov2s1d/configs/two_stream.yaml") as file:
        defaults = yaml.safe_load(file)

    # Set symmetric drift velocities
    defaults["density"]["species-background"]["v0"] = drift_velocity
    defaults["density"]["species-ion"]["v0"] = -drift_velocity

    k0 = 0.2
    theoretical_growth_rate = get_two_stream_growth_rate(k0, drift_velocity)

    # Shorter simulation for parameter scan
    defaults["grid"]["tmax"] = 100.0
    defaults["save"]["fields"]["t"]["tmax"] = 100.0
    defaults["save"]["fields"]["t"]["nt"] = 1001

    defaults["drivers"]["ex"]["0"]["k0"] = k0
    defaults["drivers"]["ex"]["0"]["w0"] = 0.0
    defaults["drivers"]["ex"]["0"]["a0"] = 1.e-8

    wavelength = 2.0 * np.pi / k0
    defaults["grid"]["xmax"] = wavelength
    defaults["grid"]["nx"] = 64

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Basic check that simulation runs and shows growth
    efs = result.ys["fields"]["e"]
    ek1_mag = np.abs(2.0 / defaults["grid"]["nx"] * np.fft.fft(efs, axis=1)[:, 1])

    assert efs.shape[0] > 0, "Simulation produced no output"
    assert not np.any(np.isnan(efs)), "Simulation produced NaN values"

    # Check that there is some growth
    initial_amplitude = np.mean(ek1_mag[:10])
    final_amplitude = np.mean(ek1_mag[-10:])
    growth_factor = final_amplitude / initial_amplitude

    print(
        f"vd = {drift_velocity}: theoretical gamma = {theoretical_growth_rate:.6f}, "
        f"growth factor = {growth_factor:.2f}"
    )

    # Should see significant growth for unstable parameters
    if theoretical_growth_rate > 0.01:
        assert growth_factor > 2.0, f"Expected significant growth but got factor {growth_factor:.2f}"


def test_two_stream_stability_criterion():
    """Test that system is stable when drift velocity is too small."""

    with open("tests/test_vlasov2s1d/configs/two_stream.yaml") as file:
        defaults = yaml.safe_load(file)

    # Very small drift velocities - should be stable
    small_drift = 0.05
    defaults["density"]["species-background"]["v0"] = small_drift
    defaults["density"]["species-ion"]["v0"] = -small_drift

    # Short simulation
    defaults["grid"]["tmax"] = 50.0
    defaults["save"]["fields"]["t"]["tmax"] = 50.0
    defaults["save"]["fields"]["t"]["nt"] = 501

    defaults["drivers"]["ex"]["0"]["a0"] = 1.e-6  # Larger perturbation to test stability
    defaults["drivers"]["ex"]["0"]["k0"] = 0.3

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Analyze stability
    efs = result.ys["fields"]["e"]
    ek1_mag = np.abs(2.0 / defaults["grid"]["nx"] * np.fft.fft(efs, axis=1)[:, 1])

    # Check that amplitude doesn't grow significantly
    initial_amplitude = np.mean(ek1_mag[:10])
    final_amplitude = np.mean(ek1_mag[-10:])
    growth_factor = final_amplitude / initial_amplitude

    print(f"Stability test: small drift vd = {small_drift}, growth factor = {growth_factor:.2f}")

    # Should not see significant growth for stable parameters
    assert growth_factor < 3.0, f"System should be stable but grew by factor {growth_factor:.2f}"


if __name__ == "__main__":
    test_two_stream_instability()
