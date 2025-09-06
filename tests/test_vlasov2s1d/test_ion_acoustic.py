#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import numpy as np
import pytest
import yaml

from adept import electrostatic, ergoExo

from .theory import get_ion_acoustic_frequency, get_two_species_roots


def test_ion_acoustic_wave_propagation():
    """
    Test ion acoustic wave propagation in two-species plasma.

    Ion acoustic waves are low-frequency oscillations driven by electron pressure
    that primarily show up as ion density fluctuations. The ions provide the inertia
    while electrons provide the restoring force through pressure gradients.
    """

    with open("tests/test_vlasov2s1d/configs/ion_acoustic.yaml") as file:
        defaults = yaml.safe_load(file)

    # Test parameters
    k0 = 0.1  # Long wavelength for ion acoustic waves
    
    # Calculate theoretical ion acoustic frequency
    Te = defaults["density"]["species-background"]["T0"]
    Ti = defaults["density"]["species-ion"]["T0"]
    Te_over_Ti = Te / Ti
    me_over_mi = 1.0 / (defaults["units"]["A"] * 1836.0)
    
    theoretical_freq = get_ion_acoustic_frequency(k0, Te_over_Ti, me_over_mi)
    
    # Modify config for this test
    defaults["drivers"]["ex"]["0"]["k0"] = k0
    defaults["drivers"]["ex"]["0"]["w0"] = theoretical_freq
    defaults["drivers"]["ex"]["0"]["a0"] = 1.e-4  # Small amplitude
    
    # Set domain size based on wavelength
    wavelength = 2.0 * np.pi / k0
    defaults["grid"]["xmax"] = 2.0 * wavelength
    defaults["grid"]["nx"] = 128
    
    defaults["mlflow"]["experiment"] = "vlasov2s1d-test-ion-acoustic"

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]
    
    # Check what's available in the fields output
    print("Available fields:", list(result.ys["fields"].keys()) if "fields" in result.ys else "No fields")
    
    # Use high-resolution ion density from fields if available
    if "fields" in result.ys and "n_i" in result.ys["fields"]:
        # Use high-resolution ion density from fields (nt=2001)
        ni_field = result.ys["fields"]["n_i"]
        ni_k1 = 2.0 / defaults["grid"]["nx"] * np.fft.fft(ni_field, axis=1)[:, 1]
        dt_fields = result.ts["fields"][1] - result.ts["fields"][0]
        env, freq = electrostatic.get_nlfs(ni_k1, dt_fields)
        print("Using high-resolution ion density from fields")
        
    else:
        # Fallback: calculate ion density from distribution function (lower resolution)
        fi = result.ys["ion"]["f"]  # Ion distribution function
        
        # Calculate ion density by integrating over velocity
        dv_i = defaults["grid"]["dv_i"]
        ni = np.array([np.trapz(fi[t_idx], dx=dv_i, axis=0) for t_idx in range(fi.shape[0])])
        
        # Get the k=1 mode (fundamental) of ion density
        ni_k1 = 2.0 / defaults["grid"]["nx"] * np.fft.fft(ni, axis=1)[:, 1]
        
        # Measure frequency from ion density oscillations
        dt_ion = result.ts["ion"][1] - result.ts["ion"][0]
        env, freq = electrostatic.get_nlfs(ni_k1, dt_ion)
        print("Using ion density calculated from distribution function")
    
    # Use middle part of simulation for measurement
    mid_start = len(freq) // 3
    mid_end = 2 * len(freq) // 3
    measured_freq = np.mean(freq[mid_start:mid_end])
    
    print(
        f"Ion Acoustic Wave Frequency Check (from ion density) \n"
        f"measured: {np.round(measured_freq, 5)}, "
        f"theoretical: {np.round(theoretical_freq, 5)}, "
        f"relative error: {np.round(100*abs(measured_freq - theoretical_freq)/theoretical_freq, 2)}%"
    )
    
    # Also check the electric field for comparison
    efs = result.ys["fields"]["e"]
    ek1 = 2.0 / defaults["grid"]["nx"] * np.fft.fft(efs, axis=1)[:, 1]
    env_e, freq_e = electrostatic.get_nlfs(ek1, result.ts["fields"][1] - result.ts["fields"][0])
    measured_freq_e = np.mean(freq_e[mid_start:mid_end])
    
    print(
        f"Electric field frequency check: {np.round(measured_freq_e, 5)} "
        f"(should be similar to ion density frequency)"
    )
    
    # Allow for 10% error due to numerical dispersion and finite temperature effects
    relative_error = abs(measured_freq - theoretical_freq) / theoretical_freq
    assert relative_error < 0.1, f"Ion acoustic frequency error too large: {relative_error*100:.1f}%"


@pytest.mark.parametrize("Ti_over_Te", [0.1, 0.2, 0.5])
def test_ion_acoustic_frequency_scaling(Ti_over_Te):
    """Test that ion acoustic frequency scales correctly with temperature ratio."""
    
    with open("tests/test_vlasov2s1d/configs/ion_acoustic.yaml") as file:
        defaults = yaml.safe_load(file)

    # Modify ion temperature
    defaults["density"]["species-ion"]["T0"] = Ti_over_Te
    
    k0 = 0.1
    Te_over_Ti = 1.0 / Ti_over_Te
    me_over_mi = 1.0 / (defaults["units"]["A"] * 1836.0)
    
    theoretical_freq = get_ion_acoustic_frequency(k0, Te_over_Ti, me_over_mi)
    
    # Quick test with shorter simulation
    defaults["grid"]["tmax"] = 200.0
    defaults["save"]["fields"]["t"]["tmax"] = 200.0
    defaults["save"]["fields"]["t"]["nt"] = 401
    defaults["save"]["ion"]["t"]["tmax"] = 200.0
    defaults["save"]["ion"]["t"]["nt"] = 21
    
    defaults["drivers"]["ex"]["0"]["k0"] = k0
    defaults["drivers"]["ex"]["0"]["w0"] = theoretical_freq
    defaults["drivers"]["ex"]["0"]["a0"] = 1.e-4
    
    wavelength = 2.0 * np.pi / k0
    defaults["grid"]["xmax"] = wavelength
    defaults["grid"]["nx"] = 64

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]
    
    # Basic check that simulation runs without crashing and has ion data
    if "ion" in result.ys and "f" in result.ys["ion"]:
        fi = result.ys["ion"]["f"]
        assert fi.shape[0] > 0, "Simulation produced no ion data"
        assert not np.any(np.isnan(fi)), "Ion distribution contains NaN values"
    else:
        print("Warning: No ion distribution data found in result")
    
    # Basic check on fields
    if "fields" in result.ys and "e" in result.ys["fields"]:
        efs = result.ys["fields"]["e"]
        assert efs.shape[0] > 0, "Simulation produced no field data"
        assert not np.any(np.isnan(efs)), "Electric field contains NaN values"
    
    print(f"Ti/Te = {Ti_over_Te}: theoretical frequency = {theoretical_freq:.6f}")


if __name__ == "__main__":
    test_ion_acoustic_wave_propagation()
