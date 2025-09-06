#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
import itertools

import numpy as np
import pytest
import yaml

from adept import electrostatic, ergoExo

from .theory import get_two_species_roots


def _modify_defaults_(defaults, rng, real_or_imag, time, field, edfdv):
    defaults["terms"]["time"] = time
    defaults["terms"]["field"] = field
    defaults["terms"]["edfdv"] = edfdv
    if field == "ampere":
        defaults["grid"]["dt"] = 0.025

    if real_or_imag == "real":
        rand_k0 = np.round(rng.uniform(0.2, 0.28), 3)
    else:
        rand_k0 = np.round(rng.uniform(0.28, 0.35), 3)

    # For two-species plasma, use electron parameters for EPW
    # Ion parameters: cold ions (Ti << Te)
    wp_e = 1.0  # normalized electron plasma frequency
    vth_e = 1.0  # normalized electron thermal velocity
    wp_i = 1.0  # ion plasma frequency (same as electron for quasi-neutrality)
    
    # Extract ion temperature from config and calculate ion thermal velocity
    ion_T0 = defaults["density"]["species-ion"]["T0"]
    me_over_mi = 1.0 / (defaults["units"]["A"] * 1836.0)
    vth_i = np.sqrt(ion_T0 * me_over_mi)

    # Get two-species dispersion relation root (electron mode for EPW)
    try:
        root = get_two_species_roots(wp_e, vth_e, wp_i, vth_i, rand_k0, mode='electron')
    except Exception:
        # Fallback to single-species if two-species fails
        root = electrostatic.get_roots_to_electrostatic_dispersion(1.0, 1.0, rand_k0)

    defaults["drivers"]["ex"]["0"]["k0"] = float(rand_k0)
    defaults["drivers"]["ex"]["0"]["w0"] = float(np.real(root))
    xmax = float(2.0 * np.pi / rand_k0)
    defaults["grid"]["xmax"] = xmax
    defaults["mlflow"]["experiment"] = "vlasov2s1d-test-resonance"

    return defaults, root


@pytest.mark.parametrize(
    "real_or_imag, time, field, edfdv",
    itertools.product(
        ["real", "imag"], ["leapfrog"], ["poisson"], ["exponential"]
    ),
)
def test_two_species_landau_damping(real_or_imag, time, field, edfdv):
    """Test Landau damping in two-species plasma (electron plasma waves with cold ions)."""
    
    with open("tests/test_vlasov2s1d/configs/resonance.yaml") as file:
        defaults = yaml.safe_load(file)

    # modify config
    rng = np.random.default_rng()
    mod_defaults, root = _modify_defaults_(defaults, rng, real_or_imag, time, field, edfdv)

    actual_damping_rate = np.imag(root)
    actual_resonance = np.real(root)

    exo = ergoExo()
    exo.setup(mod_defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]
    efs = result.ys["fields"]["e"]
    ek1 = 2.0 / mod_defaults["grid"]["nx"] * np.fft.fft(efs, axis=1)[:, 1]
    ek1_mag = np.abs(ek1)
    
    if real_or_imag == "imag":
        frslc = slice(-100, -50)
        measured_damping_rate = np.mean(
            np.gradient(ek1_mag[frslc], (result.ts["fields"][1] - result.ts["fields"][0])) / ek1_mag[frslc]
        )
        print(
            f"Two-species Landau Damping rate check \n"
            f"measured: {np.round(measured_damping_rate, 5)}, "
            f"actual: {np.round(actual_damping_rate, 5)}, "
        )

        np.testing.assert_almost_equal(measured_damping_rate, actual_damping_rate, decimal=2)
    else:
        env, freq = electrostatic.get_nlfs(ek1, result.ts["fields"][1] - result.ts["fields"][0])
        frslc = slice(-480, -240)
        print(
            f"Two-species Frequency check \n"
            f"measured: {np.round(np.mean(freq[frslc]), 5)}, "
            f"desired: {np.round(actual_resonance, 5)}, "
        )
        measured_resonance = np.mean(freq[frslc])
        np.testing.assert_almost_equal(measured_resonance, actual_resonance, decimal=2)


if __name__ == "__main__":
    test_two_species_landau_damping(real_or_imag="real", time="leapfrog", field="poisson", edfdv="exponential")
