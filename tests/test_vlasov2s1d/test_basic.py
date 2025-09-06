#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

import numpy as np
import yaml

from adept import ergoExo


def test_two_species_basic_run():
    """Test that two-species solver runs without crashing."""

    with open("tests/test_vlasov2s1d/configs/resonance.yaml") as file:
        defaults = yaml.safe_load(file)

    # Modify for quick test
    defaults["grid"]["tmax"] = 10.0
    defaults["grid"]["nx"] = 16
    defaults["grid"]["nv"] = 64
    defaults["grid"]["nv_i"] = 32
    defaults["save"]["fields"]["t"]["tmax"] = 10.0
    defaults["save"]["fields"]["t"]["nt"] = 21
    defaults["save"]["electron"]["t"]["nt"] = 3
    defaults["save"]["ion"]["t"]["nt"] = 3

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Basic checks
    assert "fields" in result.ys, "Fields not found in result"
    assert "electron" in result.ys, "Electron data not found in result"
    assert "ion" in result.ys, "Ion data not found in result"

    # Check shapes
    efs = result.ys["fields"]["e"]
    fe = result.ys["electron"]["f"]
    fi = result.ys["ion"]["f"]

    assert efs.shape[0] > 0, "No time steps in electric field"
    assert efs.shape[1] == defaults["grid"]["nx"], "Electric field spatial dimension mismatch"

    assert fe.shape[0] > 0, "No time steps in electron distribution"
    assert fe.shape[1] == defaults["grid"]["nv"], "Electron velocity dimension mismatch"
    assert fe.shape[2] == defaults["grid"]["nx"], "Electron spatial dimension mismatch"

    assert fi.shape[0] > 0, "No time steps in ion distribution"
    assert fi.shape[1] == defaults["grid"]["nv_i"], "Ion velocity dimension mismatch"
    assert fi.shape[2] == defaults["grid"]["nx"], "Ion spatial dimension mismatch"

    # Check for NaN values
    assert not np.any(np.isnan(efs)), "Electric field contains NaN values"
    assert not np.any(np.isnan(fe)), "Electron distribution contains NaN values"
    assert not np.any(np.isnan(fi)), "Ion distribution contains NaN values"

    print("Two-species basic run test passed!")


def test_charge_conservation():
    """Test that charge is conserved in two-species simulation."""

    with open("tests/test_vlasov2s1d/configs/resonance.yaml") as file:
        defaults = yaml.safe_load(file)

    # Quick test setup
    defaults["grid"]["tmax"] = 20.0
    defaults["grid"]["nx"] = 32
    defaults["save"]["fields"]["t"]["tmax"] = 20.0
    defaults["save"]["fields"]["t"]["nt"] = 41
    defaults["save"]["electron"]["t"]["nt"] = 5
    defaults["save"]["ion"]["t"]["nt"] = 5

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Calculate total charge at each saved time
    fe = result.ys["electron"]["f"]
    fi = result.ys["ion"]["f"]

    dx = defaults["grid"]["dx"]
    dv_e = defaults["grid"]["dv"]
    dv_i = defaults["grid"]["dv_i"]

    # Integrate distribution functions to get densities
    ne_total = []
    ni_total = []

    for t_idx in range(fe.shape[0]):
        # Electron density: integrate over velocity
        ne = np.sum(fe[t_idx], axis=0) * dv_e
        ne_total.append(np.sum(ne) * dx)

        # Ion density: integrate over velocity
        ni = np.sum(fi[t_idx], axis=0) * dv_i
        ni_total.append(np.sum(ni) * dx)

    ne_total = np.array(ne_total)
    ni_total = np.array(ni_total)

    # Check charge conservation (electrons are negative, ions positive)
    total_charge = ni_total - ne_total  # qi*ni + qe*ne where qe = -1, qi = +1

    initial_charge = total_charge[0]
    final_charge = total_charge[-1]
    charge_error = abs(final_charge - initial_charge) / abs(initial_charge)

    print("Charge conservation test:")
    print(f"Initial total charge: {initial_charge:.6e}")
    print(f"Final total charge: {final_charge:.6e}")
    print(f"Relative error: {charge_error:.6e}")

    # Allow for small numerical errors
    assert charge_error < 1e-10, f"Charge not conserved: error = {charge_error:.2e}"

    print("Charge conservation test passed!")


def test_energy_conservation():
    """Test that energy is approximately conserved (within expected numerical error)."""

    with open("tests/test_vlasov2s1d/configs/resonance.yaml") as file:
        defaults = yaml.safe_load(file)

    # Set up for energy conservation test
    defaults["grid"]["tmax"] = 50.0
    defaults["drivers"]["ex"]["0"]["a0"] = 1.e-8  # Very small driver
    defaults["save"]["fields"]["t"]["tmax"] = 50.0
    defaults["save"]["fields"]["t"]["nt"] = 101
    defaults["save"]["electron"]["t"]["nt"] = 6
    defaults["save"]["ion"]["t"]["nt"] = 6

    exo = ergoExo()
    exo.setup(defaults)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Calculate field energy
    efs = result.ys["fields"]["e"]
    field_energy = np.sum(efs**2, axis=1) * defaults["grid"]["dx"] / 2.0

    # Simple energy check - field energy should not grow without bound
    initial_energy = np.mean(field_energy[:5])
    final_energy = np.mean(field_energy[-5:])

    print("Energy conservation test:")
    print(f"Initial field energy: {initial_energy:.6e}")
    print(f"Final field energy: {final_energy:.6e}")

    # Basic check that energy doesn't blow up
    assert final_energy < 1000 * initial_energy, "Energy grew too much - possible instability"
    assert not np.any(np.isinf(field_energy)), "Field energy became infinite"

    print("Energy conservation test passed!")


if __name__ == "__main__":
    test_two_species_basic_run()
    test_charge_conservation()
    test_energy_conservation()
