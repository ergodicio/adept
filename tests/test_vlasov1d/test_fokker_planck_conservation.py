#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Test that Fokker-Planck collision operators conserve density and energy.

The Lenard-Bernstein and Dougherty operators should conserve:
- Density: ∫ f dv = constant
- Energy: ∫ f v² dv = constant

This test verifies that the boundary conditions in the tridiagonal solve
are implemented correctly to ensure these conservation properties.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import yaml

from adept import ergoExo


@pytest.mark.parametrize("operator_type", ["chang_cooper_dougherty", "chang_cooper", "Dougherty", "Lenard_Bernstein"])
def test_fokker_planck_conservation(operator_type):
    """
    Test that Fokker-Planck operators conserve density and energy.

    Args:
        operator_type: Either "Lenard_Bernstein" or "Dougherty"
    """
    with open("tests/test_vlasov1d/configs/fokker_planck_conservation.yaml") as file:
        config = yaml.safe_load(file)

    # Set the operator type
    config["terms"]["fokker_planck"]["type"] = operator_type

    # Run simulation
    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Extract distribution function over time
    f_xvt = result.ys["electron"]  # Shape: (nt, nx, nv)
    times = result.ts["electron"]

    # Get grid parameters from the actual config (cell-centered grid)
    # The code constructs v as: linspace(-vmax + dv/2, vmax - dv/2, nv)
    dv = config["grid"]["vmax"] * 2.0 / config["grid"]["nv"]
    v = np.linspace(-config["grid"]["vmax"] + dv / 2, config["grid"]["vmax"] - dv / 2, config["grid"]["nv"])

    # Compute density and energy over time
    nt = f_xvt.shape[0]
    nx = f_xvt.shape[1]

    # Use midpoint rule (sum * dv) consistent with the code
    # Density: ∫ f dv
    density = np.sum(f_xvt, axis=-1) * dv
    # Energy: ∫ f v² dv
    energy = np.sum(f_xvt * v**2, axis=-1) * dv

    # Check conservation for each spatial point
    for ix in range(nx):
        # Initial values
        density_0 = density[0, ix]
        energy_0 = energy[0, ix]

        # Check density conservation (should be exact to machine precision)
        density_rel_error = np.abs(density[:, ix] - density_0) / density_0
        max_density_error = np.max(density_rel_error)

        print(f"\n{operator_type} - Spatial point {ix}:")
        print(f"  Max density relative error: {max_density_error:.2e}")

        # Density should be conserved to within 1e-10 relative error
        assert max_density_error < 1e-10, (
            f"{operator_type}: Density not conserved at x[{ix}]. Max relative error: {max_density_error:.2e}"
        )

        # Check energy conservation
        energy_rel_error = np.abs(energy[:, ix] - energy_0) / energy_0
        max_energy_error = np.max(energy_rel_error)

        print(f"  Max energy relative error: {max_energy_error:.2e}")

        # Energy should be conserved to within 1e-6 relative error
        assert max_energy_error < 1e-6, (
            f"{operator_type}: Energy not conserved at x[{ix}]. Max relative error: {max_energy_error:.2e}"
        )

    print(f"\n{operator_type}: Conservation test passed!")
    print(f"  Density conserved to within {np.max(np.abs(density - density[0:1, :]) / density[0:1, :]):.2e}")
    print(f"  Energy conserved to within {np.max(np.abs(energy - energy[0:1, :]) / energy[0:1, :]):.2e}")


@pytest.mark.parametrize("operator_type", ["chang_cooper", "Lenard_Bernstein", "Dougherty"])
def test_fokker_planck_thermalization(operator_type):
    """
    Test that Fokker-Planck operators drive distribution toward Maxwellian.

    This is a secondary test to verify the operators work correctly beyond
    just conservation. The distribution should relax toward a Maxwellian
    with the same density and energy.

    Args:
        operator_type: Either "Lenard_Bernstein" or "Dougherty"
    """
    with open("tests/test_vlasov1d/configs/fokker_planck_conservation.yaml") as file:
        config = yaml.safe_load(file)

    # Set the operator type and increase collision frequency for faster thermalization
    config["terms"]["fokker_planck"]["type"] = operator_type
    config["terms"]["fokker_planck"]["space"]["baseline"] = 1.0  # Stronger collisions
    config["grid"]["tmax"] = 20.0  # Longer time for thermalization
    config["save"]["electron"]["t"]["tmax"] = 20.0
    config["save"]["electron"]["t"]["nt"] = 201
    config["save"]["fields"]["t"]["tmax"] = 20.0
    config["save"]["fields"]["t"]["nt"] = 201

    # Run simulation
    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)
    result = result["solver result"]

    # Extract distribution function
    f_xvt = result.ys["electron"]

    # Get grid parameters from the actual config (cell-centered grid)
    dv = config["grid"]["vmax"] * 2.0 / config["grid"]["nv"]
    v = np.linspace(-config["grid"]["vmax"] + dv / 2, config["grid"]["vmax"] - dv / 2, config["grid"]["nv"])

    # Check thermalization for first spatial point
    ix = 0
    f_initial = f_xvt[0, ix, :]
    f_final = f_xvt[-1, ix, :]

    # Compute temperature of final distribution using midpoint rule
    n_final = np.sum(f_final) * dv
    T_final = np.sum(f_final * v**2) * dv / n_final

    # Expected Maxwellian
    f_maxwellian = n_final / np.sqrt(2.0 * np.pi * T_final) * np.exp(-(v**2) / (2.0 * T_final))

    # Compute L2 error between final distribution and Maxwellian using midpoint rule
    l2_error = np.sqrt(np.sum((f_final - f_maxwellian) ** 2) * dv)
    l2_norm = np.sqrt(np.sum(f_maxwellian**2) * dv)
    relative_error = l2_error / l2_norm

    print(f"\n{operator_type} - Thermalization test:")
    print(f"  L2 error to Maxwellian: {relative_error:.2e}")

    # Final distribution should be close to Maxwellian (within 10% L2 error)
    # This is a loose bound since we're testing conservation primarily
    assert relative_error < 0.1, (
        f"{operator_type}: Distribution did not thermalize properly. L2 error to Maxwellian: {relative_error:.2e}"
    )


if __name__ == "__main__":
    # Run tests for both operators
    for operator in ["Lenard_Bernstein", "Dougherty", "chang_cooper"]:
        print(f"\n{'=' * 60}")
        print(f"Testing {operator}")
        print("=" * 60)
        test_fokker_planck_conservation(operator)
        test_fokker_planck_thermalization(operator)
