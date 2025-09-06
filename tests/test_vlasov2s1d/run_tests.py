#!/usr/bin/env python3
"""
Test runner for two-species Vlasov tests.

This script demonstrates how to run the various two-species Vlasov tests.
"""

import sys
import traceback


def run_basic_tests():
    """Run basic functionality tests."""
    print("=" * 60)
    print("RUNNING BASIC TWO-SPECIES TESTS")
    print("=" * 60)
    
    try:
        from tests.test_vlasov2s1d.test_basic import (
            test_two_species_basic_run,
            test_charge_conservation,
            test_energy_conservation
        )
        
        print("1. Basic run test...")
        test_two_species_basic_run()
        print("✓ PASSED\n")
        
        print("2. Charge conservation test...")
        test_charge_conservation()
        print("✓ PASSED\n")
        
        print("3. Energy conservation test...")
        test_energy_conservation()
        print("✓ PASSED\n")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def run_physics_tests():
    """Run physics validation tests."""
    print("=" * 60)
    print("RUNNING PHYSICS VALIDATION TESTS")
    print("=" * 60)
    
    success = True
    
    # Landau damping test
    try:
        print("1. Landau damping test...")
        from tests.test_vlasov2s1d.test_landau_damping import test_two_species_landau_damping
        test_two_species_landau_damping("real", "leapfrog", "poisson", "exponential")
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        success = False
    
    # Ion acoustic wave test
    try:
        print("2. Ion acoustic wave test...")
        from tests.test_vlasov2s1d.test_ion_acoustic import test_ion_acoustic_wave_propagation
        test_ion_acoustic_wave_propagation()
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        success = False
        
    return success


def run_parameter_tests():
    """Run parameter scaling tests."""
    print("=" * 60)
    print("RUNNING PARAMETER SCALING TESTS")
    print("=" * 60)
    
    success = True
    
    # Ion acoustic frequency scaling
    try:
        print("1. Ion acoustic frequency scaling...")
        from tests.test_vlasov2s1d.test_ion_acoustic import test_ion_acoustic_frequency_scaling
        for Ti_over_Te in [0.1, 0.2]:  # Test subset for demo
            test_ion_acoustic_frequency_scaling(Ti_over_Te)
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        success = False
        
    return success


def main():
    """Run all tests."""
    print("Two-Species Vlasov Code Test Suite")
    print("==================================\n")
    
    success = True
    
    # Run basic tests first
    if not run_basic_tests():
        success = False
        print("Basic tests failed - skipping physics tests")
        sys.exit(1)
    
    # Run physics validation tests
    if not run_physics_tests():
        success = False
    
    # Run parameter scaling tests
    if not run_parameter_tests():
        success = False
    
    # Summary
    print("=" * 60)
    if success:
        print("ALL TESTS PASSED! ✓")
        print("The two-species Vlasov solver is working correctly.")
    else:
        print("SOME TESTS FAILED ✗")
        print("Check the output above for details.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
