# Spectrax-1D Test Suite

This directory contains tests for the Spectrax-1D ADEPTModule, which wraps the external spectrax library for solving the Vlasov-Maxwell equations using Hermite-Fourier decomposition.

## Test Files

### test_landau_damping.py

Tests Landau damping of electron plasma waves (EPWs) for the klambda_D = 0.266 regime.

#### test_landau_damping_klambda_266()

- **Purpose**: Verify that spectrax-1d correctly reproduces Landau damping for a kinetic EPW
- **Configuration**: Uses simplified config with klambda_D = 0.266 (k = 2π, vth_e = 0.06)
- **Method**:
  1. Runs simulation with initial density perturbation
  2. Extracts field amplitude (ne_k1) from default save
  3. Measures damping rate via exponential fit
  4. Compares to theoretical value from electrostatic dispersion relation
- **Expected behavior**: Damping rate should match theory within 25% relative error

#### test_landau_damping_driven()

- **Purpose**: Verify that external field driver works correctly
- **Configuration**: Uses full config with external Ex driver
- **Method**: Runs simulation with driven EPW and checks for successful completion
- **Expected behavior**: Simulation completes without errors

## Configuration Files

### configs/landau-damping-test.yaml

Simplified configuration optimized for fast testing:
- Reduced grid resolution (Nx=17, Nn=32)
- Shorter simulation time (tmax=100)
- No external drivers
- Minimal save configuration

Parameters:
- klambda_D ≈ 0.266 (k = 2π/Lx, vth_e = 0.06)
- Bohm-Gross frequency ω ≈ 1.1
- Initial perturbation: dn/n ~ 1e-12 (linear regime)

## Running the Tests

Run all tests:
```bash
pytest tests/test_spectrax1d/
```

Run specific test:
```bash
pytest tests/test_spectrax1d/test_landau_damping.py::test_landau_damping_klambda_266
```

Run with verbose output:
```bash
pytest tests/test_spectrax1d/ -v -s
```

## Physics Background

### Landau Damping

Landau damping is a kinetic effect where electron plasma waves are damped due to wave-particle resonance. The damping rate depends on the wave number k normalized to the Debye length λD:

- **Fluid regime** (klambda_D << 1): Weak damping, Bohm-Gross dispersion
- **Kinetic regime** (klambda_D ~ 0.3): Strong damping, resonant wave-particle interaction
- **Heavily damped** (klambda_D > 0.4): Wave becomes evanescent

The test case with klambda_D = 0.266 is in the kinetic regime where Landau damping is significant but the wave is still clearly observable.

### Hermite-Fourier Method

The spectrax library uses a Hermite-Fourier decomposition:
- **Fourier basis** for spatial variation (x, y, z)
- **Hermite polynomials** for velocity variation (vx, vy, vz)

This spectral method provides:
- High accuracy for smooth distribution functions
- Efficient resolution of kinetic effects
- Direct representation of field-particle coupling

## Expected Results

For klambda_D = 0.266:
- Theoretical damping rate: γ ≈ -0.0135 (negative = exponential decay)
- Theoretical frequency: ω ≈ 1.1044 (Bohm-Gross with kinetic correction)
- Simulation should reproduce these values within ~25% relative error

The 25% tolerance accounts for:
1. Hermite mode truncation effects (finite Nn)
2. Reduced resolution in test config (for speed)
3. Finite-time windowing effects in damping rate measurement
