# Two-Species Vlasov Code Tests

This directory contains comprehensive tests for the two-species Vlasov solver (`vlasov-2s1d`). The tests validate both the numerical implementation and the physics correctness of the solver.

## Test Structure

### Basic Tests (`test_basic.py`)
- **Basic functionality**: Ensures the solver runs without crashing
- **Charge conservation**: Verifies quasi-neutrality is maintained
- **Energy conservation**: Checks energy doesn't blow up unphysically

### Physics Validation Tests

#### Landau Damping (`test_landau_damping.py`)
Tests electron plasma wave damping in the presence of cold ions:
- Validates both frequency and damping rate
- Uses two-species dispersion relation theory
- Parametrized over different numerical schemes (where available)

#### Ion Acoustic Waves (`test_ion_acoustic.py`)
Tests low-frequency ion acoustic wave propagation:
- **Primary measurement**: Ion density fluctuations (the main physics)
- **Secondary check**: Electric field oscillations (should correlate)
- Validates wave frequency against theoretical predictions
- Tests frequency scaling with temperature ratio Ti/Te
- Covers the regime where Te > Ti (standard ion acoustic conditions)

### Configuration Files (`configs/`)
- `resonance.yaml`: Base config for Landau damping tests (cold ions)
- `ion_acoustic.yaml`: Config for ion acoustic wave tests (Ti < Te)

### Theory Module (`theory.py`)
Contains analytical functions for:
- Two-species plasma dispersion relations
- Ion acoustic frequency calculations

## Running the Tests

### Individual Tests
```bash
# Run a specific test
python -m pytest tests/test_vlasov2s1d/test_basic.py::test_two_species_basic_run

# Run all tests in a file
python -m pytest tests/test_vlasov2s1d/test_landau_damping.py

# Run with verbose output
python -m pytest tests/test_vlasov2s1d/ -v
```

### Test Suite Runner
Use the provided runner script for a comprehensive test:
```bash
python tests/test_vlasov2s1d/run_tests.py
```

This will run:
1. Basic functionality tests
2. Physics validation tests  
3. Parameter scaling tests

## Test Parameters

### Physical Parameters Tested
- **Temperature ratios**: Ti/Te from 0.1 to 0.5
- **Wavenumbers**: 0.1 to 0.35 (in Debye length units)

### Numerical Parameters
- **Grid sizes**: 16-128 spatial points, 32-512 velocity points
- **Time steps**: Adaptive based on physics (0.1-0.5)
- **Simulation times**: 10-1000 plasma periods depending on physics

## Physics Tested

### Landau Damping
- **Regime**: Electron plasma waves with cold ions (Ti << Te)
- **Theory**: Two-species generalization of single-species Landau damping
- **Validation**: Frequency within 2% and damping rate within 2% of theory

### Ion Acoustic Waves  
- **Regime**: Long wavelength (k*λD << 1), warm electrons, cold ions
- **Theory**: ω ≈ k*cs where cs = sqrt((Te + Ti)/mi)
- **Primary observable**: Ion density oscillations ni(x,t)
- **Validation**: Frequency within 10% of theory (allows for finite-T effects)

## Expected Test Results

### Successful Tests Should Show:
- No NaN or infinite values in simulation output
- Proper conservation of charge (< 1e-10 relative error)
- Bounded energy evolution (no runaway growth)
- Physics frequencies/growth rates matching theory within specified tolerances

### Common Failure Modes:
- **Numerical instability**: Usually from time step too large or inadequate resolution
- **Physics mismatch**: Often from being outside the theoretical regime of validity
- **Conservation violations**: May indicate bugs in the solver implementation

## Extending the Tests

To add new physics tests:

1. **Create new config**: Add YAML file in `configs/` directory
2. **Add theory**: Implement analytical functions in `theory.py`
3. **Write test**: Create test function following existing patterns
4. **Update runner**: Add test to `run_tests.py` if desired

## Notes on Two-Species Physics

The two-species Vlasov solver handles electrons and ions with different temperatures and drift velocities. Key physics:

- **Quasi-neutrality**: Total charge should be conserved
- **Mass ratio**: me/mi ≈ 1/1836 creates separation of time scales
- **Multiple modes**: Both electron plasma waves (high frequency) and ion acoustic waves (low frequency)
- **Instabilities**: Ion-acoustic and other kinetic instabilities possible

## Dependencies

Tests require:
- `numpy`, `scipy` for numerical calculations
- `yaml` for configuration file parsing
- `pytest` for test framework
- `adept` package with `vlasov-2s1d` solver
- `adept.electrostatic` module for analytical dispersion relations
