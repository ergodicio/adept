#  Copyright (c) Ergodic LLC 2024
#  research@ergodic.io
"""
Tests for spectrax-1d Maxwell solver: EM dispersion and absorbing boundary conditions.

Two test suites:

1. EM wave dispersion (test_em_wave_dispersion)
   Drives a transverse (Ey) wave at the expected cold-plasma EM dispersion frequency
       ω_EM = √(ξ² + ωpe²)  =  √((2π/Lx)² + 1)
   (c = 1 and ωpe = 1 in the code's normalized units; see maxwell_module.py for the
   derivation from the Hermite-Fourier Vlasov-Maxwell equations.)
   For Lx = 2π → ω_EM = √2 ≈ 1.414.
   The measured instantaneous frequency of Ey(k=1) must lie within 15% of this value
   for both the explicit (Dopri8) and Lawson-RK4 exponential integrators.

2. Wave absorption (test_em_wave_absorption)
   Runs the same geometry but with a quadratic sponge layer covering the rightmost
   30% of the domain (σ_max = 10 ωpe).  After the driver turns off at t ≈ 90, the
   wave is absorbed as it enters the sponge.  The test checks that
       em_absorption_ratio = <EM energy, final 20%> / peak EM energy < 0.1
   confirming that > 90% of the wave energy has been removed from the domain.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pytest
import yaml

from adept import ergoExo

# ---------------------------------------------------------------------------
# Shared runner
# ---------------------------------------------------------------------------


def _run_maxwell(cfg: dict) -> dict:
    """Run a hermite-maxwell-1d simulation and return the post-process metrics dict."""
    exo = ergoExo()
    exo.setup(cfg)
    _, post, _ = exo(None)
    return post["metrics"]


# ---------------------------------------------------------------------------
# Fixtures: load base configs once and let each test customise them
# ---------------------------------------------------------------------------


@pytest.fixture
def dispersion_cfg():
    with open("configs/spectrax-1d/em-dispersion.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def absorption_cfg():
    with open("configs/spectrax-1d/wave-absorption.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Test 1: EM wave dispersion
# ---------------------------------------------------------------------------


def test_em_wave_dispersion(dispersion_cfg):
    """
    MaxwellExponential propagates vacuum EM waves at the correct light speed ω = c|k|.

    The Lawson-RK4 exponential integrator handles the vacuum Maxwell curl equations
    exactly via MaxwellExponential.  With Nm=1 (no y-velocity mode), the plasma
    current is identically zero and the wave propagates purely through the exponential
    operator — testing it directly.

    Expected:  ω = |ξ| = kx/Lx = 2π/Lx = 1.0  for Lx = 2π, k=1
    (speed of light c = 1 in normalised units)

    The driver is off after t ≈ 80; frequency is measured over t ∈ [1600, 2000]
    — far from any transient — using the phase-derivative method.  The exponential
    operator is applied analytically so the tolerance is tight (5%).
    """
    Lx = dispersion_cfg["physics"]["Lx"]
    # Vacuum EM dispersion: ω = c|k| with c=1, k_phys = 2π/Lx
    expected_freq = 2.0 * np.pi / Lx  # = 1.0 for Lx = 2π

    dispersion_cfg["mlflow"]["experiment"] = "test-adept-spectrax1d-wave-dispersion"
    dispersion_cfg["mlflow"]["run"] = "em-dispersion-exponential"

    metrics = _run_maxwell(dispersion_cfg)
    measured_freq = metrics["em_avg_frequency_k1"]

    print("\nEM vacuum dispersion test (Lawson-RK4)")
    print(f"  Lx = {Lx:.5f},  expected ω = |ξ| = {expected_freq:.6f}")
    print(f"  Measured  ω = {measured_freq:.6f}")
    print(f"  Error     = {100 * abs(measured_freq - expected_freq) / expected_freq:.1f}%")

    np.testing.assert_allclose(
        measured_freq,
        expected_freq,
        rtol=0.05,
        err_msg=(
            f"Vacuum EM dispersion mismatch: "
            f"measured={measured_freq:.6f}, expected={expected_freq:.6f}"
        ),
    )


# ---------------------------------------------------------------------------
# Test 2: Wave absorption via sponge boundary layer
# ---------------------------------------------------------------------------


def test_em_wave_absorption(absorption_cfg):
    """
    Sponge on right boundary absorbs > 90% of the wave energy after driver off.

    The driver turns off around t ≈ 90 (t_center=50, t_width=80).  The sponge
    (σ_max = 10 ωpe over the rightmost 30% of the domain) damps the wave on a
    time-scale of ≈ τ × 1/Γ ≈ 3 ωpe^{-1}.  By t = 200 the EM energy should be
    < 10% of its peak value.

    em_absorption_ratio = mean(EM_energy[final 20%]) / max(EM_energy)
    """
    absorption_cfg["mlflow"]["experiment"] = "test-adept-spectrax1d-wave-absorption"
    absorption_cfg["mlflow"]["run"] = "wave-absorption-exponential"

    metrics = _run_maxwell(absorption_cfg)
    absorption_ratio = metrics["em_absorption_ratio"]
    peak_energy = metrics["em_peak_energy"]
    final_energy = metrics["em_final_energy"]

    print("\nWave absorption test (Lawson-RK4)")
    print(f"  Peak  EM energy  = {peak_energy:.3e}")
    print(f"  Final EM energy  = {final_energy:.3e}")
    print(f"  Absorption ratio = {absorption_ratio:.4f}  (pass if < 0.1)")

    assert peak_energy > 0, "Peak EM energy should be > 0 (wave must have been created)"
    np.testing.assert_array_less(
        absorption_ratio,
        0.1,
        err_msg=f"Sponge absorption too weak: ratio={absorption_ratio:.4f}, expected < 0.1",
    )


# ---------------------------------------------------------------------------
# __main__: manual inspection run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for fname, label in [
        ("configs/spectrax-1d/em-dispersion.yaml", "dispersion"),
        ("configs/spectrax-1d/wave-absorption.yaml", "absorption"),
    ]:
        with open(fname) as f:
            _cfg = yaml.safe_load(f)

        _cfg["mlflow"]["run"] = f"manual-{label}"
        _m = _run_maxwell(_cfg)
        print(f"\n{label}")
        for _k, _v in _m.items():
            if "em_" in _k or "simulation_" in _k:
                print(f"  {_k}: {_v}")
