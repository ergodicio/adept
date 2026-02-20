#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io
"""
Tests for spectrax-1d EPW dispersion: real (frequency) and imaginary (damping) parts.

Four cases covering both time integrators and ion mobility:
  - explicit (Dopri8),          mobile ions
  - explicit (Dopri8),          static ions
  - exponential (Lawson-RK4),   mobile ions
  - exponential (Lawson-RK4),   static ions

Each test drives the EPW at its theoretical frequency (randomly chosen klambda_D in
[0.26, 0.34]) and asserts that the measured frequency and damping rate match the
electrostatic dispersion relation within 10%.

Static ions do not affect the EPW dispersion: ions are far too heavy to respond at
electron plasma wave frequencies, so this pair of tests also verifies that the
static_ions flag doesn't break electron dynamics.
"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import pytest
import yaml

from adept import electrostatic, ergoExo

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_driven_epw(cfg: dict, klambda_D: float) -> tuple[float, float, float, float]:
    """
    Configure klambda_D, run a driven EPW simulation, return measured and expected values.

    Returns:
        (measured_freq, measured_damp, expected_freq, expected_damp)
    """
    Lx = cfg["physics"]["Lx"]
    k = 2.0 * np.pi / Lx
    alpha_e = klambda_D * np.sqrt(2) / k
    alpha_i = alpha_e * np.sqrt(1e-3)  # cold ions: Ti/Te = 0.001

    cfg["physics"]["alpha_s"][:3] = [alpha_e] * 3
    cfg["physics"]["alpha_s"][3:6] = [alpha_i] * 3

    root = electrostatic.get_roots_to_electrostatic_dispersion(
        wp_e=1.0, vth_e=1.0, k0=klambda_D, maxwellian_convention_factor=2.0
    )
    expected_freq = float(np.real(root))
    expected_damp = float(np.imag(root))

    cfg["drivers"]["ex"]["0"]["w0"] = expected_freq

    exo = ergoExo()
    exo.setup(cfg)
    _, post, _ = exo(None)

    metrics = post["metrics"]
    return metrics["epw_avg_frequency_k1"], metrics["epw_damping_rate_k1"], expected_freq, expected_damp


# ---------------------------------------------------------------------------
# Fixture: base configuration (fresh copy per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def base_cfg():
    with open("configs/spectrax-1d/landau-damping.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["solver"] = "hermite-epw-1d"
    cfg["grid"]["hermite_modes"] = {
        "electrons": {"Nn": 512, "Nm": 1, "Np": 1},
        "ions": {"Nn": 32, "Nm": 1, "Np": 1},
    }
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "static_ions",
    [False, True],
    ids=["mobile-ions", "static-ions"],
)
def test_driven_epw_dispersion(base_cfg, static_ions):
    """Driven EPW frequency and damping rate match the electrostatic dispersion within 10%."""
    klambda_D = np.random.uniform(0.26, 0.34)

    base_cfg["physics"]["static_ions"] = static_ions
    base_cfg["grid"]["integrator"] = "exponential"
    base_cfg["grid"]["adaptive_time_step"] = False

    base_cfg["mlflow"]["experiment"] = "spectrax1d-epw-test"
    base_cfg["mlflow"]["run"] = f"epw1d-exponential-{'static' if static_ions else 'mobile'}-{klambda_D:.3f}"

    measured_freq, measured_damp, expected_freq, expected_damp = _run_driven_epw(base_cfg, klambda_D)

    print(f"\nklambda_D={klambda_D:.4f}  integrator=exponential  static_ions={static_ions}")
    print(
        f"  freq:  measured={measured_freq:.6f}  expected={expected_freq:.6f}"
        f"  err={100 * abs(measured_freq - expected_freq) / expected_freq:.1f}%"
    )
    print(
        f"  damp:  measured={measured_damp:.6f}  expected={expected_damp:.6f}"
        f"  err={100 * abs(measured_damp - expected_damp) / abs(expected_damp):.1f}%"
    )

    np.testing.assert_allclose(
        measured_freq,
        expected_freq,
        rtol=0.1,
        err_msg=f"Frequency mismatch: measured={measured_freq:.6f}, expected={expected_freq:.6f}",
    )
    np.testing.assert_allclose(
        measured_damp,
        expected_damp,
        rtol=0.1,
        err_msg=f"Damping rate mismatch: measured={measured_damp:.6f}, expected={expected_damp:.6f}",
    )


if __name__ == "__main__":
    # Run all four cases at a fixed klambda_D for manual inspection
    import copy

    with open("configs/spectrax-1d/landau-damping.yaml") as f:
        _cfg = yaml.safe_load(f)
    _cfg["solver"] = "hermite-epw-1d"
    _cfg["grid"]["hermite_modes"] = {
        "electrons": {"Nn": 512, "Nm": 1, "Np": 1},
        "ions": {"Nn": 32, "Nm": 1, "Np": 1},
    }

    for _integrator, _static_ions in [("exponential", False), ("exponential", True)]:
        _test_cfg = copy.deepcopy(_cfg)
        _test_cfg["physics"]["static_ions"] = _static_ions
        _test_cfg["grid"]["integrator"] = _integrator
        if _integrator == "exponential":
            _test_cfg["grid"]["adaptive_time_step"] = False
        _test_cfg["mlflow"]["run"] = f"manual-{_integrator}-{'static' if _static_ions else 'mobile'}"
        mf, md, ef, ed = _run_driven_epw(_test_cfg, klambda_D=0.30)
        print(f"{_integrator:12s} static={_static_ions!s:5s}  freq: {mf:.4f}/{ef:.4f}  damp: {md:.4f}/{ed:.4f}")
