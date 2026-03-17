# Copyright (c) Ergodic LLC 2023
# research@ergodic.io
"""Test EM/EP diagnostic split for Vlasov-1D.

Validates that the EM wave decomposition correctly separates rightgoing (EP)
and leftgoing (EM) components using:
    EP = E_y + c*B_z  (rightgoing)
    EM = E_y - c*B_z  (leftgoing)

For a purely rightgoing wave in vacuum, EM should be nearly zero. In a plasma
with dispersion ω² = ωpe² + c²k², we have ω ≠ ck, which causes some deviation.
For low-density plasma (0.01*n_crit), the leakage ratio is typically ~3%.
"""

import numpy as np
import yaml

from adept import ergoExo


def _load_base_config() -> dict:
    """Load the base config for em/ep tests."""
    with open("tests/test_vlasov1d/configs/em_ep_split.yaml") as f:
        return yaml.safe_load(f)


def _make_driver(
    intensity: str = "1e12W/m^2",
    wavelength: str = "0.5um",
    leftgoing: bool = False,
    space_center: str = "2.0um",
) -> dict:
    """Create a driver configuration."""
    return {
        "params": {
            "intensity": intensity,
            "wavelength": wavelength,
            "leftgoing": leftgoing,
        },
        "envelope": {
            "time": {"center": "1ps", "rise": "0.05ps", "width": "1.9ps"},
            "space": {"center": space_center, "rise": "1.0um", "width": "3.0um"},
        },
    }


def _run_simulation(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run simulation and return (ep, em) arrays."""
    exo = ergoExo()
    exo.setup(config)
    result, datasets, run_id = exo(None)

    fields_data = datasets["fields"]["fields"]
    ep = fields_data["fields-ep"].values
    em = fields_data["fields-em"].values

    return ep, em


def _compute_rms(field: np.ndarray, skip_fraction: float = 0.25) -> float:
    """Compute RMS of field, skipping early transients."""
    t_start_idx = int(len(field) * skip_fraction)
    return float(np.sqrt(np.mean(field[t_start_idx:] ** 2)))


def test_em_ep_split_rightgoing():
    """Verify that a rightgoing driver produces ep >> em.

    Uses a nearly uniform density profile at ~1% of critical density.
    The EM/EP ratio should be small (< 5%) due to plasma dispersion effects.
    """
    config = _load_base_config()
    config["drivers"]["ey"] = {"0": _make_driver(leftgoing=False)}

    ep, em = _run_simulation(config)
    ep_rms = _compute_rms(ep)
    em_rms = _compute_rms(em)
    ratio = em_rms / ep_rms

    print("\nEM/EP Split Test (Rightgoing)")
    print(f"  EP RMS: {ep_rms:.6e}")
    print(f"  EM RMS: {em_rms:.6e}")
    print(f"  EM/EP ratio: {ratio:.4f}")

    assert ratio < 0.05, f"EM/EP ratio {ratio:.4f} exceeds 0.05 for rightgoing wave"


def test_em_ep_split_leftgoing():
    """Verify that a leftgoing driver produces em >> ep.

    The EM/EP ratio should be large (> 15) since most energy is in EM.
    """
    config = _load_base_config()
    # Place driver on the right side for leftgoing wave
    config["drivers"]["ey"] = {"0": _make_driver(leftgoing=True, space_center="18.0um")}

    ep, em = _run_simulation(config)
    ep_rms = _compute_rms(ep)
    em_rms = _compute_rms(em)
    ratio = em_rms / ep_rms

    print("\nEM/EP Split Test (Leftgoing)")
    print(f"  EP RMS: {ep_rms:.6e}")
    print(f"  EM RMS: {em_rms:.6e}")
    print(f"  EM/EP ratio: {ratio:.4f}")

    # For leftgoing wave, em should dominate (ratio >> 1)
    assert ratio > 15, f"EM/EP ratio {ratio:.4f} should exceed 15 for leftgoing wave"


def test_em_ep_split_mixed_waves():
    """Verify correct separation with both rightgoing and leftgoing waves.

    Uses rightgoing wave at 10x intensity of leftgoing wave.
    Should see EP/EM ratio close to sqrt(10) ≈ 3.16 (amplitude ratio).
    """
    config = _load_base_config()
    config["drivers"]["ey"] = {
        "0": _make_driver(
            intensity="1e13W/m^2",  # 10x stronger
            leftgoing=False,
            space_center="2.0um",
        ),
        "1": _make_driver(
            intensity="1e12W/m^2",
            leftgoing=True,
            space_center="18.0um",
        ),
    }

    ep, em = _run_simulation(config)
    ep_rms = _compute_rms(ep)
    em_rms = _compute_rms(em)
    ratio = ep_rms / em_rms  # Note: EP/EM ratio for this test

    print("\nEM/EP Split Test (Mixed: 10x rightgoing + 1x leftgoing)")
    print(f"  EP RMS: {ep_rms:.6e}")
    print(f"  EM RMS: {em_rms:.6e}")
    print(f"  EP/EM ratio: {ratio:.4f} (expected ~3.16 = sqrt(10))")

    # EP should be ~sqrt(10) ≈ 3.16 times EM (amplitude scales as sqrt of intensity)
    # Allow 50% tolerance due to plasma dispersion effects
    expected_ratio = np.sqrt(10)
    assert 0.5 * expected_ratio < ratio < 1.5 * expected_ratio, (
        f"EP/EM ratio {ratio:.4f} not close to expected {expected_ratio:.2f}"
    )


if __name__ == "__main__":
    test_em_ep_split_rightgoing()
    test_em_ep_split_leftgoing()
    test_em_ep_split_mixed_waves()
