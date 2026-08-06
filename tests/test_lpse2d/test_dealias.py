"""
The TPD source is a product of the pump with a plasma-wave field, evaluated pointwise in real
space. That product aliases whenever it puts content past Nyquist, and the only thing standing
between the solver and that aliasing is the mask in ``cfg["grid"]["low_pass_filter_grid"]``.

These tests check the mask by computing the same product on a 2x zero-padded grid -- which is
alias free by construction -- and asserting the two agree. The negative control matters as much as
the positive one: with the mask opened up, the comparison has to *fail*, otherwise the test is
measuring nothing.
"""

import copy
import re

import numpy as np
import pytest
import yaml

from adept._lpse2d.helpers import (
    _pump_k_support,
    get_derived_quantities,
    get_solver_quantities,
    write_units,
)

CONFIG_PATH = "tests/test_lpse2d/configs/tpd.yaml"


def _build_grid(**grid_overrides) -> tuple[dict, dict]:
    with open(CONFIG_PATH) as fi:
        cfg = yaml.safe_load(fi)
    cfg["grid"].update(grid_overrides)

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    return cfg, cfg["grid"]


def _pad(arr_k: np.ndarray) -> np.ndarray:
    """Zero-pad a k-space array to twice the size along each axis."""
    nx, ny = arr_k.shape
    padded = np.pad(
        np.fft.fftshift(arr_k),
        ((nx // 2, nx - nx // 2), (ny // 2, ny - ny // 2)),
    )
    return np.fft.ifftshift(padded)


def _truncate(arr_k_pad: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Inverse of :func:`_pad` -- keep the modes that live on the unpadded grid."""
    shifted = np.fft.fftshift(arr_k_pad)
    x0, y0 = nx // 2, ny // 2
    return np.fft.ifftshift(shifted[x0 : x0 + nx, y0 : y0 + ny])


def _product_k_aliased(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``fft2(a * b)`` the way the solver does it -- pointwise on the native grid."""
    return np.fft.fft2(a * b)


def _product_k_exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``fft2(a * b)`` with the product formed on a 2x grid, so no triad can wrap."""
    nx, ny = a.shape
    a_pad = np.fft.ifft2(_pad(np.fft.fft2(a)))
    b_pad = np.fft.ifft2(_pad(np.fft.fft2(b)))
    return _truncate(np.fft.fft2(a_pad * b_pad) * 4.0, nx, ny)


def _tpd_product_discrepancy(cfg: dict, grid: dict) -> float:
    """
    Max relative difference between the native and alias-free TPD product, over the retained band.

    Builds a plasma-wave field filling the whole retained band and a pump that is a single snapped
    plane wave, exactly as ``laser.Light.laser_update`` does, then forms ``E0_y * conj(ey)``.
    """
    mask = np.asarray(grid["low_pass_filter_grid"])
    nx, ny, kx, ky = grid["nx"], grid["ny"], grid["kx"], grid["ky"]
    derived = cfg["units"]["derived"]

    # a plasma-wave field that populates every retained mode
    rng = np.random.default_rng(0)
    phi_k = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))) * mask
    ey = np.fft.ifft2(-1j * ky[None, :] * phi_k)

    # the pump: a plane wave along x, uniform in y, at the k0 that laser.py snaps to the grid
    k0 = derived["w0"] / derived["c"] * np.sqrt(1.0 - (derived["wp0"] / derived["w0"]) ** 2)
    dk = 2.0 * np.pi / (nx * grid["dx"])
    k0_snapped = np.round(k0 / dk) * dk
    E0_y = np.exp(1j * k0_snapped * grid["x"])[:, None] * np.ones(ny)[None, :]

    aliased = _product_k_aliased(E0_y, np.conj(ey))
    exact = _product_k_exact(E0_y, np.conj(ey))

    # only the retained band is ever read back into the state, so that is what has to be right
    scale = np.abs(exact[mask > 0]).max()
    return float(np.abs((aliased - exact)[mask > 0]).max() / scale)


def test_shifted_band_mask_is_alias_free():
    """With ``dealias: shifted-band`` the native product matches the padded one to round-off."""
    cfg, grid = _build_grid(dealias="shifted-band", low_pass_filter=1.0)
    assert _tpd_product_discrepancy(cfg, grid) < 1e-12


def test_unmasked_band_aliases():
    """Negative control: without the mask the same comparison has to disagree."""
    cfg, grid = _build_grid(dealias="isotropic", low_pass_filter=1.0)
    assert _tpd_product_discrepancy(cfg, grid) > 1e-2


def test_shifted_band_keeps_more_modes_than_the_isotropic_circle():
    """
    The point of the anisotropic mask: the pump translates along x only, so restricting ky the same
    way as kx throws away band for nothing.
    """
    _, isotropic = _build_grid(dealias="isotropic")
    _, shifted = _build_grid(dealias="shifted-band", low_pass_filter=1.0)

    retained_isotropic = np.mean(np.asarray(isotropic["low_pass_filter_grid"]) > 0)
    retained_shifted = np.mean(np.asarray(shifted["low_pass_filter_grid"]) > 0)

    assert retained_shifted > 2.0 * retained_isotropic


def test_default_is_unchanged():
    """Decks that do not opt in keep exactly the mask they had before."""
    _, without = _build_grid()
    _, explicit = _build_grid(dealias="isotropic")

    np.testing.assert_array_equal(
        np.asarray(without["low_pass_filter_grid"]),
        np.asarray(explicit["low_pass_filter_grid"]),
    )


def test_low_pass_filter_still_caps_the_band():
    """``low_pass_filter`` stays a physics knob and is honoured on top of the dealias mask."""
    _, grid = _build_grid(dealias="shifted-band", low_pass_filter=0.3)

    k_mag = np.sqrt(grid["kx"][:, None] ** 2 + grid["ky"][None, :] ** 2)
    retained = np.asarray(grid["low_pass_filter_grid"]) > 0
    assert k_mag[retained].max() <= 0.3 * np.abs(grid["kx"]).max()


def test_pump_support_widens_in_ky_with_speckle():
    """A speckle profile spreads the pump over the aperture, which eats into the ky margin."""
    with open(CONFIG_PATH) as fi:
        cfg = yaml.safe_load(fi)
    write_units(cfg)
    cfg = get_derived_quantities(cfg)

    kx_plain, ky_plain = _pump_k_support(cfg)
    assert ky_plain == 0.0

    speckled = copy.deepcopy(cfg)
    speckled["drivers"]["E0"]["speckle"] = {
        "enabled": True,
        "focal_length": "3.5m",
        "beam_aperture": ["0.35m", "0.35m"],
        "n_beamlets": [24, 32],
    }
    kx_speckle, ky_speckle = _pump_k_support(speckled)

    assert kx_speckle == pytest.approx(kx_plain)
    # numerical aperture is 0.35 / (2 * 3.5) = 0.05
    assert ky_speckle == pytest.approx(0.05 * kx_plain)


def test_unknown_dealias_mode_is_rejected():
    with pytest.raises(ValueError, match=re.escape("Unknown grid.dealias")):
        _build_grid(dealias="two-thirds")
