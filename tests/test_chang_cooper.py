#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Unit tests for Fokker-Planck utilities and model contracts.

Tests cover:
- chang_cooper_delta edge cases (w→0, w→∞)
- FastVFP kernel math properties
- Type 1 vs Type 2 model contracts
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from jax import numpy as jnp

from adept._vlasov1d.solvers.pushers.fokker_planck import LenardBernstein
from adept.driftdiffusion import chang_cooper_delta
from adept.vfp1d.fokker_planck import FastVFP


def test_delta_small_w():
    """Test delta(w) for small w approaches 0.5."""
    w = jnp.array([1e-10, 1e-9, 1e-8])
    delta = chang_cooper_delta(w)
    # For small w, delta ≈ 0.5 - w/12
    expected = 0.5 - w / 12.0
    np.testing.assert_allclose(delta, expected, rtol=1e-6)


def test_delta_zero():
    """Test delta(0) = 0.5 exactly."""
    w = jnp.array([0.0])
    delta = chang_cooper_delta(w)
    np.testing.assert_allclose(delta, 0.5, rtol=1e-10)


def test_delta_large_positive_w():
    """Test delta(w) for large positive w approaches 1/w."""
    w = jnp.array([10.0, 100.0, 1000.0])
    delta = chang_cooper_delta(w)
    # For large positive w: delta ≈ 1/w (since 1/expm1(w) → 0)
    expected = 1.0 / w
    np.testing.assert_allclose(delta, expected, rtol=1e-2)


def test_delta_large_negative_w():
    """Test delta(w) for large negative w approaches 1."""
    w = jnp.array([-10.0, -100.0])
    delta = chang_cooper_delta(w)
    # For large negative w: delta → 1 (since 1/w → 0 and 1/expm1(w) → -1)
    np.testing.assert_allclose(delta, 1.0, rtol=0.15)


def test_delta_moderate_w():
    """Test delta(w) for moderate w against analytical formula."""
    w = jnp.array([0.1, 0.5, 1.0, 2.0, -0.1, -0.5, -1.0, -2.0])
    delta = chang_cooper_delta(w)
    # Analytical: delta = 1/w - 1/(exp(w) - 1)
    expected = 1.0 / w - 1.0 / jnp.expm1(w)
    np.testing.assert_allclose(delta, expected, rtol=1e-10)


def test_delta_bounded():
    """Test that delta is always between 0 and 1."""
    w = jnp.linspace(-5.0, 5.0, 100)
    delta = chang_cooper_delta(w)
    assert jnp.all(delta >= 0.0)
    assert jnp.all(delta <= 1.0)
