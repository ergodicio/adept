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


class TestChangCooperDelta:
    """Tests for the Chang-Cooper weighting function."""

    def test_delta_small_w(self):
        """Test delta(w) for small w approaches 0.5."""
        w = jnp.array([1e-10, 1e-9, 1e-8])
        delta = chang_cooper_delta(w)
        # For small w, delta ≈ 0.5 - w/12
        expected = 0.5 - w / 12.0
        np.testing.assert_allclose(delta, expected, rtol=1e-6)

    def test_delta_zero(self):
        """Test delta(0) = 0.5 exactly."""
        w = jnp.array([0.0])
        delta = chang_cooper_delta(w)
        np.testing.assert_allclose(delta, 0.5, rtol=1e-10)

    def test_delta_large_positive_w(self):
        """Test delta(w) for large positive w approaches 1/w."""
        w = jnp.array([10.0, 100.0, 1000.0])
        delta = chang_cooper_delta(w)
        # For large positive w: delta ≈ 1/w (since 1/expm1(w) → 0)
        expected = 1.0 / w
        np.testing.assert_allclose(delta, expected, rtol=1e-2)

    def test_delta_large_negative_w(self):
        """Test delta(w) for large negative w approaches 1."""
        w = jnp.array([-10.0, -100.0])
        delta = chang_cooper_delta(w)
        # For large negative w: delta → 1 (since 1/w → 0 and 1/expm1(w) → -1)
        np.testing.assert_allclose(delta, 1.0, rtol=0.15)

    def test_delta_moderate_w(self):
        """Test delta(w) for moderate w against analytical formula."""
        w = jnp.array([0.1, 0.5, 1.0, 2.0, -0.1, -0.5, -1.0, -2.0])
        delta = chang_cooper_delta(w)
        # Analytical: delta = 1/w - 1/(exp(w) - 1)
        expected = 1.0 / w - 1.0 / jnp.expm1(w)
        np.testing.assert_allclose(delta, expected, rtol=1e-10)

    def test_delta_bounded(self):
        """Test that delta is always between 0 and 1."""
        w = jnp.linspace(-5.0, 5.0, 100)
        delta = chang_cooper_delta(w)
        assert jnp.all(delta >= 0.0)
        assert jnp.all(delta <= 1.0)


class TestFastVFP:
    """Tests for the FastVFP kernel-based model."""

    @pytest.fixture
    def model(self):
        """Create a FastVFP model with test grid."""
        nv = 64
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        return FastVFP(v=v, dv=dv)

    def test_fastvfp_kernel_rank1(self, model):
        """Test that FastVFP kernel is rank-1 (√ε·√ε')."""
        # For rank-1 kernel g(ε,ε') = √ε·√ε':
        # apply_kernel(u) = √ε · Σⱼ √εⱼ · uⱼ · Δεⱼ
        # If u = √ε, then apply_kernel(u) = √ε · ⟨ε⟩
        u = model.sqrt_eps_edge
        result = model.apply_kernel(u)
        expected_inner = 4.0 * jnp.pi * jnp.sum(model.sqrt_eps_edge**2 * model.d_eps_edge)
        expected = model.sqrt_eps_edge * expected_inner
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_fastvfp_ignores_beta(self, model):
        """Test that Type 2 models ignore beta."""
        f = jnp.exp(-(model.v**2) / 2.0)
        f = f / (jnp.sum(f) * model.dv)

        D1 = model.compute_D(f, beta=jnp.array(1.0))
        D2 = model.compute_D(f, beta=jnp.array(100.0))
        D3 = model.compute_D(f, beta=None)

        np.testing.assert_allclose(D1, D2)
        np.testing.assert_allclose(D1, D3)
        assert model.compute_vbar(f) is None

    def test_fastvfp_d_at_edges(self, model):
        """Test that FastVFP returns D at cell edges."""
        f = jnp.ones_like(model.v)
        f = f / (jnp.sum(f) * model.dv)

        D = model.compute_D(f)

        # D should have shape (nv-1,) for edge values
        assert D.shape == (len(model.v) - 1,)

    def test_fastvfp_kernel_produces_positive_d(self):
        """Test that FastVFP kernel produces positive D on positive grid."""
        nv = 64
        vmax = 6.0
        dv = vmax / nv
        v = jnp.linspace(dv / 2, vmax - dv / 2, nv)
        model = FastVFP(v=v, dv=dv)

        T = 1.0
        f = jnp.exp(-(v**2) / (2 * T))
        f = f / (jnp.sum(f) * dv)

        D = model.compute_D(f)

        assert jnp.all(D > 0), "FastVFP D should be positive on positive grid"


class TestTypeComparison:
    """Compare Type 1 (beta-based) and Type 2 (kernel-based) models."""

    @pytest.fixture
    def setup(self):
        """Create both model types with identical setup."""
        nv = 64
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        lb_model = LenardBernstein(v=v, dv=dv)
        fastvfp_model = FastVFP(v=v, dv=dv)
        f = jnp.exp(-(v**2) / 2.0)
        f = f / (jnp.sum(f) * dv)
        return lb_model, fastvfp_model, f, v, dv

    def test_type1_uses_beta(self, setup):
        """Test that Type 1 models use the provided beta (Buet notation)."""
        lb_model, _, f, v, dv = setup

        # With different betas (Buet notation: β = 1/(2T)), D = 1/(2β) = T
        D1 = lb_model.compute_D(f, beta=jnp.array(1.0))  # D = 0.5
        D2 = lb_model.compute_D(f, beta=jnp.array(2.0))  # D = 0.25

        np.testing.assert_allclose(D1, 0.5)
        np.testing.assert_allclose(D2, 0.25)

    def test_type2_ignores_beta(self, setup):
        """Test that Type 2 models ignore beta."""
        _, fastvfp_model, f, v, dv = setup

        D1 = fastvfp_model.compute_D(f, beta=jnp.array(1.0))
        D2 = fastvfp_model.compute_D(f, beta=jnp.array(2.0))
        D3 = fastvfp_model.compute_D(f, beta=None)

        np.testing.assert_allclose(D1, D2)
        np.testing.assert_allclose(D1, D3)

    def test_type1_scalar_d_type2_edge_d(self, setup):
        """Test that Type 1 returns scalar D while Type 2 returns edge D."""
        lb_model, fastvfp_model, f, v, dv = setup

        D1 = lb_model.compute_D(f, beta=jnp.array(1.0))
        D2 = fastvfp_model.compute_D(f)

        # Type 1: scalar D = 0.5 for beta=1
        assert D1.shape == ()
        np.testing.assert_allclose(D1, 0.5)

        # Type 2: D at edges
        assert D2.shape == (len(v) - 1,)
