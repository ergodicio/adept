#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Unit tests for shared Fokker-Planck utilities.

Tests cover:
- chang_cooper_delta edge cases (w→0, w→∞)
- LenardBernstein vs Dougherty outputs (C, D)
- CentralDifferencing vs ChangCooper operator assembly
- Zero-flux boundary condition behavior
"""

import jax

jax.config.update("jax_enable_x64", True)

import lineax as lx
import numpy as np
import pytest
from jax import numpy as jnp

from adept.driftdiffusion import (
    CentralDifferencing,
    ChangCooper,
    Dougherty,
    LenardBernstein,
    SphericalLenardBernstein,
    chang_cooper_delta,
)


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


class TestLenardBernstein:
    """Tests for the Lenard-Bernstein physics model."""

    @pytest.fixture
    def model(self):
        """Create a Lenard-Bernstein model with test grid."""
        nv = 64
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        return LenardBernstein(v=v, dv=dv)

    def test_drift_is_velocity_at_edges(self, model):
        """Test that drift coefficient C_edge = v_edge."""
        f = jnp.ones_like(model.v)
        C_edge, D = model(f)
        np.testing.assert_array_equal(C_edge, model.v_edge)

    def test_diffusion_is_energy(self, model):
        """Test that diffusion coefficient D = <v²>."""
        # Maxwellian: f = exp(-v²/2)
        f = jnp.exp(-(model.v**2) / 2.0)
        f = f / (jnp.sum(f) * model.dv)  # Normalize to unit density
        C_edge, D = model(f)
        # For unit Maxwellian, <v²> ≈ 1
        np.testing.assert_allclose(D, 1.0, rtol=0.05)

    def test_batch_dimensions(self, model):
        """Test model handles batch dimensions correctly."""
        nx = 10
        nv = len(model.v)
        f = jnp.ones((nx, nv))
        C_edge, D = model(f)
        # C_edge should have shape (nx, nv-1) for edge values
        assert C_edge.shape == (nx, nv - 1)
        # D should have batch dimension
        assert D.shape == (nx,)


class TestDougherty:
    """Tests for the Dougherty physics model."""

    @pytest.fixture
    def model(self):
        """Create a Dougherty model with test grid."""
        nv = 64
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        return Dougherty(v=v, dv=dv)

    def test_drift_centered_at_mean(self, model):
        """Test that drift coefficient C_edge = v_edge - <v>."""
        # Shifted Maxwellian: f = exp(-(v-v0)²/2)
        v0 = 1.5
        f = jnp.exp(-((model.v - v0) ** 2) / 2.0)
        f = f / (jnp.sum(f) * model.dv)
        C_edge, D = model(f)
        # <v> should be approximately v0
        vbar = jnp.sum(f * model.v) * model.dv
        np.testing.assert_allclose(vbar, v0, rtol=0.05)
        # C_edge should be v_edge - vbar
        expected_C_edge = model.v_edge - vbar
        np.testing.assert_allclose(C_edge, expected_C_edge, rtol=1e-10)

    def test_diffusion_is_thermal_velocity(self, model):
        """Test that diffusion coefficient D = <(v-<v>)²>."""
        # Maxwellian with T = 2: f = exp(-v²/4)
        T = 2.0
        f = jnp.exp(-(model.v**2) / (2.0 * T))
        f = f / (jnp.sum(f) * model.dv)
        C_edge, D = model(f)
        # For Maxwellian centered at 0, <(v-<v>)²> = <v²> ≈ T
        np.testing.assert_allclose(D, T, rtol=0.05)

    def test_batch_dimensions(self, model):
        """Test model handles batch dimensions correctly."""
        nx = 10
        nv = len(model.v)
        f = jnp.ones((nx, nv))
        f = f / (jnp.sum(f, axis=-1, keepdims=True) * model.dv)
        C_edge, D = model(f)
        # C_edge should have shape (nx, nv-1) for edge values
        assert C_edge.shape == (nx, nv - 1)
        # D should have batch dimension
        assert D.shape == (nx,)


class TestCentralDifferencing:
    """Tests for the central differencing scheme."""

    @pytest.fixture
    def setup(self):
        """Create a scheme with test grid and distribution."""
        nv = 32
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        model = LenardBernstein(v=v, dv=dv)
        scheme = CentralDifferencing(dv=dv)
        # Unit Maxwellian
        f = jnp.exp(-(v**2) / 2.0)
        f = f / (jnp.sum(f) * dv)
        return scheme, model, f, dv, nv

    def test_operator_returns_correct_type(self, setup):
        """Test that get_operator returns a TridiagonalLinearOperator."""
        scheme, model, f, dv, nv = setup
        C_edge, D = model(f)
        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt=0.1)
        assert isinstance(op, lx.TridiagonalLinearOperator)

    def test_diagonal_positive(self, setup):
        """Test that diagonal elements are positive for small dt."""
        scheme, model, f, dv, nv = setup
        C_edge, D = model(f)
        dt = 0.001
        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
        assert jnp.all(op.diagonal > 0)

    def test_solve_preserves_shape(self, setup):
        """Test that solving preserves input shape."""
        scheme, model, f, dv, nv = setup
        C_edge, D = model(f)
        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt=0.1)
        result = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value
        assert result.shape == f.shape


class TestChangCooper:
    """Tests for the Chang-Cooper scheme."""

    @pytest.fixture
    def setup(self):
        """Create a scheme with test grid and distribution."""
        nv = 32
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        model = LenardBernstein(v=v, dv=dv)
        scheme = ChangCooper(dv=dv)
        # Unit Maxwellian
        f = jnp.exp(-(v**2) / 2.0)
        f = f / (jnp.sum(f) * dv)
        return scheme, model, f, dv, nv, v

    def test_operator_returns_correct_type(self, setup):
        """Test that get_operator returns a TridiagonalLinearOperator."""
        scheme, model, f, dv, nv, v = setup
        C_edge, D = model(f)
        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt=0.1)
        assert isinstance(op, lx.TridiagonalLinearOperator)

    def test_diagonal_positive(self, setup):
        """Test that diagonal elements are positive."""
        scheme, model, f, dv, nv, v = setup
        C_edge, D = model(f)
        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt=0.1)
        assert jnp.all(op.diagonal > 0)

    def test_solve_preserves_shape(self, setup):
        """Test that solving preserves input shape."""
        scheme, model, f, dv, nv, v = setup
        C_edge, D = model(f)
        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt=0.1)
        result = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value
        assert result.shape == f.shape


class TestSchemeComparison:
    """Compare CentralDifferencing and ChangCooper schemes."""

    @pytest.fixture
    def setup(self):
        """Create both schemes with identical setup."""
        nv = 32
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        model = LenardBernstein(v=v, dv=dv)
        central = CentralDifferencing(dv=dv)
        chang_cooper = ChangCooper(dv=dv)
        f = jnp.exp(-(v**2) / 2.0)
        f = f / (jnp.sum(f) * dv)
        return central, chang_cooper, model, f, nv, dv

    def test_both_conserve_density(self, setup):
        """Test that both schemes conserve density.

        Note: CentralDifferencing uses a non-conservative discretization of
        the advection term (C*df/dv) rather than a flux-based discretization,
        so it doesn't conserve density exactly. ChangCooper uses proper flux
        coefficients and conserves to machine precision.
        """
        central, chang_cooper, model, f, nv, dv = setup
        C_edge, D = model(f)
        dt = 0.01
        initial_density = jnp.sum(f) * dv

        # CentralDifferencing: non-conservative advection, looser tolerance
        op = central.get_operator(C_edge, D, jnp.array(1.0), dt)
        result = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value
        final_density = jnp.sum(result) * dv
        np.testing.assert_allclose(final_density, initial_density, rtol=1e-6)

        # ChangCooper: flux-based, strictly conservative
        op = chang_cooper.get_operator(C_edge, D, jnp.array(1.0), dt)
        result = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value
        final_density = jnp.sum(result) * dv
        np.testing.assert_allclose(final_density, initial_density, rtol=1e-10)


class TestDoughertyWithSchemes:
    """Test schemes with the Dougherty model."""

    @pytest.fixture
    def setup(self):
        """Create schemes with Dougherty model."""
        nv = 32
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        model = Dougherty(v=v, dv=dv)
        central = CentralDifferencing(dv=dv)
        chang_cooper = ChangCooper(dv=dv)
        # Shifted Maxwellian
        v0 = 1.0
        f = jnp.exp(-((v - v0) ** 2) / 2.0)
        f = f / (jnp.sum(f) * dv)
        return central, chang_cooper, model, f, nv, dv, v0

    def test_dougherty_operator_works(self, setup):
        """Test that schemes work with Dougherty model."""
        central, chang_cooper, model, f, nv, dv, v0 = setup
        C_edge, D = model(f)
        dt = 0.01

        for scheme in [central, chang_cooper]:
            op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
            result = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value
            assert result.shape == f.shape
            assert jnp.all(jnp.isfinite(result))


class TestBoundaryConditionBehavior:
    """Tests for boundary condition behavior in solves."""

    @pytest.fixture
    def setup(self):
        """Create grid and distribution for BC tests."""
        nv = 64
        vmax = 6.0
        dv = 2.0 * vmax / nv
        v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)
        # Unit Maxwellian
        f = jnp.exp(-(v**2) / 2.0)
        f = f / (jnp.sum(f) * dv)
        return v, dv, nv, f

    def test_zero_flux_conserves_density(self, setup):
        """Test that zero_flux BC conserves total density.

        Uses ChangCooper scheme which has a proper flux-based discretization.
        CentralDifferencing doesn't conserve exactly due to non-conservative
        advection discretization (tested separately in TestSchemeComparison).
        """
        v, dv, nv, f = setup
        model = LenardBernstein(v=v, dv=dv)
        scheme = ChangCooper(dv=dv)  # Use conservative scheme for BC test

        C_edge, D = model(f)
        dt = 0.1
        initial_density = jnp.sum(f) * dv

        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
        result = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value

        final_density = jnp.sum(result) * dv
        np.testing.assert_allclose(final_density, initial_density, rtol=1e-10)

    def test_maxwellian_equilibrium_with_zero_flux(self, setup):
        """Test that Maxwellian relaxes toward equilibrium with zero_flux BC."""
        v, dv, nv, f = setup
        model = LenardBernstein(v=v, dv=dv)
        scheme = ChangCooper(dv=dv)

        dt = 0.1
        initial_density = jnp.sum(f) * dv

        # Run multiple steps
        f_current = f.copy()
        for _ in range(10):
            C_edge, D = model(f_current)
            op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
            f_current = lx.linear_solve(op, f_current, solver=lx.AutoLinearSolver(well_posed=True)).value

        # Density should be conserved
        final_density = jnp.sum(f_current) * dv
        np.testing.assert_allclose(final_density, initial_density, rtol=1e-10)

        # Distribution should remain reasonably close to Maxwellian shape
        center_slice = slice(nv // 4, 3 * nv // 4)
        np.testing.assert_allclose(f_current[center_slice], f[center_slice], rtol=0.15)


class TestPositiveOnlyGridBC:
    """Tests for positive-only velocity grids (vfp1d-style) with zero_flux BC."""

    @pytest.fixture
    def setup(self):
        """Create Chang-Cooper scheme on positive-only grid."""
        nv = 32
        vmax = 6.0
        dv = vmax / nv
        v_pos = jnp.linspace(dv / 2, vmax - dv / 2, nv)

        model = SphericalLenardBernstein(v=v_pos, dv=dv)
        scheme = ChangCooper(dv=dv)

        # Maxwellian on positive grid
        f_pos = jnp.exp(-(v_pos**2) / 2.0)
        f_pos = f_pos / (jnp.sum(f_pos) * dv)

        return scheme, model, f_pos, v_pos, dv, nv

    def test_positive_only_grid_conserves_density(self, setup):
        """Test that solving on positive-only grid with zero_flux BC conserves density."""
        scheme, model, f_pos, v_pos, dv, nv = setup

        initial_density = jnp.sum(f_pos) * dv

        C_edge, D = model(f_pos)
        dt = 0.1

        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
        result = lx.linear_solve(op, f_pos, solver=lx.AutoLinearSolver(well_posed=True)).value

        final_density = jnp.sum(result) * dv
        np.testing.assert_allclose(final_density, initial_density, rtol=1e-10)

    def test_positive_only_grid_preserves_positivity(self, setup):
        """Test that Chang-Cooper on positive-only grid preserves positivity."""
        scheme, model, f_pos, v_pos, dv, nv = setup

        # Start with a bi-Maxwellian (non-equilibrium)
        f_bimax = 0.7 * jnp.exp(-(v_pos**2) / 2.0) + 0.3 * jnp.exp(-((v_pos - 2.0) ** 2) / 2.0)
        f_bimax = f_bimax / (jnp.sum(f_bimax) * dv)

        dt = 0.1

        f_current = f_bimax.copy()
        for _ in range(10):
            C_edge, D = model(f_current)
            op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
            f_current = lx.linear_solve(op, f_current, solver=lx.AutoLinearSolver(well_posed=True)).value

        assert jnp.all(f_current >= 0), "Chang-Cooper should preserve positivity"


class TestReflectedGridBC:
    """Tests for boundary conditions on reflected grids (legacy vfp1d-style)."""

    @pytest.fixture
    def setup(self):
        """Create Chang-Cooper scheme on reflected grid."""
        nv = 32  # Points on positive-only grid
        vmax = 6.0
        dv = vmax / nv  # Positive-only grid spacing
        v_pos = jnp.linspace(dv / 2, vmax - dv / 2, nv)  # Positive-only, cell-centered

        # Create reflected grid [-vmax, +vmax]
        refl_v = jnp.concatenate([-v_pos[::-1], v_pos])

        model = SphericalLenardBernstein(v=refl_v, dv=dv)
        scheme = ChangCooper(dv=dv)

        # Maxwellian on positive grid
        f_pos = jnp.exp(-(v_pos**2) / 2.0)
        f_pos = f_pos / (jnp.sum(f_pos) * dv)

        return scheme, model, f_pos, v_pos, refl_v, dv, nv

    def test_reflected_solve_conserves_density(self, setup):
        """Test that solving on reflected grid conserves total density."""
        scheme, model, f_pos, v_pos, refl_v, dv, nv = setup

        # Reflect f across v=0
        refl_f = jnp.concatenate([f_pos[::-1], f_pos])

        # Compute initial density on full grid
        initial_density = jnp.sum(refl_f) * dv

        C_edge, D = model(refl_f)
        dt = 0.1

        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
        result = lx.linear_solve(op, refl_f, solver=lx.AutoLinearSolver(well_posed=True)).value

        final_density = jnp.sum(result) * dv
        np.testing.assert_allclose(final_density, initial_density, rtol=1e-10)

    def test_reflected_solve_preserves_positivity(self, setup):
        """Test that Chang-Cooper on reflected grid preserves positivity."""
        scheme, model, f_pos, v_pos, refl_v, dv, nv = setup

        # Start with a bi-Maxwellian (non-equilibrium)
        f_bimax = 0.7 * jnp.exp(-(v_pos**2) / 2.0) + 0.3 * jnp.exp(-((v_pos - 2.0) ** 2) / 2.0)
        f_bimax = f_bimax / (jnp.sum(f_bimax) * dv)

        # Reflect
        refl_f = jnp.concatenate([f_bimax[::-1], f_bimax])

        dt = 0.1

        for _ in range(10):
            C_edge, D = model(refl_f)
            op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
            refl_f = lx.linear_solve(op, refl_f, solver=lx.AutoLinearSolver(well_posed=True)).value

        # Extract positive half and check positivity
        f_pos_final = refl_f[nv:]
        assert jnp.all(f_pos_final >= 0), "Chang-Cooper should preserve positivity"

    def test_symmetry_preserved_on_reflected_grid(self, setup):
        """Test that symmetric initial condition remains symmetric."""
        scheme, model, f_pos, v_pos, refl_v, dv, nv = setup

        # Reflect
        refl_f = jnp.concatenate([f_pos[::-1], f_pos])

        C_edge, D = model(refl_f)
        dt = 0.1

        op = scheme.get_operator(C_edge, D, jnp.array(1.0), dt)
        result = lx.linear_solve(op, refl_f, solver=lx.AutoLinearSolver(well_posed=True)).value

        # Check symmetry: f(-v) = f(v)
        f_neg = result[:nv][::-1]
        f_pos_result = result[nv:]
        np.testing.assert_allclose(f_neg, f_pos_result, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
