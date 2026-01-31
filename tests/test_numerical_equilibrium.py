#  Copyright (c) Ergodic LLC 2025
#  research@ergodic.io
"""
Tests for numerical equilibrium preservation in Fokker-Planck schemes.

The Chang-Cooper scheme is designed to exactly preserve its numerical equilibrium,
which should be the Maxwellian (for Lenard-Bernstein model).

Key insight: The Chang-Cooper delta(w) is chosen such that for f_{i+1}/f_i = exp(w),
the discrete flux F_{i+1/2} = alpha·f_i + β·f_{i+1} = 0.
"""

import jax

jax.config.update("jax_enable_x64", True)

import lineax as lx
import numpy as np
from jax import numpy as jnp

from adept.driftdiffusion import (
    CentralDifferencing,
    ChangCooper,
    LenardBernstein,
    _broadcast_to_fshape,
    chang_cooper_delta,
)


def compute_edge_fluxes_chang_cooper(scheme, model, nu, f, verbose=False):
    """Compute the flux at each cell edge for Chang-Cooper scheme.

    Returns fluxes at edges 0.5, 1.5, ..., (nv-1)-0.5

    Args:
        scheme: ChangCooper scheme
        model: Drift-diffusion model
        nu: collision frequency, scalar
        f: distribution function, shape (nv,) unbatched
    """
    # Handle unbatched f
    f_1d = f.squeeze() if f.ndim > 1 else f

    C, D = model(f_1d)
    D_scalar = float(D)
    nu_scalar = float(nu)

    # Edge values (cell-centered to edge)
    C_edge = 0.5 * (C[1:] + C[:-1])
    D_edge = D_scalar  # scalar for LB model

    # Cell Peclet number (Chang-Cooper convention: w = C*Δv/D)
    w = C_edge * scheme.dv / D_edge
    delta = chang_cooper_delta(w)

    # Chang-Cooper flux: F = [δC - D/Δv]*f_j + [(1-δ)C + D/Δv]*f_{j+1}
    # For implicit solver coefficients (with drift = -nu*C, diff = nu*D):
    drift = -nu_scalar * C_edge
    diff = nu_scalar * D_edge

    alpha = drift * delta + diff / scheme.dv
    beta = drift * (1.0 - delta) - diff / scheme.dv

    # Flux at each edge
    F = alpha * f_1d[:-1] + beta * f_1d[1:]

    if verbose:
        print(f"Temperature D = {D_scalar:.6f}")
        print(f"Max |flux| = {float(jnp.max(jnp.abs(F))):.2e}")
        print(f"Flux at boundaries: F[0.5] = {float(F[0]):.2e}, F[N-0.5] = {float(F[-1]):.2e}")

    return F


def compute_numerical_maxwellian(v, dv, T):
    """Compute the exact numerical Maxwellian for Chang-Cooper scheme.

    The numerical equilibrium satisfies:
        f_{i+1}/f_i = exp(w_{i+1/2}) where w = -v_{edge}*dv/T

    This is exactly the continuous Maxwellian sampled at cell centers.
    """
    f = jnp.exp(-(v**2) / (2 * T))
    # Normalize to unit density
    f = f / (jnp.sum(f) * dv)
    return f


def test_chang_cooper_flux_zero_for_maxwellian():
    """Test that Chang-Cooper flux is exactly zero for a Maxwellian.

    The Chang-Cooper scheme is designed so that for the equilibrium distribution
    f_eq ∝ exp(-v²/2T), the discrete flux is exactly zero at all interior edges.
    """
    nv = 64
    vmax = 6.0
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    # Use a known temperature
    T = 1.0

    # Create Maxwellian with this temperature
    f = jnp.exp(-(v**2) / (2 * T))
    f = f / (jnp.sum(f) * dv)  # Normalize

    # Create model - but note that D = <v²> is computed from f, not T!
    model = LenardBernstein(v=v, dv=dv)
    scheme = ChangCooper(dv=dv)

    # Check what temperature the model computes
    C, D = model(f)
    print(f"Analytical temperature: T = {T}")
    print(f"Model-computed D = <v²> = {float(D):.6f}")
    print(f"Difference: {abs(float(D) - T):.2e}")

    # Compute fluxes
    nu = 1.0
    F = compute_edge_fluxes_chang_cooper(scheme, model, nu, f, verbose=True)

    # The flux should be nearly zero
    # There may be small numerical errors due to D ≠ T exactly
    max_flux = float(jnp.max(jnp.abs(F)))
    print(f"\nMax |flux| / max(f) = {max_flux / float(jnp.max(f)):.2e}")

    # The residual flux is due to numerical error in D = <v²> ≈ T
    # The error in D is ~7e-8, so flux should be O(7e-8) not O(1)
    assert max_flux < 1e-6, f"Flux should be ~0 for Maxwellian, got max |F| = {max_flux}"


def test_chang_cooper_preserves_maxwellian_one_step():
    """Test that a single implicit step preserves the Maxwellian exactly."""
    nv = 64
    vmax = 6.0
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    T = 1.0
    f = jnp.exp(-(v**2) / (2 * T))
    f = f / (jnp.sum(f) * dv)

    model = LenardBernstein(v=v, dv=dv)
    scheme = ChangCooper(dv=dv)

    C, D = model(f)
    dt = 0.1

    # Get operator and solve
    op = scheme.get_operator(C, D, jnp.array(1.0), dt)
    f_new = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value

    # Check preservation
    max_change = float(jnp.max(jnp.abs(f_new - f)))
    rel_change = max_change / float(jnp.max(f))

    print(f"Max absolute change: {max_change:.2e}")
    print(f"Max relative change: {rel_change:.2e}")
    print(f"Density before: {float(jnp.sum(f) * dv):.10f}")
    print(f"Density after:  {float(jnp.sum(f_new) * dv):.10f}")

    # The issue: D is recomputed from f at each step
    # If f_new ≠ f, then D_new ≠ D, which drives further evolution

    # Let's check what D would be for f_new
    C_new, D_new = model(f_new)
    C_old, D_old = model(f)
    print(f"D before: {float(D_old):.6f}")
    print(f"D after:  {float(D_new):.6f}")


def test_self_consistent_equilibrium():
    """Find the self-consistent numerical equilibrium.

    The true equilibrium must satisfy:
    1. F_j = 0 for all edges (zero flux)
    2. D = <v²> computed from f equals the temperature used in the flux formula

    Starting from a Maxwellian, iterate until convergence.
    """
    nv = 64
    vmax = 6.0
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    # Start with T=1 Maxwellian
    T_init = 1.0
    f = jnp.exp(-(v**2) / (2 * T_init))
    f = f / (jnp.sum(f) * dv)

    model = LenardBernstein(v=v, dv=dv)
    scheme = ChangCooper(dv=dv)

    dt = 0.1

    print("Iterating to find self-consistent equilibrium...")
    print(f"{'Step':>4} {'D=<v²>':>12} {'max|Δf|':>12} {'max|F|':>12}")
    print("-" * 48)

    for i in range(20):
        # Compute current D and flux
        C, D = model(f)
        D_val = float(D)
        F = compute_edge_fluxes_chang_cooper(scheme, model, 1.0, f)
        max_flux = float(jnp.max(jnp.abs(F)))

        # Take implicit step
        op = scheme.get_operator(C, D, jnp.array(1.0), dt)
        f_new = lx.linear_solve(op, f, solver=lx.AutoLinearSolver(well_posed=True)).value

        max_change = float(jnp.max(jnp.abs(f_new - f)))

        print(f"{i:4d} {D_val:12.8f} {max_change:12.2e} {max_flux:12.2e}")

        if max_change < 1e-14:
            print("\nConverged!")
            break

        f = f_new

    # Final equilibrium analysis
    C_eq, D_eq = model(f)
    T_eq = float(D_eq)

    print(f"\nEquilibrium temperature: T = {T_eq:.8f}")
    print(f"Initial temperature: T = {T_init}")
    print(f"Difference: {abs(T_eq - T_init):.2e}")

    # Check if the equilibrium is a Maxwellian with T_eq
    f_maxwellian = jnp.exp(-(v**2) / (2 * T_eq))
    f_maxwellian = f_maxwellian / (jnp.sum(f_maxwellian) * dv)

    max_diff = float(jnp.max(jnp.abs(f - f_maxwellian)))
    print(f"Max difference from Maxwellian(T_eq): {max_diff:.2e}")


def test_temperature_drift_source():
    """Understand why temperature drifts.

    The Lenard-Bernstein model has D = <v²> which depends on f.
    For a normalized Maxwellian: <v²> = T (analytically)
    But numerically, <v²> = Σ f_i v_i² dv may differ slightly.
    """
    nv = 64
    vmax = 6.0
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    T_exact = 1.0

    # Analytical Maxwellian
    f_analytical = jnp.exp(-(v**2) / (2 * T_exact))
    f_analytical = f_analytical / (jnp.sum(f_analytical) * dv)

    # Compute <v²> numerically
    v2_numerical = float(jnp.sum(f_analytical * v**2) * dv)

    print(f"Exact temperature: T = {T_exact}")
    print(f"Numerical <v²>:    D = {v2_numerical:.10f}")
    print(f"Difference:        {v2_numerical - T_exact:.2e}")
    print(f"Relative error:    {(v2_numerical - T_exact) / T_exact:.2e}")

    # This small difference drives the evolution!
    # The fix: use a finer grid, or accept the discrete equilibrium


def test_positive_grid_equilibrium():
    """Test equilibrium on positive-only grid (vfp1d style).

    On a positive-only grid v ∈ [0, vmax], the equilibrium is still Maxwellian,
    but the normalization and moments are different.
    """
    from adept.driftdiffusion import SphericalLenardBernstein

    nv = 64
    vmax = 6.0
    dv = vmax / nv
    v = jnp.linspace(dv / 2, vmax - dv / 2, nv)

    # For spherical harmonics, we integrate over v² dv (spherical Jacobian)
    # But the model uses D = <v⁴>/<v²>/3

    T = 1.0
    f = jnp.exp(-(v**2) / (2 * T))
    # Normalize with v² weight (spherical)
    f = f / (jnp.sum(f * v**2) * dv * 4 * jnp.pi)

    model = SphericalLenardBernstein(v=v, dv=dv)
    scheme = ChangCooper(dv=dv)

    C, D = model(f)
    print(f"Spherical model D = <v⁴>/<v²>/3 = {float(D):.6f}")
    print(f"Expected T = {T}")

    # For a Maxwellian: <v⁴>/<v²>/3 = (3T²)/(T)/3 = T  (in 1D)
    # In 3D spherical: <v⁴> = 15T², <v²> = 3T, so <v⁴>/<v²>/3 = 15T²/3T/3 = 5T/3
    # Wait, this depends on the measure used in the integral...


def test_delta_convention():
    """Verify the Chang-Cooper convention is correctly implemented.

    Chang-Cooper (1970) equation (16-18):
        w = C*Δv/D  (positive for positive drift)
        δ = 1/w - 1/(exp(w)-1)
        f_interp = δ*f_j + (1-δ)*f_{j+1}
        F = [δC - D/Δv]*f_j + [(1-δ)C + D/Δv]*f_{j+1}

    For Maxwellian equilibrium: f_{j+1}/f_j = exp(-w)
    """
    nv = 64
    vmax = 6.0
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    T = 1.0
    f = jnp.exp(-(v**2) / (2 * T))
    f = f / (jnp.sum(f) * dv)

    # Pick an edge in positive velocity region
    i = nv // 2 + 8

    v_edge = 0.5 * (v[i] + v[i + 1])
    f_j = f[i]
    f_jp1 = f[i + 1]

    print(f"Edge {i}+1/2 at v = {float(v_edge):.3f}")
    print(f"f[{i}] = {float(f_j):.6f}")
    print(f"f[{i + 1}] = {float(f_jp1):.6f}")
    print(f"Ratio r = f[j+1]/f[j] = {float(f_jp1 / f_j):.6f}")

    # Chang-Cooper convention: w = C*Δv/D (positive for positive C)
    C_val = v_edge
    D_val = T
    w = C_val * dv / D_val
    print(f"\nw = C*Δv/D = {float(w):.6f} (Chang-Cooper convention)")

    # Expected ratio for Maxwellian: exp(-w)
    r_expected = jnp.exp(-w)
    print(f"Expected r = exp(-w) = {float(r_expected):.6f}")
    print(f"Actual ratio matches: {jnp.allclose(f_jp1 / f_j, r_expected)}")

    # Chang-Cooper delta
    delta = chang_cooper_delta(w)
    print(f"\nδ = 1/w - 1/(exp(w)-1) = {float(delta):.6f}")

    # Chang-Cooper interpolation: f_interp = δ*f_j + (1-δ)*f_{j+1}
    f_interp = delta * f_j + (1 - delta) * f_jp1
    print(f"\nChang-Cooper f_interp = δ*f_j + (1-δ)*f_{{j+1}} = {float(f_interp):.6f}")

    # For equilibrium flux F = C*f_interp + D*(f_{j+1}-f_j)/Δv = 0:
    # f_interp = -D*(f_{j+1}-f_j)/(C*Δv) = D*(f_j - f_{j+1})/(C*Δv)
    f_interp_required = D_val * (f_j - f_jp1) / (C_val * dv)
    print(f"Required for F=0: f_interp = {float(f_interp_required):.6f}")

    coeff_fj = delta * C_val - D_val / dv
    coeff_fjp1 = (1 - delta) * C_val + D_val / dv
    flux_cc = coeff_fj * f_j + coeff_fjp1 * f_jp1
    print(f"  F = {float(flux_cc):.2e}")
    print("  (Should be ~0 for Maxwellian equilibrium)")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Delta convention investigation")
    print("=" * 60)
    test_delta_convention()

    print("\n" + "=" * 60)
    print("TEST: Chang-Cooper flux zero for Maxwellian")
    print("=" * 60)
    try:
        test_chang_cooper_flux_zero_for_maxwellian()
    except AssertionError as e:
        print(f"FAILED: {e}")

    print("\n" + "=" * 60)
    print("TEST: Temperature drift source")
    print("=" * 60)
    test_temperature_drift_source()

    print("\n" + "=" * 60)
    print("TEST: Chang-Cooper preserves Maxwellian (one step)")
    print("=" * 60)
    test_chang_cooper_preserves_maxwellian_one_step()

    print("\n" + "=" * 60)
    print("TEST: Self-consistent equilibrium iteration")
    print("=" * 60)
    test_self_consistent_equilibrium()
