"""
Verify the Chang-Cooper convention carefully.

Standard Chang-Cooper formula for FLUX at edge j+1/2:
F_{j+1/2} = [(1-δ)C + D/Δv] f_j + [δC - D/Δv] f_{j+1}

This implies:
  f_interp = (1-δ)f_j + δ f_{j+1}

Let's verify what our code produces and whether it matches.
"""

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp

from adept.driftdiffusion import ChangCooper, LenardBernstein, chang_cooper_delta


def analyze_coefficients():
    """Extract and analyze the coefficients from ChangCooper scheme."""

    nv = 64
    vmax = 6.0
    dv = 2.0 * vmax / nv
    v = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, nv)

    T = 1.0
    f = jnp.exp(-(v**2) / (2 * T))
    f = f / (jnp.sum(f) * dv)

    model = LenardBernstein(v=v, dv=dv)
    scheme = ChangCooper(dv=dv)

    # Get the model outputs
    C, D = model(f[None, :])
    C = C[0]  # shape (nv,)
    D = float(D[0])

    # Edge values
    C_edge = 0.5 * (C[1:] + C[:-1])  # shape (nv-1,)
    D_edge = D  # scalar for LB

    # Cell Peclet number (Chang-Cooper convention: w = C*Δv/D)
    w = C_edge * dv / D_edge
    delta = chang_cooper_delta(w)

    # Chang-Cooper flux: F = [δC - D/Δv]*f_j + [(1-δ)C + D/Δv]*f_{j+1}
    # For implicit solver (drift = -nu*C, diff = nu*D):
    #   alpha = drift*δ + diff/Δv
    #   beta = drift*(1-δ) - diff/Δv

    nu = 1.0
    drift = -nu * C_edge
    diff = nu * D_edge

    alpha = drift * delta + diff / dv
    beta = drift * (1.0 - delta) - diff / dv

    print("=" * 60)
    print("Analyzing Chang-Cooper coefficients")
    print("=" * 60)

    # Pick an edge in positive velocity region
    j = nv // 2 + 8

    print(f"\nAt edge {j}+1/2:")
    print(f"  v_edge = {float(0.5 * (v[j] + v[j + 1])):.4f}")
    print(f"  C_edge = {float(C_edge[j]):.4f}")
    print(f"  D = {D:.4f}")
    print(f"  dv = {dv:.4f}")
    print(f"  w = C*dv/D = {float(w[j]):.4f} (Chang-Cooper convention)")
    print(f"  δ = {float(delta[j]):.4f}")

    print(f"\nCode coefficients (with nu={nu}):")
    print(f"  alpha (coeff of f_j)   = {float(alpha[j]):.6f}")
    print(f"  beta  (coeff of f_{{j+1}}) = {float(beta[j]):.6f}")

    # Chang-Cooper formula coefficients (equation 16):
    # coeff(f_j)   = δC - D/Δv
    # coeff(f_j+1) = (1-δ)C + D/Δv
    std_coeff_fj = delta[j] * C_edge[j] - D_edge / dv
    std_coeff_fjp1 = (1 - delta[j]) * C_edge[j] + D_edge / dv

    print("\nStandard formula coefficients (physical flux):")
    print(f"  coeff(f_j)   = (1-δ)C + D/Δv = {float(std_coeff_fj):.6f}")
    print(f"  coeff(f_{{j+1}}) = δC - D/Δv   = {float(std_coeff_fjp1):.6f}")

    # The code has drift = -nu*C, so let's factor that out
    print("\nRelation to code:")
    print("  alpha = drift*(1-δ) + diff/dv = -nu*C*(1-δ) + nu*D/dv")
    print("        = nu * [D/dv - C*(1-δ)]")
    print("        = nu * [D/dv - C + C*δ]")
    print(f"  Numerical: {float(alpha[j]):.6f}")
    print(f"  Check:     {float(nu * (D_edge / dv - C_edge[j] + C_edge[j] * delta[j])):.6f}")

    print("\n  beta = drift*δ - diff/dv = -nu*C*δ - nu*D/dv")
    print("       = -nu * [C*δ + D/dv]")
    print(f"  Numerical: {float(beta[j]):.6f}")
    print(f"  Check:     {float(-nu * (C_edge[j] * delta[j] + D_edge / dv)):.6f}")

    # Now verify the flux is zero for Maxwellian
    f_j = f[j]
    f_jp1 = f[j + 1]

    print("\nFlux verification:")
    print(f"  f_j = {float(f_j):.6f}")
    print(f"  f_{{j+1}} = {float(f_jp1):.6f}")
    print(f"  ratio = {float(f_jp1 / f_j):.6f}")
    print(f"  exp(w) = {float(jnp.exp(w[j])):.6f}")

    flux_code = alpha[j] * f_j + beta[j] * f_jp1
    flux_std = std_coeff_fj * f_j + std_coeff_fjp1 * f_jp1

    print(f"\n  Flux from code: {float(flux_code):.2e}")
    print(f"  Flux from std:  {float(flux_std):.2e}")

    # Verify the interpolation
    f_interp_std = (1 - delta[j]) * f_j + delta[j] * f_jp1
    print("\nInterpolation check:")
    print(f"  f_interp = (1-δ)f_j + δ f_{{j+1}} = {float(f_interp_std):.6f}")

    # For physical flux F = C*f + D*df/dv, equilibrium F=0 means:
    # f_interp = -D*(f_{j+1}-f_j)/(C*dv)
    req_interp_physical = -D_edge * (f_jp1 - f_j) / (C_edge[j] * dv)
    print(f"  Required for F_physical=0: f_interp = {float(req_interp_physical):.6f}")

    # For user's formula F = C*f_interp - D*(f_{j+1}-f_j)/Δv, equilibrium F=0 means:
    # C*f_interp = D*(f_{j+1}-f_j)/Δv
    # f_interp = D*(f_{j+1}-f_j)/(C*dv)
    req_interp_user = D_edge * (f_jp1 - f_j) / (C_edge[j] * dv)
    print(f"  Required for F_user=0:     f_interp = {float(req_interp_user):.6f}")

    print("\n  Which matches the actual f_interp? ")
    print(f"    Physical formula: {'YES' if abs(f_interp_std - req_interp_physical) < 1e-10 else 'NO'}")
    print(f"    User's formula:   {'YES' if abs(f_interp_std - req_interp_user) < 1e-10 else 'NO'}")

    # Also verify both flux formulas
    df_dv = (f_jp1 - f_j) / dv
    F_physical = C_edge[j] * f_interp_std + D_edge * df_dv
    F_user = C_edge[j] * f_interp_std - D_edge * df_dv

    print("\nFlux values with f_interp = (1-δ)f_j + δ f_{j+1}:")
    print(f"  F_physical = C*f_interp + D*df/dv = {float(F_physical):.6e}")
    print(f"  F_user     = C*f_interp - D*df/dv = {float(F_user):.6e}")


if __name__ == "__main__":
    analyze_coefficients()
