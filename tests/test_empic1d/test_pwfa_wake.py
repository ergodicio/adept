"""PWFA linear-wake validation.

A rigid, relativistic electron drive beam (a heavy species so it barely responds
to its own wake) propagates through cold plasma. The simulated longitudinal
wake is compared to 1D linear cold-plasma theory,

    ∂_ξ² E_z + k_p² E_z = -∂_ξ n_b  ⇒  E_z(ξ) = ∫_ξ^∞ n_b(ξ') cos(k_p(ξ'-ξ)) dξ',

with k_p = ω_pe / v_b. Units: equilibrium plasma density = 1 ⇒ ω_pe = 1.

The beam density profile is represented by **per-particle weights** on a fixed
particle grid (the linear-in-weights deposition that the Inc 4 optimization will
differentiate through). We also check the transformer ratio of the (symmetric)
Gaussian driver respects the R ≤ 2 bound.
"""

import numpy as np
from jax import numpy as jnp

from adept._empic1d.diagnostics import face_positions, to_comoving, transformer_ratio
from adept._empic1d.solvers.pushers.field import charge_density_nodes, solve_ex_from_gauss
from adept._empic1d.solvers.vector_field import longitudinal_step

# Setup
C = 5.0
V_B = 4.9  # beam velocity (0.98 c) ⇒ γ_b ≈ 5.0
GAMMA_B = 1.0 / np.sqrt(1.0 - (V_B / C) ** 2)
U_B = GAMMA_B * V_B
K_P = 1.0 / V_B  # ω_pe / v_b (ω_pe = 1)
LAMBDA_P = 2.0 * np.pi / K_P

L = 160.0
NX = 320
DX = L / NX
DT = 0.05
SHAPE = "tsc"

X_B0 = 100.0  # beam co-moving position
SIGMA_B = 3.0  # beam length (≪ λ_p ≈ 30.8)
N_B_PEAK = 0.05  # peak beam density / n_0  (linear regime)
BEAM_MASS = 1000.0  # heavy ⇒ rigid driver


def _gaussian_nb(xi):
    return N_B_PEAK * np.exp(-((xi - X_B0) ** 2) / (2.0 * SIGMA_B**2))


def _build_state():
    # Cold quiet background plasma, density 1.
    ppc = 200
    n_p = NX * ppc
    xp = (np.arange(n_p) + 0.5) * (L / n_p)
    wp = np.full(n_p, L / n_p)  # ⇒ density 1
    up = np.zeros((n_p, 3))

    # Drive beam: fixed particle grid over ±4σ, density set by per-particle weights.
    n_beam = 4000
    half = 4.0 * SIGMA_B
    xb = np.linspace(X_B0 - half, X_B0 + half, n_beam)
    spacing = xb[1] - xb[0]
    wb = _gaussian_nb(xb) * spacing  # weight ⇒ deposited density ≈ n_b(x)
    ub = np.zeros((n_beam, 3))
    ub[:, 0] = U_B

    species = {
        "electron": {"x": jnp.array(xp), "u": jnp.array(up), "w": jnp.array(wp)},
        "beam": {"x": jnp.array(xb), "u": jnp.array(ub), "w": jnp.array(wb)},
    }
    species_params = {
        "electron": {"charge": -1.0, "qm": -1.0},
        "beam": {"charge": -1.0, "qm": -1.0 / BEAM_MASS},
    }

    # Initial field from total charge (plasma + beam + neutralizing ion background).
    rho_e = charge_density_nodes(species["electron"]["x"], species["electron"]["w"], -1.0, NX, DX, 0.0, SHAPE)
    rho_b = charge_density_nodes(species["beam"]["x"], species["beam"]["w"], -1.0, NX, DX, 0.0, SHAPE)
    background = -float(jnp.mean(rho_e))  # uniform ions neutralize the plasma
    rho = rho_e + rho_b + background
    rho = rho - jnp.mean(rho)  # remove net (beam) charge offset for periodic solvability
    e_face = solve_ex_from_gauss(rho, DX)

    state = {"species": species, "E": e_face}
    params = dict(species_params=species_params, dt=DT, c=C, nx=NX, dx=DX, xmin=0.0, length=L, shape=SHAPE)
    return state, params


def _linear_wake_theory(xi):
    """E_z(ξ) = ∫_ξ^∞ n_b(ξ') cos(k_p(ξ'-ξ)) dξ' via reverse-cumulative integrals."""
    nb = _gaussian_nb(xi)
    dxi = xi[1] - xi[0]
    rev_c = np.cumsum((nb * np.cos(K_P * xi))[::-1])[::-1] * dxi
    rev_s = np.cumsum((nb * np.sin(K_P * xi))[::-1])[::-1] * dxi
    return np.cos(K_P * xi) * rev_c + np.sin(K_P * xi) * rev_s


def _simulate():
    import jax

    state, params = _build_state()
    t_final = 8.0
    n_steps = round(t_final / DT)

    def scan_fn(s, _):
        return longitudinal_step(s, **params), None

    final, _ = jax.lax.scan(scan_fn, state, None, length=n_steps)
    t = n_steps * DT

    x_face = face_positions(NX, DX, 0.0)
    xi, e_sim = to_comoving(x_face, final["E"], V_B, t, L, X_B0)
    return np.asarray(xi), np.asarray(e_sim), t


def test_pwfa_linear_wake_matches_theory():
    xi, e_sim, _ = _simulate()
    e_theory = _linear_wake_theory(xi)

    # Compare in the established wake a bit behind the driver (avoid the leading
    # transient and the periodic far edge).
    region = (xi > X_B0 - 1.5 * LAMBDA_P) & (xi < X_B0 - 2.0 * SIGMA_B)
    es, et = e_sim[region], e_theory[region]

    corr = np.corrcoef(es, et)[0, 1]
    amp_ratio = np.std(es) / np.std(et)
    assert corr > 0.85, f"wake shape correlation {corr:.3f}"
    assert 0.7 < amp_ratio < 1.4, f"wake amplitude ratio {amp_ratio:.3f}"


def test_symmetric_beam_transformer_ratio_bounded():
    xi, e_sim, _ = _simulate()
    r = float(transformer_ratio(jnp.array(xi), jnp.array(e_sim), X_B0, 2.0 * SIGMA_B))
    # A symmetric (Gaussian) driver cannot exceed R = 2.
    assert 0.3 < r < 2.1, f"transformer ratio {r:.3f}"
