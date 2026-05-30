"""Differentiable PWFA drive-beam optimization (Inc 4).

The headline result: backpropagating through the relativistic PIC solver, we
optimize the drive-beam longitudinal profile — represented by **per-particle
weights at fixed total charge** (`w = Q·softmax(θ)`) — to maximize the
transformer ratio. A symmetric driver is bounded by R ≤ 2; gradient-based
shaping finds an asymmetric (ramped) profile that decisively beats it,
reproducing the Bane–Chen–Wilson picture.

`test_transformer_ratio_gradient_flows` (fast) just checks the objective is
differentiable through the solver. The full optimization is marked ``slow``.
"""

import jax
import numpy as np
import optax
import pytest
from jax import numpy as jnp

from adept._empic1d.diagnostics import (
    face_positions,
    soft_transformer_ratio,
    to_comoving,
    transformer_ratio,
)
from adept._empic1d.solvers.pushers.field import charge_density_nodes, solve_ex_from_gauss
from adept._empic1d.solvers.vector_field import longitudinal_step

# Setup (matches the validated wake regime: c=5, v_b=0.98c, ω_pe=1)
C = 5.0
V_B = 4.9
GAMMA_B = 1.0 / np.sqrt(1.0 - (V_B / C) ** 2)
U_B = GAMMA_B * V_B
K_P = 1.0 / V_B
LAMBDA_P = 2.0 * np.pi / K_P

L = 160.0
NX = 320
DX = L / NX
DT = 0.05
SHAPE = "tsc"
N_STEPS = 120

X_B0 = 80.0
L_B = 30.0  # bunch length ~ λ_p ⇒ room for a ramp to build R > 2
N_B = 48
Q = 1.5  # total beam weight (fixed charge)
BEAM_MASS = 1000.0  # heavy ⇒ rigid driver
HALFWIDTH = 0.5 * L_B

# Static (θ-independent) plasma + params, built once.
_NP = NX * 50
_XP = jnp.array((np.arange(_NP) + 0.5) * (L / _NP))
_WP = jnp.array(np.full(_NP, L / _NP))
_UP = jnp.zeros((_NP, 3))
_XB = jnp.linspace(X_B0 - 0.5 * L_B, X_B0 + 0.5 * L_B, N_B)
_SPECIES_PARAMS = {
    "electron": {"charge": -1.0, "qm": -1.0},
    "beam": {"charge": -1.0, "qm": -1.0 / BEAM_MASS},
}
_BG = -float(jnp.mean(charge_density_nodes(_XP, _WP, -1.0, NX, DX, 0.0, SHAPE)))


def _weights(logits):
    return Q * jax.nn.softmax(logits)  # ⇒ Σw = Q exactly (fixed charge)


def _build_state(logits):
    wb = _weights(logits)
    ub = jnp.zeros((N_B, 3)).at[:, 0].set(U_B)
    species = {
        "electron": {"x": _XP, "u": _UP, "w": _WP},
        "beam": {"x": _XB, "u": ub, "w": wb},
    }
    rho = (
        charge_density_nodes(_XP, _WP, -1.0, NX, DX, 0.0, SHAPE)
        + charge_density_nodes(_XB, wb, -1.0, NX, DX, 0.0, SHAPE)
        + _BG
    )
    rho = rho - jnp.mean(rho)
    return {"species": species, "E": solve_ex_from_gauss(rho, DX)}


def _wake(logits):
    state = _build_state(logits)

    def scan_fn(s, _):
        nxt = longitudinal_step(
            s, species_params=_SPECIES_PARAMS, dt=DT, c=C, nx=NX, dx=DX, xmin=0.0, length=L, shape=SHAPE
        )
        return nxt, None

    final, _ = jax.lax.scan(scan_fn, state, None, length=N_STEPS)
    x_face = face_positions(NX, DX, 0.0)
    return to_comoving(x_face, final["E"], V_B, N_STEPS * DT, L, X_B0)


def _soft_objective(logits):
    xi, e = _wake(logits)
    return soft_transformer_ratio(xi, e, X_B0, HALFWIDTH, beta=40.0)


def _hard_ratio(logits):
    xi, e = _wake(logits)
    return float(transformer_ratio(xi, e, X_B0, HALFWIDTH))


def _gaussian_logits():
    return jnp.array(-((np.asarray(_XB) - X_B0) ** 2) / (2.0 * (0.25 * L_B) ** 2))


def _profile_centroid(logits):
    """Normalized first moment of the profile along the bunch; ~0 if symmetric."""
    w = np.asarray(_weights(logits))
    return float(np.sum(w * np.linspace(-1.0, 1.0, N_B)) / np.sum(w))


def test_transformer_ratio_gradient_flows():
    """The transformer ratio is differentiable through the PIC solver."""
    val, grad = jax.value_and_grad(_soft_objective)(_gaussian_logits())
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.linalg.norm(grad)) > 1e-6


@pytest.mark.slow
def test_pwfa_optimization_beats_symmetric_bound():
    logits = _gaussian_logits()
    r_baseline = _hard_ratio(logits)

    loss_and_grad = jax.jit(jax.value_and_grad(lambda lg: -_soft_objective(lg)))
    opt = optax.adam(0.1)
    opt_state = opt.init(logits)
    for _ in range(30):
        _, grad = loss_and_grad(logits)
        updates, opt_state = opt.update(grad, opt_state)
        logits = optax.apply_updates(logits, updates)

    r_opt = _hard_ratio(logits)

    # Symmetric driver ⇒ R ≤ 2; shaped driver beats it substantially.
    assert r_baseline < 2.0
    assert r_opt > 2.5
    assert r_opt > 1.8 * r_baseline
    # The optimum is an asymmetric (ramped) profile, not the symmetric baseline.
    assert abs(_profile_centroid(logits)) > 0.2
