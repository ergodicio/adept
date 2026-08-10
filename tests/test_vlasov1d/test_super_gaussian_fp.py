#  Copyright (c) Ergodic LLC 2026
#  research@ergodic.io
"""
Tests for the super-Gaussian-preserving Fokker-Planck operator (vlasov1d).

The SuperGaussianDougherty model relaxes toward f ∝ exp(-β·|v-vbar|^m) instead
of a Maxwellian, e.g. to maintain a Langdon/DLM inverse-bremsstrahlung-heated
distribution against collisional relaxation. These tests verify:

- a super-Gaussian of the configured order is a fixed point (no secular drift),
- a Maxwellian relaxes TO the super-Gaussian (kurtosis moves to the SG value),
- density is conserved exactly; the transient energy error is O(dv²),
- m=2 reduces to the Dougherty (Chang-Cooper) operator.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma

from adept._vlasov1d.solvers.pushers.fokker_planck import Collisions

VMAX = 6.0


class Grid(NamedTuple):
    v: np.ndarray
    dv: float


def make_grid(nv: int) -> Grid:
    dv = 2.0 * VMAX / nv
    return Grid(v=np.linspace(-VMAX + dv / 2.0, VMAX - dv / 2.0, nv), dv=dv)


GRID = make_grid(128)


# =============================================================================
# Helpers
# =============================================================================


def make_collisions(fp_type: str, m: float | None = None, sc_steps: int = 3, grid: Grid = GRID) -> Collisions:
    """Build a production Collisions instance for a single-x-point test grid."""
    fp_cfg = {
        "type": fp_type,
        "is_on": True,
        "self_consistent_beta": {"enabled": sc_steps > 0, "max_steps": sc_steps},
    }
    if m is not None:
        fp_cfg["m"] = m
    cfg = {
        "grid": {"species_grids": {"electron": {"v": grid.v, "dv": grid.dv}}},
        "terms": {"fokker_planck": fp_cfg, "krook": {"is_on": False}},
    }
    return Collisions(cfg)


def step_n(collisions: Collisions, f: jnp.ndarray, nsteps: int, nu: float = 1.0, dt: float = 0.1) -> jnp.ndarray:
    """Apply nsteps implicit collision steps to f of shape (nx, nv)."""
    nu_fp = jnp.full(f.shape[0], nu)
    nu_k = jnp.zeros(f.shape[0])

    @jax.jit
    def body(f_carry, _):
        return collisions(nu_fp, nu_k, f_carry, dt), None

    f_final, _ = jax.lax.scan(body, f, length=nsteps)
    return f_final


def supergaussian(m: float, T: float, v0: float = 0.0, grid: Grid = GRID) -> np.ndarray:
    """Unit-density super-Gaussian with variance T: f ∝ exp(-(|v-v0|/vm)^m)."""
    vm = np.sqrt(T * gamma(1.0 / m) / gamma(3.0 / m))
    f = np.exp(-(np.abs((grid.v - v0) / vm) ** m))
    return f / np.sum(f * grid.dv)


def maxwellian(T: float, v0: float = 0.0, grid: Grid = GRID) -> np.ndarray:
    return supergaussian(2.0, T, v0, grid)


def moments(f: np.ndarray, grid: Grid = GRID) -> tuple[float, float, float, float]:
    """Return (density, vbar, T, kurtosis) of f(v); kurtosis = ⟨(v-vbar)⁴⟩/T²."""
    f = np.asarray(f)
    n = np.sum(f * grid.dv)
    vbar = np.sum(f * grid.v * grid.dv) / n
    T = np.sum(f * (grid.v - vbar) ** 2 * grid.dv) / n
    kurt = np.sum(f * (grid.v - vbar) ** 4 * grid.dv) / n / T**2
    return n, vbar, T, kurt


def sg_kurtosis(m: float) -> float:
    """Analytic kurtosis ⟨v⁴⟩/⟨v²⟩² of a 1D super-Gaussian of order m."""
    return gamma(5.0 / m) * gamma(1.0 / m) / gamma(3.0 / m) ** 2


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("m", [3.0, 4.0])
def test_supergaussian_is_fixed_point(m):
    """A super-Gaussian of the configured order stays put; conservation holds."""
    collisions = make_collisions("super_gaussian", m=m)
    f0 = jnp.asarray(supergaussian(m, T=1.0))[None, :]
    f1 = step_n(collisions, f0, nsteps=100)  # 10 collision times

    n0, _, T0, _ = moments(f0[0])
    n1, _, T1, k1 = moments(f1[0])

    assert abs(n1 / n0 - 1.0) < 1e-12, f"density drift: {n1 / n0 - 1.0:.2e}"
    assert abs(T1 / T0 - 1.0) < 5e-3, f"temperature drift: {T1 / T0 - 1.0:.2e}"
    # Shape is maintained: kurtosis stays at the super-Gaussian value
    assert abs(k1 - sg_kurtosis(m)) < 0.02, f"kurtosis {k1:.4f} left SG value {sg_kurtosis(m):.4f}"
    # And f itself barely moves (analytic vs self-consistent discrete equilibrium)
    rel_l2 = float(jnp.linalg.norm(f1 - f0) / jnp.linalg.norm(f0))
    assert rel_l2 < 5e-3, f"super-Gaussian not a fixed point: rel L2 change {rel_l2:.2e}"
    assert float(jnp.min(f1)) > -1e-20, "positivity violated"


def test_no_secular_drift_at_equilibrium():
    """Once settled on the discrete equilibrium, a long run does not move it."""
    m = 3.0
    collisions = make_collisions("super_gaussian", m=m)
    f0 = jnp.asarray(supergaussian(m, T=1.0))[None, :]
    # Settle onto the self-consistent discrete equilibrium first
    f_eq = step_n(collisions, f0, nsteps=100)
    # Then run 10x longer: a secular drift would accumulate, a fixed point won't
    f_end = step_n(collisions, f_eq, nsteps=1000)

    rel_l2 = float(jnp.linalg.norm(f_end - f_eq) / jnp.linalg.norm(f_eq))
    assert rel_l2 < 1e-8, f"secular drift at equilibrium: rel L2 change {rel_l2:.2e} over 100 tau"
    _, _, T_eq, _ = moments(f_eq[0])
    _, _, T_end, _ = moments(f_end[0])
    assert abs(T_end / T_eq - 1.0) < 1e-8, f"secular T drift: {T_end / T_eq - 1.0:.2e}"


def test_dougherty_destroys_supergaussian_control():
    """Control: the standard Dougherty operator relaxes the same IC to Maxwellian."""
    collisions = make_collisions("chang_cooper_dougherty")
    f0 = jnp.asarray(supergaussian(3.0, T=1.0))[None, :]
    f1 = step_n(collisions, f0, nsteps=100)
    _, _, _, k1 = moments(f1[0])
    assert abs(k1 - 3.0) < 0.02, f"Dougherty control did not maxwellianize: kurtosis {k1:.4f}"


def test_maxwellian_relaxes_to_supergaussian():
    """A Maxwellian IC relaxes to the super-Gaussian shape; energy error is O(nu·dt).

    Transient energy conservation for m≠2 is limited by operator splitting
    (see SuperGaussianDougherty.compute_beta): β is frozen from f^n during the
    implicit step, so a full Maxwellian→SG conversion accumulates a one-time
    O(nu·dt) temperature offset. Verify the shape conversion and the
    first-order convergence of the offset in dt (nu·dt is small in production).
    """
    m = 3.0
    T_drift = {}
    for dt in (0.1, 0.05):
        collisions = make_collisions("super_gaussian", m=m)
        f0 = jnp.asarray(maxwellian(T=1.0))[None, :]
        f1 = step_n(collisions, f0, nsteps=round(20.0 / dt), dt=dt)  # 20 collision times

        n0, _, T0, _ = moments(f0[0])
        n1, _, T1, k1 = moments(f1[0])
        T_drift[dt] = abs(T1 / T0 - 1.0)

        assert abs(n1 / n0 - 1.0) < 1e-12
        assert abs(k1 - sg_kurtosis(m)) < 0.02, f"dt={dt}: kurtosis {k1:.4f} != SG target {sg_kurtosis(m):.4f}"

        # Compare against the analytic super-Gaussian with the realized (n, T)
        f_target = n1 * supergaussian(m, T=T1)
        rel_l2 = float(np.linalg.norm(np.asarray(f1[0]) - f_target) / np.linalg.norm(f_target))
        assert rel_l2 < 1e-2, f"dt={dt}: did not relax to super-Gaussian: rel L2 {rel_l2:.2e}"

    assert T_drift[0.1] < 2e-2, f"transient energy error too large: {T_drift[0.1]:.2e}"
    # O(nu·dt) convergence: halving dt should roughly halve the offset (allow 1.7x)
    assert T_drift[0.05] < T_drift[0.1] / 1.7, f"energy error not O(dt): {T_drift[0.1]:.2e} -> {T_drift[0.05]:.2e}"


def test_shifted_supergaussian_conserves_momentum():
    """A drifting IC keeps its momentum and relaxes to a shifted super-Gaussian."""
    m = 3.0
    v_shift = 1.5
    collisions = make_collisions("super_gaussian", m=m)
    f0 = jnp.asarray(maxwellian(T=0.5, v0=v_shift))[None, :]
    f1 = step_n(collisions, f0, nsteps=200)

    _, vbar0, T0, _ = moments(f0[0])
    _, vbar1, T1, k1 = moments(f1[0])

    assert abs(vbar1 - vbar0) < 5e-5, f"momentum drift: {vbar1 - vbar0:.2e}"
    assert abs(T1 / T0 - 1.0) < 2e-2, f"transient energy error too large: {T1 / T0 - 1.0:.2e}"
    assert abs(k1 - sg_kurtosis(m)) < 0.02, f"kurtosis {k1:.4f} != SG target {sg_kurtosis(m):.4f}"


def test_m2_reduces_to_dougherty():
    """super_gaussian with m=2 matches chang_cooper_dougherty step for step."""
    # sc_steps=0 so the Dougherty beta is exactly 1/(2·T_discrete), matching
    # the super-Gaussian energy-conserving closure n/(2·⟨(v-vbar)²⟩) at m=2
    sg = make_collisions("super_gaussian", m=2.0, sc_steps=0)
    dough = make_collisions("chang_cooper_dougherty", sc_steps=0)

    # Non-equilibrium two-temperature IC exercises the full operator
    f0 = 0.7 * maxwellian(T=0.5) + 0.3 * maxwellian(T=2.0, v0=0.5)
    f0 = jnp.asarray(f0)[None, :]

    f_sg = step_n(sg, f0, nsteps=10)
    f_dough = step_n(dough, f0, nsteps=10)

    np.testing.assert_allclose(np.asarray(f_sg), np.asarray(f_dough), rtol=1e-10, atol=1e-14)


def test_self_consistent_beta_supergaussian_shape():
    """find_self_consistent_beta(m=3) returns β whose discrete SG matches T of f."""
    from adept.driftdiffusion import discrete_temperature, find_self_consistent_beta

    m = 3.0
    f = jnp.asarray(0.6 * maxwellian(T=0.8) + 0.4 * supergaussian(4.0, T=1.5))[None, :]
    v = jnp.asarray(GRID.v)
    beta = find_self_consistent_beta(f, v, GRID.dv, spherical=False, m=m)

    f_sg = jnp.exp(-beta[:, None] * jnp.abs(v[None, :]) ** m)
    T_sg = discrete_temperature(f_sg, v, GRID.dv)
    T_f = discrete_temperature(f, v, GRID.dv)
    np.testing.assert_allclose(np.asarray(T_sg), np.asarray(T_f), rtol=1e-8)
